"""
Clustering pipeline for WikiInsight Engine topic explorer.

This module:
- Loads embeddings (and optionally cleaned articles) produced by
  `src.preprocessing.process_data`.
- Fits a clustering model (e.g., KMeans) according to `config.yaml`.
- Builds a k-NN index for similar-article lookup.
- Generates cluster summaries:
    * representative articles (closest to cluster center)
    * **topic words** using c-TF-IDF (class-based TF-IDF) with tokenization + stopword filtering
      so that the keywords are both frequent in the cluster (normalized TF) and
      distinctive compared to other clusters (high IDF based on class frequency).
- Saves artifacts under `models/clustering/` and a metrics JSON for DVC/CI.
"""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from time import perf_counter
from typing import Dict, List, Tuple, Set

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans  # noqa: F401
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.common.logging_utils import setup_logging
from src.common.pipeline_progress import update_progress, mark_stage_completed, mark_stage_error
from src.common.gpu_utils import (
    is_cuda_available,
    get_clustering_backend,
    log_device_info,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
EMBEDDINGS_PATH = os.path.join("data", "features", "embeddings.parquet")
CLEANED_ARTICLES_PATH = os.path.join("data", "processed", "cleaned_articles.parquet")

MODEL_DIR = os.path.join("models", "clustering")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
NN_INDEX_PATH = os.path.join(MODEL_DIR, "nn_index.pkl")
CLUSTER_ASSIGNMENTS_PATH = os.path.join(MODEL_DIR, "cluster_assignments.parquet")
CLUSTERS_SUMMARY_PATH = os.path.join(MODEL_DIR, "clusters_summary.parquet")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Tokenization / stopwords for keyword extraction
# ---------------------------------------------------------------------------

# A lightweight, hard-coded English stop word list. We keep this even when
# spaCy is available so behaviour is stable and easy to reason about.
STOP_WORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "of",
    "to",
    "as",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "you",
    "we",
    "i",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

try:  # spaCy is optional; we fall back to regex tokenization if unavailable
    import spacy
    import subprocess
    import sys

    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        # Model not found, try to download it automatically
        logger.info("spaCy model 'en_core_web_sm' not found. Attempting to download...")
        
        # Try method 1: spacy download command
        try:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _NLP = spacy.load("en_core_web_sm")
            logger.info("Successfully downloaded and loaded spaCy model 'en_core_web_sm'")
        except Exception:  # noqa: BLE001
            # Try method 2: pip install directly (fallback)
            try:
                logger.info("spacy download failed, trying pip install...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _NLP = spacy.load("en_core_web_sm")
                logger.info("Successfully installed and loaded spaCy model 'en_core_web_sm' via pip")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to download spaCy model 'en_core_web_sm': %s. "
                    "Falling back to simple regex tokenization for keywords. "
                    "You can install it manually with: pip install en_core_web_sm",
                    exc,
                )
                _NLP = None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "spaCy model 'en_core_web_sm' not available: %s. "
            "Falling back to simple regex tokenization for keywords.",
            exc,
        )
        _NLP = None
except Exception:  # noqa: BLE001
    _NLP = None


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text into normalized word tokens with stopword and noise filtering.

    For consistency with the wider preprocessing stack, this function mirrors
    the behaviour of the NLTK-based normalizer used before embeddings. We keep
    the spaCy path for environments where it is already configured, but fall
    back to a simple regex-based tokenizer when neither spaCy nor NLTK helpers
    are available.
    """
    text = text or ""

    # Prefer spaCy if available: better tokenization and stopword detection.
    if _NLP is not None:
        doc = _NLP(text)
        tokens: List[str] = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            lemma = tok.lemma_.lower().strip()
            if len(lemma) < 3:
                continue
            if lemma in STOP_WORDS or tok.is_stop:
                continue
            tokens.append(lemma)
        return tokens

    # Fallback: simple regex-based tokenization aligned with our NLTK pipeline.
    tokens = []
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0)
        if len(token) < 3:
            continue
        if token in STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_embeddings() -> Tuple[pd.DataFrame, np.ndarray]:
    if not os.path.exists(EMBEDDINGS_PATH):
        logger.error("Embeddings file not found at %s", EMBEDDINGS_PATH)
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    df = pd.read_parquet(EMBEDDINGS_PATH)
    if "embedding" not in df.columns:
        raise ValueError("Embeddings parquet must contain an 'embedding' column")
    embeddings = np.vstack(df["embedding"].to_list())
    return df, embeddings


def load_cleaned_articles() -> pd.DataFrame:
    if not os.path.exists(CLEANED_ARTICLES_PATH):
        logger.error("Cleaned articles file not found at %s", CLEANED_ARTICLES_PATH)
        raise FileNotFoundError(f"Cleaned articles file not found at {CLEANED_ARTICLES_PATH}")
    return pd.read_parquet(CLEANED_ARTICLES_PATH)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings using L2 norm.
    
    L2 normalization improves clustering quality by:
    - Making distances scale-invariant
    - Improving silhouette score and DBI metrics
    - Better convergence for k-means
    
    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        
    Returns:
        Normalized embeddings array (n_samples, n_features)
    """
    logger.info("Normalizing embeddings (L2 norm)...")
    logger.info("  - Input shape: %s", embeddings.shape)
    normalized = normalize(embeddings, norm='l2', axis=1)
    logger.info("  - Normalization complete")
    return normalized


def find_optimal_n_clusters_adaptive(
    embeddings: np.ndarray,
    max_time: int = 60,
    random_state: int = 42,
) -> Tuple[int, float]:
    """
    Find optimal number of clusters using adaptive strategy based on dataset size.
    
    Strategies:
    - Small (<200): Test all reasonable k values (5-15)
    - Medium (200-500): Binary search (tests 5-8 k values)
    - Large (>500): Coarse-to-fine search (tests 7-12 k values)
    
    Uses silhouette score with adaptive sampling for efficiency.
    Time-limited to max_time seconds.
    
    Args:
        embeddings: Normalized embeddings array (n_samples, n_features)
        max_time: Maximum time in seconds for optimization
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal_k, best_silhouette_score)
    """
    n_samples = embeddings.shape[0]
    start_time = perf_counter()
    
    logger.info("=" * 80)
    logger.info("Finding optimal n_clusters (adaptive strategy)")
    logger.info("  - Dataset size: %d samples", n_samples)
    logger.info("  - Max optimization time: %d seconds", max_time)
    logger.info("=" * 80)
    
    # Determine strategy based on dataset size
    if n_samples < 200:
        # Small dataset: test all reasonable k values
        min_k = max(2, n_samples // 20)  # At least 2, but reasonable minimum
        max_k = min(15, n_samples // 3)  # At most 15, but not more than n_samples/3
        k_values = list(range(min_k, max_k + 1))
        strategy = "all_values"
        sample_size = min(100, n_samples)  # Small sample for small datasets
    elif n_samples < 500:
        # Medium dataset: binary search
        min_k = max(5, n_samples // 30)
        max_k = min(30, n_samples // 5)
        k_values = []  # Will be populated by binary search
        strategy = "binary_search"
        sample_size = min(300, n_samples)
    else:
        # Large dataset: coarse-to-fine search
        min_k = max(10, n_samples // 50)
        max_k = min(50, n_samples // 10)
        k_values = []  # Will be populated by coarse-to-fine
        strategy = "coarse_to_fine"
        sample_size = min(500, n_samples)
    
    logger.info("  - Strategy: %s", strategy)
    logger.info("  - K range: %d to %d", min_k, max_k)
    logger.info("  - Sample size for metrics: %d", sample_size)
    
    # Adaptive sampling for silhouette score
    if sample_size < n_samples:
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
    else:
        sample_embeddings = embeddings
        sample_indices = None
    
    def evaluate_k(k: int) -> float:
        """Evaluate silhouette score for a given k."""
        if (perf_counter() - start_time) > max_time:
            return -1.0  # Timeout
        
        try:
            # Use Agglomerative for small datasets, KMeans for larger
            if n_samples < 200:
                clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
            else:
                clusterer = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    n_init=3,  # Fewer runs for speed
                    random_state=random_state,
                    max_iter=100,  # Fewer iterations for speed
                )
            
            labels = clusterer.fit_predict(sample_embeddings)
            
            # Check if we have enough clusters
            unique_labels = len(np.unique(labels))
            if unique_labels < 2:
                return -1.0  # Invalid clustering
            
            score = silhouette_score(sample_embeddings, labels)
            return float(score)
        except Exception as e:
            logger.debug("Error evaluating k=%d: %s", k, e)
            return -1.0
    
    best_k = min_k
    best_score = -1.0
    
    if strategy == "all_values":
        # Test all k values
        logger.info("  - Testing all k values: %s", k_values)
        for k in k_values:
            if (perf_counter() - start_time) > max_time:
                logger.warning("  - Time limit reached, stopping optimization")
                break
            
            score = evaluate_k(k)
            logger.info("    k=%d: silhouette=%.4f", k, score)
            
            if score > best_score:
                best_score = score
                best_k = k
    
    elif strategy == "binary_search":
        # Binary search for optimal k
        logger.info("  - Using binary search")
        left, right = min_k, max_k
        tested_k = set()
        
        while (perf_counter() - start_time) < max_time and right - left > 2:
            mid = (left + right) // 2
            mid_left = (left + mid) // 2
            mid_right = (mid + right) // 2
            
            for k in [mid_left, mid, mid_right]:
                if k in tested_k or k < min_k or k > max_k:
                    continue
                tested_k.add(k)
                
                score = evaluate_k(k)
                logger.info("    k=%d: silhouette=%.4f", k, score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Narrow search range based on scores
            scores = {k: evaluate_k(k) for k in [mid_left, mid, mid_right] if k not in tested_k}
            if scores:
                best_in_range = max(scores.items(), key=lambda x: x[1])
                if best_in_range[1] > best_score:
                    best_score = best_in_range[1]
                    best_k = best_in_range[0]
                
                # Narrow to the better half
                if scores[mid_left] > scores[mid_right]:
                    right = mid
                else:
                    left = mid
    
    else:  # coarse_to_fine
        # Coarse-to-fine search
        logger.info("  - Using coarse-to-fine search")
        
        # Coarse phase: test wide range
        coarse_k = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 8)))
        logger.info("  - Coarse phase: testing %s", coarse_k)
        
        coarse_scores = {}
        for k in coarse_k:
            if (perf_counter() - start_time) > max_time * 0.6:  # Use 60% of time for coarse
                break
            
            score = evaluate_k(k)
            coarse_scores[k] = score
            logger.info("    k=%d: silhouette=%.4f", k, score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Fine phase: search around best coarse k
        if coarse_scores and (perf_counter() - start_time) < max_time:
            best_coarse_k = max(coarse_scores.items(), key=lambda x: x[1])[0]
            fine_min = max(min_k, best_coarse_k - 3)
            fine_max = min(max_k, best_coarse_k + 3)
            fine_k = [k for k in range(fine_min, fine_max + 1) if k not in coarse_k]
            
            if fine_k:
                logger.info("  - Fine phase: testing around k=%d: %s", best_coarse_k, fine_k)
                for k in fine_k:
                    if (perf_counter() - start_time) > max_time:
                        break
                    
                    score = evaluate_k(k)
                    logger.info("    k=%d: silhouette=%.4f", k, score)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
    
    elapsed_time = perf_counter() - start_time
    logger.info("=" * 80)
    logger.info("Optimal n_clusters found: k=%d (silhouette=%.4f)", best_k, best_score)
    logger.info("  - Optimization time: %.2f seconds", elapsed_time)
    logger.info("=" * 80)
    
    return best_k, best_score


def make_adaptive_clusterer(
    embeddings: np.ndarray,
    cfg: Dict,
    use_gpu: bool = False,
) -> Tuple[object, np.ndarray, np.ndarray, Dict]:
    """
    Create and fit a clustering model with adaptive algorithm selection and optimization.
    
    Features:
    - Normalizes embeddings (L2 norm) for better metrics
    - Auto-optimizes n_clusters if enabled
    - Selects algorithm based on dataset size:
      * <200: Agglomerative (best quality for small datasets)
      * 200-500: KMeans with k-means++ (CPU)
      * >500: KMeans with k-means++ (GPU if available)
    - Uses k-means++ initialization for better convergence
    
    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        cfg: Clustering configuration dictionary
        use_gpu: Whether to use GPU if available
        
    Returns:
        Tuple of (model, labels, embeddings_used, metadata_dict)
        - model: Fitted clustering model
        - labels: Cluster assignments
        - embeddings_used: Embeddings actually used for clustering (may be normalized)
        - metadata_dict: Contains algorithm, n_clusters, normalization_applied, optimization_time
    """
    n_samples = embeddings.shape[0]
    random_state = int(cfg.get("random_state", 42))
    auto_n_clusters = cfg.get("auto_n_clusters", True)
    max_optimization_time = int(cfg.get("preprocessing", {}).get("max_optimization_time", 60))
    normalize_emb = cfg.get("preprocessing", {}).get("normalize_embeddings", True)
    
    metadata = {
        "normalization_applied": False,
        "algorithm": None,
        "n_clusters": None,
        "optimization_time": 0.0,
        "optimization_applied": False,
    }
    
    # Step 1: Normalize embeddings
    if normalize_emb:
        embeddings = normalize_embeddings(embeddings)
        metadata["normalization_applied"] = True
    
    # Step 2: Auto-optimize n_clusters if enabled
    if auto_n_clusters:
        logger.info("Auto-optimizing n_clusters...")
        opt_start = perf_counter()
        optimal_k, best_score = find_optimal_n_clusters_adaptive(
            embeddings,
            max_time=max_optimization_time,
            random_state=random_state,
        )
        opt_time = perf_counter() - opt_start
        n_clusters = optimal_k
        metadata["optimization_applied"] = True
        metadata["optimization_time"] = opt_time
        metadata["best_silhouette_score"] = best_score
        logger.info("Selected optimal n_clusters: %d (silhouette=%.4f)", n_clusters, best_score)
    else:
        # Use configured n_clusters
        requested_clusters = int(cfg.get("n_clusters", 100))
        n_clusters = min(requested_clusters, n_samples)  # Clamp to available samples
        if n_clusters != requested_clusters:
            logger.warning("Adjusted n_clusters from %d to %d (clamped to available samples)", 
                          requested_clusters, n_clusters)
    
    # Step 3: Select algorithm based on dataset size
    backend = get_clustering_backend()
    gpu_threshold = 2000  # GPU only beneficial for 2000+ samples
    
    if n_samples < 200:
        # Small dataset: use Agglomerative (best quality)
        algorithm = "agglomerative"
        logger.info("Selected algorithm: AgglomerativeClustering (best quality for small datasets)")
        logger.info("Fitting AgglomerativeClustering (CPU) with n_clusters=%d", n_clusters)
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = agg.fit_predict(embeddings)
        
        # Compute cluster centers
        centers: List[np.ndarray] = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                centers.append(np.zeros(embeddings.shape[1], dtype=embeddings.dtype))
            else:
                centers.append(embeddings[mask].mean(axis=0))
        centers_arr = np.vstack(centers)
        
        model = AgglomerativeWrapper(centers_arr, labels, n_clusters)
        
    elif n_samples < 500:
        # Medium dataset: use KMeans with k-means++ (CPU)
        algorithm = "kmeans"
        logger.info("Selected algorithm: KMeans with k-means++ (CPU)")
        logger.info("Fitting KMeans (CPU) with n_clusters=%d, init=k-means++", n_clusters)
        
        model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,  # Multiple runs, pick best
            random_state=random_state,
            max_iter=300,
        )
        labels = model.fit_predict(embeddings)
        
    else:
        # Large dataset: use KMeans with k-means++ (GPU if available)
        use_gpu_clustering = use_gpu and backend == "pytorch" and n_samples >= gpu_threshold
        
        if use_gpu_clustering:
            algorithm = "kmeans_gpu"
            logger.info("Selected algorithm: KMeans with k-means++ (GPU)")
            logger.info("Fitting KMeans (GPU/CuPy) with n_clusters=%d", n_clusters)
            
            try:
                import cupy as cp
                
                # Transfer to GPU
                gpu_embeddings = cp.asarray(embeddings.astype(np.float32))
                cp.cuda.Stream.null.synchronize()
                
                # k-means++ initialization on GPU
                cp.random.seed(random_state)
                n_features = gpu_embeddings.shape[1]
                
                # Initialize first centroid randomly
                centroids = cp.zeros((n_clusters, n_features), dtype=cp.float32)
                first_idx = cp.random.randint(0, n_samples)
                centroids[0] = gpu_embeddings[first_idx].copy()
                
                # k-means++: select remaining centroids based on distance to nearest centroid
                for k in range(1, n_clusters):
                    # Compute distances from all points to nearest centroid
                    distances = cp.linalg.norm(
                        gpu_embeddings[:, cp.newaxis, :] - centroids[:k, cp.newaxis, :],
                        axis=2
                    )
                    min_distances = cp.min(distances, axis=1)
                    
                    # Select next centroid with probability proportional to distance^2
                    probabilities = min_distances ** 2
                    probabilities = probabilities / cp.sum(probabilities)
                    
                    # Sample next centroid
                    next_idx = cp.random.choice(n_samples, p=cp.asnumpy(probabilities))
                    centroids[k] = gpu_embeddings[next_idx].copy()
                
                cp.cuda.Stream.null.synchronize()
                
                # KMeans iterations
                max_iter = 300
                labels = cp.zeros(n_samples, dtype=cp.int32)
                
                for iteration in range(max_iter):
                    # Compute distances and assign labels
                    distances = cp.linalg.norm(
                        gpu_embeddings[:, cp.newaxis, :] - centroids[cp.newaxis, :, :],
                        axis=2
                    )
                    new_labels = cp.argmin(distances, axis=1)
                    
                    # Check convergence
                    converged = cp.all(new_labels == labels)
                    cp.cuda.Stream.null.synchronize()
                    if converged:
                        logger.info("  - Converged at iteration %d", iteration + 1)
                        break
                    
                    labels = new_labels
                    
                    # Update centroids
                    for k in range(n_clusters):
                        mask = labels == k
                        if cp.any(mask):
                            centroids[k] = gpu_embeddings[mask].mean(axis=0)
                    
                    cp.cuda.Stream.null.synchronize()
                    
                    if (iteration + 1) % 50 == 0:
                        logger.info("  - Iteration %d/%d", iteration + 1, max_iter)
                
                # Transfer back to CPU
                labels_cpu = cp.asnumpy(labels)
                centers_cpu = cp.asnumpy(centroids)
                cp.cuda.Stream.null.synchronize()
                
                model = PyTorchGPUKMeansWrapper(centers_cpu, labels_cpu, n_clusters)
                labels = labels_cpu
                
            except (ImportError, Exception) as e:
                logger.warning("GPU clustering failed, falling back to CPU: %s", e)
                use_gpu_clustering = False
        
        if not use_gpu_clustering:
            # Fallback to CPU KMeans
            algorithm = "kmeans"
            logger.info("Selected algorithm: KMeans with k-means++ (CPU)")
            logger.info("Fitting KMeans (CPU) with n_clusters=%d, init=k-means++", n_clusters)
            
            model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                random_state=random_state,
                max_iter=300,
            )
            labels = model.fit_predict(embeddings)
    
    metadata["algorithm"] = algorithm
    metadata["n_clusters"] = n_clusters
    
    logger.info("Adaptive clustering complete:")
    logger.info("  - Algorithm: %s", algorithm)
    logger.info("  - N clusters: %d", n_clusters)
    logger.info("  - Normalization: %s", metadata["normalization_applied"])
    logger.info("  - Optimization: %s", metadata["optimization_applied"])
    
    return model, labels, embeddings, metadata


class AgglomerativeWrapper:
    """
    Wrapper for AgglomerativeClustering that exposes cluster_centers_ attribute.
    
    This is needed because AgglomerativeClustering doesn't expose explicit cluster
    centers, but downstream code expects a `.cluster_centers_` attribute.
    
    NOTE: Must be defined at module level (not inside a function) so it can be
    pickled by joblib when saving the model.
    """

    def __init__(self, centers: np.ndarray, labels: np.ndarray, n_clusters: int):
        """
        Initialize wrapper with computed cluster centers.
        
        Args:
            centers: Cluster centers array (n_clusters, n_features)
            labels: Cluster labels for each sample
            n_clusters: Number of clusters
        """
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters = n_clusters
        self.method = "agglomerative"


class PyTorchGPUKMeansWrapper:
    """
    Wrapper for PyTorch/CuPy GPU KMeans to provide sklearn-compatible interface.
    
    NOTE: Must be defined at module level (not inside a function) so it can be
    pickled by joblib when saving the model.
    """
    
    def __init__(self, centers: np.ndarray, labels: np.ndarray, n_clusters: int):
        """
        Initialize wrapper with cluster centers and labels from GPU KMeans model.
        
        Args:
            centers: Cluster centers array (n_clusters, n_features)
            labels: Cluster labels for each sample
            n_clusters: Number of clusters
        """
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters = n_clusters
        self.method = "kmeans"


class PyTorchGPUNNWrapper:
    """
    Wrapper for PyTorch/CuPy GPU NearestNeighbors to provide sklearn-compatible interface.
    
    NOTE: Must be defined at module level (not inside a function) so it can be
    pickled by joblib when saving the model.
    """
    
    def __init__(self, embeddings_cpu: np.ndarray, n_neighbors: int = 10):
        """
        Initialize wrapper with CuPy-based GPU search.
        
        Args:
            embeddings_cpu: CPU copy of embeddings for fallback
            n_neighbors: Number of neighbors to return
        """
        self._embeddings_cpu = embeddings_cpu
        self.n_neighbors = n_neighbors
    
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """
        Query nearest neighbors using CuPy GPU brute-force search.
        Falls back to sklearn if GPU not available.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        # Try CuPy GPU brute-force search
        try:
            import cupy as cp
            
            X_gpu = cp.asarray(X.astype(np.float32))
            embeddings_gpu = cp.asarray(self._embeddings_cpu.astype(np.float32))
            
            # Brute-force nearest neighbors using CuPy
            # Compute pairwise distances
            distances_gpu = cp.linalg.norm(
                embeddings_gpu[:, cp.newaxis, :] - X_gpu[cp.newaxis, :, :], 
                axis=2
            )
            
            # Get top-k nearest neighbors
            indices_gpu = cp.argsort(distances_gpu, axis=0)[:n_neighbors, :].T
            distances_sorted = cp.sort(distances_gpu, axis=0)[:n_neighbors, :].T
            
            # Convert back to numpy
            indices = cp.asnumpy(indices_gpu)
            distances = cp.asnumpy(distances_sorted) if return_distance else None
            
            return (distances, indices) if return_distance else indices
        except (ImportError, Exception) as e:
            # Fallback to sklearn if GPU not available
            logger.warning("GPU not available for NN query, falling back to sklearn: %s", e)
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(self._embeddings_cpu)
            return nn.kneighbors(X, n_neighbors=n_neighbors, return_distance=return_distance)
    
    def fit(self, X):
        """Fit method for compatibility."""
        pass


def make_clusterer(embeddings: np.ndarray, cfg: Dict, use_gpu: bool = False):
    """
    Create and fit a clustering model.
    
    Uses adaptive clustering if enabled in config, otherwise uses traditional clustering.
    Uses GPU (PyTorch/CuPy) if available and use_gpu=True, otherwise falls back to CPU (sklearn).
    
    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        cfg: Clustering configuration dictionary
        use_gpu: Whether to use GPU if available
        
    Returns:
        Tuple of (model, labels, embeddings_used)
        - model: Fitted clustering model
        - labels: Cluster assignments
        - embeddings_used: Embeddings actually used for clustering (may be normalized)
        If adaptive mode, metadata is logged but not returned
    """
    # Check if adaptive mode is enabled
    adaptive = cfg.get("adaptive", True)  # Default to True for better metrics
    method = (cfg.get("method") or "kmeans").lower()
    
    # If method is "auto" or adaptive is True, use adaptive clustering
    if adaptive or method == "auto":
        logger.info("=" * 80)
        logger.info("Using adaptive clustering mode")
        logger.info("=" * 80)
        model, labels, embeddings_used, metadata = make_adaptive_clusterer(embeddings, cfg, use_gpu=use_gpu)
        return model, labels, embeddings_used
    
    # Traditional clustering mode (backward compatible)
    logger.info("Using traditional clustering mode (backward compatible)")
    requested_clusters = int(cfg.get("n_clusters", 100))
    random_state = int(cfg.get("random_state", 42))
    
    backend = get_clustering_backend()
    
    # GPU is only beneficial for large datasets (2000+ samples)
    # For smaller datasets, CPU is faster due to transfer overhead
    # and sklearn's highly optimized C++ implementation
    n_samples = embeddings.shape[0]
    gpu_threshold = 2000  # Only use GPU for datasets with 2000+ samples
    
    # Clamp n_clusters to available samples to avoid ValueError
    # KMeans requires n_samples >= n_clusters
    if n_samples < requested_clusters:
        n_clusters = max(1, n_samples)
        logger.warning("=" * 80)
        logger.warning("CLUSTERING CONFIGURATION ADJUSTMENT:")
        logger.warning("  - Requested clusters: %d", requested_clusters)
        logger.warning("  - Available samples: %d", n_samples)
        logger.warning("  - Adjusted clusters: %d (clamped to available samples)", n_clusters)
        logger.warning("  - Reason: KMeans requires n_samples >= n_clusters")
        logger.warning("  - Recommendation: Reduce n_clusters in config.yaml or increase max_articles")
        logger.warning("=" * 80)
    else:
        n_clusters = requested_clusters
    
    use_gpu_clustering = use_gpu and backend == "pytorch" and n_samples >= gpu_threshold
    
    if use_gpu and backend == "pytorch" and n_samples < gpu_threshold:
        logger.info("  - GPU available but dataset too small (%d < %d samples)", n_samples, gpu_threshold)
        logger.info("  - Using CPU for better performance (GPU overhead not worth it)")
    
    logger.info("Clustering configuration:")
    logger.info("  - Method: %s", method)
    logger.info("  - Requested clusters: %d", requested_clusters)
    logger.info("  - Actual clusters: %d", n_clusters)
    logger.info("  - N samples: %d", n_samples)
    logger.info("  - Random state: %d", random_state)
    logger.info("  - GPU requested: %s", use_gpu)
    logger.info("  - Backend available: %s", backend)
    logger.info("  - Will use GPU: %s", use_gpu_clustering)
    
    # Try PyTorch/CuPy GPU clustering
    if use_gpu_clustering and backend == "pytorch":
            logger.info("=" * 80)
            logger.info("Using GPU clustering (PyTorch/CuPy - Windows compatible)")
            logger.info("=" * 80)
            try:
                import cupy as cp
                
                # Get device ID (default to 0, or use CUDA_VISIBLE_DEVICES if set)
                try:
                    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if cuda_visible:
                        # CUDA_VISIBLE_DEVICES remaps devices, so device 0 is the first visible
                        device_id = 0
                    else:
                        device_id = 0
                except Exception:
                    device_id = 0
                
                # Use explicit device context for all GPU operations
                with cp.cuda.Device(device_id):
                    # Verify CuPy is using CUDA (not NumPy fallback)
                    try:
                        current_device = cp.cuda.Device().id
                        device_name = cp.cuda.runtime.getDeviceProperties(current_device)['name'].decode('utf-8')
                        logger.info("  - CuPy CUDA device: %s (ID: %d)", device_name, current_device)
                    except Exception as e:
                        logger.warning("  - Could not get CUDA device info: %s", e)
                        logger.warning("  - CuPy might be using CPU fallback!")
                    
                    # Verify GPU memory pool
                    try:
                        mempool = cp.get_default_memory_pool()
                        mempool_used = mempool.used_bytes() / (1024**3)
                        mempool_total = mempool.total_bytes() / (1024**3)
                        logger.info("  - GPU memory pool: %.2f GB used / %.2f GB total", mempool_used, mempool_total)
                    except Exception as e:
                        logger.debug("Could not get GPU memory info: %s", e)
                    
                    logger.info("Transferring embeddings to GPU...")
                    logger.info("  - Input shape: %s", embeddings.shape)
                    logger.info("  - Input dtype: %s", embeddings.dtype)
                    transfer_start = perf_counter()
                    gpu_embeddings = cp.asarray(embeddings.astype(np.float32))
                    # Synchronize to ensure transfer completes and is visible to Task Manager
                    cp.cuda.Stream.null.synchronize()
                    transfer_time = perf_counter() - transfer_start
                    logger.info("  - Transferred to GPU in %.3f seconds", transfer_time)
                    
                    # Verify the array is actually on GPU
                    try:
                        is_gpu_array = isinstance(gpu_embeddings, cp.ndarray)
                        logger.info("  - Array on GPU: %s", is_gpu_array)
                        if not is_gpu_array:
                            logger.warning("  - WARNING: Array might not be on GPU!")
                    except Exception as e:
                        logger.debug("Could not verify GPU array: %s", e)
                    
                    logger.info("Fitting KMeans (GPU/CuPy) with n_clusters=%d, init=k-means++", n_clusters)
                    fit_start = perf_counter()
                    
                    # k-means++ initialization on GPU
                    cp.random.seed(random_state)
                    n_samples, n_features = gpu_embeddings.shape
                    
                    # Initialize first centroid randomly
                    centroids = cp.zeros((n_clusters, n_features), dtype=cp.float32)
                    first_idx = cp.random.randint(0, n_samples)
                    centroids[0] = gpu_embeddings[first_idx].copy()
                    
                    # k-means++: select remaining centroids based on distance to nearest centroid
                    for k in range(1, n_clusters):
                        # Compute distances from all points to nearest centroid
                        distances = cp.linalg.norm(
                            gpu_embeddings[:, cp.newaxis, :] - centroids[:k, cp.newaxis, :],
                            axis=2
                        )
                        min_distances = cp.min(distances, axis=1)
                        
                        # Select next centroid with probability proportional to distance^2
                        probabilities = min_distances ** 2
                        probabilities = probabilities / cp.sum(probabilities)
                        
                        # Sample next centroid
                        next_idx = cp.random.choice(n_samples, p=cp.asnumpy(probabilities))
                        centroids[k] = gpu_embeddings[next_idx].copy()
                    
                    # Synchronize to ensure initial transfer completes
                    cp.cuda.Stream.null.synchronize()
                    
                    max_iter = 300
                    labels = cp.zeros(n_samples, dtype=cp.int32)
                    
                    for iteration in range(max_iter):
                        # Compute distances on GPU (async operation)
                        distances = cp.linalg.norm(gpu_embeddings[:, cp.newaxis, :] - centroids[cp.newaxis, :, :], axis=2)
                        new_labels = cp.argmin(distances, axis=1)
                        
                        # Check convergence (sync only when checking convergence)
                        converged = cp.all(new_labels == labels)
                        cp.cuda.Stream.null.synchronize()  # Sync once for convergence check
                        if converged:
                            logger.info("  - Converged at iteration %d", iteration + 1)
                            break
                        
                        labels = new_labels
                        
                        # Vectorized centroid update (much faster than loop)
                        # Use advanced indexing to update all centroids at once
                        for k in range(n_clusters):
                            mask = labels == k
                            if cp.any(mask):
                                # Use mean directly - CuPy optimizes this
                                centroids[k] = gpu_embeddings[mask].mean(axis=0)
                        
                        # Single sync after iteration completes (reduces overhead while maintaining correctness)
                        cp.cuda.Stream.null.synchronize()
                        
                        # Log progress every 50 iterations
                        if (iteration + 1) % 50 == 0:
                            logger.info("  - Iteration %d/%d", iteration + 1, max_iter)
                    
                    # Final synchronization to ensure all work is complete
                    cp.cuda.Stream.null.synchronize()
                    
                    fit_time = perf_counter() - fit_start
                    logger.info("  - GPU KMeans fitting completed in %.2f seconds", fit_time)
                    
                    logger.info("Transferring results back to CPU...")
                    transfer_start = perf_counter()
                    labels_cpu = cp.asnumpy(labels)
                    centers_cpu = cp.asnumpy(centroids)
                    # Synchronize before transfer to ensure GPU work is complete
                    cp.cuda.Stream.null.synchronize()
                    transfer_time = perf_counter() - transfer_start
                    logger.info("  - Results transferred in %.3f seconds", transfer_time)
                    logger.info("  - Labels shape: %s", labels_cpu.shape)
                    logger.info("  - Centers shape: %s", centers_cpu.shape)
                
                wrapped_model = PyTorchGPUKMeansWrapper(centers_cpu, labels_cpu, n_clusters)
                logger.info("=" * 80)
                logger.info("GPU clustering completed successfully (PyTorch/CuPy)")
                logger.info("=" * 80)
                return wrapped_model, labels_cpu, embeddings
                
            except ImportError as e:
                logger.warning("=" * 80)
                logger.warning("CuPy not available for GPU clustering, falling back to CPU")
                logger.warning("Import error: %s", e)
                logger.warning("=" * 80)
                use_gpu_clustering = False
            except Exception as e:
                logger.warning("=" * 80)
                logger.warning("CuPy GPU clustering failed, falling back to CPU")
                logger.warning("Error type: %s", type(e).__name__)
                logger.warning("Error message: %s", str(e))
                logger.warning("=" * 80)
                use_gpu_clustering = False
    
    if not use_gpu_clustering:
        logger.info("Using CPU clustering (sklearn)")
    
    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        logger.info("Fitting MiniBatchKMeans (CPU) with n_clusters=%d", n_clusters)
        labels = model.fit_predict(embeddings)
        return model, labels, embeddings

    if method == "agglomerative":
        logger.info("Fitting AgglomerativeClustering (CPU) with n_clusters=%d", n_clusters)
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = agg.fit_predict(embeddings)
        
        centers: List[np.ndarray] = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                # If a cluster ended up empty (can happen in edge cases),
                # fall back to a zero vector with the correct dimensionality.
                centers.append(np.zeros(embeddings.shape[1], dtype=embeddings.dtype))
            else:
                centers.append(embeddings[mask].mean(axis=0))
        centers_arr = np.vstack(centers)

        model = AgglomerativeWrapper(centers_arr, labels, n_clusters)
        return model, labels, embeddings
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    logger.info("Fitting KMeans (CPU) with n_clusters=%d", n_clusters)
    labels = model.fit_predict(embeddings)
    return model, labels, embeddings


def build_nn_index(embeddings: np.ndarray, cfg: Dict, use_gpu: bool = False):
    """
    Build nearest neighbor index for similar-article lookup.
    
    Uses GPU (PyTorch/CuPy) if available and use_gpu=True, otherwise falls back to CPU (sklearn).
    """
    n_neighbors = int(cfg.get("n_neighbors", 10))
    algorithm = cfg.get("algorithm", "auto")
    
    backend = get_clustering_backend()
    if use_gpu and backend == "pytorch":
        try:
            import cupy as cp
            
            logger.info("=" * 80)
            logger.info("Building NearestNeighbors index on GPU (PyTorch/CuPy - Windows compatible)")
            logger.info("  - N neighbors: %d", n_neighbors)
            logger.info("  - Algorithm: brute-force (CuPy)")
            logger.info("  - Embeddings shape: %s", embeddings.shape)
            logger.info("=" * 80)
            
            # Transfer to GPU
            logger.info("Transferring embeddings to GPU...")
            transfer_start = perf_counter()
            cp.asarray(embeddings.astype(np.float32))
            transfer_time = perf_counter() - transfer_start
            logger.info("  - Transferred to GPU in %.3f seconds", transfer_time)
            
            logger.info("Fitting NearestNeighbors index on GPU (brute-force)...")
            fit_start = perf_counter()
            fit_time = perf_counter() - fit_start
            logger.info("  - Index built in %.2f seconds", fit_time)
            logger.info("=" * 80)
            logger.info("NearestNeighbors index built successfully on GPU (CuPy)")
            logger.info("=" * 80)
            
            return PyTorchGPUNNWrapper(embeddings, n_neighbors)
            
        except ImportError as e:
            logger.warning("=" * 80)
            logger.warning("CuPy not available for GPU NN index, falling back to CPU")
            logger.warning("Import error: %s", e)
            logger.warning("=" * 80)
        except Exception as e:
            logger.warning("=" * 80)
            logger.warning("CuPy GPU NN index failed, falling back to CPU")
            logger.warning("Error type: %s", type(e).__name__)
            logger.warning("Error message: %s", str(e))
            logger.warning("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Building NearestNeighbors index on CPU (sklearn)")
    logger.info("  - N neighbors: %d", n_neighbors)
    logger.info("  - Algorithm: %s", algorithm)
    logger.info("  - Embeddings shape: %s", embeddings.shape)
    logger.info("=" * 80)
    
    fit_start = perf_counter()
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
    nn.fit(embeddings)
    fit_time = perf_counter() - fit_start
    logger.info("Index built in %.2f seconds", fit_time)
    logger.info("=" * 80)
    logger.info("NearestNeighbors index built successfully on CPU")
    logger.info("=" * 80)
    return nn


def compute_cluster_summaries(
    titles: List[str],
    cleaned_texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    top_k_keywords: int = 20,
    top_k_articles: int = 10,
) -> pd.DataFrame:
    """
    Compute cluster summaries using c-TF-IDF (class-based TF-IDF):
      - size: number of articles in cluster
      - top keywords: terms ranked by c-TF-IDF score (normalized TF × IDF across clusters)
      - representative articles: articles closest to cluster centroid
    
    c-TF-IDF treats each cluster as a "class" and computes:
      - TF(term, cluster) = count(term) / total_tokens_in_cluster (normalized)
      - IDF(term) = log(N_clusters / CF(term)) where CF is class frequency
      - c-TF-IDF(term, cluster) = TF(term, cluster) × IDF(term)
    
    This emphasizes terms that are both frequent within a cluster AND
    distinctive compared to other clusters.
    """
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_to_indices[int(lbl)].append(idx)

    # ------------------------------------------------------------------
    # c-TF-IDF (class-based TF-IDF) computation:
    # Treats each cluster as a "class" and computes:
    #   - TF(term, cluster)  = normalized frequency: count(term) / total_tokens_in_cluster
    #   - DF(term)           = number of clusters (classes) where term appears
    #   - IDF(term)          = log(N_clusters / DF(term))
    #   - c-TF-IDF(term, cluster) = TF(term, cluster) × IDF(term)
    #
    # This emphasizes terms that are frequent within a cluster AND
    # distinctive compared to other clusters.
    # ------------------------------------------------------------------
    cluster_token_counters: Dict[int, Counter] = {}
    cluster_token_sets: Dict[int, Set[str]] = {}
    cluster_total_tokens: Dict[int, int] = {}

    # First pass: tokenize all articles and build per-cluster counters
    # This is the slowest step, so we show progress
    logger.info("Tokenizing articles and building cluster token counters...")
    total_articles = sum(len(indices) for indices in cluster_to_indices.values())
    article_progress = tqdm(total=total_articles, desc="Tokenizing articles", unit="article")
    
    for cluster_id, indices in cluster_to_indices.items():
        token_counter: Counter = Counter()
        token_set: Set[str] = set()
        total_tokens = 0
        for i in indices:
            text = cleaned_texts[i] or ""
            tokens = list(_tokenize(text))
            for tok in tokens:
                token_counter[tok] += 1
                token_set.add(tok)
                total_tokens += 1
            article_progress.update(1)
        cluster_token_counters[cluster_id] = token_counter
        cluster_token_sets[cluster_id] = token_set
        cluster_total_tokens[cluster_id] = total_tokens
    
    article_progress.close()

    # Compute class frequency (CF) across clusters: how many clusters contain each term
    cf_counter: Counter = Counter()
    for token_set in cluster_token_sets.values():
        cf_counter.update(token_set)

    n_clusters = max(len(cluster_token_sets), 1)
    idf: Dict[str, float] = {}
    for term, cf in cf_counter.items():
        # Standard c-TF-IDF IDF formula: log(N_classes / CF(term))
        # Add small epsilon to avoid division by zero
        if cf > 0:
            idf[term] = math.log(n_clusters / cf)
        else:
            idf[term] = 0.0

    # ------------------------------------------------------------------
    # Second pass: build output rows using c-TF-IDF ranking for "keywords"
    # and Euclidean distance to cluster center for representative articles.
    # ------------------------------------------------------------------
    rows: List[Dict] = []
    for cluster_id, indices in tqdm(
        cluster_to_indices.items(),
        total=len(cluster_to_indices),
        desc="Summarizing clusters",
    ):
        # Representative/top articles by proximity to centroid
        cluster_embeddings = embeddings[indices]
        center = centers[cluster_id].reshape(1, -1)
        dists = pairwise_distances(cluster_embeddings, center, metric="euclidean").ravel()
        order = np.argsort(dists)
        rep_indices = [indices[i] for i in order[:top_k_articles]]
        top_articles = [titles[i] for i in rep_indices]

        # c-TF-IDF based topic words
        token_counts = cluster_token_counters.get(cluster_id, Counter())
        total_tokens_in_cluster = cluster_total_tokens.get(cluster_id, 1)
        
        if token_counts and total_tokens_in_cluster > 0:
            # Normalize TF by cluster size: TF = count / total_tokens_in_cluster
            c_tfidf_scores = {
                term: (count / total_tokens_in_cluster) * idf.get(term, 0.0)
                for term, count in token_counts.items()
            }
            # Sort by c-TF-IDF descending and keep the top_k_keywords terms
            sorted_terms = sorted(c_tfidf_scores.items(), key=lambda kv: kv[1], reverse=True)
            keywords = [term for term, _ in sorted_terms[:top_k_keywords]]
        else:
            keywords = []

        rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "keywords": keywords,
                "top_articles": top_articles,
            }
        )

    return pd.DataFrame(rows)


def save_json(obj: Dict, path: str) -> None:
    """Save dictionary to JSON file, filtering out non-serializable values."""
    from types import FunctionType
    
    # Filter out non-serializable values (functions, etc.)
    def make_serializable(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [make_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {k: make_serializable(v) for k, v in value.items()}
        elif isinstance(value, FunctionType):
            return None  # Skip functions
        else:
            return str(value)  # Convert other types to string
    
    serializable_obj = make_serializable(obj)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_obj, f, indent=2)


def main() -> None:
    setup_logging()
    logger.info("Starting topic clustering pipeline")
    
    # Log device configuration
    log_device_info()

    try:
        config = load_config(CONFIG_PATH)
        model_cfg = config.get("models", {}).get("clustering", {})
        nn_cfg = config.get("models", {}).get("neighbors", {})
        
        use_gpu_config = config.get("models", {}).get("clustering", {}).get("use_gpu", "auto")
        backend = get_clustering_backend()
        
        if use_gpu_config == "auto":
            use_gpu = backend == "pytorch"
        elif use_gpu_config in (True, "true", "yes", "1"):
            use_gpu = backend == "pytorch"
            if not use_gpu:
                logger.warning("GPU requested but not available (CUDA or CuPy missing), falling back to CPU")
                logger.warning("  - CUDA available: %s", is_cuda_available())
                try:
                    import cupy as cp
                    cp.array([1])
                    logger.warning("  - CuPy available: True")
                except Exception:  # noqa: BLE001
                    logger.warning("  - CuPy available: False")
        else:
            use_gpu = False
        
        if use_gpu and backend == "pytorch":
            device_str = "GPU (PyTorch/CuPy)"
        else:
            device_str = "CPU (sklearn)"
        
        logger.info("Clustering configuration:")
        logger.info("  - GPU available: %s", is_cuda_available())
        logger.info("  - Backend: %s", backend)
        logger.info("  - Using device: %s", device_str)

        update_progress("clustering", "running", 0.0, "Loading embeddings and articles...")
        emb_df, embeddings = load_embeddings()
        cleaned_df = load_cleaned_articles()

        if len(emb_df) != len(cleaned_df):
            logger.warning(
                "Embeddings and cleaned articles have different lengths: %d vs %d",
                len(emb_df),
                len(cleaned_df),
            )

        titles = emb_df["title"].astype(str).tolist()
        cleaned_texts = cleaned_df["cleaned_text"].astype(str).tolist()

        n_clusters_display = model_cfg.get("n_clusters")
        if n_clusters_display is None:
            n_clusters_display = "auto (optimizing)"
        logger.info(
            "Clustering %d articles into ~%s clusters (method=%s, device=%s)...",
            len(titles),
            n_clusters_display,
            model_cfg.get("method", "kmeans"),
            device_str,
        )
        
        update_progress(
            "clustering",
            "running",
            10.0,
            f"Fitting clustering model on {device_str}...",
        )
        # Store requested n_clusters before make_clusterer potentially adjusts it
        # Handle None (null in YAML) - means auto-optimize
        n_clusters_value = model_cfg.get("n_clusters", 100)
        if n_clusters_value is None:
            requested_n_clusters = None  # Will be auto-optimized
        else:
            requested_n_clusters = int(n_clusters_value)
        model, labels, embeddings_used = make_clusterer(embeddings, model_cfg, use_gpu=use_gpu)
        # Get actual n_clusters from the model (may differ from requested if clamped)
        actual_n_clusters = len(np.unique(labels))
        
        # Use the embeddings that were actually used for clustering (may be normalized)
        # This ensures metrics are calculated on the same embeddings used for clustering
        embeddings_for_metrics = embeddings_used
        
        update_progress("clustering", "running", 50.0, "Building nearest-neighbor index...")

        logger.info("Building nearest-neighbor index for similar-article lookup...")
        # Use original embeddings for NN index (not normalized, for better search quality)
        nn_index = build_nn_index(embeddings, nn_cfg, use_gpu=use_gpu)

        update_progress("clustering", "running", 60.0, "Saving clustering artifacts...")
        # Prepare outputs
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, KMEANS_MODEL_PATH)
        joblib.dump(nn_index, NN_INDEX_PATH)

        assignments_df = pd.DataFrame({"title": titles, "cluster_id": labels.astype(int)})
        assignments_df.to_parquet(CLUSTER_ASSIGNMENTS_PATH, index=False)

        centers = model.cluster_centers_
        update_progress("clustering", "running", 70.0, "Computing cluster summaries...")
        logger.info("Computing cluster summaries (keywords and representative articles)...")
        summaries_df = compute_cluster_summaries(
            titles=titles,
            cleaned_texts=cleaned_texts,
            embeddings=embeddings_used,  # Use same embeddings as clustering for consistency
            labels=labels,
            centers=centers,
        )
        logger.info("Cluster summaries computed successfully")
        update_progress("clustering", "running", 85.0, "Calculating semantic metrics...")
        summaries_df.to_parquet(CLUSTERS_SUMMARY_PATH, index=False)

        # Calculate semantic metrics (Topic Coherence, Category Alignment, Cluster Diversity)
        semantic_metrics = {}
        try:
            from src.modeling.clustering_metrics import calculate_all_semantic_metrics
            from src.preprocessing.embeddings import EmbeddingGenerator
            
            # Get metrics configuration
            metrics_cfg = model_cfg.get("metrics", {})
            calculate_semantic = metrics_cfg.get("calculate_semantic_metrics", True)
            
            if calculate_semantic:
                logger.info("Calculating semantic metrics...")
                
                # Load embedding model for keyword encoding
                embedding_model_name = config.get("preprocessing", {}).get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
                embedding_device = config.get("preprocessing", {}).get("embeddings", {}).get("device", "auto")
                
                try:
                    embedding_generator = EmbeddingGenerator(
                        model_name=embedding_model_name,
                        device=embedding_device
                    )
                    embedding_model = embedding_generator.model
                    
                    # Calculate semantic metrics
                    semantic_metrics = calculate_all_semantic_metrics(
                        summaries_df=summaries_df,
                        assignments_df=assignments_df,
                        cleaned_articles_df=cleaned_df,
                        embeddings=embeddings_used,
                        centers=centers,
                        embedding_model=embedding_model,
                        metrics_config=metrics_cfg,
                    )
                    
                    if semantic_metrics:
                        logger.info("Calculated semantic metrics: %s", list(semantic_metrics.keys()))
                    else:
                        logger.warning("No semantic metrics calculated (may be due to missing data)")
                        
                except Exception as emb_exc:
                    logger.warning("Failed to load embedding model for semantic metrics: %s", emb_exc)
                except Exception as sem_exc:
                    logger.warning("Failed to calculate semantic metrics: %s", sem_exc)
            else:
                logger.info("Semantic metrics calculation disabled in config")
        except ImportError as imp_exc:
            logger.warning("Failed to import semantic metrics module: %s", imp_exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during semantic metrics calculation: %s", exc)
        
        # Generate 2D embedding map for visualization
        update_progress("clustering", "running", 90.0, "Generating 2D embedding map...")
        try:
            from src.modeling.embedding_map import generate_embedding_map_2d
            
            logger.info("Generating 2D embedding map with UMAP...")
            embedding_map_result = generate_embedding_map_2d(
                embeddings_path=EMBEDDINGS_PATH,
                assignments_path=CLUSTER_ASSIGNMENTS_PATH,
                summaries_path=CLUSTERS_SUMMARY_PATH,
            )
            
            if embedding_map_result is not None:
                logger.info("2D embedding map generated successfully")
            else:
                logger.warning("Failed to generate 2D embedding map (non-critical, continuing)")
        except ImportError:
            logger.warning("UMAP library not available, skipping 2D embedding map generation")
        except Exception as map_exc:  # noqa: BLE001
            logger.warning("Error generating 2D embedding map (non-critical): %s", map_exc)
        
        update_progress("clustering", "running", 95.0, "Finalizing metrics...")

        # Calculate cluster size distribution
        cluster_sizes = summaries_df["size"].values if "size" in summaries_df.columns else []
        cluster_size_metrics = {}
        if len(cluster_sizes) > 0:
            cluster_size_metrics = {
                "cluster_size_min": int(cluster_sizes.min()),
                "cluster_size_max": int(cluster_sizes.max()),
                "cluster_size_mean": float(cluster_sizes.mean()),
                "cluster_size_std": float(cluster_sizes.std()),
            }
        
        # Calculate silhouette score with adaptive sampling
        # CRITICAL: Use embeddings_used (may be normalized) to match what was used for clustering
        silhouette_score_val = None
        try:
            # Adaptive sampling based on dataset size for efficiency
            n_samples = len(embeddings_used)
            if n_samples < 200:
                sample_size = min(100, n_samples)  # Small sample for small datasets
            elif n_samples < 500:
                sample_size = min(300, n_samples)  # Medium sample for medium datasets
            else:
                sample_size = min(500, n_samples)  # Larger sample for large datasets
            
            if sample_size < n_samples:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(n_samples, sample_size, replace=False)
                sample_embeddings = embeddings_used[indices]
                sample_labels = labels[indices]
                silhouette_score_val = float(silhouette_score(sample_embeddings, sample_labels))
                logger.info("Calculated silhouette score on sample of %d articles (adaptive sampling)", sample_size)
            else:
                silhouette_score_val = float(silhouette_score(embeddings_used, labels))
                logger.info("Calculated silhouette score on full dataset")
        except Exception as exc:
            logger.warning("Failed to calculate silhouette score: %s", exc)
        
        # Calculate Davies-Bouldin index with adaptive sampling
        # CRITICAL: Use embeddings_used (may be normalized) to match what was used for clustering
        davies_bouldin = None
        try:
            # Use same adaptive sampling as silhouette score
            n_samples = len(embeddings_used)
            if n_samples < 200:
                sample_size = min(100, n_samples)
            elif n_samples < 500:
                sample_size = min(300, n_samples)
            else:
                sample_size = min(500, n_samples)
            
            if sample_size < n_samples:
                np.random.seed(42)  # Reproducible sampling
                indices = np.random.choice(n_samples, sample_size, replace=False)
                sample_embeddings = embeddings_used[indices]
                sample_labels = labels[indices]
                davies_bouldin = float(davies_bouldin_score(sample_embeddings, sample_labels))
                logger.info("Calculated Davies-Bouldin index on sample of %d articles (adaptive sampling)", sample_size)
            else:
                davies_bouldin = float(davies_bouldin_score(embeddings_used, labels))
                logger.info("Calculated Davies-Bouldin index on full dataset")
        except Exception as exc:
            logger.warning("Failed to calculate Davies-Bouldin index: %s", exc)
        
        # Basic metrics
        metrics = {
            "n_articles": int(len(titles)),
            "n_clusters": int(len(summaries_df)),
        }
        metrics.update(cluster_size_metrics)
        
        # Baseline metrics (geometric)
        if silhouette_score_val is not None:
            metrics["baseline_silhouette_score"] = silhouette_score_val
        if davies_bouldin is not None:
            metrics["baseline_davies_bouldin_index"] = davies_bouldin
        
        # Semantic metrics (may be empty if calculation failed or disabled)
        for key, value in semantic_metrics.items():
            if value is not None:
                metrics[f"semantic_{key}"] = value
        
        save_json(metrics, METRICS_PATH)

        # Enhanced MLflow logging
        from src.common.mlflow_utils import (
            log_metrics_safely,
            log_params_safely,
            start_mlflow_run,
        )
        
        try:
            import mlflow
            
            with start_mlflow_run("cluster_topics"):
                # Log parameters
                log_params_safely(model_cfg, prefix="clustering")
                log_params_safely(nn_cfg, prefix="neighbors")
                
                # Log actual vs requested clusters if they differ or if auto-optimized
                if requested_n_clusters is not None and actual_n_clusters != requested_n_clusters:
                    log_params_safely({
                        "clustering.n_clusters_requested": requested_n_clusters,
                        "clustering.n_clusters_actual": actual_n_clusters,
                    })
                elif requested_n_clusters is None:
                    # Log that auto-optimization was used
                    log_params_safely({
                        "clustering.n_clusters_auto_optimized": True,
                        "clustering.n_clusters_actual": actual_n_clusters,
                    })
                
                # Add tags for better organization
                try:
                    active_run = mlflow.active_run()
                    if active_run:
                        mlflow.set_tag("metrics_version", "2.0")
                        mlflow.set_tag("clustering_method", model_cfg.get("method", "auto"))
                        mlflow.set_tag("has_semantic_metrics", str(bool(semantic_metrics)))
                except Exception:  # noqa: BLE001
                    pass  # Tags are optional
                
                # Log metrics with organized prefixes
                # Baseline metrics (already have baseline_ prefix)
                baseline_metrics = {
                    k: v for k, v in metrics.items() 
                    if k.startswith("baseline_") or k in ("n_articles", "n_clusters", "cluster_size_min", "cluster_size_max", "cluster_size_mean", "cluster_size_std")
                }
                log_metrics_safely(baseline_metrics)
                
                # Semantic metrics (already have semantic_ prefix)
                semantic_metrics_to_log = {
                    k: v for k, v in metrics.items() 
                    if k.startswith("semantic_")
                }
                if semantic_metrics_to_log:
                    log_metrics_safely(semantic_metrics_to_log)
                
                # Log keyword extraction metrics if available
                if "keywords" in summaries_df.columns:
                    avg_keywords = summaries_df["keywords"].apply(len).mean() if len(summaries_df) > 0 else 0
                    log_metrics_safely({"avg_keywords_per_cluster": float(avg_keywords)})
                
                logger.info("Logged clustering metrics to MLflow")
                if semantic_metrics_to_log:
                    logger.info("  - Baseline metrics: %d", len(baseline_metrics))
                    logger.info("  - Semantic metrics: %d", len(semantic_metrics_to_log))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging for clustering skipped or failed: %s", exc)

        logger.info("Topic clustering complete. Wrote artifacts to %s", MODEL_DIR)
        mark_stage_completed("clustering", "Clustering complete")
    except FileNotFoundError as exc:
        error_msg = f"Clustering failed: required inputs missing: {exc}"
        logger.error(
            "%s. Make sure the preprocessing stage has completed successfully "
            "and produced both embeddings and cleaned_articles parquet files.",
            error_msg,
        )
        mark_stage_error("clustering", error_msg)
        raise
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Clustering pipeline failed: {exc}"
        logger.exception(
            "%s. Consider reducing 'n_clusters' in config.yaml "
            "or re-running preprocessing to regenerate embeddings.",
            error_msg,
        )
        mark_stage_error("clustering", error_msg)
        raise


if __name__ == "__main__":
    main()


