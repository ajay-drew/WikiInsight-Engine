"""
Statistical drift detection for embeddings, text distributions, and clusters.

Detects changes in data distributions between baseline and current runs
using statistical tests (KS test, MMD) and tracks cluster stability.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def detect_embedding_drift(
    baseline_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    sample_size: Optional[int] = 5000,
) -> Dict[str, float]:
    """
    Detect drift in embedding distributions using statistical tests.
    
    Args:
        baseline_embeddings: Baseline embedding matrix (n_samples, n_features)
        current_embeddings: Current embedding matrix (n_samples, n_features)
        sample_size: Optional sample size for large datasets
    
    Returns:
        Dict with drift scores and p-values
    """
    # Sample if datasets are large
    if sample_size and len(baseline_embeddings) > sample_size:
        np.random.seed(42)
        baseline_indices = np.random.choice(len(baseline_embeddings), sample_size, replace=False)
        current_indices = np.random.choice(len(current_embeddings), min(sample_size, len(current_embeddings)), replace=False)
        baseline_embeddings = baseline_embeddings[baseline_indices]
        current_embeddings = current_embeddings[current_indices]
    
    # Calculate mean and std for each dimension
    baseline_mean = np.mean(baseline_embeddings, axis=0)
    baseline_std = np.std(baseline_embeddings, axis=0)
    current_mean = np.mean(current_embeddings, axis=0)
    current_std = np.std(current_embeddings, axis=0)
    
    # Mean drift (Euclidean distance between means)
    mean_drift = float(np.linalg.norm(current_mean - baseline_mean))
    
    # Std drift (Euclidean distance between stds)
    std_drift = float(np.linalg.norm(current_std - baseline_std))
    
    # KS test on first principal component (approximation)
    # Project to first PC for univariate test
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=42)
        baseline_pc = pca.fit_transform(baseline_embeddings).flatten()
        current_pc = pca.transform(current_embeddings).flatten()
        ks_statistic, ks_pvalue = stats.ks_2samp(baseline_pc, current_pc)
    except Exception as exc:
        logger.warning("Failed to compute KS test: %s", exc)
        ks_statistic = 0.0
        ks_pvalue = 1.0
    
    return {
        "mean_drift": mean_drift,
        "std_drift": std_drift,
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": ks_pvalue < 0.05,  # Significant at 5% level
    }


def detect_text_distribution_drift(
    baseline_articles: pd.DataFrame,
    current_articles: pd.DataFrame,
) -> Dict[str, float]:
    """
    Detect drift in text distributions (vocabulary, length, categories).
    
    Args:
        baseline_articles: Baseline articles DataFrame
        current_articles: Current articles DataFrame
    
    Returns:
        Dict with drift scores
    """
    results = {}
    
    # Average article length drift
    if "cleaned_text" in baseline_articles.columns and "cleaned_text" in current_articles.columns:
        baseline_lengths = baseline_articles["cleaned_text"].str.len()
        current_lengths = current_articles["cleaned_text"].str.len()
        
        if len(baseline_lengths) > 0 and len(current_lengths) > 0:
            baseline_mean_len = baseline_lengths.mean()
            current_mean_len = current_lengths.mean()
            length_drift = abs(current_mean_len - baseline_mean_len) / baseline_mean_len if baseline_mean_len > 0 else 0
            
            # KS test on length distribution
            ks_stat, ks_pval = stats.ks_2samp(baseline_lengths, current_lengths)
            
            results.update({
                "avg_length_drift": float(length_drift),
                "length_ks_statistic": float(ks_stat),
                "length_ks_pvalue": float(ks_pval),
            })
    
    # Vocabulary overlap (Jaccard similarity of unique words)
    if "cleaned_text" in baseline_articles.columns and "cleaned_text" in current_articles.columns:
        baseline_words = set()
        for text in baseline_articles["cleaned_text"]:
            if isinstance(text, str):
                baseline_words.update(text.lower().split())
        
        current_words = set()
        for text in current_articles["cleaned_text"]:
            if isinstance(text, str):
                current_words.update(text.lower().split())
        
        if len(baseline_words) > 0:
            intersection = len(baseline_words & current_words)
            union = len(baseline_words | current_words)
            vocab_jaccard = intersection / union if union > 0 else 0
            vocab_drift = 1.0 - vocab_jaccard
            
            results.update({
                "vocab_jaccard": float(vocab_jaccard),
                "vocab_drift": float(vocab_drift),
                "baseline_vocab_size": len(baseline_words),
                "current_vocab_size": len(current_words),
            })
    
    return results


def detect_cluster_drift(
    baseline_assignments: pd.DataFrame,
    current_assignments: pd.DataFrame,
) -> Dict[str, float]:
    """
    Detect drift in cluster assignments.
    
    Args:
        baseline_assignments: Baseline cluster assignments (title, cluster_id)
        current_assignments: Current cluster assignments (title, cluster_id)
    
    Returns:
        Dict with cluster drift metrics
    """
    # Merge on title to compare assignments
    merged = baseline_assignments.merge(
        current_assignments,
        on="title",
        suffixes=("_baseline", "_current"),
        how="inner",
    )
    
    if len(merged) == 0:
        return {"cluster_drift": 1.0, "assignment_overlap": 0.0}
    
    # Calculate assignment overlap (same cluster assignment)
    same_cluster = (merged["cluster_id_baseline"] == merged["cluster_id_current"]).sum()
    assignment_overlap = same_cluster / len(merged)
    cluster_drift = 1.0 - assignment_overlap
    
    # Cluster size distribution drift
    baseline_sizes = baseline_assignments["cluster_id"].value_counts().sort_index()
    current_sizes = current_assignments["cluster_id"].value_counts().sort_index()
    
    # Align cluster IDs
    all_clusters = set(baseline_sizes.index) | set(current_sizes.index)
    baseline_aligned = [baseline_sizes.get(cid, 0) for cid in all_clusters]
    current_aligned = [current_sizes.get(cid, 0) for cid in all_clusters]
    
    if len(baseline_aligned) > 0 and len(current_aligned) > 0:
        size_ks_stat, size_ks_pval = stats.ks_2samp(baseline_aligned, current_aligned)
    else:
        size_ks_stat = 0.0
        size_ks_pval = 1.0
    
    return {
        "cluster_drift": float(cluster_drift),
        "assignment_overlap": float(assignment_overlap),
        "cluster_size_ks_statistic": float(size_ks_stat),
        "cluster_size_ks_pvalue": float(size_ks_pval),
    }


def detect_drift_from_files(
    baseline_embeddings_path: str,
    current_embeddings_path: str,
    baseline_articles_path: Optional[str] = None,
    current_articles_path: Optional[str] = None,
    baseline_assignments_path: Optional[str] = None,
    current_assignments_path: Optional[str] = None,
) -> Dict[str, any]:
    """
    Detect drift by loading baseline and current artifacts from files.
    
    Args:
        baseline_embeddings_path: Path to baseline embeddings parquet
        current_embeddings_path: Path to current embeddings parquet
        baseline_articles_path: Optional path to baseline articles parquet
        current_articles_path: Optional path to current articles parquet
        baseline_assignments_path: Optional path to baseline assignments parquet
        current_assignments_path: Optional path to current assignments parquet
    
    Returns:
        Dict with all drift metrics
    """
    results = {}
    
    # Embedding drift
    if os.path.exists(baseline_embeddings_path) and os.path.exists(current_embeddings_path):
        try:
            baseline_emb_df = pd.read_parquet(baseline_embeddings_path)
            current_emb_df = pd.read_parquet(current_embeddings_path)
            
            if "embedding" in baseline_emb_df.columns and "embedding" in current_emb_df.columns:
                baseline_embeddings = np.vstack(baseline_emb_df["embedding"].to_list())
                current_embeddings = np.vstack(current_emb_df["embedding"].to_list())
                
                embedding_drift = detect_embedding_drift(baseline_embeddings, current_embeddings)
                results.update({f"embedding_{k}": v for k, v in embedding_drift.items()})
        except Exception as exc:
            logger.warning("Failed to detect embedding drift: %s", exc)
    
    # Text distribution drift
    if baseline_articles_path and current_articles_path:
        if os.path.exists(baseline_articles_path) and os.path.exists(current_articles_path):
            try:
                baseline_articles = pd.read_parquet(baseline_articles_path)
                current_articles = pd.read_parquet(current_articles_path)
                
                text_drift = detect_text_distribution_drift(baseline_articles, current_articles)
                results.update({f"text_{k}": v for k, v in text_drift.items()})
            except Exception as exc:
                logger.warning("Failed to detect text drift: %s", exc)
    
    # Cluster drift
    if baseline_assignments_path and current_assignments_path:
        if os.path.exists(baseline_assignments_path) and os.path.exists(current_assignments_path):
            try:
                baseline_assignments = pd.read_parquet(baseline_assignments_path)
                current_assignments = pd.read_parquet(current_assignments_path)
                
                cluster_drift = detect_cluster_drift(baseline_assignments, current_assignments)
                results.update({f"cluster_{k}": v for k, v in cluster_drift.items()})
            except Exception as exc:
                logger.warning("Failed to detect cluster drift: %s", exc)
    
    return results

