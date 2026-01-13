"""
Semantic clustering quality metrics.

Calculates semantic metrics to complement baseline geometric metrics:
- Topic Coherence: Semantic similarity within clusters
- Category Alignment: Alignment with Wikipedia categories
- Cluster Diversity: Distinctness between clusters
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

logger = logging.getLogger(__name__)


def calculate_topic_coherence_keywords(
    summaries_df: pd.DataFrame,
    embedding_model,
) -> Optional[float]:
    """
    Calculate topic coherence using cluster keywords.
    
    Measures semantic similarity of keywords within each cluster.
    Higher coherence = keywords are semantically related.
    
    Args:
        summaries_df: DataFrame with 'cluster_id' and 'keywords' columns
        embedding_model: Sentence transformer model with .encode() method
        
    Returns:
        Average coherence score (0-1, higher is better) or None if calculation fails
    """
    try:
        coherence_scores = []
        
        for _, row in summaries_df.iterrows():
            keywords = row.get("keywords", [])
            
            if not keywords or len(keywords) < 2:
                continue  # Need at least 2 keywords to measure coherence
            
            # Get embeddings for keywords
            try:
                keyword_embeddings = embedding_model.encode(keywords)
            except Exception as e:
                logger.debug("Failed to encode keywords for cluster %d: %s", row.get("cluster_id"), e)
                continue
            
            # Calculate average pairwise cosine similarity
            similarity_matrix = cosine_similarity(keyword_embeddings)
            
            # Average similarity (excluding diagonal)
            n = len(keywords)
            if n > 1:
                avg_similarity = (similarity_matrix.sum() - n) / (n * (n - 1))
                coherence_scores.append(avg_similarity)
        
        if coherence_scores:
            return float(np.mean(coherence_scores))
        else:
            logger.debug("No valid clusters for keyword coherence calculation")
            return None
            
    except Exception as exc:
        logger.warning("Failed to calculate topic coherence (keywords): %s", exc)
        return None


def calculate_topic_coherence_articles(
    assignments_df: pd.DataFrame,
    embeddings: np.ndarray,
    title_to_index: Dict[str, int],
) -> Optional[float]:
    """
    Calculate topic coherence using article embeddings within clusters.
    
    Measures semantic similarity of articles within each cluster.
    Higher coherence = articles are semantically related.
    
    Args:
        assignments_df: DataFrame with 'title' and 'cluster_id' columns
        embeddings: Embedding matrix (n_articles, n_features)
        title_to_index: Mapping from article title to embedding index
        
    Returns:
        Average coherence score (0-1, higher is better) or None if calculation fails
    """
    try:
        coherence_scores = []
        
        # Group articles by cluster
        for cluster_id, group in assignments_df.groupby("cluster_id"):
            cluster_articles = group["title"].tolist()
            
            if len(cluster_articles) < 2:
                continue  # Need at least 2 articles
            
            # Get embedding indices for articles in this cluster
            indices = []
            for title in cluster_articles:
                if title in title_to_index:
                    indices.append(title_to_index[title])
            
            if len(indices) < 2:
                continue
            
            # Get embeddings for articles in this cluster
            cluster_embeddings = embeddings[indices]
            
            # Calculate average pairwise cosine similarity
            similarity_matrix = cosine_similarity(cluster_embeddings)
            
            # Average similarity (excluding diagonal)
            n = len(indices)
            if n > 1:
                avg_similarity = (similarity_matrix.sum() - n) / (n * (n - 1))
                coherence_scores.append(avg_similarity)
        
        if coherence_scores:
            return float(np.mean(coherence_scores))
        else:
            logger.debug("No valid clusters for article coherence calculation")
            return None
            
    except Exception as exc:
        logger.warning("Failed to calculate topic coherence (articles): %s", exc)
        return None


def calculate_category_alignment(
    assignments_df: pd.DataFrame,
    cleaned_articles_df: pd.DataFrame,
) -> Optional[float]:
    """
    Calculate category alignment: how well clusters align with Wikipedia categories.
    
    Measures Jaccard similarity of category sets within clusters.
    Higher alignment = articles in cluster share common categories.
    
    Args:
        assignments_df: DataFrame with 'title' and 'cluster_id' columns
        cleaned_articles_df: DataFrame with 'title' and 'categories' columns
        
    Returns:
        Average alignment score (0-1, higher is better) or None if calculation fails
    """
    try:
        alignment_scores = []
        
        # Merge to get categories for each article
        merged = assignments_df.merge(
            cleaned_articles_df[["title", "categories"]],
            on="title",
            how="left"
        )
        
        # Group by cluster
        for cluster_id, group in merged.groupby("cluster_id"):
            categories_list = group["categories"].dropna().tolist()
            
            if len(categories_list) < 2:
                continue  # Need at least 2 articles with categories
            
            # Calculate pairwise Jaccard similarity
            overlaps = []
            for i in range(len(categories_list)):
                for j in range(i + 1, len(categories_list)):
                    cats_i = set(categories_list[i]) if isinstance(categories_list[i], list) else set()
                    cats_j = set(categories_list[j]) if isinstance(categories_list[j], list) else set()
                    
                    if len(cats_i | cats_j) > 0:  # Avoid division by zero
                        overlap = len(cats_i & cats_j) / len(cats_i | cats_j)
                        overlaps.append(overlap)
            
            if overlaps:
                avg_overlap = np.mean(overlaps)
                alignment_scores.append(avg_overlap)
        
        if alignment_scores:
            return float(np.mean(alignment_scores))
        else:
            logger.debug("No valid clusters for category alignment calculation")
            return None
            
    except Exception as exc:
        logger.warning("Failed to calculate category alignment: %s", exc)
        return None


def calculate_cluster_diversity_centroids(
    centers: np.ndarray,
) -> Optional[float]:
    """
    Calculate cluster diversity using cluster centroids.
    
    Measures cosine distance between cluster centroids.
    Higher diversity = clusters are more distinct.
    
    Args:
        centers: Cluster centroids array (n_clusters, n_features)
        
    Returns:
        Average diversity score (0-1, higher is better) or None if calculation fails
    """
    try:
        if len(centers) < 2:
            logger.debug("Need at least 2 clusters for diversity calculation")
            return None
        
        # Calculate pairwise cosine distances
        distances = cosine_distances(centers)
        
        # Average distance (excluding diagonal)
        n = len(centers)
        avg_distance = (distances.sum() - n) / (n * (n - 1))
        
        # Convert distance to similarity (1 - distance)
        # Higher similarity = lower diversity, so we want lower similarity = higher diversity
        diversity = 1.0 - avg_distance
        
        return float(diversity)
        
    except Exception as exc:
        logger.warning("Failed to calculate cluster diversity (centroids): %s", exc)
        return None


def calculate_cluster_diversity_keywords(
    summaries_df: pd.DataFrame,
    embedding_model,
) -> Optional[float]:
    """
    Calculate cluster diversity using cluster keywords.
    
    Measures cosine distance between average keyword embeddings per cluster.
    Higher diversity = clusters are more distinct.
    
    Args:
        summaries_df: DataFrame with 'cluster_id' and 'keywords' columns
        embedding_model: Sentence transformer model with .encode() method
        
    Returns:
        Average diversity score (0-1, higher is better) or None if calculation fails
    """
    try:
        cluster_keywords = {}
        
        # Get average embedding of keywords for each cluster
        for _, row in summaries_df.iterrows():
            cluster_id = row.get("cluster_id")
            keywords = row.get("keywords", [])
            
            if not keywords:
                continue
            
            try:
                # Get embeddings for keywords
                keyword_embeddings = embedding_model.encode(keywords)
                # Average embedding for this cluster's keywords
                cluster_keywords[cluster_id] = keyword_embeddings.mean(axis=0)
            except Exception as e:
                logger.debug("Failed to encode keywords for cluster %d: %s", cluster_id, e)
                continue
        
        if len(cluster_keywords) < 2:
            logger.debug("Need at least 2 clusters with keywords for diversity calculation")
            return None
        
        # Convert to array
        centroids = np.array(list(cluster_keywords.values()))
        
        # Calculate pairwise cosine distances
        distances = cosine_distances(centroids)
        
        # Average distance (excluding diagonal)
        n = len(centroids)
        avg_distance = (distances.sum() - n) / (n * (n - 1))
        
        # Convert distance to similarity (1 - distance)
        diversity = 1.0 - avg_distance
        
        return float(diversity)
        
    except Exception as exc:
        logger.warning("Failed to calculate cluster diversity (keywords): %s", exc)
        return None


def calculate_all_semantic_metrics(
    summaries_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    cleaned_articles_df: pd.DataFrame,
    embeddings: np.ndarray,
    centers: np.ndarray,
    embedding_model,
    metrics_config: Optional[Dict] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate all semantic metrics.
    
    Args:
        summaries_df: DataFrame with cluster summaries (keywords, top_articles)
        assignments_df: DataFrame with article-to-cluster assignments
        cleaned_articles_df: DataFrame with article data including categories
        embeddings: Embedding matrix (n_articles, n_features)
        centers: Cluster centroids (n_clusters, n_features)
        embedding_model: Sentence transformer model for encoding keywords
        metrics_config: Optional config dict with 'coherence_method' and 'diversity_method'
        
    Returns:
        Dictionary of metric names to values (None if calculation failed)
    """
    if metrics_config is None:
        metrics_config = {}
    
    coherence_method = metrics_config.get("coherence_method", "keywords")
    diversity_method = metrics_config.get("diversity_method", "centroids")
    
    results = {}
    
    # Topic Coherence
    if coherence_method in ("keywords", "both"):
        coherence_keywords = calculate_topic_coherence_keywords(summaries_df, embedding_model)
        if coherence_keywords is not None:
            results["topic_coherence_keywords"] = coherence_keywords
    
    if coherence_method in ("articles", "both"):
        # Create title to index mapping
        title_to_index = {}
        for idx, title in enumerate(cleaned_articles_df["title"]):
            title_to_index[str(title)] = idx
        
        coherence_articles = calculate_topic_coherence_articles(
            assignments_df, embeddings, title_to_index
        )
        if coherence_articles is not None:
            results["topic_coherence_articles"] = coherence_articles
    
    # Category Alignment
    alignment = calculate_category_alignment(assignments_df, cleaned_articles_df)
    if alignment is not None:
        results["category_alignment"] = alignment
    
    # Cluster Diversity
    if diversity_method in ("centroids", "both"):
        diversity_centroids = calculate_cluster_diversity_centroids(centers)
        if diversity_centroids is not None:
            results["cluster_diversity_centroids"] = diversity_centroids
    
    if diversity_method in ("keywords", "both"):
        diversity_keywords = calculate_cluster_diversity_keywords(summaries_df, embedding_model)
        if diversity_keywords is not None:
            results["cluster_diversity_keywords"] = diversity_keywords
    
    return results
