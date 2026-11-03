"""DriftLens analytics package — embeddings and drift computation."""

from __future__ import annotations

from driftlens.analytics.drift import (
    all_terms_drift,
    compute_centroid,
    compute_centroids,
    compute_centroids_by_community,
    compute_centroids_by_slice,
    compute_drift_series,
    compute_pairwise_drift,
    compute_term_centroids,
    cosine_distance,
    cosine_similarity,
    drift_score,
    drift_velocity,
    rank_by_velocity,
    rank_drift,
    term_drift_across_slices,
    top_displaced_terms,
    top_displaced_terms_by_velocity,
)
from driftlens.analytics.embedder import Embedder

__all__ = [
    "Embedder",
    "all_terms_drift",
    "compute_centroid",
    "compute_centroids",
    "compute_centroids_by_community",
    "compute_centroids_by_slice",
    "compute_drift_series",
    "compute_pairwise_drift",
    "compute_term_centroids",
    "cosine_distance",
    "cosine_similarity",
    "drift_score",
    "drift_velocity",
    "rank_by_velocity",
    "rank_drift",
    "term_drift_across_slices",
    "top_displaced_terms",
    "top_displaced_terms_by_velocity",
]
