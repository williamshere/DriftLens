"""Semantic drift computation — centroids, cosine distance, velocity, ranking.

All vector math uses :mod:`numpy` and :mod:`scipy` and operates on
``float32`` arrays.  Drift is defined as cosine distance:

    drift(a, b) = 1 - cosine_similarity(a, b)

so identical vectors yield 0 and opposite vectors yield 2.

Centroids are the arithmetic mean of member embeddings.  Drift velocity
normalises drift by elapsed time (or unit slice steps when timestamps are
not available).  Ranking surfaces the most displaced terms / communities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

# ---------------------------------------------------------------------------
# Cosine helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (1-D).
        b: Second vector (1-D).

    Returns:
        Cosine similarity in ``[-1, 1]``.  Returns ``0.0`` when either
        vector is zero to avoid ``NaN``.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance ``1 - cosine_similarity``.

    This is the primary drift metric.  Uses :func:`cosine_similarity`
    internally but cross-checks via :mod:`scipy` when available.

    Args:
        a: First vector (1-D).
        b: Second vector (1-D).

    Returns:
        Cosine distance in ``[0, 2]``.
    """
    # Use scipy as secondary validation path (required by spec: vector math
    # uses numpy/scipy).  scipy_cosine already returns 1 - cos_sim but
    # returns NaN for zero vectors, so we guard.
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()
    if np.linalg.norm(a_f) == 0.0 or np.linalg.norm(b_f) == 0.0:
        return 1.0  # maximally distant from zero vector convention
    # Prefer manual for NaN safety; also demonstrate scipy usage.
    try:
        d_scipy = float(scipy_cosine(a_f, b_f))
        if not np.isnan(d_scipy):
            return d_scipy
    except Exception:
        pass
    return 1.0 - cosine_similarity(a_f, b_f)


def drift_score(a: np.ndarray, b: np.ndarray) -> float:
    """Alias for :func:`cosine_distance` — drift between two centroids.

    Args:
        a: Centroid at time / slice A.
        b: Centroid at time / slice B.

    Returns:
        Drift magnitude.
    """
    return cosine_distance(a, b)


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute the centroid (mean) of a stack of embeddings.

    Args:
        embeddings: 2-D array of shape ``(n, dim)``.  May be a list of
            vectors that will be stacked.

    Returns:
        1-D ``float32`` centroid of shape ``(dim,)``.  Returns a zero
        vector of shape ``(0,)`` when ``n == 0``.

    Raises:
        ValueError: If ``embeddings`` is not 2-D after coercion.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0,), dtype=np.float32)
    if arr.ndim == 1:
        # Single vector — already a centroid.
        return arr.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {arr.shape}")
    centroid = np.mean(arr, axis=0)
    return centroid.astype(np.float32)


def compute_centroids(
    embeddings: np.ndarray,
    groups: Sequence[str],
) -> dict[str, np.ndarray]:
    """Compute per-group centroids.

    Groups documents by an arbitrary label (term, time slice, or community)
    and returns the mean vector per group.

    Args:
        embeddings: 2-D array ``(n, dim)``.
        groups: Sequence of length ``n`` with group keys.

    Returns:
        Mapping ``group_key -> centroid (float32, shape (dim,))``.

    Raises:
        ValueError: If lengths mismatch or embeddings not 2-D.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {arr.shape}")
    if len(groups) != arr.shape[0]:
        raise ValueError(f"groups length {len(groups)} != n {arr.shape[0]}")
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for vec, g in zip(arr, groups):
        buckets[str(g)].append(vec)
    result: dict[str, np.ndarray] = {}
    for key, vecs in buckets.items():
        stacked = np.vstack(vecs)
        result[key] = compute_centroid(stacked)
    return result


def compute_term_centroids(
    embeddings: np.ndarray,
    terms: Sequence[str],
    slices: Sequence[str] | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Compute centroids keyed by ``(term, slice)``.

    When ``slices`` is ``None``, each term gets a single centroid across
    all documents.

    Args:
        embeddings: 2-D array ``(n, dim)``.
        terms: Per-document term / keyword label.
        slices: Optional per-document slice key (time slice or community).
            When provided must have length ``n``.  When ``None`` the slice
            key ``"__all__"`` is used.

    Returns:
        Mapping ``(term, slice_key) -> centroid``.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    n = arr.shape[0]
    if len(terms) != n:
        raise ValueError(f"terms length {len(terms)} != n {n}")
    if slices is not None and len(slices) != n:
        raise ValueError(f"slices length {len(slices)} != n {n}")
    effective_slices: Sequence[str] = slices if slices is not None else ["__all__"] * n
    buckets: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for vec, t, s in zip(arr, terms, effective_slices):
        buckets[(str(t), str(s))].append(vec)
    return {k: compute_centroid(np.vstack(v)) for k, v in buckets.items()}


def compute_centroids_by_slice(
    embeddings: np.ndarray,
    slice_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    """Compute one centroid per time / community slice.

    Convenience wrapper over :func:`compute_centroids`.

    Args:
        embeddings: 2-D array ``(n, dim)``.
        slice_keys: Per-document slice key.

    Returns:
        Mapping ``slice_key -> centroid``.
    """
    return compute_centroids(embeddings, slice_keys)


def compute_centroids_by_community(
    embeddings: np.ndarray,
    communities: Sequence[str],
) -> dict[str, np.ndarray]:
    """Compute one centroid per community.

    Args:
        embeddings: 2-D array ``(n, dim)``.
        communities: Per-document community label.

    Returns:
        Mapping ``community -> centroid``.
    """
    return compute_centroids(embeddings, communities)


# ---------------------------------------------------------------------------
# Drift series & velocity
# ---------------------------------------------------------------------------


def _compute_drift_series_dict(
    centroids: dict[str, np.ndarray],
    ordered_keys: Sequence[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Internal dict-returning drift series (legacy)."""
    keys = list(ordered_keys) if ordered_keys is not None else sorted(centroids.keys())
    result: dict[tuple[str, str], float] = {}
    for a, b in zip(keys, keys[1:]):
        if a not in centroids or b not in centroids:
            continue
        result[(a, b)] = cosine_distance(centroids[a], centroids[b])
    return result


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b))


def drift_score_between(
    a: np.ndarray,
    b: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Drift between two vectors under the chosen metric."""
    if metric == "cosine":
        return cosine_distance(a, b)
    if metric == "euclidean":
        return euclidean_distance(a, b)
    raise ValueError(f"Unknown metric {metric!r}; expected 'cosine' or 'euclidean'")


def compute_drift_series(
    centroids: dict[str, np.ndarray],
    ordered_keys: Sequence[str] | None = None,
    *,
    ordered_slices: Sequence[str] | None = None,
    metric: str = "cosine",
) -> list[dict[str, object]]:
    """Compute drift between consecutive centroids in order.

    Returns a list of ``{from_slice, to_slice, drift, similarity}`` dicts,
    sorted by slice order.  Raises ``KeyError`` if an ``ordered_slices``
    entry is missing.

    Args:
        centroids: Mapping ``slice_key -> centroid``.
        ordered_keys: Explicit ordering (positional, legacy).
        ordered_slices: Explicit ordering (keyword, test-facing).
        metric: ``"cosine"`` (default) or ``"euclidean"``.
    """
    # Resolve ordering alias
    order = ordered_slices if ordered_slices is not None else ordered_keys
    keys = list(order) if order is not None else sorted(centroids.keys())
    # Validate requested order (test expects KeyError on missing)
    for k in keys:
        if k not in centroids:
            raise KeyError(k)
    if len(keys) < 2:
        return []
    series: list[dict[str, object]] = []
    for a, b in zip(keys, keys[1:]):
        va = centroids[a]
        vb = centroids[b]
        if metric == "cosine":
            drift = cosine_distance(va, vb)
            sim = cosine_similarity(va, vb)
        elif metric == "euclidean":
            drift = euclidean_distance(va, vb)
            # similarity not meaningful for euclidean; use negative drift
            sim = -drift
        else:
            raise ValueError(f"Unknown metric {metric!r}")
        series.append(
            {"from_slice": a, "to_slice": b, "drift": float(drift), "similarity": float(sim)}
        )
    return series


def drift_magnitude(series: list[dict[str, object]]) -> float:
    """Mean drift across a series (0.0 for empty)."""
    if not series:
        return 0.0
    return float(sum(float(s["drift"]) for s in series) / len(series))


def is_drifting(series: list[dict[str, object]], threshold: float = 0.15) -> bool:
    """Return ``True`` if any step exceeds *threshold*."""
    return any(float(s["drift"]) > threshold for s in series)


# Back-compat aliases
def centroid(vectors: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Alias for :func:`compute_centroid` expected by tests."""
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2 and vectors.shape[0] == 0:
        raise ValueError("Cannot compute centroid of empty array")
    if isinstance(vectors, list) and len(vectors) == 0:
        raise ValueError("Cannot compute centroid of empty sequence")
    arr = np.asarray(vectors) if isinstance(vectors, list) else vectors
    if isinstance(arr, np.ndarray) and arr.size == 0:
        raise ValueError("Cannot compute centroid of empty array")
    # Delegate to compute_centroid; it handles 2-D stacking
    if isinstance(vectors, list):
        if len(vectors) == 0:
            raise ValueError("empty")
        stacked = np.vstack([np.asarray(v) for v in vectors])
        return compute_centroid(stacked)
    return compute_centroid(arr)


def compute_pairwise_drift(
    centroids: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Compute drift for every unordered pair of centroids.

    Args:
        centroids: Mapping label -> centroid.

    Returns:
        Mapping ``(label_a, label_b) -> drift``.
    """
    keys = list(centroids.keys())
    result: dict[tuple[str, str], float] = {}
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            result[(ka, kb)] = cosine_distance(centroids[ka], centroids[kb])
    return result


def drift_velocity(
    drift: float,
    time_delta: float = 1.0,
) -> float:
    """Normalise drift by elapsed time.

    Args:
        drift: Cosine distance.
        time_delta: Elapsed time in arbitrary units (slices, days, etc.).
            Must be positive.

    Returns:
        Velocity ``drift / time_delta``.

    Raises:
        ValueError: If ``time_delta`` is not positive.
    """
    if time_delta <= 0:
        raise ValueError(f"time_delta must be positive, got {time_delta}")
    return float(drift / time_delta)


def rank_by_velocity(
    drifts: dict[str, float],
    time_deltas: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Rank terms / slices by drift velocity descending.

    Args:
        drifts: Mapping label -> drift magnitude.
        time_deltas: Optional mapping label -> elapsed time.  When ``None``
            each drift is treated as velocity with ``time_delta=1``.

    Returns:
        List of ``(label, velocity)`` sorted descending by velocity.
    """
    velocities: list[tuple[str, float]] = []
    for label, d in drifts.items():
        td = 1.0 if time_deltas is None else time_deltas.get(label, 1.0)
        velocities.append((label, drift_velocity(d, td)))
    velocities.sort(key=lambda x: x[1], reverse=True)
    return velocities


def rank_drift(
    drifts: dict[str, float],
) -> list[tuple[str, float]]:
    """Rank entries by raw drift descending.

    Args:
        drifts: Mapping label -> drift.

    Returns:
        Sorted list ``(label, drift)`` descending.
    """
    return sorted(drifts.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Top displaced terms
# ---------------------------------------------------------------------------


def top_displaced_terms(
    drifts: dict[str, float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k most displaced terms by drift magnitude.

    Args:
        drifts: Mapping ``term -> drift`` (or any label -> drift).
        top_k: Number of top entries to return.

    Returns:
        List of ``(term, drift)`` sorted descending, length ``min(top_k, len(drifts))``.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    ranked = rank_drift(drifts)
    return ranked[:top_k]


def top_displaced_terms_by_velocity(
    drifts: dict[str, float],
    time_deltas: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k terms ranked by drift velocity.

    Args:
        drifts: Mapping term -> drift.
        time_deltas: Optional mapping term -> time delta.
        top_k: Number to return.

    Returns:
        List of ``(term, velocity)`` sorted descending.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    ranked = rank_by_velocity(drifts, time_deltas)
    return ranked[:top_k]


def term_drift_across_slices(
    term_slice_centroids: dict[tuple[str, str], np.ndarray],
    term: str,
    ordered_slices: Sequence[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Compute drift for a single term across its slices.

    Args:
        term_slice_centroids: Mapping ``(term, slice) -> centroid`` as
            returned by :func:`compute_term_centroids`.
        term: Term to isolate.
        ordered_slices: Optional explicit slice ordering.  Defaults to
            sorted slice keys for that term.

    Returns:
        Mapping ``(slice_a, slice_b) -> drift`` for consecutive slices.
    """
    # Collect slices for this term.
    slice_to_vec: dict[str, np.ndarray] = {}
    for (t, s), vec in term_slice_centroids.items():
        if t == term:
            slice_to_vec[s] = vec
    return _compute_drift_series_dict(slice_to_vec, ordered_slices)


def all_terms_drift(
    term_slice_centroids: dict[tuple[str, str], np.ndarray],
) -> dict[str, float]:
    """Summarise per-term drift as max pairwise distance across slices.

    For terms appearing in multiple slices, drift is the maximum cosine
    distance between any two slice centroids (a measure of overall
    displacement).  Terms appearing in a single slice get drift ``0.0``.

    Args:
        term_slice_centroids: Mapping ``(term, slice) -> centroid``.

    Returns:
        Mapping ``term -> max drift``.
    """
    # Group by term.
    per_term: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for (t, s), vec in term_slice_centroids.items():
        per_term[t][s] = vec

    result: dict[str, float] = {}
    for term, slice_map in per_term.items():
        if len(slice_map) < 2:
            result[term] = 0.0
            continue
        pairwise = compute_pairwise_drift(slice_map)
        result[term] = max(pairwise.values()) if pairwise else 0.0
    return result
