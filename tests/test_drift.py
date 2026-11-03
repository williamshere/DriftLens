"""Tests for drift math — cosine, centroid, drift series."""

from __future__ import annotations

import math

import numpy as np
import pytest

from driftlens.analytics.drift import (
    centroid,
    cosine_distance,
    cosine_similarity,
    compute_drift_series,
    drift_magnitude,
    drift_score_between,
    euclidean_distance,
    is_drifting,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)
        assert cosine_similarity(b, a) == pytest.approx(0.0)
        assert cosine_similarity(a, a) == pytest.approx(0.0)

    def test_cosine_distance_range(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0)
        assert cosine_distance(a, a) == pytest.approx(0.0)

    def test_euclidean_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0)
        assert euclidean_distance(a, a) == pytest.approx(0.0)


class TestCentroid:
    def test_single_vector(self):
        v = np.array([1.0, 2.0, 3.0])
        c = centroid([v])
        np.testing.assert_allclose(c, v.astype(np.float32))

    def test_mean_of_two(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        c = centroid([a, b])
        np.testing.assert_allclose(c, np.array([0.5, 0.5], dtype=np.float32))

    def test_stacked_array(self):
        arr = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        c = centroid(arr)
        np.testing.assert_allclose(c, np.array([2 / 3, 2 / 3], dtype=np.float32))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            centroid([])

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            centroid(np.zeros((0, 3)))


class TestDriftScore:
    def test_cosine_no_drift(self):
        v = np.array([1.0, 0.0, 0.0])
        assert drift_score_between(v, v, metric="cosine") == pytest.approx(0.0)

    def test_euclidean_no_drift(self):
        v = np.array([1.0, 2.0])
        assert drift_score_between(v, v, metric="euclidean") == pytest.approx(0.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            drift_score_between(np.array([1.0]), np.array([0.0]), metric="manhattan")  # type: ignore[arg-type]


class TestDriftSeries:
    def test_two_slices(self):
        centroids = {
            "2020-01": np.array([1.0, 0.0]),
            "2020-02": np.array([0.0, 1.0]),
        }
        series = compute_drift_series(centroids, metric="cosine")
        assert len(series) == 1
        assert series[0]["from_slice"] == "2020-01"
        assert series[0]["to_slice"] == "2020-02"
        assert series[0]["drift"] == pytest.approx(1.0)  # orthogonal => distance 1
        assert series[0]["similarity"] == pytest.approx(0.0)

    def test_ordered_slices_respected(self):
        centroids = {
            "b": np.array([1.0, 0.0]),
            "a": np.array([0.0, 1.0]),
            "c": np.array([1.0, 1.0]),
        }
        series = compute_drift_series(centroids, ordered_slices=["a", "b", "c"])
        assert series[0]["from_slice"] == "a"
        assert series[1]["from_slice"] == "b"

    def test_missing_slice_raises(self):
        centroids = {"a": np.array([1.0, 0.0])}
        with pytest.raises(KeyError):
            compute_drift_series(centroids, ordered_slices=["a", "missing"])

    def test_single_slice_empty(self):
        assert compute_drift_series({"only": np.array([1.0, 0.0])}) == []

    def test_empty_empty(self):
        assert compute_drift_series({}) == []

    def test_drift_magnitude(self):
        centroids = {
            "s1": np.array([1.0, 0.0]),
            "s2": np.array([0.0, 1.0]),
            "s3": np.array([1.0, 0.0]),
        }
        series = compute_drift_series(centroids, ordered_slices=["s1", "s2", "s3"])
        # s1->s2 drift 1.0, s2->s3 drift 1.0 => mean 1.0
        assert drift_magnitude(series) == pytest.approx(1.0)
        assert drift_magnitude([]) == pytest.approx(0.0)

    def test_is_drifting(self):
        centroids = {
            "s1": np.array([1.0, 0.0, 0.0]),
            "s2": np.array([0.99, 0.01, 0.0]),
        }
        series = compute_drift_series(centroids)
        # Very small drift should not trigger default threshold 0.15
        assert not is_drifting(series, threshold=0.15)
        # But huge drift should
        centroids2 = {"s1": np.array([1.0, 0.0]), "s2": np.array([-1.0, 0.0])}
        series2 = compute_drift_series(centroids2)
        assert is_drifting(series2, threshold=0.15)

    def test_euclidean_series(self):
        centroids = {
            "s1": np.array([0.0, 0.0]),
            "s2": np.array([3.0, 4.0]),
        }
        series = compute_drift_series(centroids, metric="euclidean")
        assert series[0]["drift"] == pytest.approx(5.0)
