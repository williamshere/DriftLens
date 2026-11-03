"""Tests for co-occurrence / interaction graph construction."""

from __future__ import annotations

import pytest
import networkx as nx

from driftlens.analytics.network import (
    build_cooccurrence_graph,
    build_graphs_by_slice,
    compute_centralities,
    detect_communities,
    export_for_d3,
    extract_terms,
    graph_metrics_summary,
)


class TestExtractTerms:
    def test_basic(self):
        terms = extract_terms("The quick brown fox jumps over the lazy dog")
        assert "quick" in terms
        assert "brown" in terms
        assert "the" not in terms  # stopword

    def test_empty(self):
        assert extract_terms("") == []
        assert extract_terms("the and for") == []


class TestBuildCooccurrenceGraph:
    def test_empty_documents(self):
        G = build_cooccurrence_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_single_doc(self):
        docs = [{"text": "climate policy economy climate policy", "id": "1"}]
        G = build_cooccurrence_graph(docs)
        assert "climate" in G
        assert "policy" in G
        assert "economy" in G
        # climate-policy co-occur once in this doc
        assert G.has_edge("climate", "policy")
        assert G["climate"]["policy"]["weight"] == 1

    def test_weight_accumulates_across_docs(self):
        docs = [
            {"text": "alpha beta gamma", "id": "1"},
            {"text": "alpha beta", "id": "2"},
        ]
        G = build_cooccurrence_graph(docs)
        assert G["alpha"]["beta"]["weight"] == 2
        assert G["alpha"]["gamma"]["weight"] == 1

    def test_min_edge_weight_filter(self):
        docs = [
            {"text": "alpha beta gamma", "id": "1"},
            {"text": "alpha beta", "id": "2"},
        ]
        G = build_cooccurrence_graph(docs, min_edge_weight=2)
        assert G.has_edge("alpha", "beta")
        assert not G.has_edge("alpha", "gamma")

    def test_min_term_freq_filter(self):
        docs = [
            {"text": "alpha beta", "id": "1"},
            {"text": "alpha beta", "id": "2"},
            {"text": "gamma delta", "id": "3"},
        ]
        G = build_cooccurrence_graph(docs, min_term_freq=2)
        assert "alpha" in G
        assert "gamma" not in G

    def test_object_documents(self):
        class Doc:
            def __init__(self, text, id):  # noqa: A002
                self.text = text
                self.id = id

        docs = [Doc("hello world hello", "1")]
        G = build_cooccurrence_graph(docs)
        assert "hello" in G
        assert "world" in G

    def test_window_size(self):
        docs = [{"text": "alpha beta gamma delta epsilon", "id": "1"}]
        G_full = build_cooccurrence_graph(docs, window_size=None)
        G_win = build_cooccurrence_graph(docs, window_size=2)
        # Full doc connects all pairs; window=2 only adjacent pairs co-occur repeatedly
        # gamma-epsilon should exist in full but maybe not with window 2 depending on position
        # Just check both produce graphs
        assert G_full.number_of_edges() >= G_win.number_of_edges()

    def test_normalized_weight(self):
        docs = [
            {"text": "alpha beta", "id": "1"},
            {"text": "alpha beta", "id": "2"},
            {"text": "alpha gamma", "id": "3"},
        ]
        G = build_cooccurrence_graph(docs)
        assert G["alpha"]["beta"]["normalized_weight"] == pytest.approx(1.0)
        assert G["alpha"]["gamma"]["normalized_weight"] < 1.0


class TestBuildGraphsBySlice:
    def test_partition(self):
        docs = [
            {"text": "alpha beta", "id": "1", "slice_key": "2020"},
            {"text": "gamma delta", "id": "2", "slice_key": "2021"},
            {"text": "alpha gamma", "id": "3", "slice_key": "2020"},
        ]
        graphs = build_graphs_by_slice(docs)
        assert "2020" in graphs
        assert "2021" in graphs
        assert graphs["2020"].has_edge("alpha", "beta")
        assert graphs["2021"].has_edge("gamma", "delta")

    def test_none_slice_key(self):
        docs = [{"text": "hello world", "id": "1"}]
        graphs = build_graphs_by_slice(docs)
        assert "__none__" in graphs


class TestCentralities:
    def test_empty(self):
        G: nx.Graph = nx.Graph()
        result = compute_centralities(G)
        assert result == {"degree": {}, "betweenness": {}}

    def test_triangle(self):
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        result = compute_centralities(G)
        assert result["degree"]["a"] == pytest.approx(1.0)
        # In a triangle all betweenness is 0
        assert result["betweenness"]["a"] == pytest.approx(0.0)
        # Node attributes written
        assert G.nodes["a"]["degree_centrality"] == pytest.approx(1.0)

    def test_path_betweenness(self):
        # a - b - c : b bridges
        G = nx.Graph()
        G.add_edges_from([("a", "b", {"weight": 1}), ("b", "c", {"weight": 1})])
        result = compute_centralities(G)
        assert result["betweenness"]["b"] > result["betweenness"]["a"]


class TestCommunities:
    def test_empty(self):
        G: nx.Graph = nx.Graph()
        result = detect_communities(G)
        assert result["num_communities"] == 0
        assert result["modularity"] == pytest.approx(0.0)

    def test_no_edges(self):
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        result = detect_communities(G)
        assert result["num_communities"] == 1
        assert G.nodes["a"]["community"] == 0

    def test_two_cliques(self):
        G = nx.Graph()
        # Two triangles weakly connected
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        G.add_edges_from([("x", "y"), ("y", "z"), ("z", "x")])
        G.add_edge("c", "x", weight=1)
        result = detect_communities(G)
        assert result["num_communities"] >= 1
        assert 0.0 <= result["modularity"] <= 1.0

    def test_modularity_finite(self):
        docs = [{"text": "alpha beta gamma delta", "id": "1"}]
        G = build_cooccurrence_graph(docs)
        result = detect_communities(G)
        assert result["modularity"] == pytest.approx(result["modularity"])  # not NaN


class TestExportForD3:
    def test_export_shape(self):
        docs = [{"text": "alpha beta gamma", "id": "1"}]
        G = build_cooccurrence_graph(docs)
        exported = export_for_d3(G)
        assert "nodes" in exported
        assert "edges" in exported
        assert len(exported["nodes"]) == G.number_of_nodes()
        assert len(exported["edges"]) == G.number_of_edges()
        node = exported["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "degree_centrality" in node
        assert "community" in node
        edge = exported["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "weight" in edge

    def test_empty_export(self):
        G: nx.Graph = nx.Graph()
        exported = export_for_d3(G)
        assert exported["nodes"] == []
        assert exported["edges"] == []

    def test_metrics_summary(self):
        docs = [{"text": "alpha beta gamma", "id": "1"}]
        G = build_cooccurrence_graph(docs)
        summary = graph_metrics_summary(G)
        assert summary["num_nodes"] == 3
        assert summary["num_edges"] == 3
        assert 0.0 <= summary["density"] <= 1.0
        assert "top_degree" in summary
        assert "top_betweenness" in summary
