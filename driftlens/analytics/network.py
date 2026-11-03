"""Co-occurrence / interaction graph construction per time slice.

Nodes are actors or concepts (extracted terms), edges are weighted co-occurrence
counts within documents.  The resulting :class:`networkx.Graph` carries:

* node attributes: ``label``, ``type``, ``frequency``
* edge attributes: ``weight`` (co-occurrence count), ``normalized_weight``
* graph-level metrics: degree & betweenness centrality, modularity

Export helpers produce ``{nodes, edges}`` payloads consumable by D3.js / PyVis
(without requiring the caller to know NetworkX internals).
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Term extraction helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,30}")

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "had",
        "been",
        "being",
        "will",
        "would",
        "could",
        "should",
        "about",
        "into",
        "through",
        "between",
        "their",
        "there",
        "which",
        "while",
        "where",
        "when",
        "what",
        "also",
        "more",
        "most",
        "some",
        "such",
        "than",
        "then",
        "them",
        "these",
        "those",
        "only",
        "other",
        "very",
        "just",
        "over",
        "under",
        "after",
        "before",
        "because",
        "each",
        "both",
        "even",
    }
)


def extract_terms(
    text: str,
    *,
    min_length: int = 2,
    lower: bool = True,
    stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Extract candidate terms (actors/concepts) from *text*.

    A lightweight regex tokenizer — suitable when no NER model is available.
    Filters stop-words and short tokens.
    """
    if stopwords is None:
        stopwords = _STOPWORDS
    raw = _TOKEN_RE.findall(text.lower() if lower else text)
    terms: list[str] = []
    for tok in raw:
        if len(tok) < min_length:
            continue
        if tok.lower() in stopwords:
            continue
        terms.append(tok.lower() if lower else tok)
    return terms


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_cooccurrence_graph(
    documents: Iterable[dict[str, Any] | Any],
    *,
    window_size: int | None = None,
    min_edge_weight: int = 1,
    min_term_freq: int = 1,
) -> nx.Graph:
    """Build a weighted co-occurrence graph from *documents*.

    Two terms co-occur when they appear in the same document (or, when
    *window_size* is set, within a sliding window of N tokens).  Nodes are
    unique terms; undirected edges are weighted by co-occurrence count.

    Args:
        documents: Iterable of ``{"text": str, "id": str, ...}`` dicts or
            objects with ``.text`` / ``.id`` attributes (e.g. Document).
        window_size: If ``None`` (default) co-occurrence is per-document.
            Otherwise co-occurrence is counted within a sliding window of
            *window_size* tokens.
        min_edge_weight: Drop edges with weight < this threshold.
        min_term_freq: Drop terms with total frequency < this threshold (and
            their incident edges).

    Returns:
        :class:`networkx.Graph` with ``weight`` on every edge and ``frequency``
        on every node.  Isolated nodes are retained when they pass the
        frequency filter.
    """
    term_freq: Counter[str] = Counter()
    edge_weights: Counter[tuple[str, str]] = Counter()

    for doc in documents:
        if isinstance(doc, dict):
            text = str(doc.get("text") or doc.get("content") or "")
        else:
            text = str(getattr(doc, "text", "") or "")
        terms = extract_terms(text)
        if not terms:
            continue
        # Frequency (unique terms per doc counted once for doc-frequency,
        # but we count all occurrences for node size).
        term_freq.update(terms)

        # Unique terms in this doc/window for co-occurrence
        if window_size is None:
            uniq = sorted(set(terms))
            for a, b in itertools.combinations(uniq, 2):
                edge_weights[(a, b)] += 1
        else:
            # Sliding window co-occurrence
            for i in range(len(terms)):
                window = terms[i : i + window_size]
                uniq_win = sorted(set(window))
                for a, b in itertools.combinations(uniq_win, 2):
                    edge_weights[(a, b)] += 1

    # Filter by min_term_freq
    if min_term_freq > 1:
        allowed_terms = {t for t, c in term_freq.items() if c >= min_term_freq}
        term_freq = Counter({t: c for t, c in term_freq.items() if t in allowed_terms})
        edge_weights = Counter({e: w for e, w in edge_weights.items() if e[0] in allowed_terms and e[1] in allowed_terms})

    # Filter by min_edge_weight
    if min_edge_weight > 1:
        edge_weights = Counter({e: w for e, w in edge_weights.items() if w >= min_edge_weight})

    G: nx.Graph = nx.Graph()
    for term, freq in term_freq.items():
        G.add_node(term, label=term, type="concept", frequency=int(freq))
    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=int(w))

    # Normalized weight in [0, 1] for visualization scaling
    if G.number_of_edges() > 0:
        max_w = max(d["weight"] for _, _, d in G.edges(data=True))
        for _, _, d in G.edges(data=True):
            d["normalized_weight"] = float(d["weight"] / max_w) if max_w else 0.0
    return G


def build_graphs_by_slice(
    documents: Iterable[Any],
    *,
    slice_key_attr: str = "slice_key",
    **kwargs: Any,
) -> dict[str, nx.Graph]:
    """Build one co-occurrence graph per distinct slice key.

    Args:
        documents: Iterable of documents with ``slice_key`` (attr or dict key).
        slice_key_attr: Attribute / dict key used to partition slices.
        **kwargs: Forwarded to :func:`build_cooccurrence_graph`.

    Returns:
        Mapping ``slice_key -> Graph``.  Documents with ``None`` slice_key are
        grouped under ``"__none__"``.
    """
    buckets: dict[str, list[Any]] = defaultdict(list)
    for doc in documents:
        if isinstance(doc, dict):
            key = doc.get(slice_key_attr)
        else:
            key = getattr(doc, slice_key_attr, None)
        key_str = str(key) if key is not None else "__none__"
        buckets[key_str].append(doc)
    return {k: build_cooccurrence_graph(v, **kwargs) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_centralities(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute degree and betweenness centrality for *G*.

    Returns:
        ``{"degree": {node: score}, "betweenness": {node: score}}``.
        Betweenness uses edge ``weight`` as distance when present (inverted so
        higher co-occurrence = shorter distance) to surface bridging concepts.
        For empty / single-node graphs both dicts are empty/zero-filled.

    Side-effect: node attributes ``degree_centrality`` and
    ``betweenness_centrality`` are written onto *G*.
    """
    if G.number_of_nodes() == 0:
        return {"degree": {}, "betweenness": {}}

    degree = nx.degree_centrality(G)

    # Weighted betweenness: invert weight so strong ties = short distance.
    # Use 1/weight as length; fall back to unweighted when no weights.
    has_weights = any("weight" in d for _, _, d in G.edges(data=True))
    if has_weights and G.number_of_edges() > 0:
        # Create length attribute
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1))
            d["_length"] = 1.0 / w if w != 0 else 1.0
        betweenness = nx.betweenness_centrality(G, weight="_length", normalized=True)
        # cleanup temp attribute
        for _, _, d in G.edges(data=True):
            d.pop("_length", None)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)

    for n in G.nodes:
        G.nodes[n]["degree_centrality"] = float(degree.get(n, 0.0))
        G.nodes[n]["betweenness_centrality"] = float(betweenness.get(n, 0.0))

    # Ensure plain floats
    degree_f = {k: float(v) for k, v in degree.items()}
    betweenness_f = {k: float(v) for k, v in betweenness.items()}
    return {"degree": degree_f, "betweenness": betweenness_f}


def detect_communities(G: nx.Graph) -> dict[str, Any]:
    """Detect communities and compute modularity.

    Tries greedy modularity optimisation (no extra deps).  Falls back to
    single-community assignment when the graph is too small / disconnected.

    Returns:
        ``{"communities": [[nodes]], "modularity": float, "num_communities": int}``
        — also writes ``community`` node attribute (int id).

    Modularity is NaN-safe: 0.0 is returned when undefined (e.g. no edges).
    """
    if G.number_of_nodes() == 0:
        return {"communities": [], "modularity": 0.0, "num_communities": 0}
    if G.number_of_edges() == 0:
        for n in G.nodes:
            G.nodes[n]["community"] = 0
        return {"communities": [list(G.nodes)], "modularity": 0.0, "num_communities": 1}

    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity

        communities = list(greedy_modularity_communities(G, weight="weight"))
        # modularity function expects partition as iterable of sets
        mod = float(modularity(G, communities, weight="weight"))
        # Handle NaN (e.g. isolated graphs)
        if not np.isfinite(mod):
            mod = 0.0
        # Assign community id to nodes
        for cid, comm in enumerate(communities):
            for n in comm:
                G.nodes[n]["community"] = int(cid)
        # Ensure every node has a community (isolated nodes may be missed)
        for n in G.nodes:
            if "community" not in G.nodes[n]:
                G.nodes[n]["community"] = int(len(communities))
        communities_list = [sorted(list(c)) for c in communities]
        # Include any unassigned isolates as extra communities
        assigned = {n for c in communities for n in c}
        isolates = [n for n in G.nodes if n not in assigned]
        if isolates:
            communities_list.append(sorted(isolates))
        return {
            "communities": communities_list,
            "modularity": mod,
            "num_communities": len(communities_list),
        }
    except Exception:
        # Defensive fallback
        for n in G.nodes:
            G.nodes[n]["community"] = 0
        return {"communities": [list(G.nodes)], "modularity": 0.0, "num_communities": 1}


# ---------------------------------------------------------------------------
# Export for D3 / PyVis
# ---------------------------------------------------------------------------


def export_for_d3(
    G: nx.Graph,
    *,
    include_centralities: bool = True,
    include_communities: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Export *G* as ``{nodes, edges}`` for D3.js / PyVis.

    Each node: ``{id, label, frequency, degree_centrality, betweenness_centrality, community}``
    Each edge: ``{source, target, weight, normalized_weight}``

    Args:
        G: Graph to export.
        include_centralities: Compute centralities if not already present.
        include_communities: Compute communities if not already present.

    Returns:
        Dict with ``nodes`` and ``edges`` lists (JSON-serialisable).
    """
    if include_centralities and G.number_of_nodes() > 0:
        # Only compute when missing
        if not any("degree_centrality" in G.nodes[n] for n in G.nodes):
            compute_centralities(G)
    if include_communities and G.number_of_nodes() > 0:
        if not any("community" in G.nodes[n] for n in G.nodes):
            detect_communities(G)

    nodes: list[dict[str, Any]] = []
    for nid, attrs in G.nodes(data=True):
        nodes.append(
            {
                "id": str(nid),
                "label": str(attrs.get("label", nid)),
                "frequency": int(attrs.get("frequency", 1)),
                "type": str(attrs.get("type", "concept")),
                "degree_centrality": float(attrs.get("degree_centrality", 0.0)),
                "betweenness_centrality": float(attrs.get("betweenness_centrality", 0.0)),
                "community": int(attrs.get("community", 0)),
            }
        )

    edges: list[dict[str, Any]] = []
    for u, v, d in G.edges(data=True):
        edges.append(
            {
                "source": str(u),
                "target": str(v),
                "weight": int(d.get("weight", 1)),
                "normalized_weight": float(d.get("normalized_weight", 1.0)),
            }
        )

    # Stable sort for deterministic output
    nodes.sort(key=lambda n: n["id"])
    edges.sort(key=lambda e: (e["source"], e["target"]))
    return {"nodes": nodes, "edges": edges}


def graph_metrics_summary(G: nx.Graph) -> dict[str, Any]:
    """One-shot summary: centralities, communities, density, counts.

    Convenience for API responses that want overview stats without exporting
    the full graph.
    """
    centralities = compute_centralities(G)
    communities = detect_communities(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = float(nx.density(G)) if n > 1 else 0.0
    # Top nodes by degree centrality
    top_degree = sorted(centralities["degree"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_betweenness = sorted(centralities["betweenness"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "num_nodes": int(n),
        "num_edges": int(m),
        "density": density,
        "modularity": float(communities["modularity"]),
        "num_communities": int(communities["num_communities"]),
        "top_degree": [{"id": k, "score": float(v)} for k, v in top_degree],
        "top_betweenness": [{"id": k, "score": float(v)} for k, v in top_betweenness],
    }
