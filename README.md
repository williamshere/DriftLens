# DriftLens

**Semantic drift detection across document corpora.**

DriftLens tracks how the meaning of terms, actors, and concepts evolves over time. It ingests document collections partitioned by time slices, encodes them with transformer embeddings, computes per-term centroid drift, and builds co-occurrence networks to surface structural change.

## Features

- **Temporal / categorical slicing** — partition a corpus by date, period, or category
- **Embedding pipeline** — `sentence-transformers` with auto device detection (`cuda`, `mps`, `cpu`)
- **Drift metrics** — cosine / Euclidean drift scores and per-step drift series
- **Co-occurrence graphs** — per-slice NetworkX graphs with degree / betweenness centrality, modularity, and `{nodes, edges}` export for D3.js / PyVis
- **SQLite persistence** — documents, embeddings, and term centroids with WAL mode
- **FastAPI service** — `GET /api/drift?term=` and `GET /api/network?slice=`
- **Click CLI** — ingest, embed, drift, and serve from the terminal

## Installation

```bash
# Python >= 3.12 required
pip install -e .

# With dev/test extras
pip install -e ".[dev]"

# Or with uv
uv sync --extra dev
```

Dependencies are declared in `pyproject.toml`:

- `pydantic`, `pydantic-settings`, `numpy`, `scipy`, `networkx`, `fastapi`, `uvicorn`, `click`, `sentence-transformers`

## Quick Start

### 1. Prepare a corpus file

**JSON** — list of documents:
```json
[
  {"id": "doc-1", "text": "Climate policy shapes economic outcomes.", "date": "2020-01-15"},
  {"id": "doc-2", "text": "Economic policy and climate governance.", "date": "2021-06-02"}
]
```

**CSV** — must contain at least `id` and `text` columns.

### 2. CLI usage

```bash
# Ingest (slice_by = field used as slice_key, e.g. 'date' -> YYYY-MM)
driftlens ingest --file corpus.json --slice-by date
driftlens ingest --file corpus.csv --slice-by category

# Encode documents (uses all-MiniLM-L6-v2 by default)
driftlens process-embeddings
driftlens process-embeddings --model all-MiniLM-L6-v2 --batch-size 32 --limit 100

# Compute drift
driftlens compute-drift --terms climate,economy
driftlens compute-drift --metric cosine --top-k 10

# Serve the API
driftlens serve --port 8000
driftlens serve --host 0.0.0.0 --port 8000 --reload
```

Set the database path or model via env or flags:

```bash
DRIFTLENS_DB_PATH=./data/driftlens.db DRIFTLENS_EMBEDDING_MODEL=all-MiniLM-L6-v2 driftlens serve
driftlens --db-path ./data/custom.db ingest --file corpus.json --slice-by date
```

### 3. API usage

Start the server then query:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/health

# Drift series for a term
curl "http://127.0.0.1:8000/api/drift?term=climate"
curl "http://127.0.0.1:8000/api/drift?term=economy&metric=euclidean"

# Co-occurrence network for a slice (D3-ready)
curl "http://127.0.0.1:8000/api/network?slice=2020-01"
curl "http://127.0.0.1:8000/api/network"  # all documents
```

**Response shapes**

`GET /api/drift?term=climate`:
```json
{
  "term": "climate",
  "metric": "cosine",
  "slices": ["2020-01", "2021-01"],
  "series": [
    {"from_slice": "2020-01", "to_slice": "2021-01", "drift": 0.42, "similarity": 0.58}
  ]
}
```

`GET /api/network?slice=2020-01`:
```json
{
  "slice": "2020-01",
  "nodes": [{"id": "climate", "label": "climate", "frequency": 12, "degree_centrality": 0.8, "community": 0}],
  "edges": [{"source": "climate", "target": "policy", "weight": 5, "normalized_weight": 1.0}],
  "metrics": {"num_nodes": 42, "num_edges": 120, "density": 0.14, "modularity": 0.38}
}
```

## Python API

```python
from driftlens.analytics.drift import cosine_similarity, compute_drift_series, centroid
from driftlens.analytics.network import build_cooccurrence_graph, export_for_d3
from driftlens.db.session import Database, Document
import numpy as np

# Build a graph
docs = [{"text": "climate policy economy", "id": "1"}]
G = build_cooccurrence_graph(docs)
payload = export_for_d3(G)  # {"nodes": [...], "edges": [...]}

# Drift
centroids = {"2020-01": np.array([1.0, 0.0]), "2021-01": np.array([0.0, 1.0])}
series = compute_drift_series(centroids)

# DB
db = Database("driftlens.db")
db.init()
with db.session() as conn:
    db.insert_document(conn, Document(id="1", text="hello world", slice_key="2020-01"))
```

## Configuration

All settings are in `driftlens/config.py` via `pydantic-settings` with env prefix `DRIFTLENS_`:

| Env var | Default | Description |
|---|---|---|
| `DRIFTLENS_DB_PATH` | `driftlens.db` | SQLite file |
| `DRIFTLENS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `DRIFTLENS_BATCH_SIZE` | `32` | Encoding batch size |
| `DRIFTLENS_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `DRIFTLENS_LOG_LEVEL` | `INFO` | Log level |
| `DRIFTLENS_SLICE_STRATEGY` | `temporal` | `temporal` / `categorical` / `none` |

## Testing

```bash
pytest -q
pytest tests/test_drift.py tests/test_network.py -v
```

## Project Structure

```
driftlens/
  __init__.py
  config.py
  cli.py                  # Click CLI
  analytics/
    drift.py              # centroid, cosine drift
    network.py            # co-occurrence graphs
  api/
    app.py                # FastAPI app
  db/
    schema.sql
    session.py            # Database, Document, Embedding, TermCentroid
tests/
  test_drift.py
  test_network.py
```

## License

MIT
