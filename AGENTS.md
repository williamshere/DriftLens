# Antigravity Guidelines for DriftLens

This document configures and guides Antigravity agents working in this repository.

## Python Environment & Virtualenv
- The project virtual environment is located at `.venv/`.
- Python binary: `.venv/Scripts/python.exe`.
- All dependencies (including `driftlens` editable package) are installed in `.venv`.
- Always run scripts and commands using `.venv/Scripts/python.exe` or `.venv/Scripts/driftlens.exe`.

## Key Commands
- **Run Example Pipeline:**
  ```powershell
  .venv\Scripts\python.exe run_example.py
  ```
- **CLI Ingestion & Processing:**
  ```powershell
  .venv\Scripts\driftlens.exe --db-path example.db ingest --file example_corpus.csv --slice-by timestamp
  .venv\Scripts\driftlens.exe --db-path example.db process-embeddings
  .venv\Scripts\driftlens.exe --db-path example.db compute-drift --metric cosine
  ```
- **FastAPI Server:**
  ```powershell
  .venv\Scripts\driftlens.exe --db-path example.db serve --port 8000
  ```
- **Run Tests:**
  ```powershell
  .venv\Scripts\python.exe -m pytest tests/ -v
  ```

## Architecture Overview
- `driftlens/db/session.py`: SQLite storage (WAL mode) for documents, embeddings, and term centroids.
- `driftlens/analytics/embedder.py`: Embedding computation with fallback mock generator when `sentence-transformers` is absent.
- `driftlens/analytics/drift.py`: Centroid computation, cosine/Euclidean distance tracking, and multi-slice trajectory series.
- `driftlens/analytics/network.py`: Co-occurrence graph construction, community detection, centrality metrics, and D3 export.
- `driftlens/api/app.py`: FastAPI endpoints for health, drift series (`/api/drift`), and network graphs (`/api/network`).

