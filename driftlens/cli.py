"""DriftLens CLI — Click-based entry point.

Commands
--------
* ``ingest`` — load documents from JSON/CSV into SQLite, partitioned by slice.
* ``process-embeddings`` — encode document texts with sentence-transformers.
* ``compute-drift`` — aggregate term centroids & report drift scores.
* ``serve`` — start the FastAPI server via uvicorn.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np

from driftlens.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def _load_documents_from_file(path: Path) -> list[dict[str, Any]]:
    """Load documents from JSON or CSV file.

    JSON: expects a list of ``{id, text, ...}`` objects or ``{"documents": [...]}``.
    CSV: expects at least ``id`` and ``text`` columns.
    """
    if not path.exists():
        raise click.ClickException(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "documents" in data:
            data = data["documents"]
        if not isinstance(data, list):
            raise click.ClickException("JSON file must contain a list of documents or {\"documents\": [...]}")
        return data  # type: ignore[return-value]
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise click.ClickException("CSV must have at least a 'text' column")
            return list(reader)
    raise click.ClickException(f"Unsupported file type {suffix!r}; use .json or .csv")


def _slice_key_for_doc(doc: dict[str, Any], slice_by: str | None) -> str | None:
    """Derive slice_key from a document dict."""
    if slice_by is None or slice_by == "none":
        return None
    # If slice_by is a field name in the doc, use it directly.
    if slice_by in doc:
        val = doc[slice_by]
        return str(val) if val is not None else None
    # Common alias: date field may contain full timestamp; slice by YYYY-MM
    if slice_by in ("temporal", "date") and "date" in doc:
        raw = str(doc["date"])
        # Try to extract YYYY-MM
        return raw[:7] if len(raw) >= 7 else raw
    # Fallback: treat slice_by as date field name
    if slice_by in doc:
        return str(doc[slice_by])
    return None


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--db-path", default=None, help="Path to SQLite DB (overrides config).")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def app(ctx: click.Context, db_path: str | None, verbose: bool) -> None:
    """DriftLens — semantic drift detection across document corpora."""
    _setup_logging("DEBUG" if verbose else "INFO")
    settings = Settings()  # type: ignore[call-arg]
    if db_path:
        settings.db_path = Path(db_path)
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command()
@click.option("--file", "file_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="JSON or CSV file to ingest.")
@click.option("--slice-by", default=None, help="Document field to use as slice key (e.g. 'date', 'category'). Use 'none' for no slicing.")
@click.option("--id-field", default="id", help="Field containing document ID (auto-generated if missing).")
@click.option("--text-field", default="text", help="Field containing document text.")
@click.pass_context
def ingest(
    ctx: click.Context,
    file_path: Path,
    slice_by: str | None,
    id_field: str,
    text_field: str,
) -> None:
    """Ingest documents from a file into the database."""
    from driftlens.db.session import Database, Document

    settings: Settings = ctx.obj["settings"]
    docs_raw = _load_documents_from_file(file_path)

    db = Database(settings.db_path)
    db.init_db()
    count = 0
    with db.session() as conn:
        for idx, raw in enumerate(docs_raw):
            doc_id = str(raw.get(id_field) or raw.get("id") or f"doc-{idx}")
            text = raw.get(text_field) or raw.get("content") or ""
            if not text:
                logger.warning("Skipping document %s with empty text", doc_id)
                continue
            title = raw.get("title")
            slice_key = _slice_key_for_doc(raw, slice_by)
            # Preserve extra fields as meta
            meta = {k: v for k, v in raw.items() if k not in (id_field, text_field, "text", "content", "title", slice_by or "")}
            doc = Document(id=doc_id, text=str(text), title=str(title) if title else None, slice_key=slice_key, meta=meta or None)
            db.insert_document(document=doc, conn=conn)
            count += 1

    click.echo(f"Ingested {count} documents from {file_path} into {settings.db_path} (slice_by={slice_by!r})")


# ---------------------------------------------------------------------------
# process-embeddings
# ---------------------------------------------------------------------------


@app.command("process-embeddings")
@click.option("--model", default=None, help="Sentence-transformers model (overrides config).")
@click.option("--batch-size", default=None, type=int, help="Encoding batch size.")
@click.option("--limit", default=None, type=int, help="Only process first N documents (for testing).")
@click.pass_context
def process_embeddings(
    ctx: click.Context,
    model: str | None,
    batch_size: int | None,
    limit: int | None,
) -> None:
    """Encode document texts and store embeddings."""
    from driftlens.db.session import Database, Embedding

    settings: Settings = ctx.obj["settings"]
    model_name = model or settings.embedding_model
    bs = batch_size or settings.batch_size

    db = Database(settings.db_path)
    db.init_db()

    with db.session() as conn:
        docs = db.list_documents(conn=conn, limit=limit)
    if not docs:
        click.echo("No documents found; run 'driftlens ingest' first.", err=True)
        sys.exit(1)

    click.echo(f"Encoding {len(docs)} documents with model '{model_name}' (batch_size={bs}) ...")

    # Lazy import sentence_transformers (heavy)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        st_model = SentenceTransformer(model_name, device=settings.resolved_device())
        texts = [d.text for d in docs]
        vectors = st_model.encode(texts, batch_size=bs, show_progress_bar=True, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)
    except ImportError:
        logger.warning("sentence-transformers not installed; using random mock embeddings for demo")
        # Mock embeddings — deterministic via seeded RNG for reproducibility
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((len(docs), settings.embedding_dim)).astype(np.float32)
        # Normalize to unit length to mimic real embeddings
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-9)
    except Exception as exc:
        click.echo(f"Embedding encoding failed: {exc}", err=True)
        sys.exit(1)

    # Store
    with db.session() as conn:
        for doc, vec in zip(docs, vectors):
            emb = Embedding(doc_id=doc.id, vector=np.asarray(vec, dtype=np.float32), dim=int(vec.size), model=model_name)
            db.upsert_embedding(embedding=emb, conn=conn)

    click.echo(f"Stored {len(docs)} embeddings (dim={vectors.shape[1]}, model={model_name})")


# ---------------------------------------------------------------------------
# compute-drift
# ---------------------------------------------------------------------------


@app.command("compute-drift")
@click.option("--terms", default=None, help="Comma-separated terms to compute drift for (default: all terms with centroids).")
@click.option("--metric", default="cosine", type=click.Choice(["cosine", "euclidean"]), help="Distance metric.")
@click.option("--top-k", default=None, type=int, help="Only report top K terms by drift magnitude.")
@click.pass_context
def compute_drift(
    ctx: click.Context,
    terms: str | None,
    metric: str,
    top_k: int | None,
) -> None:
    """Compute and display drift scores per term across slices."""
    from driftlens.analytics.drift import compute_drift_series, drift_magnitude
    from driftlens.db.session import Database, TermCentroid

    settings: Settings = ctx.obj["settings"]
    db = Database(settings.db_path)
    db.init_db()

    # If term centroids don't exist, try to build them from embeddings
    # (simple aggregation: per-term centroid via TF grouping).
    with db.session() as conn:
        all_centroids = db.list_centroids(conn=conn)

    if not all_centroids:
        click.echo("No term centroids found. Building centroids from embeddings ...")
        _build_term_centroids(db)
        with db.session() as conn:
            all_centroids = db.list_centroids(conn=conn)
        if not all_centroids:
            click.echo("No centroids could be built (no embeddings or no terms).", err=True)
            sys.exit(1)

    # Filter terms if requested
    if terms:
        wanted = {t.strip() for t in terms.split(",") if t.strip()}
        all_centroids = [c for c in all_centroids if c.term in wanted]
        if not all_centroids:
            click.echo(f"No centroids found for terms: {sorted(wanted)}", err=True)
            sys.exit(1)

    # Group by term -> {slice: vector}
    from collections import defaultdict

    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for c in all_centroids:
        grouped[c.term][c.slice_key] = c.vector

    results: list[dict[str, Any]] = []
    for term, cent_map in grouped.items():
        series = compute_drift_series(cent_map, metric=metric)
        mag = drift_magnitude(series)
        results.append({"term": term, "magnitude": mag, "series": series, "num_slices": len(cent_map)})

    results.sort(key=lambda r: r["magnitude"], reverse=True)  # type: ignore[return-value]
    if top_k is not None:
        results = results[:top_k]

    # Pretty output
    for r in results:
        click.echo(f"\nTerm: {r['term']}  |  slices={r['num_slices']}  |  drift_magnitude={r['magnitude']:.4f}")
        for step in r["series"]:  # type: ignore[attr-defined]
            click.echo(f"  {step['from_slice']} -> {step['to_slice']}: drift={step['drift']:.4f}  sim={step['similarity']:.4f}")

    if not results:
        click.echo("No drift results.")


def _build_term_centroids(db: Any) -> None:
    """Build term centroids from stored embeddings and documents.

    For each distinct term (extracted via network.extract_terms) and slice,
    compute centroid of embeddings for docs containing that term.
    """
    from collections import defaultdict

    from driftlens.analytics.drift import centroid
    from driftlens.analytics.network import extract_terms
    from driftlens.db.session import TermCentroid

    with db.session() as conn:
        docs = db.list_documents(conn=conn)
        # Build map doc_id -> embedding vector
        embeddings = {e.doc_id: e.vector for e in db.list_embeddings(conn=conn)}

    if not embeddings:
        return

    # Map (term, slice) -> list of vectors
    buckets: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    # Need doc slice info; build id->slice
    doc_slice: dict[str, str] = {}
    doc_text: dict[str, str] = {}
    for d in docs:
        doc_slice[d.id] = d.slice_key or "__none__"
        doc_text[d.id] = d.text

    for doc_id, vec in embeddings.items():
        text = doc_text.get(doc_id, "")
        terms = set(extract_terms(text))
        sk = doc_slice.get(doc_id, "__none__")
        for t in terms:
            buckets[(t, sk)].append(vec)

    # Persist centroids
    with db.session() as conn:
        # Need model name from any embedding (or default)
        from driftlens.config import Settings

        settings = Settings()  # type: ignore[call-arg]
        model = settings.embedding_model
        # Try to get actual model from first embedding row
        embs = db.list_embeddings(conn=conn)
        if embs:
            model = embs[0].model
        for (term, slice_key), vecs in buckets.items():
            cent = centroid(vecs)
            tc = TermCentroid(term=term, slice_key=slice_key, vector=cent, dim=int(cent.size), doc_count=len(vecs), model=model)
            db.upsert_centroid(centroid=tc, conn=conn)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", help="Host to bind.")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev).")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str, reload: bool) -> None:
    """Start the DriftLens API server."""
    import uvicorn  # type: ignore[import-not-found]

    settings: Settings = ctx.obj["settings"]
    _setup_logging(settings.log_level)
    click.echo(f"Starting DriftLens API on http://{host}:{port} (db={settings.db_path})")
    uvicorn.run("driftlens.api.app:app", host=host, port=port, reload=reload, log_level=settings.log_level.lower())


# Alias for pyproject entry point ``driftlens = "driftlens.cli:app"``
# (Click group is callable directly).
