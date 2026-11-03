"""FastAPI application for DriftLens — drift & network endpoints.

Endpoints
---------
* ``GET /health`` / ``GET /api/health`` — liveness probe.
* ``GET /api/drift?term=`` — drift series for a term across slices.
* ``GET /api/network?slice=`` — co-occurrence graph for a slice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from driftlens.config import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional :class:`Settings` instance.  When ``None`` a
            default :class:`Settings` is constructed (reads env / .env).

    Returns:
        Configured :class:`FastAPI` instance.
    """
    settings = settings or Settings()  # type: ignore[call-arg]

    app = FastAPI(
        title="DriftLens API",
        description="Semantic drift detection across document corpora",
        version="0.1.0",
    )

    # CORS — permissive for local dev; tighten in production via env.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_db():
        from driftlens.db.session import Database

        db = Database(settings.db_path)
        db.connect()
        # Ensure schema exists (idempotent)
        try:
            db.init()
        except Exception as exc:  # pragma: no cover
            logger.warning("DB init warning: %s", exc)
        return db

    # ------------------------------------------------------------------
    # Health & Dashboard
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/dashboard", response_class=HTMLResponse, tags=["dashboard"])
    def dashboard() -> HTMLResponse:
        dashboard_path = Path(__file__).parent / "dashboard.html"
        if dashboard_path.exists():
            return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>DriftLens Dashboard</h2><p>API is running.</p>")

    @app.get("/health", tags=["health"])
    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": "0.1.0"}

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------

    @app.get("/api/drift", tags=["drift"])
    def get_drift(
        term: str = Query(..., description="Term to compute drift for", min_length=1),
        metric: str = Query("cosine", description="Distance metric: cosine or euclidean"),
    ) -> dict[str, Any]:
        """Return drift series for *term* across all slices.

        Reads ``term_centroids`` from SQLite.  Returns 404 when the term has
        no centroids, 400 for invalid metric.
        """
        if metric not in ("cosine", "euclidean"):
            raise HTTPException(status_code=400, detail="metric must be 'cosine' or 'euclidean'")

        from driftlens.analytics.drift import compute_drift_series
        from driftlens.db.session import deserialize_vector

        db = _get_db()
        try:
            with db.session() as conn:
                centroids_raw = db.list_centroids(conn, term=term)
                if not centroids_raw:
                    raise HTTPException(status_code=404, detail=f"No centroids found for term '{term}'")
                # Build slice -> vector map
                centroids: dict[str, Any] = {}
                for c in centroids_raw:
                    centroids[c.slice_key] = c.vector
                series = compute_drift_series(centroids, metric=metric)
                # Also include raw centroids metadata
                slices_info = [
                    {"slice_key": c.slice_key, "doc_count": c.doc_count, "dim": c.dim}
                    for c in centroids_raw
                ]
                return {
                    "term": term,
                    "metric": metric,
                    "slices": sorted(centroids.keys()),
                    "slices_info": slices_info,
                    "series": series,
                    "num_slices": len(centroids),
                }
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("drift endpoint error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            # Don't close shared connection aggressively; just leave it.
            pass

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    @app.get("/api/network", tags=["network"])
    def get_network(
        slice: str | None = Query(None, alias="slice", description="Slice key to build graph for (None = all)"),
        min_edge_weight: int = Query(1, ge=1, description="Minimum co-occurrence weight"),
        min_term_freq: int = Query(1, ge=1, description="Minimum term frequency"),
    ) -> dict[str, Any]:
        """Return co-occurrence graph for a slice as ``{nodes, edges}``.

        Query param ``slice`` selects a single time slice.  When omitted the
        graph is built from all documents.
        """
        from driftlens.analytics.network import build_cooccurrence_graph, export_for_d3, graph_metrics_summary

        db = _get_db()
        try:
            with db.session() as conn:
                if slice is not None:
                    docs = db.list_documents(conn, slice_key=slice)
                    if not docs:
                        # Check if slice exists at all (maybe no docs yet)
                        # Return empty graph rather than 404 to keep API lenient
                        # but surface warning.
                        pass
                else:
                    docs = db.list_documents(conn)
                if not docs:
                    return {
                        "slice": slice,
                        "nodes": [],
                        "edges": [],
                        "metrics": {
                            "num_nodes": 0,
                            "num_edges": 0,
                            "density": 0.0,
                            "modularity": 0.0,
                            "num_communities": 0,
                            "top_degree": [],
                            "top_betweenness": [],
                        },
                    }
                G = build_cooccurrence_graph(
                    docs,  # type: ignore[arg-type]
                    min_edge_weight=min_edge_weight,
                    min_term_freq=min_term_freq,
                )
                exported = export_for_d3(G)
                metrics = graph_metrics_summary(G)
                return {"slice": slice, **exported, "metrics": metrics}
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("network endpoint error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Error handling — ensure JSON errors
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):  # type: ignore[no-untyped-def]
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


# Default app instance for ``uvicorn driftlens.api.app:app``
app = create_app()
