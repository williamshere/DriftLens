"""Thread-safe SQLite session with WAL mode and vector serialisation.

Vectors are stored as raw ``float32`` bytes (``np.ndarray.tobytes()``) and
rehydrated with ``np.frombuffer(..., dtype=np.float32)`` — compact, fast,
and portable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialise a 1-D float32 vector to bytes.

    Args:
        vec: 1-D numpy array. Will be cast to ``float32`` if needed.

    Returns:
        Raw bytes suitable for BLOB storage.
    """
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return arr.tobytes()


def deserialize_vector(blob: bytes, dim: int | None = None) -> np.ndarray:
    """Deserialise bytes back to a 1-D float32 array.

    Args:
        blob: Raw bytes from the ``vector`` BLOB column.
        dim: Optional expected dimensionality for validation.

    Returns:
        1-D ``float32`` array (copy).
    """
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if dim is not None and arr.size != dim:
        raise ValueError(f"Vector dim mismatch: expected {dim}, got {arr.size}")
    return arr


# ---------------------------------------------------------------------------
# Dataclasses for row mapping
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Document:
    """Represents a row in the ``documents`` table."""

    id: str
    text: str
    title: str | None = None
    slice_key: str | None = None
    meta: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class Embedding:
    """Represents a row in the ``embeddings`` table."""

    doc_id: str
    vector: np.ndarray
    dim: int
    model: str
    created_at: str | None = None


@dataclass(slots=True)
class TermCentroid:
    """Represents a row in the ``term_centroids`` table."""

    term: str
    slice_key: str
    vector: np.ndarray
    dim: int
    doc_count: int
    model: str
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Row mappers (internal)
# ---------------------------------------------------------------------------


def _row_to_document(row: sqlite3.Row) -> Document:
    meta = json.loads(row["meta_json"]) if row["meta_json"] else None
    return Document(
        id=row["id"],
        text=row["text"],
        title=row["title"],
        slice_key=row["slice_key"],
        meta=meta,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_embedding(row: sqlite3.Row) -> Embedding:
    blob: bytes = row["vector"]
    dim: int = row["dim"]
    vec = deserialize_vector(blob, dim=dim)
    return Embedding(
        doc_id=row["doc_id"],
        vector=vec,
        dim=dim,
        model=row["model"],
        created_at=row["created_at"],
    )


def _row_to_centroid(row: sqlite3.Row) -> TermCentroid:
    blob: bytes = row["vector"]
    dim: int = row["dim"]
    vec = deserialize_vector(blob, dim=dim)
    return TermCentroid(
        term=row["term"],
        slice_key=row["slice_key"],
        vector=vec,
        dim=dim,
        doc_count=row["doc_count"],
        model=row["model"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# Core Database class
# ---------------------------------------------------------------------------


class Database:
    """Thread-safe SQLite database with WAL mode.

    Supports both simple self-managed connections and explicit
    ``session()`` / ``conn``-passing usage.

    Example::

        db = Database("driftlens.db")
        db.init_db()

        # Simple API (no conn needed)
        db.insert_document(doc_id="1", text="hello world")

        # Explicit session / transaction
        with db.session() as conn:
            db.insert_document(doc_id="2", text="hi", conn=conn)

        # Context-manager for lifecycle
        with Database("driftlens.db") as db:
            db.insert_document(doc_id="3", text="hey")
    """

    def __init__(
        self,
        db_path: str | Path = "driftlens.db",
        *,
        timeout: float = 30.0,
    ) -> None:
        self.db_path: Path | str = db_path
        self.timeout = timeout
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        # Ensure parent dir exists (except for :memory:)
        if isinstance(db_path, Path) and str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        elif isinstance(db_path, str) and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL & foreign-keys enabled.

        Each call returns a *new* connection.  For the shared
        transaction-aware connection use :meth:`session` or the
        :attr:`conn` property.
        """
        if isinstance(self.db_path, str) and self.db_path == ":memory:":
            conn = sqlite3.connect(":memory:", timeout=self.timeout, check_same_thread=False)
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=self.timeout, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        with self._lock:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _shared_connect(self) -> sqlite3.Connection:
        """Return/reuse the shared connection for ``session()`` / ``conn``."""
        with self._lock:
            if self._conn is not None:
                return self._conn
            if isinstance(self.db_path, str) and self.db_path == ":memory:":
                conn = sqlite3.connect(":memory:", timeout=self.timeout, check_same_thread=False)
            else:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.timeout,
                    check_same_thread=False,
                    isolation_level=None,
                )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn = conn
            logger.debug("Connected to SQLite at %s (WAL mode)", self.db_path)
            return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Shared connection, opened on demand."""
        return self._shared_connect()

    def close(self) -> None:
        """Close the shared connection if open."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                logger.debug("Closed SQLite connection at %s", self.db_path)

    def init_db(self) -> None:
        """Create tables / indexes from :file:`schema.sql` (idempotent)."""
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._lock:
            with self.connect() as conn:
                conn.executescript(schema)
                conn.commit()
        logger.info("Database schema initialised at %s", self.db_path)

    # Alias for compatibility with earlier spec
    init = init_db

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> Database:
        self._shared_connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a shared connection inside an explicit transaction.

        Commits on success, rolls back on exception. The connection
        itself is *not* closed — only the transaction is finalised.
        """
        conn = self._shared_connect()
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise

    # ==================================================================
    # Document CRUD
    # ==================================================================

    def insert_document(
        self,
        doc_id: str | None = None,
        text: str | None = None,
        title: str | None = None,
        slice_key: str | None = None,
        meta: dict[str, Any] | None = None,
        *,
        conn: sqlite3.Connection | None = None,
        document: Document | None = None,
    ) -> str:
        """Insert or replace a single document row.

        Supports two calling conventions for backwards compatibility:

        1. ``db.insert_document(doc_id="1", text="hello")``
        2. ``db.insert_document(document=Document(id="1", text="hello"))``

        Args:
            doc_id: Primary key. Generated as UUID4 when ``None``.
            text: Document body (required unless *document* is given).
            title: Optional title.
            slice_key: Temporal / community slice key.
            meta: Arbitrary JSON-serialisable metadata.
            conn: Optional explicit connection (uses a new one if ``None``).
            document: Alternative :class:`Document` instance.

        Returns:
            The document id used.
        """
        if document is not None:
            doc_id = document.id
            text = document.text
            title = document.title
            slice_key = document.slice_key
            meta = document.meta
        if text is None:
            raise ValueError("text is required")
        doc_id = doc_id or str(uuid.uuid4())
        meta_json = json.dumps(meta) if meta is not None else None

        def _exec(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT OR REPLACE INTO documents (id, text, title, slice_key, meta_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, text, title, slice_key, meta_json),
            )

        if conn is not None:
            with self._lock:
                _exec(conn)
                # commit only if not inside an explicit session transaction
                if not conn.in_transaction:
                    conn.commit()
        else:
            with self._lock:
                with self.connect() as c:
                    _exec(c)
                    c.commit()
        logger.debug("Upserted document id=%s slice=%s", doc_id, slice_key)
        return doc_id

    def insert_documents(
        self,
        docs: list[dict[str, Any]] | list[Document],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> list[str]:
        """Bulk insert multiple documents.

        Each dict must contain ``text`` and may contain ``id``, ``title``,
        ``slice_key``, and ``meta`` (or ``meta_json``).

        Args:
            docs: List of document dicts or :class:`Document` objects.
            conn: Optional explicit connection.

        Returns:
            List of document ids in insertion order.
        """
        ids: list[str] = []

        def _exec_batch(c: sqlite3.Connection) -> None:
            for doc in docs:
                if isinstance(doc, Document):
                    doc_id = doc.id
                    text = doc.text
                    title = doc.title
                    slice_key = doc.slice_key
                    meta_json = json.dumps(doc.meta) if doc.meta is not None else None
                else:
                    doc_id = doc.get("id") or str(uuid.uuid4())
                    text: str = doc["text"]
                    title = doc.get("title")
                    slice_key = doc.get("slice_key")
                    meta = doc.get("meta", doc.get("meta_json"))
                    if isinstance(meta, dict):
                        meta_json = json.dumps(meta)
                    elif isinstance(meta, str):
                        meta_json = meta
                    else:
                        meta_json = None
                c.execute(
                    "INSERT OR REPLACE INTO documents (id, text, title, slice_key, meta_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, text, title, slice_key, meta_json),
                )
                ids.append(doc_id)

        if conn is not None:
            with self._lock:
                _exec_batch(conn)
                if not conn.in_transaction:
                    conn.commit()
        else:
            with self._lock:
                with self.connect() as c:
                    _exec_batch(c)
                    c.commit()
        return ids

    def count_documents(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        slice_key: str | None = None,
    ) -> int:
        """Return total document count, optionally filtered by slice."""
        sql = "SELECT COUNT(*) FROM documents"
        params: tuple[Any, ...] = ()
        if slice_key is not None:
            sql += " WHERE slice_key = ?"
            params = (slice_key,)

        def _query(c: sqlite3.Connection) -> int:
            cur = c.execute(sql, params)
            return int(cur.fetchone()[0])

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def fetch_documents(
        self,
        limit: int | None = None,
        *,
        conn: sqlite3.Connection | None = None,
        slice_key: str | None = None,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        """Fetch raw document rows.

        Args:
            limit: Optional row limit.
            conn: Optional explicit connection.
            slice_key: Optional slice filter.
            offset: Row offset.

        Returns:
            List of row objects.
        """
        # Keep raw Row return for backwards compatibility
        docs = self.list_documents(conn=conn, slice_key=slice_key, limit=limit, offset=offset)
        # But fetch_documents historically returned sqlite3.Row; we emulate via re-query
        # For compatibility we return Document-like rows? Instead return list of Documents as dicts.
        # To preserve Row interface, re-fetch as rows:
        if conn is not None:
            with self._lock:
                if slice_key is not None:
                    cur = conn.execute(
                        "SELECT * FROM documents WHERE slice_key = ? ORDER BY id LIMIT ? OFFSET ?",
                        (slice_key, limit if limit is not None else -1, offset),
                    )
                elif limit is not None:
                    cur = conn.execute("SELECT * FROM documents ORDER BY id LIMIT ?", (limit,))
                else:
                    cur = conn.execute("SELECT * FROM documents ORDER BY id")
                return list(cur.fetchall())
        with self.connect() as c:
            if slice_key is not None:
                cur = c.execute(
                    "SELECT * FROM documents WHERE slice_key = ? ORDER BY id LIMIT ? OFFSET ?",
                    (slice_key, limit if limit is not None else -1, offset),
                )
            elif limit is not None:
                cur = c.execute("SELECT * FROM documents ORDER BY id LIMIT ?", (limit,))
            else:
                cur = c.execute("SELECT * FROM documents ORDER BY id")
            return list(cur.fetchall())

    def list_documents(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        slice_key: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Document]:
        """List documents as :class:`Document` objects, optionally filtered."""
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if slice_key is not None:
            sql += " WHERE slice_key = ?"
            params.append(slice_key)
        sql += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([limit if limit is not None else -1, offset])

        def _query(c: sqlite3.Connection) -> list[Document]:
            rows = c.execute(sql, params).fetchall()
            return [_row_to_document(r) for r in rows]

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def get_document(
        self,
        doc_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> Document | None:
        """Fetch a single document by id, or ``None`` if not found."""
        sql = "SELECT * FROM documents WHERE id = ?"

        def _query(c: sqlite3.Connection) -> Document | None:
            row = c.execute(sql, (doc_id,)).fetchone()
            if row is None:
                return None
            return _row_to_document(row)

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def delete_document(
        self,
        doc_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Delete a document (cascades to embeddings). Returns True if deleted."""
        def _exec(c: sqlite3.Connection) -> bool:
            cur = c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            if not c.in_transaction:
                c.commit()
            return cur.rowcount > 0

        if conn is not None:
            with self._lock:
                result = _exec(conn)
                # if inside session transaction, commit is deferred
                return result
        with self._lock:
            with self.connect() as c:
                return _exec(c)

    def iter_documents_batched(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        batch_size: int = 100,
    ) -> Iterator[list[Document]]:
        """Yield documents in batches (useful for embedding pipelines)."""
        offset = 0
        while True:
            batch = self.list_documents(conn=conn, limit=batch_size, offset=offset)
            if not batch:
                break
            yield batch
            # For conn-based iteration we need to avoid infinite loop; advance offset
            offset += len(batch)
            # If conn was None we created new connection each time, so offset tracking is correct

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def insert_embedding(
        self,
        doc_id: str | None = None,
        vector: np.ndarray | None = None,
        model: str = "all-MiniLM-L6-v2",
        *,
        conn: sqlite3.Connection | None = None,
        embedding: Any | None = None,
    ) -> None:
        """Store an embedding vector for a document (insert or replace).

        Supports ``embedding=Embedding(...)`` as alternative.

        Args:
            doc_id: Foreign key to ``documents.id``.
            vector: 1-D float32 array.
            model: Embedding model identifier.
            conn: Optional explicit connection.
            embedding: Optional :class:`Embedding` instance.
        """
        if embedding is not None:
            doc_id = getattr(embedding, "doc_id", doc_id)
            vector = getattr(embedding, "vector", vector)
            model = getattr(embedding, "model", model)
        if doc_id is None or vector is None:
            raise ValueError("doc_id and vector are required (or provide embedding=)")
        vec = np.asarray(vector, dtype=np.float32)
        blob: bytes = vec.tobytes()
        dim: int = int(vec.shape[0])

        def _exec(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT OR REPLACE INTO embeddings (doc_id, vector, dim, model) VALUES (?, ?, ?, ?)",
                (doc_id, blob, dim, model),
            )
            if not c.in_transaction:
                c.commit()

        if conn is not None:
            with self._lock:
                _exec(conn)
        else:
            with self._lock:
                with self.connect() as c:
                    _exec(c)
        logger.debug("Upserted embedding doc_id=%s model=%s dim=%d", doc_id, model, dim)

    # Alias for richer API naming
    upsert_embedding = insert_embedding

    def fetch_embedding(
        self,
        doc_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> np.ndarray | None:
        """Fetch an embedding vector by document id.

        Args:
            doc_id: Document primary key.
            conn: Optional explicit connection.

        Returns:
            Float32 array or ``None`` if not found.
        """
        emb = self.get_embedding(doc_id, conn=conn)
        if emb is None:
            return None
        return emb.vector

    def get_embedding(
        self,
        doc_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> Embedding | None:
        """Fetch embedding as :class:`Embedding` object, or ``None``."""
        sql = "SELECT * FROM embeddings WHERE doc_id = ?"

        def _query(c: sqlite3.Connection) -> Embedding | None:
            row = c.execute(sql, (doc_id,)).fetchone()
            if row is None:
                return None
            return _row_to_embedding(row)

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def list_embeddings(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        model: str | None = None,
    ) -> list[Embedding]:
        """List embeddings, optionally filtered by model."""
        sql = "SELECT * FROM embeddings"
        params: tuple[Any, ...] = ()
        if model is not None:
            sql += " WHERE model = ?"
            params = (model,)
        sql += " ORDER BY doc_id"

        def _query(c: sqlite3.Connection) -> list[Embedding]:
            rows = c.execute(sql, params).fetchall()
            return [_row_to_embedding(r) for r in rows]

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def delete_embedding(
        self,
        doc_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Delete an embedding. Returns True if deleted."""
        def _exec(c: sqlite3.Connection) -> bool:
            cur = c.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
            if not c.in_transaction:
                c.commit()
            return cur.rowcount > 0

        if conn is not None:
            with self._lock:
                return _exec(conn)
        with self._lock:
            with self.connect() as c:
                return _exec(c)

    def count_embeddings(
        self,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Count total embeddings."""
        def _query(c: sqlite3.Connection) -> int:
            row = c.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0])

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    # ------------------------------------------------------------------
    # Term-centroid helpers
    # ------------------------------------------------------------------

    def upsert_centroid(
        self,
        term: str | None = None,
        slice_key: str | None = None,
        vector: np.ndarray | None = None,
        model: str = "all-MiniLM-L6-v2",
        doc_count: int = 0,
        *,
        conn: sqlite3.Connection | None = None,
        centroid: TermCentroid | None = None,
    ) -> None:
        """Insert or replace a term centroid.

        Supports both signatures:

        1. ``db.upsert_centroid(term="hello", slice_key="2024", vector=vec)``
        2. ``db.upsert_centroid(centroid=TermCentroid(...))``

        Args:
            term: Term / keyword.
            slice_key: Slice identifier.
            vector: Centroid vector (float32).
            model: Embedding model identifier.
            doc_count: Number of documents contributing.
            conn: Optional explicit connection.
            centroid: Alternative :class:`TermCentroid` instance.
        """
        if centroid is not None:
            term = centroid.term
            slice_key = centroid.slice_key
            vector = centroid.vector
            model = centroid.model
            doc_count = centroid.doc_count
        if term is None or slice_key is None or vector is None:
            raise ValueError("term, slice_key and vector are required (or provide centroid=)")
        vec = np.asarray(vector, dtype=np.float32)
        blob: bytes = vec.tobytes()
        dim: int = int(vec.shape[0])

        def _exec(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT OR REPLACE INTO term_centroids "
                "(term, slice_key, vector, dim, doc_count, model) VALUES (?, ?, ?, ?, ?, ?)",
                (term, slice_key, blob, dim, doc_count, model),
            )
            if not c.in_transaction:
                c.commit()

        if conn is not None:
            with self._lock:
                _exec(conn)
        else:
            with self._lock:
                with self.connect() as c:
                    _exec(c)
        logger.debug("Upserted centroid term=%s slice=%s count=%d", term, slice_key, doc_count)

    def fetch_centroid(
        self,
        term: str,
        slice_key: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> np.ndarray | None:
        """Fetch a stored centroid vector.

        Args:
            term: Term.
            slice_key: Slice key.
            conn: Optional explicit connection.

        Returns:
            Float32 array or ``None``.
        """
        c_obj = self.get_centroid(term, slice_key, conn=conn)
        if c_obj is None:
            return None
        return c_obj.vector

    def get_centroid(
        self,
        term: str,
        slice_key: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> TermCentroid | None:
        """Fetch a centroid as :class:`TermCentroid`, or ``None``."""
        sql = "SELECT * FROM term_centroids WHERE term = ? AND slice_key = ?"

        def _query(c: sqlite3.Connection) -> TermCentroid | None:
            row = c.execute(sql, (term, slice_key)).fetchone()
            if row is None:
                return None
            return _row_to_centroid(row)

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def list_centroids(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        term: str | None = None,
        slice_key: str | None = None,
    ) -> list[TermCentroid]:
        """List centroids, optionally filtered by term and/or slice."""
        conditions: list[str] = []
        params: list[Any] = []
        if term is not None:
            conditions.append("term = ?")
            params.append(term)
        if slice_key is not None:
            conditions.append("slice_key = ?")
            params.append(slice_key)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM term_centroids {where} ORDER BY term, slice_key"

        def _query(c: sqlite3.Connection) -> list[TermCentroid]:
            rows = c.execute(sql, params).fetchall()
            return [_row_to_centroid(r) for r in rows]

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)

    def delete_centroid(
        self,
        term: str,
        slice_key: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Delete a centroid. Returns True if deleted."""
        def _exec(c: sqlite3.Connection) -> bool:
            cur = c.execute(
                "DELETE FROM term_centroids WHERE term = ? AND slice_key = ?", (term, slice_key)
            )
            if not c.in_transaction:
                c.commit()
            return cur.rowcount > 0

        if conn is not None:
            with self._lock:
                return _exec(conn)
        with self._lock:
            with self.connect() as c:
                return _exec(c)

    def count_centroids(
        self,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Count total centroids."""
        def _query(c: sqlite3.Connection) -> int:
            row = c.execute("SELECT COUNT(*) FROM term_centroids").fetchone()
            return int(row[0])

        if conn is not None:
            with self._lock:
                return _query(conn)
        with self.connect() as c:
            return _query(c)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_default_db: Database | None = None


def get_db(db_path: str | Path | None = None) -> Database:
    """Return a :class:`Database` handle.

    Args:
        db_path: Optional path override. Falls back to ``Settings().db_path``
            or ``driftlens.db`` when ``None``.

    Returns:
        Database instance (not yet connected).
    """
    if db_path is not None:
        return Database(db_path)
    try:
        from driftlens.config import settings as _settings

        return Database(_settings.db_path)
    except Exception:
        return Database("driftlens.db")


def init_db(db_path: str | Path | None = None) -> Database:
    """Initialise the database schema and return the handle.

    Args:
        db_path: Optional path override.

    Returns:
        Initialised :class:`Database`.
    """
    db = get_db(db_path)
    db.init_db()
    return db
