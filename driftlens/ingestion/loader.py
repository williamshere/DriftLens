"""Corpus ingestion — JSONL / CSV loading, cleaning, slicing, validation, DB insert.

Handles corpora where each record contains ``timestamp`` / ``author`` /
``community`` / ``text`` (plus optional ``title`` / ``id``).  Records are:

* Cleaned (whitespace normalisation, empty filtering).
* Validated with :mod:`pydantic`.
* Time-sliced by ``month`` / ``week`` / ``quarter`` via :func:`get_time_slice`.
* Inserted into SQLite via :mod:`driftlens.db.session`.

Time slicing helpers also support community-based and generic slice keys for
the drift analytics layer.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

SliceGranularity = Literal["month", "week", "quarter"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RawDocument(BaseModel):
    """Validated document record as it appears in JSONL / CSV.

    Attributes:
        text: Document body (required, non-empty after stripping).
        timestamp: Optional timestamp — ISO-8601 string or datetime.  Parsed
            to :class:`datetime` via the validator.
        author: Optional author identifier.
        community: Optional community / subreddit / forum label.
        title: Optional title.
        doc_id: Optional explicit id (alias ``id``).
        slice_key: Optional pre-computed slice key.
    """

    text: str = Field(..., min_length=1, description="Document body")
    timestamp: datetime | None = Field(default=None, description="Document timestamp")
    author: str | None = Field(default=None, description="Author identifier")
    community: str | None = Field(default=None, description="Community label")
    title: str | None = Field(default=None, description="Optional title")
    doc_id: str | None = Field(default=None, alias="id", description="Explicit document id")
    slice_key: str | None = Field(default=None, description="Pre-computed slice key")

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("text", mode="before")
    @classmethod
    def _clean_text_field(cls, v: Any) -> str:
        """Strip and validate text is non-empty.

        Args:
            v: Raw value.

        Returns:
            Stripped non-empty string.

        Raises:
            ValueError: If empty after stripping.
        """
        if not isinstance(v, str):
            raise ValueError("text must be a string")
        cleaned = clean_text(v)
        if not cleaned:
            raise ValueError("text must be non-empty after cleaning")
        return cleaned

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, v: Any) -> datetime | None:
        """Parse timestamp from various string formats.

        Args:
            v: ISO-8601 string, epoch seconds, or datetime.

        Returns:
            Timezone-aware datetime or ``None``.
        """
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            # Try ISO-8601 first.
            try:
                # Handle trailing Z.
                iso = v.replace("Z", "+00:00") if v.endswith("Z") else v
                dt = datetime.fromisoformat(iso)
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            # Try common formats.
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(v, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse timestamp: {v!r}")
        raise ValueError(f"Unsupported timestamp type: {type(v)}")


class ValidatedDocument(BaseModel):
    """Normalised document ready for DB insertion.

    Attributes:
        doc_id: Stable document id.
        text: Cleaned body.
        timestamp: Parsed timestamp or ``None``.
        author: Author or ``None``.
        community: Community or ``None``.
        title: Title or ``None``.
        slice_key: Computed slice key.
        meta: Extra metadata dict.
    """

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    timestamp: datetime | None = None
    author: str | None = None
    community: str | None = None
    title: str | None = None
    slice_key: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise whitespace and strip surrounding space.

    Collapses runs of whitespace (including newlines / tabs) to a single
    space and strips leading / trailing whitespace.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.  Empty string when input is whitespace-only.
    """
    if not isinstance(text, str):
        return ""
    # Replace whitespace runs with single space, then strip.
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    return cleaned


def is_valid_text(text: str, min_length: int = 1) -> bool:
    """Check whether text is valid after cleaning.

    Args:
        text: Raw or cleaned text.
        min_length: Minimum length threshold after cleaning.

    Returns:
        ``True`` when valid.
    """
    return len(clean_text(text)) >= min_length


# ---------------------------------------------------------------------------
# Time slicing
# ---------------------------------------------------------------------------


def get_time_slice(timestamp: datetime, granularity: SliceGranularity = "month") -> str:
    """Map a timestamp to its slice key.

    Args:
        timestamp: Aware or naive datetime (naive treated as UTC).
        granularity: ``"month"`` -> ``YYYY-MM``, ``"week"`` -> ``YYYY-Www``,
            ``"quarter"`` -> ``YYYY-Qn``.

    Returns:
        Slice key string.

    Raises:
        ValueError: For unknown granularity.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if granularity == "month":
        return timestamp.strftime("%Y-%m")
    if granularity == "week":
        year, week, _ = timestamp.isocalendar()
        return f"{year}-W{week:02d}"
    if granularity == "quarter":
        q = (timestamp.month - 1) // 3 + 1
        return f"{timestamp.year}-Q{q}"
    raise ValueError(f"Unknown granularity: {granularity!r}. Expected month/week/quarter.")


def slice_documents(
    docs: list[ValidatedDocument],
    granularity: SliceGranularity = "month",
) -> dict[str, list[ValidatedDocument]]:
    """Group documents by time slice.

    Args:
        docs: Validated documents.
        granularity: Slicing strategy.

    Returns:
        Mapping ``slice_key -> documents``.  Documents with ``None``
        timestamp fall under ``"__unknown__"``.
    """
    groups: dict[str, list[ValidatedDocument]] = {}
    for doc in docs:
        key = get_time_slice(doc.timestamp, granularity) if doc.timestamp is not None else "__unknown__"
        groups.setdefault(key, []).append(doc)
    return groups


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _raw_to_validated(raw: RawDocument, granularity: SliceGranularity = "month") -> ValidatedDocument:
    """Convert a RawDocument to ValidatedDocument, computing slice_key.

    Args:
        raw: Validated raw record.
        granularity: Time-slice granularity for automatic slice_key.

    Returns:
        ValidatedDocument with computed ``slice_key`` and ``meta``.
    """
    # Compute slice_key: prefer raw.slice_key, else derive from timestamp, else community, else unknown.
    slice_key: str | None = raw.slice_key
    if slice_key is None and raw.timestamp is not None:
        slice_key = get_time_slice(raw.timestamp, granularity)
    if slice_key is None and raw.community is not None:
        slice_key = str(raw.community)
    if slice_key is None:
        slice_key = "__unknown__"

    # Collect extra fields into meta.
    known_fields = {"text", "timestamp", "author", "community", "title", "id", "doc_id", "slice_key"}
    extra: dict[str, Any] = {}
    # model_extra holds unknown fields when extra="allow".
    if hasattr(raw, "model_extra") and raw.model_extra:
        for k, v in raw.model_extra.items():
            if k not in known_fields:
                extra[k] = v

    return ValidatedDocument(
        doc_id=raw.doc_id or str(uuid.uuid4()),
        text=clean_text(raw.text),
        timestamp=raw.timestamp,
        author=raw.author,
        community=raw.community,
        title=raw.title,
        slice_key=slice_key,
        meta=extra,
    )


def load_jsonl(
    path: str | Path,
    granularity: SliceGranularity = "month",
    skip_invalid: bool = True,
) -> list[ValidatedDocument]:
    """Load and validate a JSONL file.

    Each line must be a JSON object with at least ``text`` and optionally
    ``timestamp``, ``author``, ``community``, ``title``, ``id``.

    Args:
        path: Path to ``.jsonl`` file.
        granularity: Time-slice granularity.
        skip_invalid: When ``True`` invalid lines are logged and skipped;
            when ``False`` the first :class:`ValidationError` is raised.

    Returns:
        List of validated documents.

    Raises:
        FileNotFoundError: If path does not exist.
        ValidationError: If ``skip_invalid`` is ``False`` and a record fails.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    docs: list[ValidatedDocument] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                raw = RawDocument.model_validate(obj)
                docs.append(_raw_to_validated(raw, granularity))
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                if skip_invalid:
                    logger.warning("Skipping invalid JSONL line %d in %s: %s", lineno, p, exc)
                    continue
                raise
    logger.info("Loaded %d documents from JSONL %s", len(docs), p)
    return docs


def load_csv(
    path: str | Path,
    granularity: SliceGranularity = "month",
    skip_invalid: bool = True,
    text_column: str = "text",
    timestamp_column: str = "timestamp",
    author_column: str = "author",
    community_column: str = "community",
) -> list[ValidatedDocument]:
    """Load and validate a CSV file.

    Expects a header row.  Recognised columns: ``text`` (required),
    ``timestamp``, ``author``, ``community``, ``title``, ``id``.  Extra
    columns are collected into ``meta``.

    Args:
        path: Path to ``.csv`` file.
        granularity: Time-slice granularity.
        skip_invalid: Whether to skip invalid rows.
        text_column: Name of the text column.
        timestamp_column: Name of the timestamp column.
        author_column: Name of the author column.
        community_column: Name of the community column.

    Returns:
        List of validated documents.

    Raises:
        FileNotFoundError: If path does not exist.
        ValidationError: If ``skip_invalid`` is ``False`` and a row fails.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    docs: list[ValidatedDocument] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {p}")
        for lineno, row in enumerate(reader, start=2):
            # Normalise empty strings to None for optional fields.
            # Extract text.
            raw_text = row.get(text_column, "")
            if raw_text is None:
                raw_text = ""
            raw_text = str(raw_text)
            if not raw_text.strip():
                if skip_invalid:
                    logger.warning("Skipping CSV row %d in %s: empty text", lineno, p)
                    continue
                raise ValidationError.from_exception_data(
                    title="ValidatedDocument",
                    line_errors=[{"type": "value_error", "loc": ("text",), "msg": "empty", "input": raw_text}],
                )

            # Build raw object dict.
            obj: dict[str, Any] = {"text": raw_text}
            # Map known columns.
            ts_val = row.get(timestamp_column)
            if ts_val not in (None, ""):
                obj["timestamp"] = ts_val
            author_val = row.get(author_column)
            if author_val not in (None, ""):
                obj["author"] = author_val
            comm_val = row.get(community_column)
            if comm_val not in (None, ""):
                obj["community"] = comm_val
            # Optional id / title.
            for col in ("id", "title", "slice_key"):
                if col in row and row[col] not in (None, ""):
                    obj[col] = row[col]
            # Collect any extra columns.
            for k, v in row.items():
                if k not in {text_column, timestamp_column, author_column, community_column, "id", "title", "slice_key"} and v not in (None, ""):
                    obj[k] = v

            try:
                raw = RawDocument.model_validate(obj)
                docs.append(_raw_to_validated(raw, granularity))
            except (ValidationError, ValueError) as exc:
                if skip_invalid:
                    logger.warning("Skipping invalid CSV row %d in %s: %s", lineno, p, exc)
                    continue
                raise
    logger.info("Loaded %d documents from CSV %s", len(docs), p)
    return docs


def load_documents(
    path: str | Path,
    granularity: SliceGranularity = "month",
    skip_invalid: bool = True,
) -> list[ValidatedDocument]:
    """Load documents from JSONL or CSV, dispatching by file extension.

    Args:
        path: Path to ``.jsonl``, ``.json`` (treated as JSONL), or ``.csv``.
        granularity: Time-slice granularity.
        skip_invalid: Whether to skip invalid records.

    Returns:
        List of validated documents.

    Raises:
        ValueError: For unsupported extensions.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return load_jsonl(p, granularity=granularity, skip_invalid=skip_invalid)
    if suffix == ".csv":
        return load_csv(p, granularity=granularity, skip_invalid=skip_invalid)
    raise ValueError(f"Unsupported file type {suffix!r} for {p}. Expected .jsonl, .json, or .csv.")


# ---------------------------------------------------------------------------
# DB insertion
# ---------------------------------------------------------------------------


def insert_documents(
    docs: list[ValidatedDocument],
    db_path: str | Path | None = None,
) -> list[str]:
    """Insert validated documents into the SQLite database.

    Uses :mod:`driftlens.db.session` for persistence.

    Args:
        docs: Validated documents.
        db_path: Optional override for the database path.

    Returns:
        List of inserted document ids.
    """
    from driftlens.db.session import get_db

    db = get_db(db_path)
    db.init_db()
    rows: list[dict[str, Any]] = []
    for d in docs:
        meta: dict[str, Any] = dict(d.meta) if d.meta else {}
        if d.author is not None:
            meta["author"] = d.author
        if d.community is not None:
            meta["community"] = d.community
        if d.timestamp is not None:
            meta["timestamp"] = d.timestamp.isoformat()
        rows.append(
            {
                "id": d.doc_id,
                "text": d.text,
                "title": d.title,
                "slice_key": d.slice_key,
                "meta": meta,
            }
        )
    return db.insert_documents(rows)


def load_and_insert(
    path: str | Path,
    granularity: SliceGranularity = "month",
    skip_invalid: bool = True,
    db_path: str | Path | None = None,
) -> list[str]:
    """Load a corpus file and insert all records into the database.

    Convenience wrapper combining :func:`load_documents` and
    :func:`insert_documents`.

    Args:
        path: Corpus file path (JSONL or CSV).
        granularity: Time-slice granularity.
        skip_invalid: Whether to skip invalid records.
        db_path: Optional database path override.

    Returns:
        List of inserted document ids.
    """
    docs = load_documents(path, granularity=granularity, skip_invalid=skip_invalid)
    if not docs:
        logger.warning("No valid documents found in %s", path)
        return []
    return insert_documents(docs, db_path=db_path)
