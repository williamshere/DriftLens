"""DriftLens ingestion package — corpus loading and DB insertion."""

from __future__ import annotations

from driftlens.ingestion.loader import (
    RawDocument,
    ValidatedDocument,
    clean_text,
    get_time_slice,
    insert_documents,
    is_valid_text,
    load_and_insert,
    load_csv,
    load_documents,
    load_jsonl,
    slice_documents,
)

__all__ = [
    "RawDocument",
    "ValidatedDocument",
    "clean_text",
    "get_time_slice",
    "insert_documents",
    "is_valid_text",
    "load_and_insert",
    "load_csv",
    "load_documents",
    "load_jsonl",
    "slice_documents",
]
