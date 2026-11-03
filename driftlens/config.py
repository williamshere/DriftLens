"""Application configuration via Pydantic Settings.

Centralises all tunable parameters for DriftLens so that environment
variables, ``.env`` files, and direct construction all behave consistently.

Environment variables are prefixed with ``DRIFTLENS_`` (e.g.
``DRIFTLENS_DB_PATH``, ``DRIFTLENS_EMBEDDING_MODEL``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Auto-detect the best available compute device.

    Tries ``cuda`` via :mod:`torch` if installed, then ``mps`` on Apple
    Silicon, falling back to ``cpu``.  Never raises — always returns a
    valid device string.
    """
    # Try CUDA first.
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            logger.info("Auto-detected device: cuda")
            return "cuda"
        # Apple Metal Performance Shaders
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            logger.info("Auto-detected device: mps")
            return "mps"
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Device detection fallback triggered: %s", exc)

    logger.info("Auto-detected device: cpu")
    return "cpu"


class SliceConfig(BaseSettings):
    """Configuration for temporal / categorical slicing of a corpus."""

    model_config = SettingsConfigDict(env_prefix="DRIFTLENS_SLICE_", env_nested_delimiter="__")

    strategy: Literal["temporal", "categorical", "none"] = Field(
        default="temporal",
        description="How to partition documents into slices.",
    )
    date_field: str = Field(
        default="date",
        description="Document field containing the slice key (e.g. date or category).",
    )
    num_slices: int | None = Field(
        default=None,
        description="Target number of slices; None = one slice per distinct key.",
    )
    min_docs_per_slice: int = Field(
        default=10,
        ge=1,
        description="Minimum documents required for a slice to be retained.",
    )


class Settings(BaseSettings):
    """Top-level DriftLens settings.

    Attributes:
        embedding_model: HuggingFace / sentence-transformers model identifier.
        db_path: Path to the SQLite database file.
        batch_size: Batch size for embedding encoding.
        device: Compute device (``cpu``, ``cuda``, ``mps`` or ``auto``).
        slice: Nested slice configuration.
        log_level: Root log level.
        embedding_dim: Expected embedding dimensionality (validated at runtime).
    """

    model_config = SettingsConfigDict(
        env_prefix="DRIFTLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model identifier.",
    )
    db_path: Path = Field(
        default=Path("driftlens.db"),
        description="Path to the SQLite database file.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for embedding encoding.",
    )
    device: str = Field(
        default="auto",
        description="Compute device: cpu, cuda, mps, or auto (detect).",
    )
    embedding_dim: int = Field(
        default=384,
        ge=1,
        description="Expected embedding dimensionality for all-MiniLM-L6-v2.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level.",
    )

    slice: SliceConfig = Field(default_factory=SliceConfig)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, v: str) -> str:
        """Resolve ``auto`` to an actual device string."""
        if v == "auto":
            return _detect_device()
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed} or 'auto', got {v!r}")
        return v

    @field_validator("db_path", mode="before")
    @classmethod
    def _coerce_db_path(cls, v: str | Path) -> Path:
        return Path(v) if isinstance(v, str) else v

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolved_device(self) -> str:
        """Return the resolved device string (handles lingering ``auto``)."""
        if self.device == "auto":
            return _detect_device()
        return self.device

    def ensure_db_dir(self) -> Path:
        """Ensure the parent directory for :attr:`db_path` exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return self.db_path


# Module-level default instance (lazy-friendly; callers may create their own).
settings = Settings()  # type: ignore[call-arg]
