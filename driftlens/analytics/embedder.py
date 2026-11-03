"""SentenceTransformer embedding wrapper with auto device and batch handling.

Provides :class:`Embedder`, a thin, typed wrapper around
``sentence_transformers.SentenceTransformer`` that:

* Auto-detects ``cuda`` / ``cpu`` (and ``mps`` when available) so callers
  can pass ``device="auto"`` or omit the argument entirely.
* Processes texts in configurable batches to bound memory.
* Exposes :meth:`Embedder.encode_documents` as the primary ingestion-facing
  entry-point while also offering :meth:`Embedder.encode` for general use.
* Always returns :class:`numpy.ndarray` with ``dtype=float32`` for
  consistent downstream storage (BLOB float32 in SQLite).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect the best available device.

    Returns:
        ``"cuda"`` if CUDA is available, ``"mps"`` on Apple Silicon when
        available, otherwise ``"cpu"``.
    """
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Device detection failed, falling back to cpu: %s", exc)
    return "cpu"


class Embedder:
    """Wrapper around :class:`sentence_transformers.SentenceTransformer`.

    Args:
        model_name: HuggingFace / sentence-transformers identifier.  Defaults
            to ``Settings().embedding_model`` (``all-MiniLM-L6-v2``) when
            ``None``.
        device: Compute device — ``"cpu"``, ``"cuda"``, ``"mps"`` or
            ``"auto"``.  ``"auto"`` triggers :func:`_detect_device`.
        batch_size: Default batch size for encoding.
        normalize: Whether to L2-normalise embeddings.  Defaults to ``False``
            so raw float32 vectors are stored; set ``True`` when cosine
            similarity without explicit normalisation is desired.
        show_progress: Whether to show progress bar during encoding.

    Example:
        >>> embedder = Embedder(device="auto")
        >>> vecs = embedder.encode_documents(["hello world", "semantic drift"])
        >>> vecs.dtype  # np.float32
        dtype('float32')
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = False,
        show_progress: bool = False,
    ) -> None:
        # Resolve defaults from config when available.
        default_model = "all-MiniLM-L6-v2"
        default_batch = 32
        try:
            from driftlens.config import settings as _settings

            default_model = _settings.embedding_model
            default_batch = _settings.batch_size
            if device is None:
                device = _settings.device
        except Exception:
            pass

        self.model_name: str = model_name or default_model
        self.batch_size: int = batch_size or default_batch
        self.normalize: bool = normalize
        self.show_progress: bool = show_progress

        # Resolve device string.
        if device is None or device == "auto":
            self.device: str = _detect_device()
        else:
            if device not in {"cpu", "cuda", "mps"}:
                raise ValueError(f"device must be one of cpu/cuda/mps/auto, got {device!r}")
            # If cuda/mps requested but unavailable, fall back to cpu with warning.
            detected = _detect_device()
            if device == "cuda" and detected != "cuda":
                logger.warning("cuda requested but not available; falling back to cpu")
                self.device = "cpu"
            elif device == "mps" and detected not in {"mps", "cuda"}:
                logger.warning("mps requested but not available; falling back to cpu")
                self.device = "cpu"
            else:
                self.device = device

        self._model: object | None = None
        logger.info("Embedder initialised: model=%s device=%s batch_size=%d", self.model_name, self.device, self.batch_size)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> object:
        """Lazily load the SentenceTransformer model.

        Returns:
            The loaded SentenceTransformer instance.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for Embedder. Install with: pip install sentence-transformers"
            ) from exc
        # SentenceTransformer handles device internally; pass device explicitly.
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int | None = None,
        normalize: bool | None = None,
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """Encode one or more texts into embeddings.

        Args:
            texts: Single string or sequence of strings.
            batch_size: Override batch size for this call.
            normalize: Override L2 normalisation for this call.
            show_progress: Override progress-bar flag.

        Returns:
            2-D ``float32`` array of shape ``(n, dim)``.  For a single string
            input a 2-D array with one row is still returned.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if len(texts) == 0:
            # Return empty array with correct dim if model is known; else (0, 0).
            dim = self._infer_dim()
            return np.empty((0, dim), dtype=np.float32)

        model = self._load_model()
        bs = batch_size or self.batch_size
        do_normalize = self.normalize if normalize is None else normalize
        do_progress = self.show_progress if show_progress is None else show_progress

        # SentenceTransformer.encode handles batching internally, but we also
        # chunk manually to guarantee bounded memory when texts is very large.
        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            # type: ignore[attr-defined] — SentenceTransformer has encode
            embs = model.encode(  # type: ignore[union-attr]
                chunk,
                batch_size=bs,
                normalize_embeddings=do_normalize,
                show_progress_bar=do_progress,
                convert_to_numpy=True,
            )
            arr = np.asarray(embs, dtype=np.float32)
            all_embs.append(arr)

        result = np.vstack(all_embs) if len(all_embs) > 1 else all_embs[0]
        # Ensure float32 even if model returned float64.
        if result.dtype != np.float32:
            result = result.astype(np.float32)
        return result

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int | None = None,
        normalize: bool | None = None,
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """Encode a batch of document texts.

        Primary entry-point for ingestion pipelines.  Thin alias over
        :meth:`encode` with a name that signals document-level usage.

        Args:
            documents: Sequence of document body strings.
            batch_size: Override batch size.
            normalize: Override normalisation flag.
            show_progress: Override progress flag.

        Returns:
            2-D ``float32`` array of shape ``(len(documents), dim)``.
            Returns shape ``(0, dim)`` for an empty input.
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single document text.

        Args:
            text: Document body.

        Returns:
            1-D ``float32`` vector.
        """
        arr = self.encode([text])
        return arr[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_dim(self) -> int:
        """Infer embedding dimension without loading full model if possible.

        Returns:
            Embedding dimensionality.
        """
        # Try config default.
        try:
            from driftlens.config import settings as _settings

            return int(_settings.embedding_dim)
        except Exception:
            pass
        # Try model attribute if already loaded.
        if self._model is not None:
            try:
                return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]
            except Exception:
                pass
        # Fallback for well-known model.
        if self.model_name == "all-MiniLM-L6-v2":
            return 384
        return 0

    @property
    def dimension(self) -> int:
        """Embedding dimensionality for the configured model.

        Returns:
            Integer dimension.
        """
        if self._model is not None:
            try:
                return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]
            except Exception:
                pass
        return self._infer_dim()

    def __repr__(self) -> str:
        return f"Embedder(model={self.model_name!r}, device={self.device!r}, dim={self.dimension})"
