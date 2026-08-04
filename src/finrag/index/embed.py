"""Embedding backends.

Two implementations behind one interface:

- SentenceTransformerEmbedder: what you actually run.
- HashEmbedder: deterministic, zero-dependency, no model download. CI uses this
  so the full ingest -> index -> retrieve -> eval loop runs in seconds on a cold
  runner. It is genuinely bad at semantics, which is the point: if a test passes
  with hash embeddings it is testing plumbing, not model quality.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from finrag.config import EmbeddingConfig, settings


class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Return (len(texts), dim) float32, L2-normalized when configured."""

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], is_query=True)[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashEmbedder(Embedder):
    """Hashed bag-of-character-ngrams. Deterministic across machines and runs."""

    def __init__(self, dim: int = 384, ngram: int = 4):
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            for j in range(max(1, len(lowered) - self.ngram + 1)):
                gram = lowered[j : j + self.ngram]
                h = int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                    "little",
                )
                out[i, h % self.dim] += 1.0
        return _normalize(out)


class SentenceTransformerEmbedder(Embedder):
    """Wraps sentence-transformers.

    BGE-family models expect an instruction prefix on the *query* side only.
    Getting this wrong silently costs several points of recall, and it is one of
    the most common bugs in hand-rolled RAG stacks.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, cfg: EmbeddingConfig | None = None):
        cfg = cfg or settings.embedding
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Install the full extras "
                "with `pip install -e '.[models]'`, or set "
                "FINRAG_EMBEDDING_BACKEND=hash to run offline."
            ) from exc

        self._model = SentenceTransformer(cfg.model_name)
        self._cfg = cfg
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prepared = list(texts)
        if is_query and "bge" in self._cfg.model_name.lower():
            prepared = [self.QUERY_INSTRUCTION + t for t in prepared]

        vectors = self._model.encode(
            prepared,
            batch_size=self._cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._cfg.normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def build_embedder(cfg: EmbeddingConfig | None = None) -> Embedder:
    cfg = cfg or settings.embedding
    if cfg.backend == "hash":
        return HashEmbedder(dim=cfg.dim)
    if cfg.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg)
    raise ValueError(f"Unknown embedding backend: {cfg.backend}")
