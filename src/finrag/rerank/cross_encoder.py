"""Cross-encoder reranking.

The retriever is a bi-encoder: query and document are embedded independently, so
the model never sees them together and cannot reason about their interaction. It
is fast enough to score a whole corpus and correspondingly shallow.

A cross-encoder scores (query, document) jointly. Far more accurate, far too
slow to run over a corpus -- so the standard pattern is retrieve wide with the
bi-encoder, then rerank a small candidate set. Here: fuse to 30 candidates,
rerank, keep 8.

This stage is usually the single largest precision win in a RAG system and the
most commonly skipped. The eval harness quantifies it; see `make eval`.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.config import RerankConfig, settings
from finrag.index.hybrid import RetrievedChunk


class NoOpReranker:
    """Used in CI and as the eval baseline."""

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(candidates)


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig | None = None):
        cfg = cfg or settings.rerank
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with `pip install -e '.[models]'` or set "
                "FINRAG_RERANK_BACKEND=none."
            ) from exc
        self._model = CrossEncoder(cfg.model_name)
        self._cfg = cfg

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.embedding_text) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self._cfg.batch_size)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))


def build_reranker(cfg: RerankConfig | None = None):
    cfg = cfg or settings.rerank
    if cfg.backend in ("none", "noop"):
        return NoOpReranker()
    if cfg.backend == "cross-encoder":
        return CrossEncoderReranker(cfg)
    raise ValueError(f"Unknown rerank backend: {cfg.backend}")
