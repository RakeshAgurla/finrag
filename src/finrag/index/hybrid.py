"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

Why hybrid, concretely:

Dense retrieval is strong on paraphrase ("margin pressure" -> "gross profit
declined") and weak on exact tokens. Filings are dense with exact tokens that
carry all the meaning: "Item 1A", "ASC 606", "$1.2 billion", "Basel III",
specific subsidiary names. Ask a pure-vector system for the wording of a
covenant and it returns thematically adjacent prose from the wrong year.

BM25 nails those and fails on paraphrase. Fusing them recovers both. The eval
harness in finrag.eval measures exactly how much -- run it before believing any
of this.

RRF over weighted score-blending is the default because RRF needs no score
normalization between two systems whose scores are on incomparable scales, and
it is markedly more robust when one retriever returns garbage for a given query.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from finrag.config import RetrievalConfig, settings
from finrag.index.embed import Embedder, build_embedder
from finrag.ingest.chunk import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        c = self.chunk
        return f"[{c.ticker} {c.fiscal_year} {c.form_type} Item {c.section_item}]"


class BM25:
    """Okapi BM25. ~80 lines and no dependency, versus pulling in a library that
    would need pinning and offers nothing we need beyond this."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_tokens = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_lens = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_lens.mean()) if self.n_docs else 0.0

        self.term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for tokens in corpus_tokens:
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)
            for tok in tf:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self.idf = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        scores = np.zeros(self.n_docs, dtype=np.float32)

        for term in q_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf_map in enumerate(self.term_freqs):
                tf = tf_map.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lens[doc_id] / (self.avg_len or 1)
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom

        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class DenseIndex:
    """FAISS when available, exact numpy dot product otherwise.

    At this corpus size (tens of thousands of chunks) brute force is fast enough,
    and it keeps FAISS out of the CI dependency graph. The FAISS path is what
    runs locally and is what would scale past a few hundred thousand vectors.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)
        self._faiss_index = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss_index = index
        except ImportError:
            pass

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(
                query_vec.reshape(1, -1).astype(np.float32), top_k
            )
            return [
                (int(i), float(s))
                for i, s in zip(ids[0], scores[0])
                if i >= 0
            ]
        scores = self.vectors @ query_vec.astype(np.float32)
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]], k: int = 60
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        cfg: RetrievalConfig | None = None,
        reranker=None,
    ):
        self.chunks = chunks
        self.cfg = cfg or settings.retrieval
        self.embedder = embedder or build_embedder()
        self.reranker = reranker

        texts = [c.embedding_text for c in chunks]
        self.dense = DenseIndex(self.embedder.encode(texts))
        self.bm25 = BM25([tokenize(t) for t in texts])

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.cfg
        top_k = top_k or cfg.final_top_k

        dense_hits = self.dense.search(
            self.embedder.encode_query(query), cfg.dense_top_k
        )
        lexical_hits = self.bm25.search(query, cfg.lexical_top_k)

        dense_ranks = {doc: r for r, (doc, _) in enumerate(dense_hits, 1)}
        lexical_ranks = {doc: r for r, (doc, _) in enumerate(lexical_hits, 1)}

        if cfg.fusion == "rrf":
            fused = reciprocal_rank_fusion([dense_hits, lexical_hits], cfg.rrf_k)
        else:
            fused = self._weighted_fusion(dense_hits, lexical_hits, cfg.alpha)

        candidates = sorted(fused.items(), key=lambda kv: -kv[1])

        if filters:
            candidates = [
                (doc_id, score)
                for doc_id, score in candidates
                if self._matches(self.chunks[doc_id], filters)
            ]

        results = [
            RetrievedChunk(
                chunk=self.chunks[doc_id],
                score=score,
                dense_rank=dense_ranks.get(doc_id),
                lexical_rank=lexical_ranks.get(doc_id),
            )
            for doc_id, score in candidates[: cfg.rerank_candidates]
        ]

        if cfg.rerank and self.reranker is not None and results:
            results = self.reranker.rerank(query, results)

        return results[:top_k]

    @staticmethod
    def _weighted_fusion(dense_hits, lexical_hits, alpha: float) -> dict[int, float]:
        """Min-max normalize each list before blending; raw scores are not
        comparable between cosine similarity and BM25."""

        def norm(hits):
            if not hits:
                return {}
            scores = [s for _, s in hits]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0
            return {doc: (s - lo) / span for doc, s in hits}

        d, l = norm(dense_hits), norm(lexical_hits)
        return {
            doc: alpha * d.get(doc, 0.0) + (1 - alpha) * l.get(doc, 0.0)
            for doc in set(d) | set(l)
        }

    @staticmethod
    def _matches(chunk: Chunk, filters: dict) -> bool:
        for key, want in filters.items():
            got = getattr(chunk, key, None)
            if isinstance(want, (list, tuple, set)):
                if got not in want:
                    return False
            elif got != want:
                return False
        return True


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict()) + "\n")
