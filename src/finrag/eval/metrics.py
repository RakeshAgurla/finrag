"""Retrieval metrics.

Implemented directly rather than pulled from a library so the definitions are
auditable. Every number this repo publishes is computed here.

A note on what these do and do not tell you:

- recall@k answers "did we put the answer in the context window at all". If
  recall@k is low, no amount of prompt engineering downstream will save the
  answer. This is the metric to optimize first.
- MRR answers "how high did the first correct chunk land". Matters because
  generators attend unevenly across long contexts.
- nDCG@k is the one to report when relevance is graded rather than binary, and
  it is the only one here that rewards ordering among multiple relevant chunks.

Reporting a single averaged number without the per-query spread hides the
failure modes that actually matter, so EvalResult keeps per-query rows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Binary: did at least one relevant chunk make the cut."""
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevance: dict[str, float], k: int
) -> float:
    """Standard nDCG with binary or graded relevance.

    relevance maps chunk_id -> gain (0 = irrelevant, 1 = relevant, 2 = directly
    answers the question).
    """
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids[:k]):
        gain = relevance.get(chunk_id, 0.0)
        if gain:
            dcg += (2**gain - 1) / math.log2(i + 2)

    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        (2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_gains) if g
    )
    return dcg / idcg if idcg else 0.0


@dataclass
class QueryResult:
    query_id: str
    question: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.metrics.get("hit_rate@5", 0.0) == 0.0


@dataclass
class EvalResult:
    config_name: str
    per_query: list[QueryResult]

    @property
    def aggregate(self) -> dict[str, float]:
        if not self.per_query:
            return {}
        keys = self.per_query[0].metrics.keys()
        return {
            key: mean(q.metrics.get(key, 0.0) for q in self.per_query) for key in keys
        }

    @property
    def failures(self) -> list[QueryResult]:
        """The rows worth reading. Aggregates tell you whether you improved;
        these tell you what to fix next."""
        return [q for q in self.per_query if q.failed]

    def to_row(self) -> dict:
        return {"config": self.config_name, **{k: round(v, 4) for k, v in self.aggregate.items()}}


def evaluate_query(
    query_id: str,
    question: str,
    retrieved_ids: Sequence[str],
    relevance: dict[str, float],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> QueryResult:
    relevant_ids = {cid for cid, gain in relevance.items() if gain > 0}
    metrics: dict[str, float] = {"mrr": reciprocal_rank(retrieved_ids, relevant_ids)}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevance, k)
    return QueryResult(
        query_id=query_id,
        question=question,
        retrieved_ids=list(retrieved_ids),
        relevant_ids=relevant_ids,
        metrics=metrics,
    )


def compare(baseline: EvalResult, candidate: EvalResult) -> dict[str, dict[str, float]]:
    """Delta table between two configurations.

    Relative lift is easy to quote and easy to mislead with; absolute values are
    what keep it honest. Report both.
    """
    out: dict[str, dict[str, float]] = {}
    base_agg, cand_agg = baseline.aggregate, candidate.aggregate
    for key in base_agg:
        b, c = base_agg[key], cand_agg.get(key, 0.0)
        out[key] = {
            "baseline": round(b, 4),
            "candidate": round(c, 4),
            "abs_delta": round(c - b, 4),
            "rel_delta_pct": round(((c - b) / b * 100) if b else 0.0, 2),
        }
    return out
