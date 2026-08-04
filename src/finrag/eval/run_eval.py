"""Ablation runner.

Produces the table in the README. The point is not that the final configuration
scores well -- it is that every component in the stack has a measured
contribution, so nothing is in the pipeline on vibes.

Run:  python -m finrag.eval.run_eval
      python -m finrag.eval.run_eval --backend sentence-transformers --rerank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finrag.config import RetrievalConfig, settings
from finrag.eval.dataset import SAMPLE_FILINGS, build_gold_set
from finrag.eval.metrics import EvalResult, compare, evaluate_query
from finrag.index.embed import build_embedder
from finrag.index.hybrid import HybridRetriever
from finrag.ingest.chunk import Chunk, chunk_filing


def build_corpus() -> tuple[list[Chunk], dict[str, list[str]]]:
    chunks: list[Chunk] = []
    for (ticker, year), text in SAMPLE_FILINGS.items():
        chunks.extend(chunk_filing(text, ticker=ticker, fiscal_year=year))

    by_section: dict[str, list[str]] = {}
    for chunk in chunks:
        key = f"{chunk.ticker}:{chunk.fiscal_year}:{chunk.section_item}"
        by_section.setdefault(key, []).append(chunk.chunk_id)
    return chunks, by_section


CONFIGS: dict[str, RetrievalConfig] = {
    # Ablations are cumulative so each row isolates one change.
    "dense_only": RetrievalConfig(lexical_top_k=0, rerank=False, fusion="rrf"),
    "bm25_only": RetrievalConfig(dense_top_k=0, rerank=False, fusion="rrf"),
    "hybrid_rrf": RetrievalConfig(rerank=False, fusion="rrf"),
    "hybrid_weighted": RetrievalConfig(rerank=False, fusion="weighted", alpha=0.5),
    "hybrid_rrf_rerank": RetrievalConfig(rerank=True, fusion="rrf"),
}


def run_config(
    name: str,
    cfg: RetrievalConfig,
    chunks: list[Chunk],
    gold,
    reranker=None,
) -> EvalResult:
    embedder = build_embedder()
    retriever = HybridRetriever(
        chunks, embedder=embedder, cfg=cfg, reranker=reranker if cfg.rerank else None
    )

    results = []
    for question in gold:
        hits = retriever.retrieve(question.question, top_k=10)
        results.append(
            evaluate_query(
                query_id=question.query_id,
                question=question.question,
                retrieved_ids=[h.chunk.chunk_id for h in hits],
                relevance=question.relevance,
            )
        )
    return EvalResult(config_name=name, per_query=results)


def format_table(results: list[EvalResult]) -> str:
    cols = ["recall@5", "hit_rate@5", "mrr", "ndcg@5", "ndcg@10"]
    header = "| config | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for res in results:
        agg = res.aggregate
        row = " | ".join(f"{agg.get(c, 0.0):.3f}" for c in cols)
        lines.append(f"| `{res.config_name}` | {row} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("evals/results.json"))
    parser.add_argument("--rerank", action="store_true", help="enable cross-encoder")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    chunks, by_section = build_corpus()
    gold = build_gold_set(by_section)

    reranker = None
    if args.rerank:
        from finrag.rerank.cross_encoder import build_reranker

        reranker = build_reranker()

    configs = dict(CONFIGS)
    if not args.rerank:
        configs.pop("hybrid_rrf_rerank", None)

    results = [
        run_config(name, cfg, chunks, gold, reranker) for name, cfg in configs.items()
    ]

    if not args.quiet:
        print(f"\ncorpus: {len(chunks)} chunks | gold set: {len(gold)} questions")
        print(f"embedding backend: {settings.embedding.backend}\n")
        print(format_table(results))

        baseline = next(r for r in results if r.config_name == "dense_only")
        best = max(results, key=lambda r: r.aggregate.get("ndcg@5", 0))
        if best.config_name != baseline.config_name:
            print(f"\ndelta: {baseline.config_name} -> {best.config_name}")
            for metric, d in compare(baseline, best).items():
                if metric in ("recall@5", "ndcg@5", "mrr"):
                    print(
                        f"  {metric:<12} {d['baseline']:.3f} -> {d['candidate']:.3f} "
                        f"({d['rel_delta_pct']:+.1f}%)"
                    )

        print("\nfailures in best config (read these, not the averages):")
        for q in best.failures:
            print(f"  [{q.query_id}] {q.question}")
        if not best.failures:
            print("  none")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "corpus_chunks": len(chunks),
                "gold_questions": len(gold),
                "embedding_backend": settings.embedding.backend,
                "results": [r.to_row() for r in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
