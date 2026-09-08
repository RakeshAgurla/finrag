"""Latency, throughput, and cost measurement.

Retrieval quality is only half of what determines whether a system is usable.
The eval harness answers "does it find the right chunk"; this answers "what does
it cost to ask, and how long does the user wait".

Both matter and they trade against each other. Cross-encoder reranking improves
ranking and roughly doubles query latency. Semantic embeddings beat lexical
matching on paraphrase and cost 10x more to index. Neither tradeoff is visible
in a recall number.

**Why percentiles rather than means.** A mean latency hides the tail, and the
tail is what users experience as "sometimes it hangs". A system at 40ms mean and
900ms p99 feels broken to one request in a hundred, and the mean says it is
fine.

    python -m finrag.eval.benchmark
    python -m finrag.eval.benchmark --backend sentence-transformers --rerank
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

# faiss-cpu and torch each bundle an OpenMP runtime; loading both in one
# process aborts on macOS. This benchmark compares backends in a single run,
# so it hits that.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dataclasses import asdict, dataclass, field
from pathlib import Path

from finrag.config import RetrievalConfig
from finrag.index.embed import HashEmbedder, build_embedder
from finrag.index.hybrid import HybridRetriever, load_chunks

# Queries spanning the retrieval modes, so the latency numbers are not measured
# on one easy case. Short exact-token queries are cheap; long paraphrase queries
# with many candidate matches are not.
BENCH_QUERIES = [
    "ASC 606",
    "operating income",
    "what supply chain risks were disclosed",
    "how did gross margin change year over year",
    "what does the company say about artificial intelligence",
    "what legal proceedings are pending against the company",
    "how much was spent on research and development",
    "describe the company's approach to capital allocation and shareholder returns",
]

# Published prices per million tokens, used to compute cost per query when a
# generation step is attached. Kept as data rather than hard-coded into the
# calculation because they change and a stale constant buried in a function is
# how a cost estimate silently becomes wrong.
PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def estimate_tokens(text: str) -> int:
    """~4 chars per token on English prose.

    Deliberately an estimate. An exact count needs the target model's tokenizer,
    which differs per provider and would make this depend on which model you
    happen to be using. For cost projection the error is a few percent and
    biased high, which is the safe direction.
    """
    return max(1, len(text) // 4)


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def of(cls, samples: list[float]) -> LatencyStats:
        ordered = sorted(samples)
        n = len(ordered)

        def pct(p: float) -> float:
            if n == 1:
                return ordered[0]
            idx = min(n - 1, max(0, round(p * (n - 1))))
            return ordered[idx]

        return cls(
            n=n,
            mean_ms=round(statistics.mean(ordered), 3),
            p50_ms=round(pct(0.50), 3),
            p95_ms=round(pct(0.95), 3),
            p99_ms=round(pct(0.99), 3),
            min_ms=round(ordered[0], 3),
            max_ms=round(ordered[-1], 3),
        )


@dataclass
class ContextStats:
    """Token volume of retrieved context.

    This is the number that drives generation cost. Retrieval itself is
    essentially free; what you pay for is stuffing 8 chunks into a prompt on
    every query. Halving chunk size or top_k halves the bill, which is a lever
    that only becomes visible once it is measured.
    """

    mean_tokens: float
    p95_tokens: int
    max_tokens: int
    mean_chunks: float

    @classmethod
    def of(cls, token_counts: list[int], chunk_counts: list[int]) -> ContextStats:
        ordered = sorted(token_counts)
        idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
        return cls(
            mean_tokens=round(statistics.mean(token_counts), 1),
            p95_tokens=ordered[idx],
            max_tokens=ordered[-1],
            mean_chunks=round(statistics.mean(chunk_counts), 2),
        )


@dataclass
class BenchmarkResult:
    config: str
    corpus_chunks: int
    index_build_s: float
    index_memory_mb: float
    retrieval: LatencyStats
    context: ContextStats
    cost_per_1k_queries: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def measure_index_build(chunks, embedder, cfg) -> tuple[HybridRetriever, float, float]:
    """Build the index and report wall time and resident memory growth."""
    import gc

    gc.collect()
    before = _rss_mb()

    start = time.perf_counter()
    retriever = HybridRetriever(chunks, embedder=embedder, cfg=cfg)
    elapsed = time.perf_counter() - start

    gc.collect()
    growth = max(0.0, _rss_mb() - before)
    return retriever, round(elapsed, 3), round(growth, 1)


def _rss_mb() -> float:
    """Resident set size in MB, without adding a psutil dependency."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS and kilobytes on Linux. Getting this
        # backwards is a 1024x error in the reported figure.
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return peak / divisor
    except Exception:  # noqa: BLE001
        return 0.0


def project_cost(context_tokens: float, output_tokens: int = 250) -> dict[str, float]:
    """Cost per 1,000 queries at published prices.

    Per-thousand rather than per-query because per-query rounds to zero and
    stops being comparable. The output estimate assumes a short cited answer.
    """
    out = {}
    for model, prices in PRICING.items():
        cost = (
            context_tokens / 1_000_000 * prices["input"]
            + output_tokens / 1_000_000 * prices["output"]
        ) * 1000
        out[model] = round(cost, 4)
    return out


def benchmark(
    chunks,
    backend: str,
    rerank: bool,
    repeats: int,
) -> BenchmarkResult:
    embedder = HashEmbedder() if backend == "hash" else build_embedder()
    cfg = RetrievalConfig(rerank=rerank)

    reranker = None
    if rerank:
        from finrag.rerank.cross_encoder import build_reranker

        reranker = build_reranker()

    retriever, build_s, memory_mb = measure_index_build(chunks, embedder, cfg)
    if reranker is not None:
        retriever.reranker = reranker

    # Warm up. The first query pays for lazy imports and cache population, and
    # including it in the percentiles would misattribute startup cost to steady
    # state.
    retriever.retrieve(BENCH_QUERIES[0])

    latencies: list[float] = []
    context_tokens: list[int] = []
    chunk_counts: list[int] = []

    for _ in range(repeats):
        for query in BENCH_QUERIES:
            start = time.perf_counter()
            hits = retriever.retrieve(query)
            latencies.append((time.perf_counter() - start) * 1000)

            tokens = sum(estimate_tokens(h.chunk.text) for h in hits)
            context_tokens.append(tokens)
            chunk_counts.append(len(hits))

    context = ContextStats.of(context_tokens, chunk_counts)

    label = backend + ("+rerank" if rerank else "")
    return BenchmarkResult(
        config=label,
        corpus_chunks=len(chunks),
        index_build_s=build_s,
        index_memory_mb=memory_mb,
        retrieval=LatencyStats.of(latencies),
        context=context,
        cost_per_1k_queries=project_cost(context.mean_tokens),
    )


def print_result(r: BenchmarkResult) -> None:
    print(f"\n{r.config}  ({r.corpus_chunks} chunks)")
    print(f"  index build      {r.index_build_s:>8.2f} s")
    print(f"  retrieval p50    {r.retrieval.p50_ms:>8.2f} ms")
    print(f"  retrieval p95    {r.retrieval.p95_ms:>8.2f} ms")
    print(f"  retrieval p99    {r.retrieval.p99_ms:>8.2f} ms")
    print(f"  context tokens   {r.context.mean_tokens:>8.0f} mean, "
          f"{r.context.p95_tokens} p95")
    print("  generation cost per 1k queries:")
    for model, cost in sorted(r.cost_per_1k_queries.items(), key=lambda kv: kv[1]):
        print(f"    {model:<22} ${cost:>7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path,
                        default=Path("artifacts/index/chunks.jsonl"))
    parser.add_argument("--backend", default="hash",
                        help="hash | sentence-transformers")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--compare", action="store_true",
                        help="run both backends and report the delta")
    parser.add_argument("--out", type=Path,
                        default=Path("results/benchmark.json"))
    args = parser.parse_args()

    if not args.chunks.exists():
        raise SystemExit(
            f"No corpus at {args.chunks}. Run `make ingest TICKERS=AAPL,MSFT` first."
        )

    chunks = load_chunks(args.chunks)
    tickers = sorted({c.ticker for c in chunks})
    print(f"corpus: {len(chunks)} chunks | {', '.join(tickers)}")
    print(f"queries: {len(BENCH_QUERIES)} x {args.repeats} repeats")

    results = []
    if args.compare:
        results.append(benchmark(chunks, "hash", False, args.repeats))
        results.append(benchmark(chunks, "sentence-transformers", False, args.repeats))
        if args.rerank:
            results.append(
                benchmark(chunks, "sentence-transformers", True, args.repeats)
            )
    else:
        results.append(benchmark(chunks, args.backend, args.rerank, args.repeats))

    for r in results:
        print_result(r)

    if len(results) > 1:
        base, *rest = results
        print(f"\ndelta vs {base.config}:")
        for r in rest:
            build_x = r.index_build_s / base.index_build_s if base.index_build_s else 0
            lat_x = r.retrieval.p95_ms / base.retrieval.p95_ms if base.retrieval.p95_ms else 0
            print(f"  {r.config:<28} index {build_x:>5.1f}x   p95 latency {lat_x:>5.1f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "corpus_chunks": len(chunks),
        "tickers": tickers,
        "queries": len(BENCH_QUERIES),
        "repeats": args.repeats,
        "results": [r.to_dict() for r in results],
    }, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
