# finrag

[![ci](https://github.com/RakeshAgurla/finrag/actions/workflows/ci.yml/badge.svg)](https://github.com/RakeshAgurla/finrag/actions/workflows/ci.yml)

Hybrid retrieval over SEC filings, built so that every performance claim in this
README can be reproduced by running one command.

Most RAG demos show a working query and stop there. The interesting question is
not whether a retrieval pipeline returns *something* — it always does — but
whether each component earns its place. This repo is organised around measuring
that.

```bash
git clone https://github.com/RakeshAgurla/finrag.git && cd finrag
pip install -e ".[dev]"
FINRAG_EMBEDDING_BACKEND=hash python -m finrag.eval.run_eval
```

No API keys, no model downloads, no network. Runs in under a second.

---

## The result

Ablation over a labelled question set. Each row changes exactly one thing.

| config | recall@5 | hit_rate@5 | mrr | ndcg@5 |
|---|---|---|---|---|
| `dense_only` | 0.875 | 0.875 | 0.625 | 0.686 |
| `bm25_only` | 0.875 | 0.875 | 0.688 | 0.744 |
| `hybrid_rrf` | 0.875 | 0.875 | 0.667 | 0.723 |
| `hybrid_weighted` | 0.875 | 0.875 | **0.812** | **0.814** |

Reading this honestly, which matters more than the headline:

- **Recall is flat across every configuration.** On a corpus this small, all the
  relevant chunks land in the top 5 regardless of method. Recall is the wrong
  metric here and reporting only the winning number would be misleading.
- **The real movement is in ranking.** MRR improves 30% and nDCG@5 improves 19%
  from dense-only to weighted hybrid. That is a genuine effect: ordering
  improved even though membership did not.
- **BM25 alone beats dense alone.** Expected, and it is a fact about the *fixture*,
  not about retrieval in general — the offline default uses hash embeddings,
  which have no semantic content whatsoever. Swap in
  `FINRAG_EMBEDDING_BACKEND=sentence-transformers` and the ordering changes.
- **One query fails in every configuration.** `q7` — "Did the company make any
  acquisitions?" — the answer sits only in the prior-year filing and recency
  signal buries it. That failure is real and unfixed. It is documented rather
  than tuned away, because the failing rows are the ones worth reading.

The numbers above are the CI fixture, deliberately small and deterministic. To
produce numbers on real filings, run `make ingest TICKERS=AAPL,MSFT` and label
against `evals/questions.yaml`.

---

## Why hybrid retrieval

Dense retrieval handles paraphrase — "margin pressure" finding "gross profit
declined" — and is weak on exact tokens. Filings are saturated with exact tokens
that carry the entire meaning: `Item 1A`, `ASC 606`, `$34 million`, `Basel III`,
specific subsidiary names. Ask a pure-vector system for the wording of a covenant
and it hands back thematically adjacent prose from the wrong fiscal year.

BM25 nails those and fails on paraphrase. Fusing recovers both.

**RRF vs weighted blending.** Reciprocal Rank Fusion is the default because it
needs no score normalisation between two systems whose scores are on
incomparable scales, and it degrades gracefully when one retriever returns
garbage for a given query. Weighted blending scores higher on this fixture —
`min-max` normalisation happens to suit a 5-chunk corpus — and that advantage is
expected to shrink as the corpus grows. Both are implemented; the config
switches between them, and the eval decides.

## Why section-aware chunking

A 10-K is not free text. It has mandated structure, and analyst questions are
implicitly scoped to a section: "what risks did they flag around supply chain"
means Item 1A.

A fixed 512-token window shreds those boundaries, producing chunks that start
mid-sentence in Item 7 and end inside Item 8. Retrieval then returns fragments
with no indication of company, year, or section, and the generator cannot cite.

So the chunker splits on Item boundaries first, packs within a section second,
carries sentence-level overlap so chunks never begin mid-clause, and stamps
every chunk with full provenance. Table-of-contents entries match the same
regex as real headings and are filtered by body length — a detail that silently
corrupts most naive filing parsers.

Gold labels are defined against *sections*, then resolved to whatever chunk IDs
the current chunker emits. Without that indirection, every chunking experiment
invalidates the labels and the harness becomes useless exactly when you need it.

## Why abstention

If the best retrieved chunk scores below threshold, the pipeline returns
`NOT_IN_CONTEXT` rather than answering. In a financial context a confident wrong
answer is strictly worse than no answer, and an LLM handed weak context will
produce one every time.

Citations are enforced structurally, not requested politely: chunks are numbered
in the prompt, the model must cite by number, and any citation that does not
resolve to a supplied chunk is flagged on the response. Asking a model to
"please cite your sources" and trusting the output is not a control.

---

## Architecture

```
EDGAR ──► html_to_text ──► section split ──► pack + overlap ──► chunks
                                                                  │
                        ┌─────────────────────────────────────────┤
                        ▼                                         ▼
                  bi-encoder embed                          BM25 index
                        │                                         │
                        └──────────► RRF fusion ◄─────────────────┘
                                          │
                                   cross-encoder rerank  (30 → 8)
                                          │
                              context assembly + abstention gate
                                          │
                                   LLM ──► cited answer
```

Retrieve wide with the cheap bi-encoder, rerank narrow with the expensive
cross-encoder. The bi-encoder embeds query and document independently and never
sees them together; the cross-encoder scores them jointly and is far more
accurate but far too slow to run over a corpus. This stage is usually the single
largest precision win in a RAG system and the most commonly skipped.

```
src/finrag/
├── config.py              all tunables, backend selection
├── ingest/
│   ├── edgar.py           rate-limited SEC client, disk cache
│   └── chunk.py           section-aware chunker
├── index/
│   ├── embed.py           sentence-transformers | hash (offline)
│   └── hybrid.py          BM25 + dense + RRF, from scratch
├── rerank/cross_encoder.py
├── rag/pipeline.py        abstention, citation enforcement
├── eval/
│   ├── metrics.py         recall@k, MRR, nDCG@k
│   ├── dataset.py         gold set + CI fixture
│   └── run_eval.py        ablation runner
└── api/main.py            FastAPI, /query /retrieve /healthz
```

BM25 is implemented directly (~80 lines) rather than pulled from a library that
would need pinning and offers nothing beyond this. Same for the metrics — every
published number should have an auditable definition.

## Every backend is swappable

| component | production | offline / CI |
|---|---|---|
| embeddings | `bge-small-en-v1.5` | deterministic hash |
| vector index | FAISS `IndexFlatIP` | numpy dot product |
| reranker | `ms-marco-MiniLM-L-6-v2` | no-op |
| generation | Anthropic / OpenAI | echo double |

CI runs the entire ingest → index → retrieve → evaluate loop with zero
downloads. If a test passes offline it is testing plumbing, not model quality —
which is exactly what a test should do.

## The eval is a build gate

`evals/results.json` is committed. On every push CI reruns the ablation and
compares `recall@5`, `ndcg@5`, and `mrr` against those committed numbers. A
drop beyond a 0.02 tolerance exits non-zero and fails the build.

```bash
python -m finrag.eval.run_eval --check-regression    # what CI runs
python -m finrag.eval.run_eval --update-baseline     # accept a change
```

Two details that matter. A failing run does **not** rewrite the baseline —
otherwise the next run would compare against the degraded numbers and the
gate would silently ratchet downward. And the tolerance exists because
tie-breaking between equal scores can differ across environments; it is set
as tight as that instability allows rather than as loose as is comfortable.

Changing the baseline requires an explicit flag and shows up as a diff in the
commit, which is the point: a quality regression should be a decision someone
made on purpose, not something that slid through.

## Running it

```bash
make test                       # 26 tests, ~0.2s
make eval                       # ablation table
make eval-full                  # real embeddings + cross-encoder
make ingest TICKERS=AAPL,MSFT   # requires SEC_USER_AGENT
make serve                      # uvicorn on :8000
make docker
```

```bash
curl -X POST localhost:8000/retrieve \
  -H 'content-type: application/json' \
  -d '{"question":"what supply chain risks were disclosed","top_k":5}'
```

`/retrieve` returns ranked hits with dense and lexical ranks side by side, with
no LLM call. It is cheap, deterministic, and the endpoint you actually want when
debugging answer quality.

## Performance and cost

Measured on the real corpus — 833 chunks from Apple and Microsoft 10-Ks,
FY2023–FY2026 — with 8 queries spanning exact-token and paraphrase retrieval,
25 repeats each. Reproduce with `python -m finrag.eval.benchmark --compare`.

| | hash (offline) | bge-small-en-v1.5 |
|---|---|---|
| index build | 0.87 s | 10.16 s |
| retrieval p50 | 0.67 ms | 7.71 ms |
| retrieval p95 | 1.46 ms | 8.66 ms |
| retrieval p99 | 1.49 ms | 13.39 ms |
| context tokens (mean) | 16,347 | **7,080** |
| context tokens (p95) | 45,482 | 29,106 |

### The expensive embedder is the cheap option

Semantic embeddings cost **11.6× more to index** and **5.9× more per query**.
They also cut generation cost roughly in half:

| model | hash | bge-small | saving |
|---|---|---|---|
| gpt-4o-mini | $2.60 | $1.21 | 53% |
| claude-haiku-4-5 | $17.60 | $8.33 | 53% |
| gpt-4o | $43.37 | $20.20 | 53% |
| claude-sonnet-4-6 | $52.79 | $24.99 | 53% |

*Per 1,000 queries, at published prices, assuming a 250-token answer.*

The mechanism is that better retrieval returns **shorter, more relevant**
context. The lexical fallback was pulling 16,347 tokens of loosely-related text
into the prompt on every query; semantic retrieval finds the right chunks and
stops, at 7,080.

So the intuition that better retrieval costs more is backwards at the system
level. The 7ms of extra retrieval latency is imperceptible to a user. The $28
per thousand queries is not — at any real volume, the expensive embedder pays
for itself many times over.

This is only visible because the benchmark measures **context tokens**, not just
latency. Retrieval itself is essentially free; what you pay for is what
retrieval decides to put in the prompt. Once that is measured, `top_k` and chunk
size stop being arbitrary config values and become the primary cost levers in
the system.

### Notes on the numbers

- **Percentiles, not means.** A mean latency hides the tail, and the tail is
  what users experience as "sometimes it hangs". The p50/p99 spread here is
  7.7ms to 13.4ms — tight, with no pathological tail.
- **The first query is excluded.** It pays for lazy imports and cache warming;
  including it would misattribute startup cost to steady state.
- **Token counts are estimated** at ~4 chars/token rather than tokenized
  exactly. An exact count needs the target model's tokenizer, which differs per
  provider. The error is a few percent and biased high, which is the safe
  direction for a cost projection.
- **Prices are published list prices** and are kept as data in `benchmark.py`
  rather than inlined, because they change and a stale constant buried in a
  calculation is how a cost estimate silently becomes wrong.
- Single machine, single run, Apple Silicon CPU, no GPU.

---

## Known limitations

- Tables are flattened to text during HTML extraction, so numeric questions that
  depend on table structure retrieve poorly. Table-aware extraction is the
  highest-value next change.
- No query decomposition. Multi-hop questions ("compare margin trends across
  three years") retrieve for the surface form only.
- The gold set is small. It demonstrates the harness; it does not establish
  statistical significance, and no claim here should be read as if it did.
- Recency bias is unaddressed — see `q7`.

## License

MIT
