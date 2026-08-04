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

## Running it

```bash
make test                       # 23 tests, ~0.2s
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
