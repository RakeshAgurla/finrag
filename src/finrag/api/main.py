"""FastAPI serving layer.

Kept deliberately small. The interesting engineering is upstream; this exists so
the system is deployable and observable rather than a notebook.

Includes: startup-time index load (not per-request), request-scoped latency
metrics, a /healthz that fails when the index is missing, and structured
retrieval traces on the response so failures are debuggable in production
instead of mysterious.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finrag.eval.run_eval import build_corpus
from finrag.index.hybrid import HybridRetriever
from finrag.rag.pipeline import RagPipeline

logger = logging.getLogger("finrag.api")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    chunks, _ = build_corpus()
    retriever = HybridRetriever(chunks)
    state["pipeline"] = RagPipeline(retriever)
    state["n_chunks"] = len(chunks)
    logger.info("index ready: %d chunks in %.2fs", len(chunks), time.time() - started)
    yield
    state.clear()


app = FastAPI(title="finrag", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=25)
    ticker: str | None = None
    fiscal_year: int | None = None


class SourceOut(BaseModel):
    chunk_id: str
    ticker: str
    fiscal_year: int
    section_item: str
    section_title: str
    score: float
    rerank_score: float | None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    abstained: bool
    sources: list[SourceOut]
    latency_ms: float


@app.get("/healthz")
def healthz():
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="index not loaded")
    return {"status": "ok", "chunks": state["n_chunks"]}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="index not loaded")

    filters = {}
    if req.ticker:
        filters["ticker"] = req.ticker.upper()
    if req.fiscal_year:
        filters["fiscal_year"] = req.fiscal_year

    started = time.perf_counter()
    answer = pipeline.answer(req.question, filters=filters or None)
    latency = (time.perf_counter() - started) * 1000

    if answer.unresolved_citations:
        logger.warning(
            "ungrounded citations %s for question=%r",
            answer.unresolved_citations, req.question,
        )

    return QueryResponse(
        answer=answer.text,
        grounded=answer.grounded,
        abstained=answer.abstained,
        latency_ms=round(latency, 2),
        sources=[
            SourceOut(
                chunk_id=h.chunk.chunk_id,
                ticker=h.chunk.ticker,
                fiscal_year=h.chunk.fiscal_year,
                section_item=h.chunk.section_item,
                section_title=h.chunk.section_title,
                score=round(h.score, 4),
                rerank_score=h.rerank_score,
                excerpt=h.chunk.text[:300],
            )
            for h in answer.sources[: req.top_k]
        ],
    )


@app.post("/retrieve")
def retrieve_only(req: QueryRequest):
    """Retrieval without generation. Cheap, deterministic, and the endpoint you
    actually want when debugging answer quality."""
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    hits = pipeline.retriever.retrieve(req.question, top_k=req.top_k)
    return {
        "question": req.question,
        "hits": [
            {
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 4),
                "dense_rank": h.dense_rank,
                "lexical_rank": h.lexical_rank,
                "section": f"Item {h.chunk.section_item}",
            }
            for h in hits
        ],
    }
