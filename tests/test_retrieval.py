import pytest

from finrag.config import RetrievalConfig
from finrag.eval.run_eval import build_corpus
from finrag.index.embed import HashEmbedder
from finrag.index.hybrid import HybridRetriever, reciprocal_rank_fusion, tokenize


@pytest.fixture(scope="module")
def retriever():
    chunks, _ = build_corpus()
    return HybridRetriever(chunks, embedder=HashEmbedder(), cfg=RetrievalConfig(rerank=False))


def test_bm25_finds_exact_token(retriever):
    hits = retriever.bm25.search("ASC 606", top_k=3)
    assert hits and "606" in retriever.chunks[hits[0][0]].text


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)]], k=60)
    assert fused[2] > fused[1] and fused[2] > fused[3]


def test_filters_restrict_year(retriever):
    hits = retriever.retrieve("gross margin", filters={"fiscal_year": 2022})
    assert hits and all(h.chunk.fiscal_year == 2022 for h in hits)


def test_retrieve_respects_top_k(retriever):
    assert len(retriever.retrieve("revenue", top_k=2)) <= 2


def test_tokenizer_keeps_decimals():
    assert "26.4" in tokenize("Gross margin declined to 26.4% this year")


def test_bm25_unknown_terms_safe(retriever):
    assert retriever.bm25.search("zzzzz qqqqq", top_k=5) == []
