from finrag.config import GenerationConfig, RetrievalConfig
from finrag.eval.run_eval import build_corpus
from finrag.index.embed import HashEmbedder
from finrag.index.hybrid import HybridRetriever
from finrag.rag.pipeline import CITATION_RE, RagPipeline, format_context


def _retriever():
    chunks, _ = build_corpus()
    return HybridRetriever(chunks, embedder=HashEmbedder(), cfg=RetrievalConfig(rerank=False))


def _pipeline(min_score=0.0):
    return RagPipeline(_retriever(), GenerationConfig(backend="echo", min_context_score=min_score))


def test_context_is_numbered():
    ctx = format_context(_retriever().retrieve("revenue", top_k=3))
    assert ctx.startswith("[1]") and "[2]" in ctx


def test_abstains_below_threshold():
    answer = _pipeline(min_score=99.0).answer("what is the revenue")
    assert answer.abstained and not answer.grounded


def test_answer_returns_sources():
    answer = _pipeline().answer("what drove the gross margin decline")
    assert answer.sources and answer.grounded


def test_citation_regex_extracts_numbers():
    assert [int(n) for n in CITATION_RE.findall("see [1] and [99]")] == [1, 99]
