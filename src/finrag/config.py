"""Central configuration.

Everything tunable lives here, so experiments never require touching code.
Backends are swappable so the same pipeline runs offline in CI (hash embeddings,
in-memory FAISS) and against real infrastructure locally (sentence-transformers,
Pinecone).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FINRAG_DATA_DIR", REPO_ROOT / "data"))
INDEX_DIR = Path(os.getenv("FINRAG_INDEX_DIR", REPO_ROOT / "artifacts" / "index"))


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking is the single highest-leverage knob in a filings RAG system.

    10-K sections are long and hierarchical. Fixed 512-token windows destroy the
    Item boundaries that make retrieval work, so we chunk *within* sections and
    carry the section header into every chunk as a prefix.
    """

    target_tokens: int = 450
    overlap_tokens: int = 60
    min_tokens: int = 80
    prepend_section_header: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid retrieval settings.

    alpha weights dense vs lexical in the fused score. Filings are full of exact
    identifiers (item numbers, GAAP line items, dollar figures) that dense
    retrieval alone reliably misses, which is why BM25 stays in the loop.
    """

    dense_top_k: int = 40
    lexical_top_k: int = 40
    fusion: str = "rrf"  # "rrf" | "weighted"
    rrf_k: int = 60
    alpha: float = 0.5  # only used when fusion == "weighted"
    final_top_k: int = 8
    rerank: bool = True
    rerank_candidates: int = 30


@dataclass(frozen=True)
class EmbeddingConfig:
    # "hash" is a deterministic, dependency-free fallback used in CI so the test
    # suite runs in seconds without downloading model weights.
    backend: str = os.getenv("FINRAG_EMBEDDING_BACKEND", "sentence-transformers")
    model_name: str = os.getenv("FINRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    dim: int = 384
    batch_size: int = 64
    normalize: bool = True


@dataclass(frozen=True)
class RerankConfig:
    backend: str = os.getenv("FINRAG_RERANK_BACKEND", "cross-encoder")
    model_name: str = os.getenv(
        "FINRAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    batch_size: int = 32


@dataclass(frozen=True)
class VectorStoreConfig:
    backend: str = os.getenv("FINRAG_VECTOR_BACKEND", "faiss")  # "faiss" | "pinecone"
    index_path: Path = INDEX_DIR / "faiss.index"
    metadata_path: Path = INDEX_DIR / "chunks.jsonl"
    pinecone_index: str = os.getenv("PINECONE_INDEX", "finrag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")


@dataclass(frozen=True)
class GenerationConfig:
    backend: str = os.getenv("FINRAG_LLM_BACKEND", "anthropic")
    model_name: str = os.getenv("FINRAG_LLM_MODEL", "claude-sonnet-4-6")
    max_tokens: int = 1024
    temperature: float = 0.0
    # Refuse to answer rather than hallucinate when retrieval comes back weak.
    min_context_score: float = 0.15


@dataclass(frozen=True)
class Settings:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    user_agent: str = os.getenv("SEC_USER_AGENT", "finrag research contact@example.com")


settings = Settings()
