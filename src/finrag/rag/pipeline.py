"""Answer generation.

Two design choices that drive everything here:

1. **Abstention.** If the best retrieved chunk scores below a threshold, the
   pipeline returns "not found in the provided filings" instead of answering.
   In a financial context a confident wrong answer is worse than no answer, and
   an LLM handed weak context will produce one every time.

2. **Citations are enforced structurally, not requested politely.** Chunks are
   numbered in the prompt, the model is required to cite by number, and any
   answer whose citations do not resolve to a supplied chunk is flagged. Asking
   a model to "please cite sources" and trusting it is not a control.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from finrag.config import GenerationConfig, settings
from finrag.index.hybrid import HybridRetriever, RetrievedChunk

SYSTEM_PROMPT = """You are a research assistant answering questions about SEC filings.

Rules:
- Answer ONLY from the numbered excerpts provided. Do not use outside knowledge.
- Cite every factual claim with the excerpt number in square brackets, e.g. [2].
- If the excerpts do not contain the answer, say exactly: NOT_IN_CONTEXT
- Quote figures exactly as they appear. Never estimate, round, or infer a number.
- If excerpts from different fiscal years conflict, state both and label the year.
"""

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    unresolved_citations: list[int] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.abstained and not self.unresolved_citations


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, hit in enumerate(chunks, start=1):
        c = hit.chunk
        blocks.append(
            f"[{i}] {c.ticker} | FY{c.fiscal_year} {c.form_type} | "
            f"Item {c.section_item}: {c.section_title}\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        cfg: GenerationConfig | None = None,
    ):
        self.retriever = retriever
        self.cfg = cfg or settings.generation

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        hits = self.retriever.retrieve(question, filters=filters)

        if not hits:
            return Answer(question, "NOT_IN_CONTEXT", [], abstained=True)

        best = max(h.rerank_score if h.rerank_score is not None else h.score for h in hits)
        if best < self.cfg.min_context_score:
            return Answer(
                question,
                "NOT_IN_CONTEXT — retrieval confidence below threshold.",
                hits,
                abstained=True,
            )

        text = self._generate(question, format_context(hits))
        abstained = "NOT_IN_CONTEXT" in text

        cited = {int(n) for n in CITATION_RE.findall(text)}
        unresolved = sorted(n for n in cited if n < 1 or n > len(hits))

        return Answer(
            question=question,
            text=text,
            sources=hits,
            abstained=abstained,
            unresolved_citations=unresolved,
        )

    def _generate(self, question: str, context: str) -> str:
        backend = self.cfg.backend
        user_msg = f"Excerpts:\n\n{context}\n\nQuestion: {question}"

        if backend == "echo":  # test double
            return f"[1] (echo backend) {question}"

        if backend == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            response = client.messages.create(
                model=self.cfg.model_name,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return "".join(b.text for b in response.content if b.type == "text")

        if backend == "openai":
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.cfg.model_name,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return response.choices[0].message.content or ""

        raise ValueError(f"Unknown generation backend: {backend}")
