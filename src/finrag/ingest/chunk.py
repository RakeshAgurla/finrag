"""Section-aware chunking for SEC filings.

Why not a plain recursive character splitter?

A 10-K is not free text. It has a mandated structure (Item 1 Business, Item 1A
Risk Factors, Item 7 MD&A, Item 8 Financial Statements ...). Analysts ask
questions that are implicitly scoped to a section: "what risks did they flag
around supply chain" means Item 1A, "how did management explain the margin
decline" means Item 7.

A fixed-window splitter shreds those boundaries and produces chunks that begin
mid-sentence in Item 7 and end inside Item 8. Retrieval then returns fragments
with no indication of which company, year, or section they came from, and the
generator cannot cite properly.

So: split on section boundaries first, pack within a section second, and stamp
every chunk with its provenance.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

from finrag.config import ChunkConfig, settings

# Item headings as they actually appear in EDGAR text, which is inconsistent
# about spacing, punctuation, and case.
_ITEM_PATTERN = re.compile(
    r"^\s*item\s+(\d{1,2}[A-Za-z]?)\s*[\.\:\-\u2014]?\s*(.{0,120})$",
    re.IGNORECASE | re.MULTILINE,
)

CANONICAL_SECTIONS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    fiscal_year: int
    form_type: str
    section_item: str
    section_title: str
    char_start: int
    char_end: int
    token_estimate: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded.

        Prefixing the header measurably improves retrieval on section-scoped
        questions -- the eval harness quantifies this; see evals/README.md.
        """
        if not settings.chunk.prepend_section_header:
            return self.text
        header = f"{self.ticker} {self.fiscal_year} {self.form_type} | Item {self.section_item}: {self.section_title}"
        return f"{header}\n\n{self.text}"


def estimate_tokens(text: str) -> int:
    """Cheap token estimate.

    Deliberately not tiktoken: this runs over millions of chars during ingest and
    the chunker only needs a stable proxy, not exact counts. Empirically ~4 chars
    per token on filing prose, which skews long and safe.
    """
    return max(1, len(text) // 4)


@dataclass
class Section:
    item: str
    title: str
    text: str
    char_start: int


def split_sections(raw_text: str) -> list[Section]:
    """Split a filing into Item-level sections.

    Falls back to a single synthetic section when no Item headings are found, so
    the pipeline still works on exhibits, press releases, and 8-Ks.
    """
    matches = list(_ITEM_PATTERN.finditer(raw_text))

    # EDGAR documents contain a table of contents that also matches the Item
    # pattern. Real section bodies are long; TOC entries are not. Drop any match
    # whose span to the next match is implausibly short.
    filtered: list[re.Match] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        if end - m.start() >= 400:
            filtered.append(m)

    if not filtered:
        return [Section(item="0", title="Full Document", text=raw_text, char_start=0)]

    sections: list[Section] = []
    for i, m in enumerate(filtered):
        item = m.group(1).upper()
        inline_title = (m.group(2) or "").strip(" .:-\u2014")
        title = CANONICAL_SECTIONS.get(item) or inline_title or f"Item {item}"
        start = m.start()
        end = filtered[i + 1].start() if i + 1 < len(filtered) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append(
                Section(item=item, title=title, text=body, char_start=start)
            )
    return sections


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _pack(
    paragraphs: Iterable[str], cfg: ChunkConfig
) -> Iterator[tuple[str, int, int]]:
    """Greedily pack paragraphs up to the target size with sentence-level overlap.

    Yields (text, start_offset_within_section, end_offset_within_section).
    Overlap is carried as whole trailing sentences rather than a raw character
    slice so chunks never begin mid-clause.
    """
    buffer: list[str] = []
    buffer_tokens = 0
    cursor = 0
    start_offset = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if buffer and buffer_tokens + para_tokens > cfg.target_tokens:
            text = "\n\n".join(buffer)
            yield text, start_offset, start_offset + len(text)

            # Build overlap from trailing sentences of what we just emitted.
            sentences = re.split(r"(?<=[.!?])\s+", text)
            overlap: list[str] = []
            overlap_tokens = 0
            for sent in reversed(sentences):
                t = estimate_tokens(sent)
                if overlap_tokens + t > cfg.overlap_tokens:
                    break
                overlap.insert(0, sent)
                overlap_tokens += t

            buffer = [" ".join(overlap)] if overlap else []
            buffer_tokens = overlap_tokens
            start_offset = max(0, cursor - len(" ".join(overlap)))

        buffer.append(para)
        buffer_tokens += para_tokens
        cursor += len(para) + 2

    if buffer:
        text = "\n\n".join(buffer)
        if estimate_tokens(text) >= cfg.min_tokens:
            yield text, start_offset, start_offset + len(text)


def chunk_filing(
    raw_text: str,
    ticker: str,
    fiscal_year: int,
    form_type: str = "10-K",
    cfg: ChunkConfig | None = None,
) -> list[Chunk]:
    """Turn one filing into retrievable chunks with full provenance."""
    cfg = cfg or settings.chunk
    chunks: list[Chunk] = []

    for section in split_sections(raw_text):
        paragraphs = _split_paragraphs(section.text)
        for idx, (text, rel_start, rel_end) in enumerate(_pack(paragraphs, cfg)):
            chunk_id = (
                f"{ticker}:{fiscal_year}:{form_type}:item{section.item}:{idx:04d}"
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    ticker=ticker.upper(),
                    fiscal_year=fiscal_year,
                    form_type=form_type,
                    section_item=section.item,
                    section_title=section.title,
                    char_start=section.char_start + rel_start,
                    char_end=section.char_start + rel_end,
                    token_estimate=estimate_tokens(text),
                )
            )
    return chunks
