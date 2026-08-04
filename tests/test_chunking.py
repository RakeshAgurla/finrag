from finrag.eval.dataset import SAMPLE_FILINGS
from finrag.ingest.chunk import chunk_filing, estimate_tokens, split_sections


def test_sections_are_detected():
    sections = split_sections(SAMPLE_FILINGS[("ACME", 2023)])
    assert {"1", "1A", "7", "8"} <= {s.item for s in sections}


def test_toc_entries_are_filtered_out():
    toc = "Item 1. Business ... 3\nItem 1A. Risk Factors ... 9\n"
    body = "Item 7. MD&A\n\n" + ("Real body text. " * 200)
    sections = split_sections(toc + body)
    assert len(sections) == 1 and sections[0].item == "7"


def test_chunks_carry_provenance():
    chunks = chunk_filing(SAMPLE_FILINGS[("ACME", 2023)], "ACME", 2023)
    assert chunks
    for c in chunks:
        assert c.ticker == "ACME" and c.fiscal_year == 2023
        assert c.section_item and c.chunk_id.startswith("ACME:2023")


def test_embedding_text_includes_header():
    chunks = chunk_filing(SAMPLE_FILINGS[("ACME", 2023)], "ACME", 2023)
    assert "Item" in chunks[0].embedding_text.split("\n")[0]


def test_no_section_headings_falls_back():
    sections = split_sections("Plain prose with no item headings at all.")
    assert len(sections) == 1 and sections[0].item == "0"


def test_token_estimate_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
