"""Sample corpus + gold question set.

Two reasons this exists rather than shipping real EDGAR text in the repo:

1. The test suite and CI must run with no network and no model downloads.
2. Gold labels require someone to actually read the source and decide which
   chunk answers the question. Hand-labelling real 10-Ks is the right thing to
   do for the published numbers -- see `finrag.ingest.edgar` and
   `evals/questions.yaml` -- but a deterministic fixture is what makes the
   harness testable.

The real eval set is built by running `make ingest` then labelling. This module
is the scaffold and the CI fixture, not the headline result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SAMPLE_FILINGS: dict[tuple[str, int], str] = {
    ("ACME", 2023): """
Item 1. Business

Acme Industrial Corporation designs and manufactures precision components for
the aerospace and automotive sectors. The Company operates fourteen production
facilities across North America and Europe and employs approximately 21,000
people worldwide. Revenue is reported across three segments: Aerospace Systems,
Automotive Components, and Industrial Services.

The Company sells primarily to original equipment manufacturers under long-term
supply agreements, typically ranging from three to seven years in duration.
Approximately 38% of consolidated revenue in fiscal 2023 was derived from the
Company's five largest customers.

Item 1A. Risk Factors

Our results are exposed to disruption in the supply of specialty alloys. A
substantial portion of the titanium and nickel alloys used in our aerospace
products is sourced from a limited number of qualified suppliers, and in certain
cases from a single qualified supplier. Qualification of an alternative supplier
typically requires twelve to eighteen months and customer approval. An extended
interruption in supply would materially affect our ability to meet delivery
commitments.

We face significant customer concentration risk. The loss of any one of our five
largest customers, or a material reduction in their order volumes, would have a
material adverse effect on our results of operations and financial condition.

Cybersecurity incidents could disrupt our manufacturing operations. Our
production facilities rely on networked industrial control systems. A successful
intrusion affecting those systems could halt production, compromise proprietary
designs, and expose us to liability under customer contracts and data protection
regulations.

Changes in international trade policy, including tariffs on imported raw
materials and components, may increase our costs. During fiscal 2023 the Company
incurred approximately $34 million in incremental tariff costs that it was
unable to fully recover through customer price adjustments.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

Consolidated revenue for fiscal 2023 was $4.82 billion, an increase of 7.3%
compared to $4.49 billion in fiscal 2022. The increase was driven primarily by
higher volumes in Aerospace Systems, which benefited from continued recovery in
commercial aircraft build rates, partially offset by softness in Automotive
Components.

Gross margin declined to 26.4% in fiscal 2023 from 28.1% in fiscal 2022. The
decline of 170 basis points reflects higher raw material costs, incremental
tariff expense, and unfavorable manufacturing absorption at two facilities that
underwent extended maintenance shutdowns during the third quarter.

Operating cash flow was $612 million compared to $701 million in the prior year.
The decrease was primarily attributable to an increase in working capital as the
Company built inventory of long-lead-time alloys to mitigate supply risk.

The Company returned $340 million to shareholders during fiscal 2023 through a
combination of dividends and share repurchases, and reduced total debt by $180
million.

Item 8. Financial Statements and Supplementary Data

The Company recognizes revenue in accordance with ASC 606. For long-term supply
agreements, revenue is recognized at a point in time upon transfer of control,
which generally occurs upon shipment. Certain aerospace contracts containing
customer-specific tooling with no alternative use are recognized over time using
an input method based on costs incurred.

Total assets at fiscal year end were $7.94 billion. Goodwill of $1.62 billion is
tested for impairment annually as of the first day of the fourth quarter. No
impairment was recorded in fiscal 2023.
""",
    ("ACME", 2022): """
Item 1A. Risk Factors

Our results are exposed to disruption in the supply of specialty alloys, which
are sourced from a limited number of qualified suppliers.

We face significant customer concentration risk, with approximately 41% of
fiscal 2022 revenue derived from our five largest customers.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

Consolidated revenue for fiscal 2022 was $4.49 billion. Gross margin was 28.1%.
Operating cash flow was $701 million. The Company completed the acquisition of
Cordera Precision Holdings for total consideration of $410 million in the second
quarter of fiscal 2022.
""",
}


@dataclass
class GoldQuestion:
    """A question plus graded relevance labels.

    gain scale: 2 = directly answers, 1 = relevant supporting context, 0 = not
    relevant. Grades rather than binary because nDCG is only meaningful with
    them, and because "mentions tariffs" and "quantifies the tariff cost" are
    genuinely different for an analyst.
    """

    query_id: str
    question: str
    relevance: dict[str, float] = field(default_factory=dict)
    section_hint: str | None = None
    note: str = ""


def build_gold_set(chunk_ids_by_section: dict[str, list[str]]) -> list[GoldQuestion]:
    """Attach labels to the chunk ids produced by the current chunker.

    Labels are defined against *sections*, then resolved to whichever chunk ids
    the chunker emitted for that section. This keeps the gold set valid when
    chunk size changes -- otherwise every chunking experiment invalidates the
    labels and the harness becomes useless exactly when you need it.
    """

    def ids(section: str) -> list[str]:
        return chunk_ids_by_section.get(section, [])

    def grade(section: str, gain: float) -> dict[str, float]:
        return {cid: gain for cid in ids(section)}

    return [
        GoldQuestion(
            query_id="q1",
            question="What supply chain risks did Acme disclose regarding raw materials?",
            relevance={**grade("ACME:2023:1A", 2.0), **grade("ACME:2022:1A", 1.0)},
            section_hint="1A",
            note="Paraphrase question; dense retrieval should carry this one.",
        ),
        GoldQuestion(
            query_id="q2",
            question="How much incremental tariff cost did the company incur in fiscal 2023?",
            relevance=grade("ACME:2023:1A", 2.0),
            section_hint="1A",
            note="Exact-figure question; BM25 carries it, pure dense often misses.",
        ),
        GoldQuestion(
            query_id="q3",
            question="Why did gross margin decline year over year?",
            relevance={**grade("ACME:2023:7", 2.0), **grade("ACME:2022:7", 1.0)},
            section_hint="7",
            note="Classic MD&A question. Tests year disambiguation.",
        ),
        GoldQuestion(
            query_id="q4",
            question="What revenue recognition policy does the company apply under ASC 606?",
            relevance=grade("ACME:2023:8", 2.0),
            section_hint="8",
            note="Standard identifier. Dense retrieval reliably fails on 'ASC 606'.",
        ),
        GoldQuestion(
            query_id="q5",
            question="What was operating cash flow and why did it change?",
            relevance={**grade("ACME:2023:7", 2.0), **grade("ACME:2022:7", 1.0)},
            section_hint="7",
        ),
        GoldQuestion(
            query_id="q6",
            question="How concentrated is Acme's customer base?",
            relevance={**grade("ACME:2023:1", 2.0), **grade("ACME:2023:1A", 2.0), **grade("ACME:2022:1A", 1.0)},
            note="Answer spans two sections; rewards recall over precision@1.",
        ),
        GoldQuestion(
            query_id="q7",
            question="Did the company make any acquisitions?",
            relevance=grade("ACME:2022:7", 2.0),
            note="Answer lives only in the prior year. Tests that we do not "
            "over-anchor on the most recent filing.",
        ),
        GoldQuestion(
            query_id="q8",
            question="What cybersecurity exposure does the company describe?",
            relevance=grade("ACME:2023:1A", 2.0),
            section_hint="1A",
        ),
    ]
