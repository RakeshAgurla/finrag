"""Ingest CLI: EDGAR -> chunks on disk.

    python -m finrag.ingest.cli --tickers AAPL,MSFT --form 10-K --years 3
"""

from __future__ import annotations

import argparse

from finrag.config import settings
from finrag.index.hybrid import save_chunks
from finrag.ingest.chunk import chunk_filing
from finrag.ingest.edgar import fetch_filing, list_filings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="comma-separated")
    parser.add_argument("--form", default="10-K")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="bypass disk cache")
    args = parser.parse_args()

    if "example.com" in settings.user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT with real contact info before ingesting. "
            "The SEC blocks anonymous clients. See .env.example"
        )

    all_chunks = []
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        refs = list_filings(ticker, args.form, limit=args.years)
        if not refs:
            print(f"  {ticker}: no {args.form} filings found")
            continue
        for ref in refs:
            text = fetch_filing(ref, force=args.force)
            chunks = chunk_filing(text, ref.ticker, ref.fiscal_year, ref.form_type)
            all_chunks.extend(chunks)
            print(f"  {ticker} FY{ref.fiscal_year}: {len(chunks)} chunks "
                  f"({len(text):,} chars)")

    out = settings.vector_store.metadata_path
    save_chunks(all_chunks, out)
    print(f"\n{len(all_chunks)} chunks -> {out}")
    print("Next: label questions in evals/questions.yaml, then `make eval-full`")


if __name__ == "__main__":
    main()
