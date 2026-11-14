"""Fetch and persist record-level daily mandi prices from data.gov.in.

The default mode is a read-only dry run. Use --write only after reviewing the
page count, rejected-row count and pagination-complete status.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import API_CONFIG  # noqa: E402
from database.market_price_store import MarketPriceStore  # noqa: E402
from services.data_gov_in_daily_prices import (  # noqa: E402
    DataGovInDailyPriceClient,
    normalize_market_price_records,
)


def parse_iso_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM-DD date, got {value!r}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest individual daily mandi-price records from OGD India."
    )
    parser.add_argument("--state", help="Optional exact OGD state filter.")
    parser.add_argument("--commodity", help="Optional exact OGD commodity filter.")
    parser.add_argument("--start-date", type=parse_iso_date, help="Keep observations on/after YYYY-MM-DD.")
    parser.add_argument("--end-date", type=parse_iso_date, help="Keep observations on/before YYYY-MM-DD.")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-records", type=int, help="Optional cap for a deliberately partial run.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit --write when --max-records deliberately capped the run.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create the observation table if needed and upsert validated records.",
    )
    args = parser.parse_args()

    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date cannot be after --end-date.")
    if args.allow_partial and not args.max_records:
        parser.error("--allow-partial is only meaningful with --max-records.")
    if args.write and args.max_records and not args.allow_partial:
        parser.error(
            "--write with --max-records requires --allow-partial. "
            "Otherwise remove --max-records for a complete paginated run."
        )

    client = DataGovInDailyPriceClient(
        api_key=API_CONFIG.data_gov_in_api_key,
        base_url=API_CONFIG.data_gov_in_base_url,
    )
    retrieved_at = datetime.now(timezone.utc)

    print("DATA.GOV.IN DAILY MANDI PRICE INGESTION")
    print("Resource:", "9ef84268-d588-465a-a308-a864a43d0070")
    print("State filter:", args.state or "All available states")
    print("Commodity filter:", args.commodity or "All available commodities")
    print("Mode:", "WRITE" if args.write else "DRY RUN")
    print("Page size:", args.page_size)

    fetched = client.fetch(
        state=args.state,
        commodity=args.commodity,
        page_size=args.page_size,
        max_pages=args.max_pages,
        max_records=args.max_records,
    )
    normalized = normalize_market_price_records(
        fetched.records,
        retrieved_at=retrieved_at,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print("\nFETCH SUMMARY")
    print("Pages fetched:", fetched.pages_fetched)
    print("Rows received:", len(fetched.records))
    print("API reported total:", fetched.reported_total)
    print("Pagination complete:", fetched.pagination_complete)
    print("Stop reason:", fetched.stop_reason)

    print("\nNORMALIZATION SUMMARY")
    print("Valid unique observations:", len(normalized.records))
    print("Duplicate rows collapsed:", normalized.duplicate_count)
    print("Rows rejected:", normalized.rejected_count)
    print("Rows outside date range:", normalized.out_of_range_count)
    if normalized.issues:
        print("Sample normalization issues (maximum 20):")
        for issue in normalized.issues:
            print(f"  row {issue['index']}: {issue['reason']}")

    if not fetched.pagination_complete and not (
        args.max_records and args.allow_partial
    ):
        print(
            "\nNOT WRITTEN: pagination is incomplete. Increase --max-pages "
            "or inspect the API response before persisting this batch."
        )
        return 2

    if not args.write:
        print("\nDRY RUN ONLY — PostgreSQL was not modified.")
        print("Rerun with --write after reviewing the summary.")
        return 0

    if not normalized.records:
        print("\nNOT WRITTEN: no valid observations were produced.")
        return 1

    store = MarketPriceStore.from_env()
    try:
        store.ensure_schema()
        result = store.upsert_observations(normalized.records)
    finally:
        store.dispose()

    print("\nDATABASE WRITE COMPLETE")
    print("Rows received by store:", result["received"])
    print("Rows inserted or updated:", result["upserted"])
    print("Source:", "data.gov.in")
    print("Retrieved at (UTC):", retrieved_at.isoformat())
    if not fetched.pagination_complete:
        print("WARNING: this was an explicitly allowed partial run.")
    print(
        "Arrival quantity is stored only when provided by the source; "
        "missing arrivals remain NULL."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
