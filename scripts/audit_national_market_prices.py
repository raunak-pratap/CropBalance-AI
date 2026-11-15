"""Run the read-only national market-price quality audit."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from database.market_price_store import create_database_engine  # noqa: E402
from services.national_market_price_quality_audit import (  # noqa: E402
    NationalMarketPriceQualityAuditor,
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
        description=(
            "Audit stored national mandi-price observations. "
            "This command performs SELECT queries only."
        )
    )
    parser.add_argument("--state", help="Optional state-name contains filter.")
    parser.add_argument("--district", help="Optional district-name contains filter.")
    parser.add_argument("--market", help="Optional market-name contains filter.")
    parser.add_argument("--commodity", help="Optional commodity-name contains filter.")
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        help="Include observations on/after YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        help="Include observations on/before YYYY-MM-DD.",
    )
    parser.add_argument(
        "--max-gap-examples",
        type=int,
        default=10,
        help="Maximum calendar-gap examples in the report (0-100).",
    )
    args = parser.parse_args()

    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date cannot be after --end-date.")
    if not 0 <= args.max_gap_examples <= 100:
        parser.error("--max-gap-examples must be between 0 and 100.")

    engine = None
    try:
        engine = create_database_engine()
        report = NationalMarketPriceQualityAuditor(engine).audit(
            state=args.state,
            district=args.district,
            market=args.market,
            commodity=args.commodity,
            start_date=args.start_date,
            end_date=args.end_date,
            max_gap_examples=args.max_gap_examples,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"AUDIT FAILED: database query error ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 2
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
