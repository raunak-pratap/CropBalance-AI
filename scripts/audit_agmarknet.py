import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fetcher import MandiPriceFetcher
import pandas as pd


TESTS = [
    ("wheat", "Punjab"),
    ("rice", "Punjab"),
    ("tomato", "Maharashtra"),
    ("onion", "Maharashtra"),
]

START_DATE = "2026-04-01"
END_DATE = "2026-10-03"


def audit(df, crop, state):
    df = df.sort_values("date").copy()

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    calendar_days = pd.date_range(
        start,
        end,
        freq="D",
    )

    observed_days = pd.DatetimeIndex(
        df["date"].unique()
    )

    gaps = (
        pd.Series(observed_days)
        .sort_values()
        .diff()
        .dropna()
        .dt.days
    )

    coverage = (
        len(observed_days)
        / len(calendar_days)
        * 100
    )

    max_gap = (
        int(gaps.max())
        if not gaps.empty
        else 0
    )

    market_counts = df["market_count"]

    print("\n" + "=" * 60)
    print(f"{crop.upper()} / {state.upper()}")
    print("=" * 60)

    print(f"Calendar days : {len(calendar_days)}")
    print(f"Reporting days: {len(observed_days)}")
    print(f"Coverage      : {coverage:.1f}%")
    print(f"Missing days  : {len(calendar_days) - len(observed_days)}")
    print(f"Maximum gap   : {max_gap} days")

    print(
        f"Market count  : "
        f"mean={market_counts.mean():.2f}, "
        f"max={market_counts.max():.0f}"
    )

    print(
        f"Price range   : "
        f"₹{df['modal_price'].min():.2f}"
        f" → "
        f"₹{df['modal_price'].max():.2f}"
    )

    print(
        f"Price records : {df['modal_price'].notna().sum()}"
    )

    print(
        f"Missing values: {df.isna().sum().sum()}"
    )

    print("\nGaps > 7 days:")

    large_gaps = gaps[gaps > 7]

    if large_gaps.empty:
        print("None")
    else:
        print(large_gaps.to_string(index=False))


def main():
    fetcher = MandiPriceFetcher()

    for crop, state in TESTS:

        try:
            df = fetcher.fetch(
                crop=crop,
                state=state,
                start_date=START_DATE,
                end_date=END_DATE,
            )
            source = df.attrs.get("data_source")

            if source != "agmarknet":
                print(
                    f"SKIPPED: {crop.upper()} / {state.upper()} "
                    f"returned {source} data, not AGMARKNET."
                )
                continue

            audit(
                df,
                crop,
                state,
            )

        except Exception as e:

            print("\n" + "=" * 60)
            print(f"{crop.upper()} / {state.upper()}")
            print("=" * 60)
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()