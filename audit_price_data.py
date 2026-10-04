import argparse
import pandas as pd

from fetcher import MandiPriceFetcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    args = parser.parse_args()

    print("=" * 60)
    print("AGMARKNET PRICE DATA AUDIT")
    print("=" * 60)
    print(f"Crop:       {args.crop}")
    print(f"State:      {args.state}")
    print(f"Start:      {args.start}")
    print(f"End:        {args.end}")
    print("=" * 60)

    fetcher = MandiPriceFetcher()

    df = fetcher.fetch(
        crop=args.crop,
        state=args.state,
        start_date=args.start,
        end_date=args.end,
    )

    print("\nDATA SOURCE:", df.attrs.get("data_source"))
    print("IS LIVE:", df.attrs.get("is_live"))
    print("IS HISTORICAL:", df.attrs.get("is_historical"))
    print("COMPLETE:", df.attrs.get("is_complete"))
    print("ROWS:", len(df))

    if df.empty:
        print("\n❌ No data returned.")
        return

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    expected_dates = pd.date_range(
        start=args.start,
        end=args.end,
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(df["date"].dt.normalize().unique())

    missing_dates = expected_dates.difference(actual_dates)

    print("\n" + "=" * 60)
    print("COVERAGE")
    print("=" * 60)

    print("Calendar days:", len(expected_dates))
    print("Reporting days:", len(actual_dates))
    print(
        "Coverage:     "
        f"{len(actual_dates) / len(expected_dates) * 100:.2f}%"
    )
    print("Missing days:", len(missing_dates))

    if len(missing_dates):
        print("\nFirst missing dates:")
        for d in missing_dates[:20]:
            print(" ", d.date())

    # Duplicate dates
    duplicate_dates = df[df["date"].duplicated(keep=False)]

    print("\nDuplicate date rows:", len(duplicate_dates))

    # Missing values
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(df.isna().sum())

    # Price statistics
    print("\n" + "=" * 60)
    print("PRICE STATISTICS")
    print("=" * 60)

    for column in ["min_price", "modal_price", "max_price"]:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].describe())

    # Arrivals
    if "arrivals_tonnes" in df.columns:
        print("\n" + "=" * 60)
        print("ARRIVALS")
        print("=" * 60)
        print(df["arrivals_tonnes"].describe())

    # Market count
    if "market_count" in df.columns:
        print("\n" + "=" * 60)
        print("MARKET COUNT")
        print("=" * 60)
        print(df["market_count"].describe())

    # Gaps
    dates = pd.Series(sorted(actual_dates))
    gaps = dates.diff().dt.days.dropna()

    if not gaps.empty:
        print("\n" + "=" * 60)
        print("DATE GAPS")
        print("=" * 60)

        print("Maximum gap:", int(gaps.max()), "days")

        large_gaps = gaps[gaps > 7]

        if not large_gaps.empty:
            print("\nGaps > 7 days:")
            print(large_gaps.to_string())

    # Highest modal prices
    if "modal_price" in df.columns:
        print("\n" + "=" * 60)
        print("TOP 10 MODAL PRICE DAYS")
        print("=" * 60)

        print(
            df.nlargest(10, "modal_price")[
                [
                    "date",
                    "modal_price",
                    "min_price",
                    "max_price",
                    "arrivals_tonnes",
                    "market_count",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()