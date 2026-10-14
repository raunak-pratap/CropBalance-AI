import pandas as pd

from fetcher import MandiPriceFetcher


def test_aggregation_removes_exact_duplicates_and_preserves_distinct_records():
    fetcher = object.__new__(MandiPriceFetcher)

    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "state": "Maharashtra",
                "market": "Market A",
                "variety": "Local",
                "arrivals_tonnes": 10.0,
                "min_price": 1000.0,
                "max_price": 1500.0,
                "modal_price": 1200.0,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "state": "Maharashtra",
                "market": "Market A",
                "variety": "Local",
                "arrivals_tonnes": 10.0,
                "min_price": 1000.0,
                "max_price": 1500.0,
                "modal_price": 1200.0,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "state": "Maharashtra",
                "market": "Market B",
                "variety": "Local",
                "arrivals_tonnes": 30.0,
                "min_price": 1200.0,
                "max_price": 1600.0,
                "modal_price": 1400.0,
            },
        ]
    )

    result = fetcher._aggregate_agmarknet_daily(df)

    assert len(result) == 1
    assert result.loc[0, "arrivals_tonnes"] == 40.0
    assert result.loc[0, "modal_price"] == 1350.0
    assert result.loc[0, "market_count"] == 2


def test_distinct_records_from_same_market_are_not_dropped():
    fetcher = object.__new__(MandiPriceFetcher)

    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "state": "Maharashtra",
                "market": "Market A",
                "variety": "Local",
                "arrivals_tonnes": 10.0,
                "min_price": 1000.0,
                "max_price": 1500.0,
                "modal_price": 1200.0,
            },
            {
                "date": pd.Timestamp("2026-01-01"),
                "state": "Maharashtra",
                "market": "Market A",
                "variety": "Local",
                "arrivals_tonnes": 20.0,
                "min_price": 1100.0,
                "max_price": 1600.0,
                "modal_price": 1300.0,
            },
        ]
    )

    result = fetcher._aggregate_agmarknet_daily(df)

    assert result.loc[0, "arrivals_tonnes"] == 30.0
    assert result.loc[0, "modal_price"] == (10 * 1200 + 20 * 1300) / 30
    assert result.loc[0, "market_count"] == 1
