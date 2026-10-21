import pandas as pd

from fetcher import MandiPriceFetcher


def test_cache_filters_invalid_prices_without_modifying_file(tmp_path):
    fetcher = object.__new__(MandiPriceFetcher)

    cache_path = tmp_path / "wheat_punjab_2026_01.csv"

    cached_rows = pd.DataFrame([
        {
            "date": "2026-01-01",
            "state": "Punjab",
            "market": "Valid Market",
            "variety": "Common",
            "arrivals_tonnes": 10.0,
            "min_price": 2000.0,
            "max_price": 2400.0,
            "modal_price": 2200.0,
        },
        {
            "date": "2026-01-01",
            "state": "Punjab",
            "market": "Invalid Market",
            "variety": "Common",
            "arrivals_tonnes": 100.0,
            "min_price": 2275.0,
            "max_price": 0.0,
            "modal_price": 2275.0,
        },
    ])

    cached_rows.to_csv(cache_path, index=False)
    original_contents = cache_path.read_bytes()

    fetcher._get_cache_path = lambda crop, state, year, month: cache_path

    result = fetcher._load_agmarknet_cache(
        "wheat", "Punjab", 2026, 1
    )

    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["market"] == "Valid Market"
    assert result.iloc[0]["arrivals_tonnes"] == 10.0

    # Reading and filtering the cache must not rewrite the CSV.
    assert cache_path.read_bytes() == original_contents
