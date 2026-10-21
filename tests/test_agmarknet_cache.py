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


def test_cache_with_only_invalid_rows_is_rejected(tmp_path):
    fetcher = object.__new__(MandiPriceFetcher)

    cache_path = tmp_path / "wheat_punjab_2026_01.csv"

    invalid_rows = pd.DataFrame([
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
    invalid_rows.to_csv(cache_path, index=False)

    fetcher._get_cache_path = lambda crop, state, year, month: cache_path

    # Confirm the invalid cache is rejected.
    assert fetcher._load_agmarknet_cache(
        "wheat", "Punjab", 2026, 1
    ) is None


def test_fetch_downloads_when_cache_has_no_valid_rows(tmp_path, monkeypatch):
    fetcher = object.__new__(MandiPriceFetcher)

    cache_path = tmp_path / "wheat_punjab_2026_01.csv"
    pd.DataFrame([
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
    ]).to_csv(cache_path, index=False)

    fetcher._get_cache_path = lambda crop, state, year, month: cache_path

    class FakeResolver:
        def get_state_id(self, state):
            return 1

        def get_commodity_id(self, crop):
            return 2

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "success": True,
                "markets": [{
                    "marketName": "Fresh Market",
                    "dates": [{
                        "arrivalDate": "15/01/2026",
                        "data": [{
                            "variety": "Common",
                            "arrivals": 10,
                            "minimumPrice": 2000,
                            "maximumPrice": 2400,
                            "modalPrice": 2200,
                        }],
                    }],
                }],
            }

    calls = []

    def fake_get(url, params, max_retries=3):
        calls.append(params)
        return FakeResponse()

    fetcher.metadata_resolver = FakeResolver()
    fetcher._get_with_retry = fake_get
    fetcher._save_agmarknet_cache = lambda *args, **kwargs: None

    result = fetcher._fetch_agmarknet(
        "wheat", "Punjab", "2026-01-01", "2026-01-31"
    )

    assert len(calls) == 1
    assert calls[0]["year"] == 2026
    assert calls[0]["month"] == 1
    assert not result.empty
    assert result.iloc[0]["modal_price"] == 2200
    assert "2026-01" in result.attrs["successful_months"]


def test_partial_month_failure_marks_result_incomplete(monkeypatch):
    fetcher = object.__new__(MandiPriceFetcher)

    class FakeResolver:
        def get_state_id(self, state):
            return 1

        def get_commodity_id(self, crop):
            return 2

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "success": True,
                "markets": [{
                    "marketName": "Test Market",
                    "dates": [{
                        "arrivalDate": "15/01/2026",
                        "data": [{
                            "variety": "Common",
                            "arrivals": 10,
                            "minimumPrice": 2000,
                            "maximumPrice": 2400,
                            "modalPrice": 2200,
                        }],
                    }],
                }],
            }

    def fake_get(url, params, max_retries=3):
        if params["month"] == 2:
            raise RuntimeError("Simulated API failure")
        return FakeResponse()

    fetcher.metadata_resolver = FakeResolver()
    fetcher._get_with_retry = fake_get
    fetcher._load_agmarknet_cache = (
        lambda crop, state, year, month: None
    )
    fetcher._save_agmarknet_cache = lambda *args, **kwargs: None

    result = fetcher._fetch_agmarknet(
        "wheat", "Punjab", "2026-01-01", "2026-02-28"
    )

    assert not result.empty
    assert result.attrs["is_complete"] is False
    assert "2026-01" in result.attrs["successful_months"]
    assert "2026-02" in result.attrs["failed_months"]


def test_fetch_agmarknet_filters_records_to_requested_date_range():
    fetcher = object.__new__(MandiPriceFetcher)

    class FakeResolver:
        def get_state_id(self, state):
            return 1

        def get_commodity_id(self, crop):
            return 2

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "success": True,
                "markets": [{
                    "marketName": "Test Market",
                    "dates": [
                        {
                            "arrivalDate": "31/01/2026",
                            "data": [{
                                "variety": "Common",
                                "arrivals": 10,
                                "minimumPrice": 2000,
                                "maximumPrice": 2400,
                                "modalPrice": 2200,
                            }],
                        },
                        {
                            "arrivalDate": "15/02/2026",
                            "data": [{
                                "variety": "Common",
                                "arrivals": 20,
                                "minimumPrice": 2100,
                                "maximumPrice": 2500,
                                "modalPrice": 2300,
                            }],
                        },
                        {
                            "arrivalDate": "01/03/2026",
                            "data": [{
                                "variety": "Common",
                                "arrivals": 30,
                                "minimumPrice": 2200,
                                "maximumPrice": 2600,
                                "modalPrice": 2400,
                            }],
                        },
                    ],
                }],
            }

    fetcher.metadata_resolver = FakeResolver()
    fetcher._get_with_retry = lambda *args, **kwargs: FakeResponse()
    fetcher._load_agmarknet_cache = lambda *args: None
    fetcher._save_agmarknet_cache = lambda *args, **kwargs: None

    result = fetcher._fetch_agmarknet(
        "wheat", "Punjab", "2026-02-01", "2026-02-28"
    )

    assert len(result) == 1
    assert pd.to_datetime(result.iloc[0]["date"]) == pd.Timestamp("2026-02-15")
    assert result.iloc[0]["arrivals_tonnes"] == 20


def test_aggregate_agmarknet_daily_combines_markets_and_removes_exact_duplicates():
    fetcher = object.__new__(MandiPriceFetcher)

    rows = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-01-15"),
            "state": "Punjab",
            "market": "Market A",
            "arrivals_tonnes": 10.0,
            "min_price": 2000.0,
            "max_price": 2400.0,
            "modal_price": 2200.0,
        },
        {
            "date": pd.Timestamp("2026-01-15"),
            "state": "Punjab",
            "market": "Market A",
            "arrivals_tonnes": 10.0,
            "min_price": 2000.0,
            "max_price": 2400.0,
            "modal_price": 2200.0,
        },
        {
            "date": pd.Timestamp("2026-01-15"),
            "state": "Punjab",
            "market": "Market B",
            "arrivals_tonnes": 30.0,
            "min_price": 2100.0,
            "max_price": 2500.0,
            "modal_price": 2400.0,
        },
    ])

    result = fetcher._aggregate_agmarknet_daily(rows)

    assert len(result) == 1
    daily = result.iloc[0]

    # The exact duplicate Market A row must not be counted twice.
    assert daily["arrivals_tonnes"] == 40.0
    assert daily["min_price"] == 2000.0
    assert daily["max_price"] == 2500.0
    assert daily["modal_price"] == 2350.0
    assert daily["market_count"] == 2
