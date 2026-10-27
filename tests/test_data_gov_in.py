import pandas as pd
import pytest

from fetcher import API_CONFIG, MandiPriceFetcher


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload, pages=None):
        self.payload = payload
        self.pages = pages
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({
            "url": url,
            "params": params.copy(),
            "timeout": timeout,
        })

        if self.pages is None:
            payload = self.payload
        else:
            index = min(len(self.calls) - 1, len(self.pages) - 1)
            payload = self.pages[index]

        return FakeResponse(payload)


def make_fetcher(monkeypatch, payload, pages=None):
    monkeypatch.setattr(
        API_CONFIG,
        "data_gov_in_api_key",
        "test-api-key",
    )

    fetcher = object.__new__(MandiPriceFetcher)
    fetcher.session = FakeSession(payload, pages=pages)
    return fetcher

def valid_record(
    date="15/01/2025",
    market="Market A",
    minimum=1000,
    maximum=1500,
    modal=1200,
    state="Punjab",
):
    return {
        "Arrival_Date": date,
        "State": state,
        "Market": market,
        "Min_Price": minimum,
        "Max_Price": maximum,
        "Modal_Price": modal,
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "invalid response",
        123,
    ],
)
def test_malformed_data_gov_in_payload_raises_clear_error(
    monkeypatch, payload
):
    fetcher = make_fetcher(monkeypatch, payload)

    with pytest.raises(
        RuntimeError,
        match="expected a JSON object",
    ):
        fetcher._fetch_data_gov_in(
            "wheat",
            "Punjab",
            "2025-01-01",
            "2025-01-31",
        )


def test_data_gov_in_api_error_message_is_reported(monkeypatch):
    fetcher = make_fetcher(
        monkeypatch,
        {"message": "Invalid API key"},
    )

    with pytest.raises(RuntimeError, match="Invalid API key"):
        fetcher._fetch_data_gov_in(
            "wheat",
            "Punjab",
            "2025-01-01",
            "2025-01-31",
        )


def test_data_gov_in_filters_records_outside_requested_dates(monkeypatch):
    fetcher = make_fetcher(
        monkeypatch,
        {
            "records": [
                valid_record(date="31/12/2024"),
                valid_record(date="15/01/2025"),
                valid_record(date="01/02/2025"),
            ],
            "total": 3,
        },
    )

    result = fetcher._fetch_data_gov_in(
        "wheat",
        "Punjab",
        "2025-01-01",
        "2025-01-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2025-01-15")


def test_data_gov_in_skips_invalid_prices(monkeypatch):
    fetcher = make_fetcher(
        monkeypatch,
        {
            "records": [
                valid_record(market="Valid Market"),
                valid_record(market="Zero Price", modal=0),
                valid_record(
                    market="Inconsistent Price",
                    minimum=1300,
                    modal=1200,
                ),
                valid_record(market="Missing Price", modal=None),
            ],
            "total": 4,
        },
    )

    result = fetcher._fetch_data_gov_in(
        "wheat",
        "Punjab",
        "2025-01-01",
        "2025-01-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["modal_price"] == 1200


def test_data_gov_in_aggregates_multiple_markets_by_day(monkeypatch):
    fetcher = make_fetcher(
        monkeypatch,
        {
            "records": [
                valid_record(
                    market="Market A",
                    minimum=1000,
                    maximum=1500,
                    modal=1200,
                ),
                valid_record(
                    market="Market B",
                    minimum=1100,
                    maximum=1800,
                    modal=1400,
                ),
            ],
            "total": 2,
        },
    )

    result = fetcher._fetch_data_gov_in(
        "wheat",
        "Punjab",
        "2025-01-01",
        "2025-01-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["min_price"] == 1000
    assert result.iloc[0]["max_price"] == 1800
    assert result.iloc[0]["modal_price"] == 1300
    assert result.iloc[0]["market_count"] == 2
    assert result.attrs["data_source"] == "data.gov.in"
    assert result.attrs["is_historical"] is True
    assert result.attrs["is_complete"] is False

def test_data_gov_in_ignores_duplicate_records(monkeypatch):
    duplicate = valid_record(
        market="Market A",
        minimum=1000,
        maximum=1500,
        modal=1200,
    )

    fetcher = make_fetcher(
        monkeypatch,
        {
            "records": [
                duplicate,
                duplicate.copy(),
                valid_record(
                    market="Market B",
                    minimum=1100,
                    maximum=1800,
                    modal=1400,
                ),
            ],
            "total": 3,
        },
    )

    result = fetcher._fetch_data_gov_in(
        "wheat",
        "Punjab",
        "2025-01-01",
        "2025-01-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["market_count"] == 2
    assert result.iloc[0]["min_price"] == 1000
    assert result.iloc[0]["max_price"] == 1800
    assert result.iloc[0]["modal_price"] == 1300

def test_data_gov_in_pagination_advances_offset(monkeypatch):
    first_page_records = [
        valid_record(
            market=f"Market {i}",
            minimum=1000,
            maximum=3000,
            modal=1200 + i,
        )
        for i in range(1000)
    ]

    second_page_records = [
        valid_record(
            market="Final Market",
            minimum=1000,
            maximum=3000,
            modal=2500,
        )
    ]

    fetcher = make_fetcher(
        monkeypatch,
        payload=None,
        pages=[
            {
                "records": first_page_records,
                "total": 1001,
            },
            {
                "records": second_page_records,
                "total": 1001,
            },
        ],
    )

    result = fetcher._fetch_data_gov_in(
        "wheat",
        "Punjab",
        "2025-01-01",
        "2025-01-31",
    )

    assert len(fetcher.session.calls) == 2

    offsets = [
        call["params"]["offset"]
        for call in fetcher.session.calls
    ]

    assert offsets == [0, 1000]
    assert result.iloc[0]["market_count"] == 1001
    assert result.attrs["data_source"] == "data.gov.in"

def test_data_gov_in_malformed_second_page_raises_error(monkeypatch):
    first_page_records = [
        valid_record(
            market=f"Market {i}",
            minimum=1000,
            maximum=3000,
            modal=1200 + i,
        )
        for i in range(1000)
    ]

    fetcher = make_fetcher(
        monkeypatch,
        payload=None,
        pages=[
            {
                "records": first_page_records,
                "total": 1001,
            },
            None,
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="expected a JSON object",
    ):
        fetcher._fetch_data_gov_in(
            "wheat",
            "Punjab",
            "2025-01-01",
            "2025-01-31",
        )

    assert len(fetcher.session.calls) == 2

def test_data_gov_in_duplicate_key_uses_first_record(monkeypatch):
    fetcher = make_fetcher(
        monkeypatch,
        {
            "records": [
                valid_record(
                    market="Market A",
                    minimum=1000,
                    maximum=1500,
                    modal=1200,
                ),
                valid_record(
                    market="Market A",
                    minimum=900,
                    maximum=1600,
                    modal=1200,
                ),
            ],
            "total": 2,
        },
    )

    result = fetcher._fetch_data_gov_in(
        "wheat",
        "Punjab",
        "2025-01-01",
        "2025-01-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["market_count"] == 1
    assert result.iloc[0]["min_price"] == 1000
    assert result.iloc[0]["max_price"] == 1500
    assert result.iloc[0]["modal_price"] == 1200