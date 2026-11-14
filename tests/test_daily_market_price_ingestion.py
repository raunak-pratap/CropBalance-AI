from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine

from database.market_price_store import MarketPriceStore
from services.data_gov_in_daily_prices import (
    DataGovInDailyPriceClient,
    normalize_market_price_record,
    normalize_market_price_records,
)


def valid_record(**overrides):
    record = {
        "Arrival_Date": "14/11/2026",
        "State": "Maharashtra",
        "District": "Dharashiv",
        "Market": "APMC Kalamb",
        "Commodity": "Soyabean",
        "Variety": "Yellow",
        "Grade": "FAQ",
        "Min_Price": "6800",
        "Max_Price": "7300",
        "Modal_Price": "7100",
    }
    record.update(overrides)
    return record


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        index = min(len(self.calls) - 1, len(self.pages) - 1)
        return FakeResponse(self.pages[index])


def test_normalizer_preserves_one_market_observation_without_aggregation():
    rows = [
        valid_record(Market="APMC Kalamb"),
        valid_record(Market="APMC Tuljapur", Modal_Price="6900"),
    ]

    result = normalize_market_price_records(
        rows,
        retrieved_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
    )

    assert len(result.records) == 2
    assert {row["market"] for row in result.records} == {
        "APMC Kalamb",
        "APMC Tuljapur",
    }
    assert {row["observation_date"] for row in result.records} == {date(2026, 11, 14)}


def test_normalizer_keeps_variety_and_grade_as_distinct_observation_keys():
    first = normalize_market_price_record(valid_record(Variety="Yellow", Grade="FAQ"))
    second = normalize_market_price_record(valid_record(Variety="Other", Grade="FAQ"))
    third = normalize_market_price_record(valid_record(Variety="Yellow", Grade="Grade B"))

    assert len({first["record_key"], second["record_key"], third["record_key"]}) == 3


def test_normalizer_accepts_iso_dates_and_optional_arrival_fields():
    row = normalize_market_price_record(
        valid_record(
            Arrival_Date="2026-11-14",
            Arrivals="12.5",
            Arrival_Unit="Tonnes",
            Price_Unit="Rs./Quintal",
        )
    )

    assert row["observation_date"] == date(2026, 11, 14)
    assert row["arrival_quantity"] == Decimal("12.5")
    assert row["arrival_unit"] == "Tonnes"
    assert row["price_unit"] == "Rs./Quintal"
    assert row["arrivals_available"] is True


def test_missing_arrivals_and_price_unit_remain_missing_not_inferred():
    row = normalize_market_price_record(valid_record())

    assert row["arrival_quantity"] is None
    assert row["arrival_unit"] is None
    assert row["arrivals_available"] is False
    assert row["price_unit"] is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"District": ""}, "district"),
        ({"Modal_Price": "not-a-price"}, "not numeric"),
        ({"Min_Price": "7200", "Modal_Price": "7100"}, "min_price cannot exceed modal_price"),
        ({"Min_Price": "0"}, "min_price must be greater than zero"),
        ({"Arrival_Date": "yesterday"}, "unsupported format"),
    ],
)
def test_invalid_source_rows_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        normalize_market_price_record(valid_record(**overrides))


def test_arrival_only_record_can_be_preserved_without_inventing_prices():
    row = normalize_market_price_record(
        valid_record(
            Min_Price="",
            Max_Price="",
            Modal_Price="",
            Arrivals="4.25",
            Arrival_Unit="Quintals",
        )
    )
    assert row["min_price"] is None
    assert row["max_price"] is None
    assert row["modal_price"] is None
    assert row["arrival_quantity"] == Decimal("4.25")


def test_payload_hash_changes_when_report_changes_but_record_key_does_not():
    first = normalize_market_price_record(valid_record(Modal_Price="7100"))
    corrected = normalize_market_price_record(valid_record(Modal_Price="7125"))

    assert first["record_key"] == corrected["record_key"]
    assert first["payload_hash"] != corrected["payload_hash"]


def test_batch_normalization_deduplicates_and_applies_date_window():
    records = [
        valid_record(),
        valid_record(),
        valid_record(Arrival_Date="13/11/2026", Market="APMC Bhoom"),
        valid_record(Arrival_Date="12/11/2026", Market="APMC Murum", Modal_Price="invalid"),
    ]
    result = normalize_market_price_records(
        records,
        start_date=date(2026, 11, 14),
        end_date=date(2026, 11, 14),
    )

    assert len(result.records) == 1
    assert result.duplicate_count == 1
    assert result.out_of_range_count == 1
    assert result.rejected_count == 1
    assert result.issues[0]["index"] == 3


def test_client_paginates_with_offset_and_passes_exact_filters():
    pages = [
        {
            "records": [valid_record(Market="Market A"), valid_record(Market="Market B")],
            "total": "3",
        },
        {
            "records": [valid_record(Market="Market C")],
            "total": "3",
        },
    ]
    session = FakeSession(pages)
    client = DataGovInDailyPriceClient(
        api_key="test-key",
        session=session,
    )

    result = client.fetch(
        state="Maharashtra",
        commodity="Soyabean",
        page_size=2,
    )

    assert len(result.records) == 3
    assert result.pages_fetched == 2
    assert result.pagination_complete is True
    assert [call["params"]["offset"] for call in session.calls] == [0, 2]
    assert session.calls[0]["params"]["filters[State]"] == "Maharashtra"
    assert session.calls[0]["params"]["filters[Commodity]"] == "Soyabean"


def test_client_refuses_missing_api_key():
    client = DataGovInDailyPriceClient(api_key="", session=FakeSession([]))
    with pytest.raises(RuntimeError, match="DATA_GOV_IN_API_KEY"):
        client.fetch()


def test_client_rejects_malformed_response_and_missing_records():
    client = DataGovInDailyPriceClient(
        api_key="test-key",
        session=FakeSession([{"message": "Invalid API key"}]),
    )
    with pytest.raises(RuntimeError, match="Invalid API key"):
        client.fetch()


def test_client_reports_incomplete_when_max_pages_reached():
    full_page = {
        "records": [valid_record(), valid_record(Market="Market B")],
        "total": "100",
    }
    client = DataGovInDailyPriceClient(
        api_key="test-key",
        session=FakeSession([full_page]),
    )
    result = client.fetch(page_size=2, max_pages=1)

    assert result.pagination_complete is False
    assert result.stop_reason == "max_pages_reached"


def test_client_rejects_page_size_above_api_limit():
    client = DataGovInDailyPriceClient(api_key="test-key", session=FakeSession([]))
    with pytest.raises(ValueError, match="page_size"):
        client.fetch(page_size=1001)


def test_store_upsert_is_idempotent_and_updates_corrected_price():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    store = MarketPriceStore(engine)
    store.ensure_schema()

    first = normalize_market_price_record(valid_record(Modal_Price="7100"))
    first_result = store.upsert_observations([first])
    original = store.query_observations(commodity="Soyabean")

    corrected = normalize_market_price_record(valid_record(Modal_Price="7125"))
    corrected_result = store.upsert_observations([corrected])
    after = store.query_observations(commodity="Soyabean")

    assert first_result == {"received": 1, "upserted": 1}
    assert corrected_result == {"received": 1, "upserted": 1}
    assert original["count"] == 1
    assert after["count"] == 1
    assert after["items"][0]["modal_price"] == "7125.0000"
    assert after["items"][0]["record_key"] == first["record_key"]

    store.dispose()


def test_store_keeps_market_level_rows_and_reports_observation_coverage():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    store = MarketPriceStore(engine)
    store.ensure_schema()

    rows = normalize_market_price_records([
        valid_record(Market="APMC Kalamb"),
        valid_record(Market="APMC Tuljapur"),
        valid_record(Market="APMC Paranda", Variety="Other"),
    ]).records
    store.upsert_observations(rows)

    result = store.query_observations(state="Maharashtra", limit=10)
    coverage = store.coverage()

    assert result["count"] == 3
    assert len({row["market"] for row in result["items"]}) == 3
    assert coverage["observation_count"] == 3
    assert coverage["state_count"] == 1
    assert coverage["district_count"] == 1
    assert coverage["market_count"] == 3
    assert coverage["commodity_count"] == 1

    store.dispose()


def test_store_validates_query_dates_and_pagination():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    store = MarketPriceStore(engine)
    store.ensure_schema()

    with pytest.raises(ValueError, match="start_date cannot be after"):
        store.query_observations(
            start_date=date(2026, 11, 15),
            end_date=date(2026, 11, 14),
        )
    with pytest.raises(ValueError, match="limit must be between"):
        store.query_observations(limit=501)

    store.dispose()
