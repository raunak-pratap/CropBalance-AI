from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from database.market_price_store import market_price_observations, metadata
from services.national_market_price_quality_audit import (
    NationalMarketPriceQualityAuditor,
)


def make_row(
    record_key,
    *,
    observation_date,
    market="APMC Kalamb",
    commodity="Wheat",
    variety="Lokwan",
    grade="FAQ",
    min_price="100",
    max_price="120",
    modal_price="110",
    price_unit="Rs/Quintal",
    arrival_quantity="12",
    source_record_id=None,
    retrieved_at=None,
    source="data.gov.in",
):
    retrieved = retrieved_at or datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc)
    return {
        "record_key": record_key,
        "source": source,
        "source_resource_id": "resource-1",
        "source_record_id": source_record_id,
        "observation_date": observation_date,
        "state": "Maharashtra",
        "district": "Dharashiv",
        "market": market,
        "commodity": commodity,
        "variety": variety,
        "grade": grade,
        "source_state_id": "27",
        "source_district_id": "DHR",
        "source_market_id": "KAL",
        "source_commodity_id": "WHT",
        "source_variety_id": None,
        "min_price": Decimal(min_price) if min_price is not None else None,
        "max_price": Decimal(max_price) if max_price is not None else None,
        "modal_price": Decimal(modal_price) if modal_price is not None else None,
        "price_unit": price_unit,
        "arrival_quantity": (
            Decimal(arrival_quantity) if arrival_quantity is not None else None
        ),
        "arrival_unit": "Quintal",
        "arrivals_available": arrival_quantity is not None,
        "retrieved_at": retrieved,
        "source_updated_at": None,
        "payload_hash": "hash-" + record_key,
        "raw_record": {"test": record_key},
        "created_at": retrieved,
        "updated_at": retrieved,
    }


@pytest.fixture
def audit_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(engine, tables=[market_price_observations])
    try:
        yield engine
    finally:
        engine.dispose()


def insert_rows(engine, rows):
    with engine.begin() as connection:
        connection.execute(market_price_observations.insert(), rows)


def test_audit_reports_coverage_quality_duplicates_and_calendar_gaps(audit_engine):
    first = make_row(
        "row-1",
        observation_date=date(2026, 11, 10),
        min_price="150",
        max_price="120",
        modal_price="130",
        retrieved_at=datetime(2026, 11, 15, 8, 0, tzinfo=timezone.utc),
    )
    duplicate_candidate = make_row(
        "row-2",
        observation_date=date(2026, 11, 10),
        min_price="150",
        max_price="120",
        modal_price="130",
    )
    latest = make_row(
        "row-3",
        observation_date=date(2026, 11, 14),
        retrieved_at=datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc),
    )
    incomplete = make_row(
        "row-4",
        observation_date=date(2026, 11, 12),
        market="APMC Tuljapur",
        commodity="Onion",
        min_price="0",
        max_price=None,
        modal_price=None,
        price_unit=None,
        arrival_quantity="-1",
    )
    insert_rows(audit_engine, [first, duplicate_candidate, latest, incomplete])

    report = NationalMarketPriceQualityAuditor(audit_engine).audit(
        now=datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
    )

    assert report["status"] == "ok"
    assert report["read_only"] is True
    assert report["coverage"]["observation_count"] == 4
    assert report["coverage"]["state_count"] == 1
    assert report["coverage"]["district_count"] == 1
    assert report["coverage"]["market_count"] == 2
    assert report["coverage"]["commodity_count"] == 2
    assert report["coverage"]["first_observation_date"] == "2026-11-10"
    assert report["coverage"]["last_observation_date"] == "2026-11-14"
    assert report["coverage"]["observation_age_days_at_audit"] == 1
    assert report["coverage"]["retrieval_age_hours_at_audit"] == 2.0

    quality = report["quality"]
    assert quality["missing_max_price_rows"] == 1
    assert quality["missing_modal_price_rows"] == 1
    assert quality["rows_with_any_missing_price"] == 1
    assert quality["rows_with_any_non_positive_price"] == 1
    assert quality["min_price_exceeds_max_price_rows"] == 1
    assert quality["min_price_exceeds_modal_price_rows"] == 1
    assert quality["modal_price_exceeds_max_price_rows"] == 1
    assert quality["rows_with_any_price_order_violation"] == 1
    assert quality["missing_price_unit_rows"] == 1
    assert quality["negative_arrival_quantity_rows"] == 1

    duplicates = report["duplicates"]
    assert duplicates["candidate_group_count"] == 1
    assert duplicates["records_in_candidate_groups"] == 2
    assert duplicates["excess_records_over_one_per_group"] == 1

    temporal = report["temporal_coverage"]
    assert temporal["series_count"] == 2
    assert temporal["calendar_gap_interval_count"] == 1
    assert temporal["series_with_calendar_gaps"] == 1
    assert temporal["missing_calendar_days_between_observations"] == 3
    assert temporal["largest_calendar_gap_days"] == 3
    assert temporal["largest_gap_examples"][0]["missing_calendar_days"] == 3
    assert "do not prove" in temporal["interpretation"]


def test_audit_filters_rows_without_changing_the_source_table(audit_engine):
    insert_rows(
        audit_engine,
        [
            make_row("wheat-old", observation_date=date(2026, 11, 10)),
            make_row("wheat-new", observation_date=date(2026, 11, 14)),
            make_row(
                "onion",
                observation_date=date(2026, 11, 14),
                commodity="Onion",
                market="APMC Tuljapur",
            ),
        ],
    )

    report = NationalMarketPriceQualityAuditor(audit_engine).audit(
        state="maharashtra",
        commodity="Wheat",
        start_date=date(2026, 11, 11),
        end_date=date(2026, 11, 15),
    )

    assert report["coverage"]["observation_count"] == 1
    assert report["coverage"]["market_count"] == 1
    assert report["coverage"]["commodity_count"] == 1
    assert report["coverage"]["first_observation_date"] == "2026-11-14"
    assert report["filters"]["commodity"] == "Wheat"

    with audit_engine.connect() as connection:
        total_rows = connection.execute(
            market_price_observations.select().with_only_columns(
                market_price_observations.c.record_key
            )
        ).all()
    assert len(total_rows) == 3


def test_audit_handles_an_empty_observation_table(audit_engine):
    report = NationalMarketPriceQualityAuditor(audit_engine).audit()

    assert report["coverage"]["observation_count"] == 0
    assert report["coverage"]["first_observation_date"] is None
    assert report["coverage"]["last_observation_date"] is None
    assert report["duplicates"]["candidate_group_count"] == 0
    assert report["temporal_coverage"]["series_count"] == 0
    assert report["temporal_coverage"]["largest_gap_examples"] == []


def test_audit_rejects_invalid_date_range(audit_engine):
    with pytest.raises(ValueError, match="start_date cannot be after end_date"):
        NationalMarketPriceQualityAuditor(audit_engine).audit(
            start_date=date(2026, 11, 15),
            end_date=date(2026, 11, 14),
        )


def test_audit_rejects_naive_audit_timestamp(audit_engine):
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        NationalMarketPriceQualityAuditor(audit_engine).audit(
            now=datetime(2026, 11, 15, 12, 0)
        )


def test_audit_requires_a_previously_initialized_observation_table():
    engine = create_engine("sqlite://")
    try:
        with pytest.raises(RuntimeError, match="does not exist"):
            NationalMarketPriceQualityAuditor(engine).audit()
    finally:
        engine.dispose()
