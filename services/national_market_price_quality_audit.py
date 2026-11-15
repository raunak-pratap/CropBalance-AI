"""Read-only quality and temporal-coverage audit for national mandi observations.

The auditor inspects the stored market_price_observations table only. It never
fetches upstream data, changes schema, updates rows, or deletes duplicates.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, func, inspect, or_, select
from sqlalchemy.engine import Engine

from database.market_price_store import market_price_observations


_TABLE = market_price_observations


def _serialize(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _serialized_mapping(row: Any) -> dict[str, Any]:
    return {key: _serialize(value) for key, value in dict(row).items()}


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        # SQLite may return a naive datetime for a timezone-aware column.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _build_filters(
    *,
    state: str | None,
    district: str | None,
    market: str | None,
    commodity: str | None,
    start_date: date | None,
    end_date: date | None,
) -> tuple[list[Any], dict[str, Any]]:
    filters: list[Any] = []
    normalized: dict[str, Any] = {}

    for name, value in (
        ("state", state),
        ("district", district),
        ("market", market),
        ("commodity", commodity),
    ):
        cleaned = value.strip() if value and value.strip() else None
        normalized[name] = cleaned
        if cleaned:
            escaped = (
                cleaned.casefold()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            filters.append(
                func.lower(_TABLE.c[name]).like(f"%{escaped}%", escape="\\")
            )

    normalized["start_date"] = start_date.isoformat() if start_date else None
    normalized["end_date"] = end_date.isoformat() if end_date else None
    if start_date:
        filters.append(_TABLE.c.observation_date >= start_date)
    if end_date:
        filters.append(_TABLE.c.observation_date <= end_date)

    return filters, normalized


class NationalMarketPriceQualityAuditor:
    """Produce a read-only quality report for stored market-price observations."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def audit(
        self,
        *,
        state: str | None = None,
        district: str | None = None,
        market: str | None = None,
        commodity: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        now: datetime | None = None,
        max_gap_examples: int = 10,
    ) -> dict[str, Any]:
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")
        if not 0 <= max_gap_examples <= 100:
            raise ValueError("max_gap_examples must be between 0 and 100.")

        if not inspect(self.engine).has_table(_TABLE.name):
            raise RuntimeError(
                "market_price_observations does not exist. "
                "Run the daily market-price ingestion setup before auditing."
            )

        audit_time = now or datetime.now(timezone.utc)
        if audit_time.tzinfo is None:
            raise ValueError("now must be timezone-aware.")
        audit_time = audit_time.astimezone(timezone.utc)

        filter_expressions, normalized_filters = _build_filters(
            state=state,
            district=district,
            market=market,
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
        )
        where = and_(*filter_expressions) if filter_expressions else None

        def apply_filter(statement: Any) -> Any:
            return statement.where(where) if where is not None else statement

        def count_rows(condition: Any | None = None) -> int:
            statement = select(func.count()).select_from(_TABLE)
            statement = apply_filter(statement)
            if condition is not None:
                statement = statement.where(condition)
            with self.engine.connect() as connection:
                return int(connection.execute(statement).scalar_one())

        table = _TABLE
        coverage_query = apply_filter(
            select(
                func.count().label("observation_count"),
                func.count(func.distinct(table.c.state)).label("state_count"),
                func.count(func.distinct(table.c.district)).label("district_count"),
                func.count(func.distinct(table.c.market)).label("market_count"),
                func.count(func.distinct(table.c.commodity)).label("commodity_count"),
                func.min(table.c.observation_date).label("first_observation_date"),
                func.max(table.c.observation_date).label("last_observation_date"),
                func.max(table.c.retrieved_at).label("last_retrieved_at"),
            ).select_from(table)
        )

        missing_min = table.c.min_price.is_(None)
        missing_max = table.c.max_price.is_(None)
        missing_modal = table.c.modal_price.is_(None)
        non_positive_min = and_(table.c.min_price.is_not(None), table.c.min_price <= 0)
        non_positive_max = and_(table.c.max_price.is_not(None), table.c.max_price <= 0)
        non_positive_modal = and_(
            table.c.modal_price.is_not(None), table.c.modal_price <= 0
        )
        min_exceeds_max = and_(
            table.c.min_price.is_not(None),
            table.c.max_price.is_not(None),
            table.c.min_price > table.c.max_price,
        )
        min_exceeds_modal = and_(
            table.c.min_price.is_not(None),
            table.c.modal_price.is_not(None),
            table.c.min_price > table.c.modal_price,
        )
        modal_exceeds_max = and_(
            table.c.modal_price.is_not(None),
            table.c.max_price.is_not(None),
            table.c.modal_price > table.c.max_price,
        )
        missing_price_unit = or_(
            table.c.price_unit.is_(None),
            func.trim(table.c.price_unit) == "",
        )
        missing_source_record_id = or_(
            table.c.source_record_id.is_(None),
            func.trim(table.c.source_record_id) == "",
        )
        missing_payload_hash = or_(
            table.c.payload_hash.is_(None),
            func.trim(table.c.payload_hash) == "",
        )
        negative_arrivals = and_(
            table.c.arrival_quantity.is_not(None),
            table.c.arrival_quantity < 0,
        )
        any_missing_price = or_(missing_min, missing_max, missing_modal)
        any_non_positive_price = or_(
            non_positive_min, non_positive_max, non_positive_modal
        )
        any_price_order_violation = or_(
            min_exceeds_max, min_exceeds_modal, modal_exceeds_max
        )

        with self.engine.connect() as connection:
            coverage_row = connection.execute(coverage_query).mappings().one()

        coverage = _serialized_mapping(coverage_row)
        last_observation_date = coverage.pop("last_observation_date")
        last_retrieved_at = coverage["last_retrieved_at"]

        observation_age_days = None
        if last_observation_date is not None:
            observed = date.fromisoformat(last_observation_date)
            observation_age_days = (audit_time.date() - observed).days

        retrieval_age_hours = None
        if last_retrieved_at is not None:
            retrieved = datetime.fromisoformat(last_retrieved_at)
            retrieval_age_hours = round(
                (audit_time - _utc_datetime(retrieved)).total_seconds() / 3600,
                2,
            )

        coverage["last_observation_date"] = last_observation_date
        coverage["observation_age_days_at_audit"] = observation_age_days
        coverage["retrieval_age_hours_at_audit"] = retrieval_age_hours

        quality = {
            "missing_min_price_rows": count_rows(missing_min),
            "missing_max_price_rows": count_rows(missing_max),
            "missing_modal_price_rows": count_rows(missing_modal),
            "rows_with_any_missing_price": count_rows(any_missing_price),
            "rows_with_any_non_positive_price": count_rows(any_non_positive_price),
            "non_positive_min_price_rows": count_rows(non_positive_min),
            "non_positive_max_price_rows": count_rows(non_positive_max),
            "non_positive_modal_price_rows": count_rows(non_positive_modal),
            "min_price_exceeds_max_price_rows": count_rows(min_exceeds_max),
            "min_price_exceeds_modal_price_rows": count_rows(min_exceeds_modal),
            "modal_price_exceeds_max_price_rows": count_rows(modal_exceeds_max),
            "rows_with_any_price_order_violation": count_rows(
                any_price_order_violation
            ),
            "missing_price_unit_rows": count_rows(missing_price_unit),
            "missing_arrival_quantity_rows": count_rows(
                table.c.arrival_quantity.is_(None)
            ),
            "negative_arrival_quantity_rows": count_rows(negative_arrivals),
            "missing_source_record_id_rows": count_rows(missing_source_record_id),
            "missing_payload_hash_rows": count_rows(missing_payload_hash),
        }

        def grouped_distribution(column: Any) -> list[dict[str, Any]]:
            statement = apply_filter(
                select(
                    column.label("value"),
                    func.count().label("row_count"),
                )
                .select_from(table)
                .group_by(column)
                .order_by(func.count().desc(), column)
            )
            with self.engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
            return [
                {
                    "value": row["value"],
                    "row_count": int(row["row_count"]),
                }
                for row in rows
            ]

        # Duplicate candidates share the same source/resource/date and market
        # dimensions but have different record keys. They are surfaced for
        # human review; no rows are collapsed or modified.
        duplicate_dimensions = [
            table.c.source,
            table.c.source_resource_id,
            table.c.observation_date,
            table.c.state,
            table.c.district,
            table.c.market,
            table.c.commodity,
            table.c.variety,
            table.c.grade,
        ]
        duplicate_groups_query = apply_filter(
            select(
                *duplicate_dimensions,
                func.count().label("row_count"),
            )
            .select_from(table)
            .group_by(*duplicate_dimensions)
            .having(func.count() > 1)
        )
        duplicate_groups = duplicate_groups_query.subquery("duplicate_groups")
        duplicate_group_count_query = select(func.count()).select_from(
            duplicate_groups
        )
        duplicate_row_count_query = select(
            func.coalesce(func.sum(duplicate_groups.c.row_count), 0)
        )
        duplicate_examples_query = (
            select(duplicate_groups)
            .order_by(
                duplicate_groups.c.row_count.desc(),
                duplicate_groups.c.observation_date.desc(),
            )
            .limit(max_gap_examples)
        )
        with self.engine.connect() as connection:
            duplicate_group_count = int(
                connection.execute(duplicate_group_count_query).scalar_one()
            )
            duplicate_rows = int(
                connection.execute(duplicate_row_count_query).scalar_one()
            )
            duplicate_examples = [
                _serialized_mapping(row)
                for row in connection.execute(
                    duplicate_examples_query
                ).mappings().all()
            ]

        # A series is a source/resource + location + commodity + variety/grade
        # combination. Distinct observation dates are streamed so the audit
        # does not retain every observation in memory.
        series_columns = [
            table.c.source,
            table.c.source_resource_id,
            table.c.state,
            table.c.district,
            table.c.market,
            table.c.commodity,
            table.c.variety,
            table.c.grade,
        ]
        dates_query = apply_filter(
            select(*series_columns, table.c.observation_date)
            .select_from(table)
            .distinct()
            .order_by(*series_columns, table.c.observation_date)
            .execution_options(stream_results=True)
        )

        series_count = 0
        gap_interval_count = 0
        series_with_gaps = 0
        missing_calendar_days = 0
        max_missing_calendar_days = 0
        gap_examples: list[dict[str, Any]] = []
        current_key: tuple[Any, ...] | None = None
        previous_date: date | None = None
        current_series_has_gap = False

        with self.engine.connect() as connection:
            result = connection.execute(dates_query)
            for row in result.mappings():
                key = tuple(row[column.name] for column in series_columns)
                observed_date = row["observation_date"]

                if key != current_key:
                    if current_key is not None and current_series_has_gap:
                        series_with_gaps += 1
                    current_key = key
                    previous_date = None
                    current_series_has_gap = False
                    series_count += 1

                if previous_date is not None:
                    absent_days = (observed_date - previous_date).days - 1
                    if absent_days > 0:
                        gap_interval_count += 1
                        series_has_gap = True
                        current_series_has_gap = True
                        missing_calendar_days += absent_days
                        max_missing_calendar_days = max(
                            max_missing_calendar_days, absent_days
                        )

                        if max_gap_examples:
                            example = {
                                column.name: _serialize(row[column.name])
                                for column in series_columns
                            }
                            example.update(
                                {
                                    "previous_observation_date": previous_date.isoformat(),
                                    "next_observation_date": observed_date.isoformat(),
                                    "missing_calendar_days": absent_days,
                                }
                            )
                            gap_examples.append(example)
                            gap_examples.sort(
                                key=lambda item: item["missing_calendar_days"],
                                reverse=True,
                            )
                            del gap_examples[max_gap_examples:]

                previous_date = observed_date

            if current_key is not None and current_series_has_gap:
                series_with_gaps += 1

        return {
            "status": "ok",
            "audit": "national_market_price_quality",
            "read_only": True,
            "generated_at": audit_time.isoformat(),
            "filters": normalized_filters,
            "coverage": coverage,
            "quality": quality,
            "distributions": {
                "price_units": grouped_distribution(table.c.price_unit),
                "sources": grouped_distribution(table.c.source),
            },
            "duplicates": {
                "candidate_group_count": duplicate_group_count,
                "records_in_candidate_groups": duplicate_rows,
                "excess_records_over_one_per_group": max(
                    0, duplicate_rows - duplicate_group_count
                ),
                "examples": duplicate_examples,
                "definition": (
                    "Same source, resource, observation date, state, district, "
                    "market, commodity, variety and grade."
                ),
            },
            "temporal_coverage": {
                "series_count": series_count,
                "calendar_gap_interval_count": gap_interval_count,
                "series_with_calendar_gaps": series_with_gaps,
                "missing_calendar_days_between_observations": missing_calendar_days,
                "largest_calendar_gap_days": max_missing_calendar_days,
                "largest_gap_examples": gap_examples,
                "interpretation": (
                    "Calendar gaps describe elapsed days between stored "
                    "observations only. They do not prove that a mandi was "
                    "expected to report on every intervening day."
                ),
            },
            "notes": [
                "Counts describe records stored by CropBalance, not national source completeness.",
                "Missing values are reported separately from invalid values.",
                "Arrival quantity is optional when the source does not provide it.",
                "Duplicate candidates and calendar gaps are reported only; no data is changed.",
            ],
        }
