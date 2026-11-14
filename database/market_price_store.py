"""PostgreSQL storage for record-level mandi price observations."""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    and_,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine


metadata = MetaData()

market_price_observations = Table(
    "market_price_observations",
    metadata,
    Column("record_key", String(64), primary_key=True),
    Column("source", String(40), nullable=False, index=True),
    Column("source_resource_id", String(100), nullable=False),
    Column("source_record_id", String(160), nullable=True),
    Column("observation_date", Date, nullable=False, index=True),
    Column("state", String(120), nullable=False),
    Column("district", String(160), nullable=False),
    Column("market", String(200), nullable=False),
    Column("commodity", String(160), nullable=False),
    Column("variety", String(200), nullable=True),
    Column("grade", String(160), nullable=True),
    Column("source_state_id", String(40), nullable=True),
    Column("source_district_id", String(40), nullable=True),
    Column("source_market_id", String(40), nullable=True),
    Column("source_commodity_id", String(40), nullable=True),
    Column("source_variety_id", String(40), nullable=True),
    Column("min_price", Numeric(18, 4), nullable=True),
    Column("max_price", Numeric(18, 4), nullable=True),
    Column("modal_price", Numeric(18, 4), nullable=True),
    Column("price_unit", String(80), nullable=True),
    Column("arrival_quantity", Numeric(20, 4), nullable=True),
    Column("arrival_unit", String(80), nullable=True),
    Column("arrivals_available", Boolean, nullable=False, default=False),
    Column("retrieved_at", DateTime(timezone=True), nullable=False),
    Column("source_updated_at", DateTime(timezone=True), nullable=True),
    Column("payload_hash", String(64), nullable=False),
    Column("raw_record", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index(
    "ix_market_price_geo_crop_date",
    market_price_observations.c.state,
    market_price_observations.c.district,
    market_price_observations.c.commodity,
    market_price_observations.c.observation_date,
)
Index(
    "ix_market_price_market_date",
    market_price_observations.c.market,
    market_price_observations.c.observation_date,
)


INSERT_COLUMNS = [
    column.name for column in market_price_observations.columns
]
UPDATE_COLUMNS = [
    name for name in INSERT_COLUMNS
    if name not in {"record_key", "created_at"}
]


def create_database_engine(database_url: str | None = None) -> Engine:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured.")
    return create_engine(url, pool_pre_ping=True, future=True)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _serialize_value(value) for key, value in row.items()}


class MarketPriceStore:
    """Idempotent storage and query facade for daily market-price facts."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def from_env(cls) -> "MarketPriceStore":
        return cls(create_database_engine())

    def ensure_schema(self) -> None:
        """Create only this feature's table/indexes; does not alter existing tables."""
        metadata.create_all(
            self.engine,
            tables=[market_price_observations],
            checkfirst=True,
        )

    @staticmethod
    def _db_record(record: dict[str, Any], now: datetime) -> dict[str, Any]:
        result = {key: record.get(key) for key in INSERT_COLUMNS}
        result["created_at"] = now
        result["updated_at"] = now
        result["arrivals_available"] = bool(record.get("arrivals_available"))
        return result

    def upsert_observations(
        self,
        records: Iterable[dict[str, Any]],
        *,
        batch_size: int = 500,
    ) -> dict[str, int]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        rows = list(records)
        if not rows:
            return {"received": 0, "upserted": 0}

        prepared = [
            self._db_record(record, datetime.now(timezone.utc))
            for record in rows
        ]
        upserted = 0

        with self.engine.begin() as connection:
            dialect = connection.dialect.name
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start:start + batch_size]
                if dialect == "postgresql":
                    from sqlalchemy.dialects.postgresql import insert as dialect_insert
                    statement = dialect_insert(market_price_observations).values(batch)
                    statement = statement.on_conflict_do_update(
                        index_elements=["record_key"],
                        set_={
                            name: getattr(statement.excluded, name)
                            for name in UPDATE_COLUMNS
                        },
                    )
                elif dialect == "sqlite":
                    from sqlalchemy.dialects.sqlite import insert as dialect_insert
                    statement = dialect_insert(market_price_observations).values(batch)
                    statement = statement.on_conflict_do_update(
                        index_elements=["record_key"],
                        set_={
                            name: getattr(statement.excluded, name)
                            for name in UPDATE_COLUMNS
                        },
                    )
                else:
                    for record in batch:
                        key = record["record_key"]
                        exists = connection.execute(
                            select(market_price_observations.c.record_key)
                            .where(market_price_observations.c.record_key == key)
                        ).first()
                        if exists:
                            connection.execute(
                                update(market_price_observations)
                                .where(market_price_observations.c.record_key == key)
                                .values(**{
                                    name: record[name] for name in UPDATE_COLUMNS
                                })
                            )
                        else:
                            connection.execute(insert(market_price_observations).values(**record))
                    upserted += len(batch)
                    continue

                connection.execute(statement)
                upserted += len(batch)

        return {"received": len(rows), "upserted": upserted}

    def query_observations(
        self,
        *,
        state: str | None = None,
        district: str | None = None,
        market: str | None = None,
        commodity: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("offset must be zero or greater.")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500.")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")

        filters = []
        for column_name, value in (
            ("state", state),
            ("district", district),
            ("market", market),
            ("commodity", commodity),
        ):
            if value:
                column = market_price_observations.c[column_name]
                filters.append(func.lower(column).like(f"%{value.strip().lower()}%"))
        if start_date:
            filters.append(
                market_price_observations.c.observation_date >= start_date
            )
        if end_date:
            filters.append(
                market_price_observations.c.observation_date <= end_date
            )

        where = and_(*filters) if filters else None
        count_query = select(func.count()).select_from(market_price_observations)
        data_query = select(market_price_observations)
        if where is not None:
            count_query = count_query.where(where)
            data_query = data_query.where(where)

        data_query = (
            data_query
            .order_by(
                market_price_observations.c.observation_date.desc(),
                market_price_observations.c.state,
                market_price_observations.c.district,
                market_price_observations.c.market,
            )
            .offset(offset)
            .limit(limit)
        )
        with self.engine.connect() as connection:
            total = connection.execute(count_query).scalar_one()
            rows = connection.execute(data_query).mappings().all()

        return {
            "status": "ok",
            "source": "data.gov.in",
            "is_live": False,
            "count": total,
            "offset": offset,
            "limit": limit,
            "items": [_serialize_row(dict(row)) for row in rows],
        }

    def coverage(self) -> dict[str, Any]:
        table = market_price_observations
        query = select(
            func.count().label("observation_count"),
            func.count(func.distinct(table.c.state)).label("state_count"),
            func.count(func.distinct(table.c.district)).label("district_count"),
            func.count(func.distinct(table.c.market)).label("market_count"),
            func.count(func.distinct(table.c.commodity)).label("commodity_count"),
            func.min(table.c.observation_date).label("first_observation_date"),
            func.max(table.c.observation_date).label("last_observation_date"),
            func.max(table.c.retrieved_at).label("last_retrieved_at"),
        )
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().one()
        values = _serialize_row(dict(row))
        return {
            "status": "ok",
            "source": "data.gov.in",
            "is_live": False,
            **values,
            "coverage_note": (
                "Counts describe records successfully stored by CropBalance, "
                "not the full number of markets or commodities in India."
            ),
        }

    def dispose(self) -> None:
        self.engine.dispose()
