"""Helpers for a provenance-preserving local AGMARKNET archive.

This module is deliberately separate from the production daily fetcher. The
weekly endpoint's price measure remains unverified until its response contract
has been independently established.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from decimal import Decimal, InvalidOperation
import json

import pandas as pd


OBSERVATION_COLUMNS = [
    "observation_date",
    "state",
    "state_id",
    "district",
    "district_id",
    "market",
    "market_id",
    "crop",
    "commodity_name",
    "commodity_id",
    "variety",
    "variety_id",
    "reported_price_raw",
    "reported_price",
    "price_metric",
    "price_semantics_verified",
    "price_unit",
    "arrivals_value",
    "arrivals_unit",
    "price_status",
    "data_source",
    "endpoint",
    "fetched_at_utc",
    "raw_response_file",
    "provenance_status",
    "conflict_flag",
]

RUN_COLUMNS = [
    "fetched_at_utc",
    "state",
    "district",
    "market",
    "market_id",
    "crop",
    "commodity_id",
    "endpoint",
    "status",
    "records_seen",
    "valid_price_count",
    "non_valid_price_count",
    "error",
    "raw_response_file",
]

# Variety is intentionally part of the key. Unknown must not be silently
# mapped to a named variety. Price unit is a measurement attribute, not identity.
BASE_OBSERVATION_KEY = [
    "observation_date",
    "market_id",
    "commodity_id",
    "variety",
    "data_source",
    "endpoint",
]


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_agmarknet_date(value: Any) -> pd.Timestamp:
    """Parse common date-key formats without assuming US month-first dates."""
    if value is None:
        return pd.NaT

    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass

    if isinstance(value, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(value)
        return ts.tz_localize(None).normalize() if ts.tzinfo else ts.normalize()

    text = str(value).strip()
    if not text:
        return pd.NaT

    # ISO dates are unambiguous and must be handled before day-first formats.
    try:
        ts = pd.Timestamp(datetime.fromisoformat(text.replace("Z", "+00:00")))
        return ts.tz_localize(None).normalize() if ts.tzinfo else ts.normalize()
    except (ValueError, TypeError):
        pass

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%d %b %Y", "%d.%m.%Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt)).normalize()
        except ValueError:
            continue

    # Last-resort parser is explicitly day-first for Indian date conventions.
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return pd.NaT if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def classify_reported_price(raw_value: Any) -> tuple[float | None, str]:
    """Return numeric price and quality status; never turn NR into zero."""
    if raw_value is None:
        return None, "missing_value"

    try:
        if pd.isna(raw_value):
            return None, "missing_value"
    except (TypeError, ValueError):
        pass

    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return None, "missing_value"
        if value.upper() in {"NR", "N.R.", "N/A", "NA", "NULL", "NONE", "-", "--"}:
            return None, "not_reported"

    numeric = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(numeric):
        return None, "non_numeric"

    numeric = float(numeric)
    if numeric <= 0:
        return numeric, "non_positive"

    return numeric, "reported_unverified_metric"


def _is_missing(value: Any) -> bool:
    """Scalar-safe missing-value check."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _normalize_identity_value(column: str, value: Any) -> str | None:
    """Normalize identity values so CSV strings and numeric values match.

    Numeric market and commodity identifiers such as 1655, "1655" and
    "1655.0" become the same string. Text identity fields retain their case
    and spelling; in particular, Unknown is never mapped to a named variety.
    """
    if _is_missing(value):
        return None

    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>", "null"}:
        return None

    if column in {"market_id", "commodity_id"}:
        try:
            numeric_value = Decimal(text)
            if numeric_value.is_finite() and numeric_value == numeric_value.to_integral_value():
                return str(int(numeric_value))
        except (InvalidOperation, ValueError):
            pass

    return text


def make_observation(
    *,
    date_key: Any,
    record: dict[str, Any],
    state: str,
    state_id: int,
    district: str,
    district_id: int,
    market: str,
    market_id: int,
    crop: str,
    commodity_name: str,
    commodity_id: int,
    fetched_at_utc: str,
    endpoint: str,
    raw_response_file: str | None,
) -> dict[str, Any]:
    """Convert one AGMARKNET date/value pair into a canonical observation."""
    observation_date = parse_agmarknet_date(date_key)
    if pd.isna(observation_date):
        raise ValueError(f"Unparseable AGMARKNET date key: {date_key!r}")

    raw_value = record.get(str(date_key))
    price, status = classify_reported_price(raw_value)
    unit = record.get("unitOfPrice")
    variety = record.get("variety")

    return {
        "observation_date": observation_date.date().isoformat(),
        "state": state,
        "state_id": state_id,
        "district": district,
        "district_id": district_id,
        "market": market,
        "market_id": market_id,
        "crop": crop,
        "commodity_name": commodity_name,
        "commodity_id": commodity_id,
        "variety": "Unknown" if _is_missing(variety) or not str(variety).strip() else str(variety).strip(),
        # This weekly response shape does not expose a separate variety ID.
        "variety_id": None,
        "reported_price_raw": None if _is_missing(raw_value) else str(raw_value),
        "reported_price": price,
        "price_metric": "unverified",
        "price_semantics_verified": False,
        "price_unit": None if _is_missing(unit) else str(unit),
        # This endpoint/script does not collect arrivals.
        "arrivals_value": None,
        "arrivals_unit": None,
        "price_status": status,
        "data_source": "agmarknet",
        "endpoint": endpoint,
        "fetched_at_utc": fetched_at_utc,
        "raw_response_file": raw_response_file,
        "provenance_status": (
            "raw_response_archived" if raw_response_file else "raw_response_unavailable"
        ),
        "conflict_flag": False,
    }


def normalize_legacy_archive(legacy: pd.DataFrame) -> pd.DataFrame:
    """Map the original dharashiv_prices.csv schema into the v3 contract.

    Old values remain reported values with unverified measure semantics.
    No modal-price label or arrivals value is inferred.
    """
    if legacy is None or legacy.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    source = legacy.copy()
    rows: list[dict[str, Any]] = []

    for _, record in source.iterrows():
        date_value = record.get("date", record.get("observation_date"))
        parsed_date = parse_agmarknet_date(date_value)
        if pd.isna(parsed_date):
            # Malformed legacy rows are not assigned fabricated dates.
            continue

        raw_value = record.get("reported_price")
        price, status = classify_reported_price(raw_value)
        fetched_at = record.get("fetched_at", record.get("fetched_at_utc"))

        def value_or_default(column: str, default: Any) -> Any:
            value = record.get(column)
            return default if _is_missing(value) or str(value).strip() == "" else value

        rows.append({
            "observation_date": parsed_date.date().isoformat(),
            "state": value_or_default("state", "Maharashtra"),
            "state_id": value_or_default("state_id", 20),
            "district": value_or_default("district", "Dharashiv"),
            "district_id": value_or_default("district_id", 347),
            "market": record.get("market"),
            "market_id": record.get("market_id"),
            "crop": record.get("crop"),
            "commodity_name": record.get("commodity_name"),
            "commodity_id": record.get("commodity_id"),
            "variety": value_or_default("variety", "Unknown"),
            "variety_id": record.get("variety_id"),
            "reported_price_raw": None if _is_missing(raw_value) else str(raw_value),
            "reported_price": price,
            "price_metric": "unverified",
            "price_semantics_verified": False,
            "price_unit": record.get("price_unit"),
            "arrivals_value": None,
            "arrivals_unit": None,
            "price_status": status,
            "data_source": "agmarknet",
            "endpoint": "/prices-and-arrivals/commodity-price/lastweek",
            "fetched_at_utc": fetched_at,
            "raw_response_file": None,
            "provenance_status": "legacy_import_raw_unavailable",
            "conflict_flag": False,
        })

    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def _normalized_measurement(row: pd.Series) -> tuple[Any, ...]:
    """Normalize a reported measurement for semantic duplicate detection.

    Numeric 3000 and 3000.0 are equivalent. The price unit remains part of the
    measurement signature so a unit change is retained as a conflict.
    """
    raw = row.get("reported_price_raw")
    status_value = row.get("price_status")
    status = "" if _is_missing(status_value) else str(status_value).strip().lower()

    price, classified_status = classify_reported_price(raw)

    unit_value = row.get("price_unit")
    unit = None if _is_missing(unit_value) else str(unit_value).strip().lower()
    unit = unit or None

    if price is not None and classified_status == "reported_unverified_metric":
        value = ("numeric", float(price))
    else:
        raw_text = "" if _is_missing(raw) else str(raw).strip().upper()
        value = ("non_numeric", status or classified_status, raw_text)

    return (*value, unit)


def merge_observations(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse identical observations and preserve genuine conflicts.

    Identity fields are normalized before grouping so values loaded from CSV
    as strings match newly collected numeric identifiers. Variety remains in
    the identity key to avoid silently assigning Unknown to a named variety.
    """
    frames = [
        frame.copy()
        for frame in (existing, incoming)
        if frame is not None and not frame.empty
    ]

    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    for column in OBSERVATION_COLUMNS:
        if column not in combined.columns:
            combined[column] = None

    # Infer provenance only when a status has not already been assigned.
    missing_provenance = combined["provenance_status"].isna()
    no_raw_reference = combined["raw_response_file"].isna()

    combined.loc[
        missing_provenance & no_raw_reference,
        "provenance_status",
    ] = "legacy_import_raw_unavailable"
    combined.loc[
        missing_provenance & ~no_raw_reference,
        "provenance_status",
    ] = "raw_response_archived"

    def normalize_date(value: Any) -> str | None:
        parsed = parse_agmarknet_date(value)
        if pd.isna(parsed):
            return None
        return parsed.date().isoformat()

    combined["observation_date"] = combined["observation_date"].map(normalize_date)

    identity_columns = BASE_OBSERVATION_KEY
    for column in identity_columns:
        combined[column] = combined[column].map(
            lambda value, field=column: _normalize_identity_value(field, value)
        )

    combined["_identity_complete"] = combined[identity_columns].notna().all(axis=1)
    combined["_measurement"] = combined.apply(_normalized_measurement, axis=1)

    # ISO-8601 timestamps sort lexicographically when consistently formatted.
    # The latest identical measurement retains its latest raw snapshot reference.
    combined = combined.sort_values("fetched_at_utc", na_position="first")

    complete = combined[combined["_identity_complete"]].copy()
    incomplete = combined[~combined["_identity_complete"]].copy()

    if not complete.empty:
        complete = complete.drop_duplicates(
            subset=identity_columns + ["_measurement"],
            keep="last",
        )

        conflict_counts = complete.groupby(
            identity_columns,
            dropna=False,
        )["_measurement"].transform("nunique")
        complete["conflict_flag"] = conflict_counts.gt(1)

    if not incomplete.empty:
        # Never collapse rows whose identity cannot be established safely.
        incomplete["conflict_flag"] = False

    result = pd.concat([complete, incomplete], ignore_index=True, sort=False)
    result = result.reindex(columns=OBSERVATION_COLUMNS)

    return result.sort_values(
        ["crop", "market", "observation_date", "variety", "fetched_at_utc"],
        na_position="last",
    ).reset_index(drop=True)


def append_run_log(
    existing: pd.DataFrame,
    incoming_rows: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """Append collection-run records without altering prior run history."""
    frames: list[pd.DataFrame] = []

    if existing is not None and not existing.empty:
        frames.append(existing.copy())

    new_df = pd.DataFrame(list(incoming_rows))
    if not new_df.empty:
        frames.append(new_df)

    if not frames:
        return pd.DataFrame(columns=RUN_COLUMNS)

    result = pd.concat(frames, ignore_index=True, sort=False)
    for column in RUN_COLUMNS:
        if column not in result.columns:
            result[column] = None

    return result.reindex(columns=RUN_COLUMNS)


def write_json_snapshot(payload: dict[str, Any], path: Path) -> None:
    """Write an immutable raw-response snapshot as formatted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
