"""Helpers for a provenance-preserving local AGMARKNET archive.

This module is deliberately separate from the production daily fetcher.
The weekly endpoint's price measure is kept unverified until its response
contract has been independently established.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json

import pandas as pd

OBSERVATION_COLUMNS = [
    "observation_date", "state", "state_id", "district", "district_id",
    "market", "market_id", "crop", "commodity_name", "commodity_id",
    "variety", "variety_id", "reported_price_raw", "reported_price",
    "price_metric", "price_semantics_verified", "price_unit",
    "arrivals_value", "arrivals_unit", "price_status", "data_source",
    "endpoint", "fetched_at_utc", "raw_response_file",
    "provenance_status", "conflict_flag",
]

RUN_COLUMNS = [
    "fetched_at_utc", "state", "district", "market", "market_id",
    "crop", "commodity_id", "endpoint", "status", "records_seen",
    "valid_price_count", "non_valid_price_count", "error",
    "raw_response_file",
]

BASE_OBSERVATION_KEY = [
    "observation_date",
    "market_id",
    "commodity_id",
    "variety",
    "data_source",
    "endpoint",
]

def utc_now_iso() -> str:
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
    # ISO dates are unambiguous and should be handled before day-first formats.
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


def make_observation(
    *, date_key: Any, record: dict[str, Any], state: str, state_id: int,
    district: str, district_id: int, market: str, market_id: int,
    crop: str, commodity_name: str, commodity_id: int, fetched_at_utc: str,
    endpoint: str, raw_response_file: str | None,
) -> dict[str, Any]:
    observation_date = parse_agmarknet_date(date_key)
    if pd.isna(observation_date):
        raise ValueError(f"Unparseable AGMARKNET date key: {date_key!r}")
    raw_value = record.get(str(date_key))
    price, status = classify_reported_price(raw_value)
    unit = record.get("unitOfPrice")
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
        "variety": record.get("variety") or "Unknown",
        "variety_id": None,  # Not returned by this weekly response shape.
        "reported_price_raw": None if raw_value is None or pd.isna(raw_value) else str(raw_value),
        "reported_price": price,
        "price_metric": "unverified",
        "price_semantics_verified": False,
        "price_unit": None if unit is None else str(unit),
        "arrivals_value": None,  # This endpoint/script does not collect arrivals.
        "arrivals_unit": None,
        "price_status": status,
        "data_source": "agmarknet",
        "endpoint": endpoint,
        "fetched_at_utc": fetched_at_utc,
        "raw_response_file": raw_response_file,
        "provenance_status": (
            "raw_response_archived"
            if raw_response_file
            else "raw_response_unavailable"
        ),
        "conflict_flag": False,
    }



def normalize_legacy_archive(legacy: pd.DataFrame) -> pd.DataFrame:
    """Map the original dharashiv_prices.csv schema into the v2 contract.

    Old values remain reported values with unverified measure semantics.
    No modal-price label or arrivals value is inferred.
    """
    if legacy is None or legacy.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    source = legacy.copy()
    rows = []
    for _, record in source.iterrows():
        date_value = record.get("date", record.get("observation_date"))
        parsed_date = parse_agmarknet_date(date_value)
        if pd.isna(parsed_date):
            # Preserve malformed legacy rows in the audit source CSV; do not
            # create an observation with a fabricated date.
            continue
        raw_value = record.get("reported_price")
        price, status = classify_reported_price(raw_value)
        fetched_at = record.get("fetched_at", record.get("fetched_at_utc"))
        rows.append({
            "observation_date": parsed_date.date().isoformat(),
            "state": record.get("state") or "Maharashtra",
            "state_id": record.get("state_id") or 20,
            "district": record.get("district") or "Dharashiv",
            "district_id": record.get("district_id") or 347,
            "market": record.get("market"),
            "market_id": record.get("market_id"),
            "crop": record.get("crop"),
            "commodity_name": record.get("commodity_name"),
            "commodity_id": record.get("commodity_id"),
            "variety": record.get("variety") or "Unknown",
            "variety_id": record.get("variety_id"),
            "reported_price_raw": None if raw_value is None or pd.isna(raw_value) else str(raw_value),
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



def _normalized_measurement(row: pd.Series) -> tuple:
    """Normalize a reported measurement for semantic duplicate detection."""
    raw = row.get("reported_price_raw")
    status = str(row.get("price_status") or "").strip().lower()

    price, classified_status = classify_reported_price(raw)

    unit = row.get("price_unit")
    unit = None if pd.isna(unit) else str(unit).strip().lower()
    unit = unit or None

    if price is not None and classified_status == "reported_unverified_metric":
        value = ("numeric", float(price))
    else:
        raw_text = "" if pd.isna(raw) else str(raw).strip().upper()
        value = ("non_numeric", status or classified_status, raw_text)

    # Unit is measurement metadata, not observation identity.
    return (*value, unit)


def merge_observations(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse semantic repeats and preserve genuinely conflicting reports."""
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
    missing_provenance = combined["provenance_status"].isna()

    combined.loc[
        missing_provenance & combined["raw_response_file"].isna(),
        "provenance_status",
    ] = "legacy_import_raw_unavailable"

    combined.loc[
        missing_provenance & combined["raw_response_file"].notna(),
        "provenance_status",
    ] = "raw_response_archived"

    combined["observation_date"] = combined["observation_date"].map(
        lambda x: (
            parse_agmarknet_date(x).date().isoformat()
            if not pd.isna(parse_agmarknet_date(x))
            else None
        )
    )

    # Keep records with incomplete identity fields separate from valid groups.
    identity_columns = BASE_OBSERVATION_KEY

    combined["_identity_complete"] = combined[identity_columns].notna().all(axis=1)

    # A semantic signature treats 3000 and 3000.0 as the same price.
    combined["_measurement"] = combined.apply(
        _normalized_measurement, axis=1
    )

    # Collapse repeated measurements, keeping the latest fetched record.
    combined = combined.sort_values(
        "fetched_at_utc", na_position="first"
    )

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
        # Do not merge records whose identity cannot be established safely.
        incomplete["conflict_flag"] = False

    result = pd.concat(
        [complete, incomplete],
        ignore_index=True,
        sort=False,
    )

    result = result.reindex(columns=OBSERVATION_COLUMNS)

    return result.sort_values(
        ["crop", "market", "observation_date", "variety", "fetched_at_utc"],
        na_position="last",
    ).reset_index(drop=True)



def append_run_log(existing: pd.DataFrame, incoming_rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing.copy())
    new_df = pd.DataFrame(list(incoming_rows))
    if not new_df.empty:
        frames.append(new_df)
    if not frames:
        return pd.DataFrame(columns=RUN_COLUMNS)
    result = pd.concat(frames, ignore_index=True, sort=False)
    for col in RUN_COLUMNS:
        if col not in result.columns:
            result[col] = None
    return result.reindex(columns=RUN_COLUMNS)


def write_json_snapshot(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
