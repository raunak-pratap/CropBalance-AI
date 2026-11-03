"""Incrementally archive real weekly AGMARKNET price-report observations.

This is a new local-market research collector. It does not alter fetcher.py or
feed the production daily LSTM. Price semantics remain marked unverified.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time
import sys

import pandas as pd
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.agmarknet_local_archive import (  # noqa: E402
    BASE_OBSERVATION_KEY,
    OBSERVATION_COLUMNS,
    RUN_COLUMNS,
    append_run_log,
    make_observation,
    merge_observations,
    normalize_legacy_archive,
    utc_now_iso,
    write_json_snapshot,
)

BASE_URL = "https://api.agmarknet.gov.in/v1"
ENDPOINT_PATH = "/prices-and-arrivals/commodity-price/lastweek"
ENDPOINT = BASE_URL + ENDPOINT_PATH
STATE = "Maharashtra"
STATE_ID = 20
DISTRICT = "Dharashiv"
DISTRICT_ID = 347

MARKETS = {
    "APMC Bhoom": 1959,
    "APMC Dharashiv": 1655,
    "APMC Kalamb": 1599,
    "APMC Murum": 1600,
    "APMC Paranda": 1960,
    "APMC Tuljapur": 1961,
    "APMC Umarga": 1962,
    "APMC Washi": 2053,
}
CROPS = {
    "wheat": {"id": 1, "name": "Wheat"},
    "onion": {"id": 23, "name": "Onion"},
    "soybean": {"id": 13, "name": "Soyabean"},
    "tomato": {"id": 65, "name": "Tomato"},
}
OUTPUT_DIR = Path("data/raw/agmarknet_weekly")
OBSERVATIONS_FILE = OUTPUT_DIR / "dharashiv_prices_v2.csv"
RUNS_FILE = OUTPUT_DIR / "collection_runs.csv"
RAW_DIR = OUTPUT_DIR / "responses"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://agmarknet.gov.in",
        "Referer": "https://agmarknet.gov.in/",
        "User-Agent": "CropBalance-AI/1.0",
    })
    return session


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def extract_date_keys(columns: list[dict]) -> list[str]:
    keys = []
    for column in columns:
        if not isinstance(column, dict):
            continue
        key = column.get("key")
        if key is None or str(key) in {"variety", "unitOfPrice"}:
            continue
        keys.append(str(key))
    return keys


def collect_one(session: requests.Session, crop: str, market: str, market_id: int,
                fetched_at: str) -> tuple[list[dict], dict]:
    crop_info = CROPS[crop]
    params = {
        "marketId": market_id,
        "stateId": STATE_ID,
        "commodityId": crop_info["id"],
        "includeExcel": "false",
    }
    response = session.get(ENDPOINT, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("AGMARKNET response is not a JSON object")

    request_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_path = RAW_DIR / f"{request_id}_{safe_name(market)}_{crop}.json"
    write_json_snapshot(payload, raw_path)

    if not payload.get("success"):
        message = str(payload.get("message") or "AGMARKNET success=false")
        return [], {
            "status": "api_unsuccessful", "records_seen": 0,
            "valid_price_count": 0, "non_valid_price_count": 0,
            "error": message, "raw_response_file": str(raw_path),
        }

    records = payload.get("data", [])
    columns = payload.get("columns", [])
    if not isinstance(records, list) or not isinstance(columns, list):
        raise ValueError("AGMARKNET data/columns fields have unexpected types")
    date_keys = extract_date_keys(columns)
    if not date_keys:
        raise ValueError("No date columns found in AGMARKNET weekly response")

    observations = []
    valid_count = 0
    non_valid_count = 0
    for record in records:
        if not isinstance(record, dict):
            non_valid_count += len(date_keys)
            continue
        for date_key in date_keys:
            row = make_observation(
                date_key=date_key,
                record=record,
                state=STATE,
                state_id=STATE_ID,
                district=DISTRICT,
                district_id=DISTRICT_ID,
                market=market,
                market_id=market_id,
                crop=crop,
                commodity_name=crop_info["name"],
                commodity_id=crop_info["id"],
                fetched_at_utc=fetched_at,
                endpoint=ENDPOINT_PATH,
                raw_response_file=str(raw_path),
            )
            if row["price_status"] == "reported_unverified_metric":
                valid_count += 1
            else:
                non_valid_count += 1
            observations.append(row)

    status = "success_with_observations" if observations else "success_empty"
    return observations, {
        "status": status,
        "records_seen": len(records) * len(date_keys),
        "valid_price_count": valid_count,
        "non_valid_price_count": non_valid_count,
        "error": None,
        "raw_response_file": str(raw_path),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = create_session()
    fetched_at = utc_now_iso()
    new_rows: list[dict] = []
    run_rows: list[dict] = []

    total = len(CROPS) * len(MARKETS)
    logger.info(f"Starting Dharashiv AGMARKNET weekly archive: {total} requests")
    for crop, crop_info in CROPS.items():
        for market, market_id in MARKETS.items():
            result = {
                "fetched_at_utc": fetched_at,
                "state": STATE,
                "district": DISTRICT,
                "market": market,
                "market_id": market_id,
                "crop": crop,
                "commodity_id": crop_info["id"],
                "endpoint": ENDPOINT_PATH,
            }
            try:
                rows, status = collect_one(session, crop, market, market_id, fetched_at)
                new_rows.extend(rows)
                result.update(status)
                logger.info(f"{market}/{crop}: {status['valid_price_count']} valid, "
                            f"{status['non_valid_price_count']} non-valid")
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                result.update({
                    "status": "request_or_parse_failed",
                    "records_seen": 0,
                    "valid_price_count": 0,
                    "non_valid_price_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_response_file": None,
                })
                logger.warning(f"{market}/{crop} failed: {exc}")
            run_rows.append(result)
            time.sleep(0.3)

    existing_frames = []
    if OBSERVATIONS_FILE.exists():
        existing_frames.append(pd.read_csv(OBSERVATIONS_FILE, dtype=str))
    legacy_file = OUTPUT_DIR / "dharashiv_prices.csv"
    if legacy_file.exists():
        legacy = pd.read_csv(legacy_file, dtype=str)
        existing_frames.append(normalize_legacy_archive(legacy))
    existing = (pd.concat(existing_frames, ignore_index=True, sort=False)
                if existing_frames else pd.DataFrame(columns=OBSERVATION_COLUMNS))
    incoming = pd.DataFrame(new_rows, columns=OBSERVATION_COLUMNS)
    merged = merge_observations(existing, incoming)
    merged.to_csv(OBSERVATIONS_FILE, index=False)

    old_runs = pd.read_csv(RUNS_FILE, dtype=str) if RUNS_FILE.exists() else pd.DataFrame(columns=RUN_COLUMNS)
    updated_runs = append_run_log(old_runs, run_rows)
    updated_runs.to_csv(RUNS_FILE, index=False)

    print("\n--- DHARASHIV COLLECTION SUMMARY ---")
    print("Requests attempted:", len(run_rows))
    print("New response rows:", len(new_rows))
    print("Archived rows (including missing statuses/conflict versions):", len(merged))
    if not merged.empty:
        print("Valid reported values:", int(merged["price_status"].eq("reported_unverified_metric").sum()))
        print("Non-valid/missing values:", int((~merged["price_status"].eq("reported_unverified_metric")).sum()))
        print("Conflicting observation groups:", int(merged.loc[merged["conflict_flag"].astype(str).str.lower().eq("true"), BASE_OBSERVATION_KEY].drop_duplicates().shape[0]))
    if legacy_file.exists():
        print("Legacy archive preserved/imported:", legacy_file)
    print("Observation archive:", OBSERVATIONS_FILE)
    print("Collection log:", RUNS_FILE)
    print("Raw response snapshots:", RAW_DIR)
    print("Price semantics: UNVERIFIED; do not label as modal until confirmed.")
    print("Arrivals: not supplied by this endpoint; retained as missing.")
    return 0 if any(row["status"] == "success_with_observations" for row in run_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
