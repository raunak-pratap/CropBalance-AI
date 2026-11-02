from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import requests
from loguru import logger


BASE_URL = "https://api.agmarknet.gov.in/v1"

STATE = "Maharashtra"
STATE_ID = 20
DISTRICT = "Dharashiv"

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

OUTPUT = Path("data/raw/agmarknet_weekly")
OUTPUT_FILE = OUTPUT / "dharashiv_prices.csv"
 

def create_session():
    session = requests.Session()

    session.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://agmarknet.gov.in",
        "Referer": "https://agmarknet.gov.in/",
        "User-Agent": "CropBalance-AI/1.0",
    })

    return session


def fetch_market_prices(
    session,
    crop,
    commodity_id,
    market,
    market_id,
):
    url = (
        f"{BASE_URL}/prices-and-arrivals/"
        "commodity-price/lastweek"
    )

    params = {
        "marketId": market_id,
        "stateId": STATE_ID,
        "commodityId": commodity_id,
        "includeExcel": "false",
    }

    response = session.get(
        url,
        params=params,
        timeout=25,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get("success"):
        logger.warning(
            f"{market}/{crop}: {payload.get('message')}"
        )
        return []

    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for record in payload.get("data", []):

        variety = record.get("variety", "Unknown")
        unit = record.get("unitOfPrice")

        for column in payload.get("columns", []):

            date_key = column.get("key")

            if date_key in ("variety", "unitOfPrice"):
                continue

            # Ignore missing and NR observations.
            price = pd.to_numeric(
                record.get(date_key),
                errors="coerce",
            )

            if pd.isna(price) or price <= 0:
                continue

            rows.append({
                "date": date_key,
                "state": STATE,
                "district": DISTRICT,
                "market": market,
                "market_id": market_id,
                "crop": crop,
                "commodity_name": CROPS[crop]["name"],
                "commodity_id": commodity_id,
                "variety": variety,
                "reported_price": float(price),
                "price_unit": unit,
                "data_source": "agmarknet_weekly",
                "fetched_at": fetched_at,
            })

    return rows


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    session = create_session()
    collected = []

    total_requests = len(MARKETS) * len(CROPS)

    logger.info(
        f"Starting weekly collection: {total_requests} requests"
    )

    for crop, crop_info in CROPS.items():

        logger.info(f"Collecting crop: {crop}")

        for market, market_id in MARKETS.items():

            try:
                rows = fetch_market_prices(
                    session,
                    crop,
                    crop_info["id"],
                    market,
                    market_id,
                )

                collected.extend(rows)

                logger.info(
                    f"{market} / {crop}: "
                    f"{len(rows)} observations"
                )

            except requests.RequestException as error:
                logger.warning(
                    f"{market} / {crop} failed: {error}"
                )

            time.sleep(0.3)

    if not collected:
        logger.warning("No new real observations collected.")
        return

    new_df = pd.DataFrame(collected)

    if OUTPUT_FILE.exists():
        old_df = pd.read_csv(OUTPUT_FILE)
        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )
    else:
        combined = new_df

    # Keep the most recently collected version of each observation.
    unique_keys = [
        "date",
        "market_id",
        "commodity_id",
        "variety",
    ]

    combined = (
        combined
        .drop_duplicates(
            subset=unique_keys,
            keep="last",
        )
        .sort_values(
            ["crop", "market", "date"]
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n--- COLLECTION SUMMARY ---")
    print("New observations:", len(new_df))
    print("Archived observations:", len(combined))
    print("Saved to:", OUTPUT_FILE)

    print("\nObservations by crop:")
    print(
        combined.groupby("crop").size()
    )


if __name__ == "__main__":
    main()