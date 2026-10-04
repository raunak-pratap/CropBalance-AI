"""
data/fetcher.py
---------------
Fetches mandi price data from Indian government sources (Agmarknet, eNAM)
and weather data from OpenWeatherMap.

Real API keys are loaded from .env. When keys are missing, a realistic
synthetic dataset is generated so the rest of the pipeline still runs.
"""

import os
import time
import requests
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
from services.agmarknet_metadata import AgmarknetMetadataResolver

from config import SUPPORTED_CROPS, API_CONFIG, PATH_CONFIG


# ──────────────────────────────────────────────
# Mandi Price Fetcher
# ──────────────────────────────────────────────

class MandiPriceFetcher:
    """
    Fetches daily mandi (wholesale market) price data.

    Priority order:
      1. eNAM API  (if key available)
      2. Agmarknet (if key available)
      3. Synthetic data generator (fallback for dev/testing)
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SmartFarming/1.0"})

        self.metadata_resolver = AgmarknetMetadataResolver()

    def _get_with_retry(self, url, params, max_retries=3):
        """GET request with exponential backoff for rate limits."""

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=30,
                    headers={
                        "User-Agent": "CropBalance-AI/1.0"
                    },
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")

                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        wait_time = 2 ** attempt * 5

                    logger.warning(
                        f"AGMARKNET rate limited (429). "
                        f"Retry {attempt + 1}/{max_retries} "
                        f"after {wait_time}s"
                    )

                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue

                    raise requests.HTTPError(
                        "AGMARKNET rate limit persisted after retries",
                        response=response,
                    )

                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = 2 ** attempt * 2

                logger.warning(
                    f"AGMARKNET request failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

        raise RuntimeError("AGMARKNET request failed")

    def _get_cache_path(
        self,
        crop: str,
        state: str,
        year: int,
        month: int,
    ):
        """Return the local cache path for one AGMARKNET month."""

        safe_crop = crop.lower().replace(" ", "_")
        safe_state = state.lower().replace(" ", "_")

        filename = (
            f"{safe_crop}_{safe_state}_"
            f"{year}_{month:02d}.csv"
        )

        return Path(
            "data/raw/agmarknet"
        ) / filename


    def _save_agmarknet_cache(
        self,
        df: pd.DataFrame,
        crop: str,
        state: str,
        year: int,
        month: int,
    ):
        """Save successful AGMARKNET market-level data."""

        cache_path = self._get_cache_path(
            crop,
            state,
            year,
            month,
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            cache_path,
            index=False,
        )

        logger.info(
            f"AGMARKNET cache saved: {cache_path}"
        )


    def _load_agmarknet_cache(
        self,
        crop: str,
        state: str,
        year: int,
        month: int,
    ):
        """Load cached AGMARKNET market-level data."""

        cache_path = self._get_cache_path(
            crop,
            state,
            year,
            month,
        )

        if not cache_path.exists():
            return None

        try:
            df = pd.read_csv(
                cache_path,
                parse_dates=["date"],
            )

            logger.info(
                f"AGMARKNET cache hit: {cache_path}"
            )

            return df

        except Exception as e:
            logger.warning(
                f"Could not load AGMARKNET cache "
                f"{cache_path}: {e}"
            )

            return None

    def fetch(
        self,
        crop: str,
        state: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            date, crop, state, market, min_price, max_price,
            modal_price, arrivals_tonnes
        """
        crop = crop.lower()
        if crop not in SUPPORTED_CROPS:
            raise ValueError(f"Unsupported crop: {crop}. Choose from {SUPPORTED_CROPS}")

        if API_CONFIG.enam_api_key:
            logger.info(f"Fetching {crop} prices from eNAM API")
            return self._fetch_enam(
                crop,
                state,
                start_date,
                end_date,
            )

        # AGMARKNET historical endpoint does not use the old
        # AGMARKNET_API_KEY configuration gate.
        try:
            logger.info(
                f"Fetching {crop} prices from AGMARKNET historical endpoint"
            )

            return self._fetch_agmarknet(
                crop,
                state,
                start_date,
                end_date,
            )

        except Exception as e:
            logger.error(
                f"AGMARKNET fetch failed: {e}"
            )

            empty_df = pd.DataFrame(
                columns=[
                    "date",
                    "state",
                    "arrivals_tonnes",
                    "min_price",
                    "max_price",
                    "modal_price",
                    "market_count",
                ]
            )

            empty_df.attrs["data_source"] = "agmarknet_failed"
            empty_df.attrs["is_live"] = False
            empty_df.attrs["is_complete"] = False
            empty_df.attrs["error"] = str(e)

            return empty_df

    # ── eNAM API ──────────────────────────────
    def _fetch_enam(self, crop, state, start_date, end_date) -> pd.DataFrame:
        params = {
            "commodity": crop,
            "state": state,
            "fromDate": start_date,
            "toDate": end_date,
            "api_key": API_CONFIG.enam_api_key,
        }
        try:
            resp = self.session.get(API_CONFIG.enam_base_url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            df = self._parse_enam_response(raw, crop, state)
            df.attrs["data_source"] = "enam"
            df.attrs["is_live"] = True
            return df
        except Exception as e:
            logger.error(f"eNAM fetch failed: {e} — falling back to synthetic data")
            df = self._generate_synthetic(crop, state, start_date, end_date)
            df.attrs["data_source"] = "synthetic_fallback"
            df.attrs["is_live"] = False
            return df

    def _parse_enam_response(self, raw: dict, crop: str, state: str) -> pd.DataFrame:
        records = raw.get("data", [])
        rows = []
        for r in records:
            rows.append({
                "date":             pd.to_datetime(r["trade_date"]),
                "crop":             crop,
                "state":            state,
                "market":           r.get("apmc_name", ""),
                "min_price":        float(r.get("min_price", 0)),
                "max_price":        float(r.get("max_price", 0)),
                "modal_price":      float(r.get("modal_price", 0)),
                "arrivals_tonnes":  float(r.get("arrivals", 0)),
            })
        return pd.DataFrame(rows)

    # ── Agmarknet ─────────────────────────────
    def _fetch_agmarknet(
        self,
        crop: str,
        state: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch and normalize historical mandi prices from AGMARKNET."""

        # Resolve human-readable names to AGMARKNET IDs dynamically.
        try:
            state_id = self.metadata_resolver.get_state_id(state)
            commodity_id = self.metadata_resolver.get_commodity_id(crop)

        except ValueError as e:
            raise ValueError(
                f"Could not resolve AGMARKNET IDs for "
                f"{crop} / {state}: {e}"
            ) from e

        logger.info(
            f"AGMARKNET IDs resolved: "
            f"{crop}={commodity_id}, {state}={state_id}"
        )

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate the months covering the requested date range.
        month_starts = pd.date_range(
            start=start.replace(day=1),
            end=end.replace(day=1),
            freq="MS",
        )

        all_records = []
        successful_months = []
        failed_months = []

        for month_start in month_starts:
            year = month_start.year
            month = month_start.month
            month_key = f"{year}-{month:02d}"

            params = {
                "year": year,
                "month": month,
                "includeExcel": "false",
                "stateId": state_id,
                "commodityId": commodity_id,
            }

            url = (
                "https://api.agmarknet.gov.in/v1/"
                "prices-and-arrivals/date-wise/"
                "specific-commodity"
            )

            logger.info(
                f"AGMARKNET: processing {crop} / {state} {month_key}"
            )

            # ---------------------------------------------------------
            # Try local cache first
            # ---------------------------------------------------------
            cached_df = self._load_agmarknet_cache(
                crop,
                state,
                year,
                month,
            )

            if cached_df is not None and not cached_df.empty:
                logger.info(
                    f"AGMARKNET: using cached data for {month_key}"
                )
                all_records.append(cached_df)
                successful_months.append(month_key)
                continue

            # ---------------------------------------------------------
            # No cache → download from AGMARKNET
            # ---------------------------------------------------------
            logger.info(
                f"AGMARKNET: downloading {month_key}"
            )

            try:
                response = self._get_with_retry(
                    url,
                    params,
                    max_retries=3,
                )

                response.raise_for_status()

                raw = response.json()

                if not raw.get("success"):
                    logger.warning(
                        f"AGMARKNET returned unsuccessful response for {month_key}"
                    )
                    failed_months.append(month_key)
                    continue

                month_df = self._parse_agmarknet_response(
                    raw,
                    state,
                )


                logger.info(
                    f"AGMARKNET: {month_key} → "
                    f"{len(month_df)} market-level records"
)

                if month_df.empty:
                    logger.warning(
                        f"AGMARKNET returned no records for {month_key}"
                    )
                    failed_months.append(month_key)
                    continue



                self._save_agmarknet_cache(
                    month_df,
                    crop,
                    state,
                    year,
                    month,
                )

                all_records.append(month_df)
                successful_months.append(month_key)



            except Exception as e:
                logger.warning(
                    f"AGMARKNET failed for {month_key}: {e}"
                )
                failed_months.append(month_key)

        if not all_records:
            empty_df = pd.DataFrame(
                columns=[
                    "date",
                    "state",
                    "arrivals_tonnes",
                    "min_price",
                    "max_price",
                    "modal_price",
                    "market_count",
                ]
            )

            empty_df.attrs["data_source"] = "agmarknet_failed"
            empty_df.attrs["is_live"] = False
            empty_df.attrs["is_historical"] = True
            empty_df.attrs["is_complete"] = False
            empty_df.attrs["successful_months"] = successful_months
            empty_df.attrs["failed_months"] = failed_months

            return empty_df

        market_df = pd.concat(
            all_records,
            ignore_index=True,
        )

        # Keep only the requested date range.
        market_df = market_df[
            (market_df["date"] >= start)
            & (market_df["date"] <= end)
        ].copy()

        if market_df.empty:
            raise ValueError(
                f"AGMARKNET returned no records inside {start_date} - {end_date}"
            )

        daily_df = self._aggregate_agmarknet_daily(
            market_df
        )

        daily_df.attrs["data_source"] = "agmarknet"
        daily_df.attrs["is_live"] = False
        daily_df.attrs["is_historical"] = True
        daily_df.attrs["successful_months"] = successful_months
        daily_df.attrs["failed_months"] = failed_months
        daily_df.attrs["is_complete"] = len(failed_months) == 0

        logger.info(
            f"AGMARKNET: {len(market_df)} market-level records → "
            f"{len(daily_df)} daily records"
        )

        return daily_df
    

    # ── Synthetic data generator ───────────────
    def _generate_synthetic(
        self, crop: str, state: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Generates realistic synthetic price data with:
        - Seasonal patterns   (annual cycle)
        - Weekly market dips  (mandis closed some days)
        - Long-term trend     (mild inflation)
        - Random noise        (market volatility)
        """
        # Base prices per crop (₹/quintal)
        BASE_PRICES = {
            "wheat":     2100, "rice":      3200, "tomato":    1800,
            "onion":     1500, "potato":    1200, "cotton":    6500,
            "soybean":   4200, "maize":     1900, "barley":    1700,
            "sugarcane":  350,
        }
        # Seasonal volatility (0=stable, 1=very volatile)
        VOLATILITY = {
            "wheat":     0.08, "rice":      0.10, "tomato":    0.40,
            "onion":     0.45, "potato":    0.30, "cotton":    0.12,
            "soybean":   0.15, "maize":     0.12, "barley":    0.09,
            "sugarcane": 0.05,
        }

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n = len(dates)
        rng = np.random.default_rng(seed=42)

        base     = BASE_PRICES.get(crop, 2000)
        vol      = VOLATILITY.get(crop, 0.15)
        t        = np.arange(n)

        # Seasonal sine wave (annual period)
        seasonal = base * 0.12 * np.sin(2 * np.pi * t / 365 + np.pi / 4)
        # Long-term inflation trend (~5% per year)
        trend    = base * 0.05 * t / 365
        # Random day-to-day noise
        noise    = rng.normal(0, base * vol * 0.1, n)
        # Cumulative random walk component
        walk     = np.cumsum(rng.normal(0, base * vol * 0.02, n))
        walk    -= walk.mean()  # zero-mean

        modal = np.clip(base + seasonal + trend + noise + walk, base * 0.4, base * 2.5)
        min_p = modal * rng.uniform(0.88, 0.95, n)
        max_p = modal * rng.uniform(1.05, 1.15, n)

        # Market closed on some days (lower arrivals on weekends)
        arrivals_base = rng.uniform(50, 500, n)
        arrivals_base[pd.DatetimeIndex(dates).dayofweek == 6] *= 0.3  # Sunday

        df = pd.DataFrame({
            "date":            dates,
            "crop":            crop,
            "state":           state,
            "market":          f"{state[:3].upper()}_MAIN_MANDI",
            "min_price":       np.round(min_p, 2),
            "max_price":       np.round(max_p, 2),
            "modal_price":     np.round(modal, 2),
            "arrivals_tonnes": np.round(arrivals_base, 1),
        })

        logger.info(f"Generated {len(df)} synthetic rows for {crop} in {state}")
        return df

    def _parse_agmarknet_response(
        self,
        raw: dict,
        state: str,
    ) -> pd.DataFrame:
        """Normalize AGMARKNET date-wise commodity response."""

        records = []

        for market in raw.get("markets", []):
            market_name = market.get("marketName", "Unknown")

            for day in market.get("dates", []):
                arrival_date = day.get("arrivalDate")

                for row in day.get("data", []):
                    records.append({
                        "date": pd.to_datetime(
                            arrival_date,
                            format="%d/%m/%Y",
                        ),
                        "state": state,
                        "market": market_name,
                        "variety": row.get("variety"),
                        "arrivals_tonnes": float(
                            row.get("arrivals", 0) or 0
                        ),
                        "min_price": float(
                            row.get("minimumPrice", 0) or 0
                        ),
                        "max_price": float(
                            row.get("maximumPrice", 0) or 0
                        ),
                        "modal_price": float(
                            row.get("modalPrice", 0) or 0
                        ),
                    })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def _aggregate_agmarknet_daily(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate market-level AGMARKNET data into daily state-level data."""

        if df.empty:
            return pd.DataFrame()

        def weighted_average(group, value_col):
            weights = group["arrivals_tonnes"]

            if weights.sum() <= 0:
                return group[value_col].mean()

            return (group[value_col] * weights).sum() / weights.sum()

        daily = (
            df.groupby(["date", "state"])
            .apply(
                lambda group: pd.Series({
                    "arrivals_tonnes": group["arrivals_tonnes"].sum(),

                    # Preserve actual observed price boundaries.
                    "min_price": group["min_price"].min(),
                    "max_price": group["max_price"].max(),

                    # Arrival-weighted market modal price.
                    "modal_price": weighted_average(
                        group,
                        "modal_price",
                    ),

                    "market_count": group["market"].nunique(),
                }),
                include_groups=False,
            )
            .reset_index()
        )

        daily = daily.sort_values("date").reset_index(drop=True)

        return daily


# ──────────────────────────────────────────────
# Weather Fetcher
# ──────────────────────────────────────────────

# Approximate coordinates for major agricultural states
STATE_COORDINATES = {
    "Maharashtra":     (19.75, 75.71),
    "Punjab":          (31.15, 75.34),
    "Haryana":         (29.06, 76.09),
    "Uttar Pradesh":   (26.85, 80.91),
    "Madhya Pradesh":  (22.97, 78.65),
    "Rajasthan":       (27.02, 74.22),
    "Gujarat":         (22.26, 71.19),
    "Karnataka":       (15.32, 75.72),
    "Andhra Pradesh":  (15.91, 79.74),
    "Telangana":       (17.38, 78.49),
    "West Bengal":     (22.99, 87.85),
}


class WeatherFetcher:
    """
    Fetches historical daily weather data (temperature, rainfall, humidity)
    for Indian states using OpenWeatherMap One Call API.
    Falls back to synthetic climate normals if no API key.
    """

    def __init__(self):
        self.session = requests.Session()
        self.base_url = API_CONFIG.openweather_base_url

    def fetch(self, state: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data.

        Open-Meteo is used for historical weather because the
        current OpenWeather subscription does not provide the
        historical endpoint required by the original implementation.
        """
        try:
            return self._fetch_open_meteo(
                state=state,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.warning(
                f"Open-Meteo historical weather failed: {e} "
                f"— falling back to climate normals"
            )

            df = self._generate_climate_normals(
                state,
                start_date,
                end_date,
            )

            df.attrs["data_source"] = "climate_normals"
            df.attrs["is_live"] = False
            df.attrs["source_counts"] = {
                "climate_normals": len(df)
            }

            return df

    def _fetch_open_meteo(
        self,
        state: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical daily weather from Open-Meteo."""

        lat, lon = STATE_COORDINATES.get(state, (20.5, 78.9))

        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_sum"
            ),
            "hourly": "relative_humidity_2m",
            "timezone": "Asia/Kolkata",
        }

        response = self.session.get(
            url,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # Daily weather
        daily = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
            "rainfall_mm": data["daily"]["precipitation_sum"],
        })

        # Hourly humidity → daily mean
        hourly = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "humidity_pct": data["hourly"]["relative_humidity_2m"],
        })

        hourly["date"] = hourly["time"].dt.normalize()

        humidity_daily = (
            hourly
            .groupby("date")["humidity_pct"]
            .mean()
            .reset_index()
        )

        weather_df = daily.merge(
            humidity_daily,
            on="date",
            how="left",
        )

        weather_df["state"] = state

        weather_df = weather_df[
            [
                "date",
                "state",
                "temp_max",
                "temp_min",
                "rainfall_mm",
                "humidity_pct",
            ]
        ]

        weather_df.attrs["data_source"] = "open_meteo"
        weather_df.attrs["is_live"] = False
        weather_df.attrs["source_counts"] = {
            "open_meteo": len(weather_df)
        }

        logger.info(
            f"Open-Meteo: fetched {len(weather_df)} historical "
            f"weather rows for {state}"
        )

        return weather_df

    def _fetch_owm(self, state: str, start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = STATE_COORDINATES.get(state, (20.5, 78.9))
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        rows = []
        sources = []

        for dt in dates:
            unix_ts = int(dt.timestamp())
            url = f"{self.base_url}/onecall/timemachine"
            params = {
                "lat": lat, "lon": lon,
                "dt": unix_ts,
                "appid": API_CONFIG.openweather_api_key,
                "units": "metric",
            }
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                daily = data.get("current", {})
                rows.append({
                    "date":         dt,
                    "state":        state,
                    "temp_max":     daily.get("temp", 25),
                    "temp_min":     daily.get("feels_like", 20),
                    "rainfall_mm":  daily.get("rain", {}).get("1h", 0) * 24,
                    "humidity_pct": daily.get("humidity", 60),
                })

                sources.append("openweather")

                time.sleep(0.2)  # Rate limit: 60 calls/min on free tier
            except Exception as e:
                logger.warning(f"OWM fetch failed for {dt.date()}: {e}")
                rows.append(self._climate_row(state, dt))
                sources.append("climate_normals")

        df = pd.DataFrame(rows)

        unique_sources = set(sources)

        if unique_sources == {"openweather"}:
            df.attrs["data_source"] = "openweather"
            df.attrs["is_live"] = True
        elif unique_sources == {"climate_normals"}:
            df.attrs["data_source"] = "climate_normals"
            df.attrs["is_live"] = False
        else:
            df.attrs["data_source"] = "mixed"
            df.attrs["is_live"] = False

        df.attrs["source_counts"] = {
            source: sources.count(source)
            for source in unique_sources
        }

        return df

    def _generate_climate_normals(
        self, state: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Climate normals based on Indian agricultural zones."""
        # Monthly mean temperatures °C (Jan–Dec)
        TEMP_NORMALS = {
            "Punjab":          [12, 15, 21, 28, 34, 36, 34, 33, 31, 26, 19, 13],
            "Maharashtra":     [24, 26, 29, 33, 35, 31, 28, 27, 28, 29, 26, 23],
            "Uttar Pradesh":   [14, 17, 23, 30, 35, 36, 33, 32, 30, 26, 20, 14],
            "Gujarat":         [22, 25, 29, 34, 37, 34, 30, 29, 30, 31, 27, 22],
            "Karnataka":       [25, 27, 29, 31, 30, 26, 25, 25, 25, 26, 25, 24],
        }
        # Monthly rainfall mm
        RAIN_NORMALS = {
            "Punjab":          [25,20,18,10,10,30,120,110,50,10,5,20],
            "Maharashtra":     [10,5,5,10,30,150,300,250,150,50,20,10],
            "Uttar Pradesh":   [20,15,10,5,10,60,200,250,150,30,10,20],
            "Gujarat":         [5,3,2,2,5,40,200,180,80,20,5,5],
            "Karnataka":       [10,8,10,40,110,80,70,80,130,170,50,20],
        }

        default_temps = [22, 24, 28, 33, 36, 34, 30, 29, 29, 29, 25, 22]
        default_rain  = [15, 10, 8,  8,  15, 60, 180, 160, 90, 30, 10, 15]

        temps = TEMP_NORMALS.get(state, default_temps)
        rains = RAIN_NORMALS.get(state, default_rain)

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        rng   = np.random.default_rng(seed=99)
        rows  = []

        for dt in dates:
            m   = dt.month - 1
            t   = temps[m] + rng.normal(0, 2.5)
            r   = max(0, rains[m] / 30 + rng.exponential(2))
            hum = 40 + (r / 10) * 30 + rng.uniform(-5, 5)
            rows.append({
                "date":         dt,
                "state":        state,
                "temp_max":     round(t + rng.uniform(3, 6), 1),
                "temp_min":     round(t - rng.uniform(3, 6), 1),
                "rainfall_mm":  round(r, 1),
                "humidity_pct": round(np.clip(hum, 20, 98), 1),
            })

        return pd.DataFrame(rows)

    def _climate_row(self, state: str, dt: datetime) -> dict:
        return {
            "date": dt, "state": state,
            "temp_max": 30, "temp_min": 20,
            "rainfall_mm": 0, "humidity_pct": 60,
        }
