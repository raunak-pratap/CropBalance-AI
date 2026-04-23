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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional

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
            return self._fetch_enam(crop, state, start_date, end_date)

        if API_CONFIG.agmarknet_api_key:
            logger.info(f"Fetching {crop} prices from Agmarknet")
            return self._fetch_agmarknet(crop, state, start_date, end_date)

        logger.warning("No API keys found — using synthetic data generator")
        return self._generate_synthetic(crop, state, start_date, end_date)

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
            return self._parse_enam_response(raw, crop, state)
        except Exception as e:
            logger.error(f"eNAM fetch failed: {e} — falling back to synthetic data")
            return self._generate_synthetic(crop, state, start_date, end_date)

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
    def _fetch_agmarknet(self, crop, state, start_date, end_date) -> pd.DataFrame:
        params = {
            "Commodity": crop.title(),
            "State":     state,
            "From":      start_date,
            "To":        end_date,
        }
        try:
            resp = self.session.get(API_CONFIG.agmarknet_base_url, params=params, timeout=15)
            resp.raise_for_status()
            # Agmarknet returns HTML table — parse with pandas
            tables = pd.read_html(resp.text)
            if tables:
                df = tables[0]
                df.columns = ["date", "market", "commodity", "min_price", "max_price",
                              "modal_price", "arrivals_tonnes"]
                df["crop"]  = crop
                df["state"] = state
                df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                return df.dropna(subset=["date"])
        except Exception as e:
            logger.error(f"Agmarknet fetch failed: {e} — falling back to synthetic data")
        return self._generate_synthetic(crop, state, start_date, end_date)

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
        if API_CONFIG.openweather_api_key:
            return self._fetch_owm(state, start_date, end_date)
        logger.warning("No OpenWeather key — using climate normals")
        return self._generate_climate_normals(state, start_date, end_date)

    def _fetch_owm(self, state: str, start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = STATE_COORDINATES.get(state, (20.5, 78.9))
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        rows = []

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
                time.sleep(0.2)  # Rate limit: 60 calls/min on free tier
            except Exception as e:
                logger.warning(f"OWM fetch failed for {dt.date()}: {e}")
                rows.append(self._climate_row(state, dt))

        return pd.DataFrame(rows)

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
