"""
main.py — FastAPI server for CropBalanceAI
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config import SUPPORTED_CROPS, API_CONFIG, PATH_CONFIG
from fetcher import MandiPriceFetcher, WeatherFetcher
from predictor import CropPredictor
from disease.disease_predictor import DiseasePredictor

# ── App setup ──────────────────────────────────
app = FastAPI(
    title="CropBalanceAI — Crop Price & Disease API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, dict] = {}
_predictors: Dict[str, CropPredictor] = {}
_disease_predictor: DiseasePredictor = None


# ── Schemas ────────────────────────────────────
class PredictRequest(BaseModel):
    crop:         str = Field(..., example="wheat")
    state:        str = Field(..., example="Punjab")
    days_history: int = Field(90, ge=60, le=365)


class ForecastPoint(BaseModel):
    date:      str
    price_inr: float


class PredictResponse(BaseModel):
    crop:         str
    state:        str
    forecast:     List[ForecastPoint]
    horizon_days: int
    generated_at: str
    model_info:   dict


class BatchPredictRequest(BaseModel):
    crops:        List[str]
    state:        str
    days_history: int = 90


# ── Helpers ────────────────────────────────────
def get_predictor(crop: str) -> CropPredictor:
    if crop not in _predictors:
        try:
            _predictors[crop] = CropPredictor(crop=crop)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model for '{crop}'. Run: python train.py --crop {crop}",
            )
    return _predictors[crop]


def get_disease_predictor() -> DiseasePredictor:
    global _disease_predictor
    if _disease_predictor is None:
        _disease_predictor = DiseasePredictor()
    return _disease_predictor


# ── Routes ─────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"service": "CropBalanceAI", "version": "1.0.0", "status": "running"}


@app.get("/crops", tags=["info"])
def list_crops():
    from config import CROP_DISPLAY_NAMES
    return {
        "crops": [
            {
                "id":      crop,
                "en":      CROP_DISPLAY_NAMES.get(crop, {}).get("en", crop.title()),
                "hi":      CROP_DISPLAY_NAMES.get(crop, {}).get("hi", ""),
                "trained": os.path.exists(os.path.join(PATH_CONFIG.models_dir, f"best_{crop}.pt")),
            }
            for crop in SUPPORTED_CROPS
        ]
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict_price(req: PredictRequest):
    crop  = req.crop.lower()
    state = req.state

    if crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"Unsupported crop '{crop}'")

    cache_key = f"predict:{crop}:{state}"
    cached    = _cache.get(cache_key)
    if cached and (datetime.utcnow() - datetime.fromisoformat(cached["generated_at"])).seconds < API_CONFIG.cache_ttl_seconds:
        return cached

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=req.days_history)).strftime("%Y-%m-%d")

    try:
        price_df   = MandiPriceFetcher().fetch(crop, state, start_date, end_date)
        weather_df = WeatherFetcher().fetch(state, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    recent_df = pd.merge(price_df, weather_df, on=["date", "state"], how="left")
    recent_df[["temp_max","temp_min","rainfall_mm","humidity_pct"]] = \
        recent_df[["temp_max","temp_min","rainfall_mm","humidity_pct"]].ffill().bfill()

    try:
        result = get_predictor(crop).predict(recent_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = PredictResponse(
        crop=crop, state=state,
        forecast=[ForecastPoint(**p) for p in result["forecast"]],
        horizon_days=len(result["forecast"]),
        generated_at=result["generated_at"],
        model_info={"architecture": "Stacked LSTM", "unit": "INR per quintal"},
    )
    _cache[cache_key] = response.model_dump()
    _cache[cache_key]["generated_at"] = result["generated_at"]
    return response


@app.post("/predict/batch", tags=["prediction"])
def batch_predict(req: BatchPredictRequest):
    results, errors = {}, {}
    for crop in req.crops:
        try:
            results[crop] = predict_price(PredictRequest(
                crop=crop, state=req.state, days_history=req.days_history
            ))
        except HTTPException as e:
            errors[crop] = e.detail
    return {"results": results, "errors": errors}


@app.get("/prices/live", tags=["prices"])
def get_live_prices(
    crop:  str = Query(..., example="wheat"),
    state: str = Query(..., example="Punjab"),
):
    cache_key = f"live:{crop}:{state}"
    cached = _cache.get(cache_key)
    if cached:
        age = (datetime.utcnow() - datetime.fromisoformat(cached["fetched_at"])).seconds
        if age < API_CONFIG.cache_ttl_seconds:
            return {**cached, "cached": True}

    today     = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    df = MandiPriceFetcher().fetch(crop.lower(), state, yesterday, today)
    if df.empty:
        raise HTTPException(status_code=404, detail="No price data available")

    latest = df.sort_values("date").iloc[-1]
    result = {
        "crop": crop, "state": state,
        "date":            latest["date"].strftime("%Y-%m-%d"),
        "min_price_inr":   round(float(latest["min_price"]), 2),
        "max_price_inr":   round(float(latest["max_price"]), 2),
        "modal_price_inr": round(float(latest["modal_price"]), 2),
        "arrivals_tonnes": round(float(latest["arrivals_tonnes"]), 1),
        "unit":            "INR per quintal",
        "fetched_at":      datetime.utcnow().isoformat(),
        "cached":          False,
    }
    _cache[cache_key] = result
    return result


@app.get("/prices/history/{crop}", tags=["prices"])
def get_price_history(
    crop:  str,
    state: str = Query(..., example="Punjab"),
    days:  int = Query(90, ge=7, le=730),
):
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = MandiPriceFetcher().fetch(crop.lower(), state, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    df = df.sort_values("date")
    return {
        "crop": crop, "state": state,
        "period": f"{start_date} to {end_date}",
        "records": df[["date","min_price","max_price","modal_price","arrivals_tonnes"]]
                     .assign(date=df["date"].dt.strftime("%Y-%m-%d"))
                     .to_dict(orient="records"),
    }


# ── Disease Detection ──────────────────────────
@app.post("/disease/detect", tags=["disease"])
async def detect_disease(
    image: UploadFile = File(..., description="Leaf photo (.jpg/.png)")
):
    """Upload a crop leaf photo to detect diseases."""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        result = get_disease_predictor().predict_from_bytes(image_bytes)
        return {**result, "filename": image.filename, "analyzed_at": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.exception("Disease detection error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/disease/classes", tags=["disease"])
def list_disease_classes():
    from disease.disease_model import DISEASE_CLASSES, DISEASE_META
    return {
        "total": len(DISEASE_CLASSES),
        "classes": [
            {"id": d, "severity": DISEASE_META.get(d, {}).get("severity", "unknown")}
            for d in DISEASE_CLASSES
        ],
    }