"""
main.py — FastAPI server for CropBalanceAI
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config import SUPPORTED_CROPS, API_CONFIG, PATH_CONFIG
from fetcher import MandiPriceFetcher, WeatherFetcher
from predictor import CropPredictor
from disease.disease_predictor import DiseasePredictor


# =========================================================
# App Setup
# =========================================================
app = FastAPI(
    title="CropBalanceAI",
    version="2.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, dict] = {}
_predictors: Dict[str, CropPredictor] = {}
_disease_predictor: DiseasePredictor = None


# =========================================================
# Schemas
# =========================================================
class PredictRequest(BaseModel):
    crop: str = Field(..., example="wheat")
    state: str = Field(..., example="Punjab")
    days_history: int = Field(90, ge=60, le=365)


class ForecastPoint(BaseModel):
    date: str
    price_inr: float


class PredictResponse(BaseModel):
    crop: str
    state: str
    forecast: List[ForecastPoint]
    horizon_days: int
    generated_at: str
    model_info: dict


class BatchPredictRequest(BaseModel):
    crops: List[str]
    state: str
    days_history: int = 90


class ChatRequest(BaseModel):
    message: str
    crop: str = "wheat"
    state: str = "Punjab"


# =========================================================
# Helpers
# =========================================================
def get_predictor(crop: str) -> CropPredictor:
    if crop not in _predictors:
        try:
            _predictors[crop] = CropPredictor(crop=crop)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for {crop}"
            )
    return _predictors[crop]


def get_disease_predictor() -> DiseasePredictor:
    global _disease_predictor
    if _disease_predictor is None:
        _disease_predictor = DiseasePredictor()
    return _disease_predictor


# =========================================================
# Root
# =========================================================
@app.get("/", tags=["health"])
def root():
    return {
        "service": "CropBalanceAI",
        "version": "2.0.0",
        "status": "running"
    }


# =========================================================
# Crop List
# =========================================================
@app.get("/crops", tags=["info"])
def list_crops():
    return {
        "supported_crops": SUPPORTED_CROPS
    }


# =========================================================
# Price Prediction
# =========================================================
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict_price(req: PredictRequest):
    crop = req.crop.lower()

    if crop not in SUPPORTED_CROPS:
        raise HTTPException(400, f"Unsupported crop: {crop}")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.now() - timedelta(days=req.days_history)
    ).strftime("%Y-%m-%d")

    try:
        price_df = MandiPriceFetcher().fetch(
            crop, req.state, start_date, end_date
        )
        weather_df = WeatherFetcher().fetch(
            req.state, start_date, end_date
        )

        recent_df = pd.merge(
            price_df,
            weather_df,
            on=["date", "state"],
            how="left"
        )

        result = get_predictor(crop).predict(recent_df)

        return PredictResponse(
            crop=crop,
            state=req.state,
            forecast=[
                ForecastPoint(**x)
                for x in result["forecast"]
            ],
            horizon_days=len(result["forecast"]),
            generated_at=result["generated_at"],
            model_info={
                "architecture": "LSTM",
                "unit": "INR/quintal"
            }
        )

    except Exception as e:
        raise HTTPException(500, str(e))


# =========================================================
# Batch Prediction
# =========================================================
@app.post("/predict/batch", tags=["prediction"])
def batch_predict(req: BatchPredictRequest):
    results = {}

    for crop in req.crops:
        try:
            results[crop] = predict_price(
                PredictRequest(
                    crop=crop,
                    state=req.state,
                    days_history=req.days_history
                )
            )
        except Exception as e:
            results[crop] = {"error": str(e)}

    return results


# =========================================================
# Live Prices
# =========================================================
@app.get("/prices/live", tags=["prices"])
def get_live_prices(
    crop: str = Query(...),
    state: str = Query(...)
):
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (
        datetime.now() - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    df = MandiPriceFetcher().fetch(
        crop.lower(),
        state,
        week_ago,
        today
    )

    if df.empty:
        raise HTTPException(404, "No live price found")

    latest = df.sort_values("date").iloc[-1]

    return {
        "crop": crop,
        "state": state,
        "date": latest["date"].strftime("%Y-%m-%d"),
        "modal_price_inr": round(float(latest["modal_price"]), 2),
        "min_price_inr": round(float(latest["min_price"]), 2),
        "max_price_inr": round(float(latest["max_price"]), 2)
    }


# =========================================================
# Historical Prices
# =========================================================
@app.get("/prices/history/{crop}", tags=["prices"])
def get_price_history(
    crop: str,
    state: str,
    days: int = 90
):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (
        datetime.now() - timedelta(days=days)
    ).strftime("%Y-%m-%d")

    df = MandiPriceFetcher().fetch(
        crop.lower(),
        state,
        start_date,
        end_date
    )

    return df.to_dict(orient="records")


# =========================================================
# Disease Detection
# =========================================================
@app.post("/disease/detect", tags=["disease"])
async def detect_disease(
    image: UploadFile = File(...)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid image")

    image_bytes = await image.read()

    try:
        result = get_disease_predictor().predict_from_bytes(
            image_bytes
        )

        return {
            **result,
            "filename": image.filename,
            "analyzed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.exception("Disease prediction failed")
        raise HTTPException(500, str(e))


# =========================================================
# Disease Classes
# =========================================================
@app.get("/disease/classes", tags=["disease"])
def list_disease_classes():
    from disease.disease_model import DISEASE_CLASSES

    return {
        "total": len(DISEASE_CLASSES),
        "classes": DISEASE_CLASSES
    }


# =========================================================
# AI Chatbot
# =========================================================
@app.post("/chat", tags=["chatbot"])
def chat(req: ChatRequest):
    msg = req.message.lower()

    try:
        if "price" in msg:
            result = get_live_prices(
                crop=req.crop,
                state=req.state
            )

            return {
                "response":
                    f"Current {req.crop} price in {req.state} is ₹{result['modal_price_inr']} per quintal."
            }

        elif "predict" in msg:
            result = predict_price(
                PredictRequest(
                    crop=req.crop,
                    state=req.state,
                    days_history=90
                )
            )

            avg_price = sum(
                p.price_inr for p in result.forecast
            ) / len(result.forecast)

            return {
                "response":
                    f"Predicted average {req.crop} price for next {result.horizon_days} days is ₹{round(avg_price,2)}."
            }

        elif "sell" in msg:
            live = get_live_prices(
                crop=req.crop,
                state=req.state
            )

            forecast = predict_price(
                PredictRequest(
                    crop=req.crop,
                    state=req.state,
                    days_history=90
                )
            )

            future_avg = sum(
                p.price_inr for p in forecast.forecast
            ) / len(forecast.forecast)

            if future_avg > live["modal_price_inr"]:
                advice = "Wait. Prices may increase."
            else:
                advice = "Sell now. Prices may fall."

            return {"response": advice}

        else:
            return {
                "response":
                    "Ask me about crop prices, prediction, or selling advice."
            }

    except Exception as e:
        raise HTTPException(500, str(e))