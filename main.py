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
from services.mandi_archive_service import MandiArchiveService
from services.reference_registry import ReferenceRegistry
from services.agmarknet_master_catalog import AgmarknetMasterCatalog


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
_mandi_archive_service = MandiArchiveService()
_reference_registry = ReferenceRegistry()


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
# Mandi Archive Intelligence
# =========================================================

def run_local_mandi_query(operation):
    """
    Execute a local mandi service operation and translate 
    service errors into appropriate HTTP responses.
    """
    try:
        return operation()
    
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )


@app.get("/markets/local", tags=["local_mandi"])
def get_local_mandi_markets(
    state_code: str | None = Query(
        default=None,
        description="CropBalance internal state code, for example maharashtra"
    ),
    district_code: str | None = Query(
        default=None,
        description="CropBalance internal district code, for example dharashiv"
    ),
):
    """
    Return markets in the archive, optionally filtered by registered geography.
    """
    return run_local_mandi_query(
        lambda: _mandi_archive_service.get_markets(
            state_code=state_code,
            district_code=district_code,
        )
    )

@app.get("/mandi/prices/latest", tags=["local_mandi"])
def get_latest_local_mandi_prices(
    crop: str = Query(
        ...,
        min_length=1,
        max_length=50,
        description="Crop name or alias, for example soyabean or wheat"
    ),
    state_code: str | None = Query(default=None, description="Internal CropBalance state key"),
    district_code: str | None = Query(default=None, description="Internal CropBalance district key"),
):
    """
    Return the latest archived observation for each market.
    Missing prices remain missing; they are never treated as zero.
    """
    crop = crop.strip().lower()

    if not crop:
        raise HTTPException(
            status_code=422,
            detail="Crop name cannot be empty"
        )

    return run_local_mandi_query(
        lambda: _mandi_archive_service.get_latest_prices(
            crop, state_code=state_code, district_code=district_code
        )
    )


@app.get("/mandi/prices/history", tags=["local_mandi"])
def get_local_mandi_history(
    crop: str = Query(
        ...,
        min_length=1,
        max_length=50,
        description="Crop name, for example soyabean or wheat"
    ),
    market_id: str = Query(
        ...,
        min_length=1,
        description="Source market ID, for example 1599 for Kalamb"
    ),
    state_code: str | None = Query(default=None, description="Internal CropBalance state key"),
    district_code: str | None = Query(default=None, description="Internal CropBalance district key"),
):
    """
    Return historical observations for a crop in a specific market.
    """
    crop = crop.strip().lower()
    market_id = market_id.strip()

    if not crop:
        raise HTTPException(
            status_code=422,
            detail="Both crop and market_id are required"
        )

    return run_local_mandi_query(
        lambda: _mandi_archive_service.get_market_history(
            crop=crop,
            market_id=market_id,
            state_code=state_code,
            district_code=district_code,
        )
    )

@app.get("/mandi/prices/compare", tags=["local_mandi"])
def compare_local_mandi_prices(
    crop: str = Query(
        ...,
        min_length=1,
        max_length=50,
        description="Crop name or alias, for example soybean"
    ),
    state_code: str | None = Query(default=None, description="Internal CropBalance state key"),
    district_code: str | None = Query(default=None, description="Internal CropBalance district key"),
):
    """
    Compare reported values across markets only when
    date, unit, metric label, and variety match.
    """
    crop = crop.strip().lower()

    if not crop:
        raise HTTPException(
            status_code=422,
            detail="Crop name cannot be empty."
        )

    return run_local_mandi_query(
        lambda: _mandi_archive_service.compare_markets(
            crop, state_code=state_code, district_code=district_code
        )
    )


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

# =========================================================
# National Reference Catalog
# =========================================================

def run_reference_query(operation):
    """Run a reference-catalog query and translate lookup errors to HTTP 422."""
    try:
        return operation()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/catalog/coverage", tags=["reference_catalog"])
def get_reference_catalog_coverage():
    """Report catalog counts and, importantly, the current coverage limits."""
    return _reference_registry.coverage_summary()


@app.get("/catalog/crops", tags=["reference_catalog"])
def get_reference_crops(
    category: str | None = Query(default=None, max_length=60),
    q: str | None = Query(default=None, max_length=80),
    language: str = Query(default="en", pattern="^(en|hi|mr)$"),
):
    """List starter crop taxonomy entries; this is not an exhaustive crop list."""
    items = run_reference_query(
        lambda: _reference_registry.list_crops(
            category=category,
            query=q,
            language=language,
        )
    )
    return {
        "status": "ok",
        "catalog": "crops",
        "catalog_scope": "national_starter_taxonomy_not_exhaustive",
        "is_exhaustive": False,
        "count": len(items),
        "items": items,
    }


@app.get("/catalog/states", tags=["reference_catalog"])
def get_reference_states(
    q: str | None = Query(default=None, max_length=80),
):
    """List the seeded 28 state and 8 union-territory names."""
    items = run_reference_query(
        lambda: _reference_registry.list_states(query=q)
    )
    return {
        "status": "ok",
        "catalog": "states",
        "catalog_scope": "national_state_and_union_territory_name_seed",
        "is_exhaustive": True,
        "count": len(items),
        "items": items,
        "note": (
            "Internal CropBalance codes are not LGD codes. Official LGD codes "
            "must be synchronized separately."
        ),
    }


@app.get("/catalog/districts", tags=["reference_catalog"])
def get_reference_districts(
    state_code: str | None = Query(default=None, max_length=80),
    q: str | None = Query(default=None, max_length=80),
    language: str = Query(default="en", pattern="^(en|hi|mr)$"),
):
    """List currently seeded districts, optionally filtered by internal state code."""
    items = run_reference_query(
        lambda: _reference_registry.list_districts(
            state_code=state_code,
            query=q,
            language=language,
        )
    )
    return {
        "status": "ok",
        "catalog": "districts",
        "catalog_scope": "pilot_district_seed_not_exhaustive",
        "is_exhaustive": False,
        "count": len(items),
        "items": items,
    }


@app.get("/catalog/markets", tags=["reference_catalog"])
def get_reference_markets(
    state_code: str | None = Query(default=None, max_length=80),
    district_code: str | None = Query(default=None, max_length=80),
    q: str | None = Query(default=None, max_length=80),
):
    """List currently seeded markets; seeded market coverage is Dharashiv only."""
    items = run_reference_query(
        lambda: _reference_registry.list_markets(
            state_code=state_code,
            district_code=district_code,
            query=q,
        )
    )
    return {
        "status": "ok",
        "catalog": "markets",
        "catalog_scope": "dharashiv_pilot_markets_only",
        "is_exhaustive": False,
        "count": len(items),
        "items": items,
    }


# =========================================================
# AGMARKNET National Source-Master Catalog
# =========================================================

def get_agmarknet_master_catalog() -> AgmarknetMasterCatalog:
    """Load the local synchronized AGMARKNET masters; never fetch on API reads."""
    try:
        return AgmarknetMasterCatalog()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "AGMARKNET source masters have not been synchronized yet. "
                "Run scripts/sync_agmarknet_master.py --write on the host."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def run_agmarknet_catalog_query(operation):
    try:
        return operation()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/source-catalog/agmarknet/coverage", tags=["agmarknet_catalog"])
def get_agmarknet_source_catalog_coverage():
    """Show imported source-master counts and synchronization provenance."""
    catalog = get_agmarknet_master_catalog()
    return catalog.coverage()


@app.get("/source-catalog/agmarknet/states", tags=["agmarknet_catalog"])
def get_agmarknet_source_states(
    q: str | None = Query(default=None, max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
):
    catalog = get_agmarknet_master_catalog()
    return run_agmarknet_catalog_query(
        lambda: catalog.list_records(
            "states", query=q, offset=offset, limit=limit
        )
    )


@app.get("/source-catalog/agmarknet/districts", tags=["agmarknet_catalog"])
def get_agmarknet_source_districts(
    state_id: str | None = Query(default=None, max_length=30),
    q: str | None = Query(default=None, max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
):
    catalog = get_agmarknet_master_catalog()
    return run_agmarknet_catalog_query(
        lambda: catalog.list_records(
            "districts",
            state_id=state_id,
            query=q,
            offset=offset,
            limit=limit,
        )
    )


@app.get("/source-catalog/agmarknet/markets", tags=["agmarknet_catalog"])
def get_agmarknet_source_markets(
    state_id: str | None = Query(default=None, max_length=30),
    district_id: str | None = Query(default=None, max_length=30),
    q: str | None = Query(default=None, max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
):
    catalog = get_agmarknet_master_catalog()
    return run_agmarknet_catalog_query(
        lambda: catalog.list_records(
            "markets",
            state_id=state_id,
            district_id=district_id,
            query=q,
            offset=offset,
            limit=limit,
        )
    )


@app.get("/source-catalog/agmarknet/commodities", tags=["agmarknet_catalog"])
def get_agmarknet_source_commodities(
    group_id: str | None = Query(default=None, max_length=30),
    q: str | None = Query(default=None, max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
):
    catalog = get_agmarknet_master_catalog()
    return run_agmarknet_catalog_query(
        lambda: catalog.list_records(
            "commodities",
            group_id=group_id,
            query=q,
            offset=offset,
            limit=limit,
        )
    )


@app.get("/source-catalog/agmarknet/varieties", tags=["agmarknet_catalog"])
def get_agmarknet_source_varieties(
    commodity_id: str | None = Query(default=None, max_length=30),
    q: str | None = Query(default=None, max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
):
    catalog = get_agmarknet_master_catalog()
    return run_agmarknet_catalog_query(
        lambda: catalog.list_records(
            "varieties",
            commodity_id=commodity_id,
            query=q,
            offset=offset,
            limit=limit,
        )
    )
