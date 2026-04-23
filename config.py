"""
Central configuration for the Smart Farming - Crop Price Prediction System.
Edit this file to customise crops, model hyperparameters, and API settings.
"""

from dataclasses import dataclass, field
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Supported crops (generic system)
# ──────────────────────────────────────────────
SUPPORTED_CROPS = [
    "wheat", "rice", "tomato", "onion", "potato",
    "cotton", "soybean", "maize", "barley", "sugarcane"
]

# Crop display names (English + Hindi)
CROP_DISPLAY_NAMES = {
    "wheat":     {"en": "Wheat",     "hi": "गेहूँ"},
    "rice":      {"en": "Rice",      "hi": "चावल"},
    "tomato":    {"en": "Tomato",    "hi": "टमाटर"},
    "onion":     {"en": "Onion",     "hi": "प्याज"},
    "potato":    {"en": "Potato",    "hi": "आलू"},
    "cotton":    {"en": "Cotton",    "hi": "कपास"},
    "soybean":   {"en": "Soybean",   "hi": "सोयाबीन"},
    "maize":     {"en": "Maize",     "hi": "मक्का"},
    "barley":    {"en": "Barley",    "hi": "जौ"},
    "sugarcane": {"en": "Sugarcane", "hi": "गन्ना"},
}

# Major Indian mandis
SUPPORTED_STATES = [
    "Maharashtra", "Punjab", "Haryana", "Uttar Pradesh",
    "Madhya Pradesh", "Rajasthan", "Gujarat", "Karnataka",
    "Andhra Pradesh", "Telangana", "West Bengal"
]


# ──────────────────────────────────────────────
# LSTM Model Hyperparameters
# ──────────────────────────────────────────────
@dataclass
class LSTMConfig:
    sequence_length: int = 60          # Days of history fed to LSTM
    hidden_size: int = 128             # LSTM hidden units
    num_layers: int = 3                # Stacked LSTM layers
    dropout: float = 0.2              # Dropout between layers
    bidirectional: bool = False        # Bidirectional LSTM
    forecast_horizon: int = 30         # Days to predict ahead

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 15                 # Early stopping patience
    train_split: float = 0.75
    val_split: float = 0.15
    # test_split = remaining 0.10

    # Features used for prediction
    feature_columns: List[str] = field(default_factory=lambda: [
        "modal_price",     # Main target (₹/quintal)
        "min_price",
        "max_price",
        "arrivals_tonnes", # Market supply
        "temp_max",        # Weather features
        "temp_min",
        "rainfall_mm",
        "humidity_pct",
        "month_sin",       # Cyclical time encoding
        "month_cos",
        "week_sin",
        "week_cos",
    ])
    target_column: str = "modal_price"


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
@dataclass
class PathConfig:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    models_dir: str = "models/saved"
    scalers_dir: str = "models/scalers"
    logs_dir: str = "logs"

    def create_dirs(self):
        for path in [self.data_raw, self.data_processed,
                     self.models_dir, self.scalers_dir, self.logs_dir]:
            os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────
# API / External Services
# ──────────────────────────────────────────────
@dataclass
class APIConfig:
    # Agmarknet (Government mandi prices)
    agmarknet_base_url: str = "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    agmarknet_api_key: str = field(default_factory=lambda: os.getenv("AGMARKNET_API_KEY", ""))

    # OpenWeatherMap
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))

    # eNAM (National Agriculture Market)
    enam_base_url: str = "https://enam.gov.in/web/dashboard/trade-data"
    enam_api_key: str = field(default_factory=lambda: os.getenv("ENAM_API_KEY", ""))

    # Internal FastAPI server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Redis (caching live prices)
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    cache_ttl_seconds: int = 3600      # 1 hour cache for live prices


# ──────────────────────────────────────────────
# Singleton instances
# ──────────────────────────────────────────────
LSTM_CONFIG = LSTMConfig()
PATH_CONFIG = PathConfig()
API_CONFIG  = APIConfig()
