# 🌾 Smart Farming — Crop Price Prediction

AI-powered crop price forecasting for Indian farmers using LSTM deep learning.  
Predicts mandi prices 30 days ahead using historical prices + live weather data.

---

## 📁 Project Structure

```
crop_price_prediction/
├── config.py               ← All hyperparameters & settings
├── train.py                ← Run this to train a model
├── requirements.txt
├── .env.example            ← Copy to .env, add API keys
│
├── data/
│   ├── fetcher.py          ← Agmarknet / eNAM / Weather APIs
│   └── preprocessor.py     ← Feature engineering + DataLoaders
│
├── models/
│   ├── lstm_model.py       ← LSTM architecture (PyTorch)
│   ├── trainer.py          ← Training loop + early stopping
│   ├── predictor.py        ← Inference (used by API)
│   ├── saved/              ← Trained model checkpoints (.pt)
│   └── scalers/            ← Fitted scalers (.pkl)
│
├── api/
│   └── main.py             ← FastAPI REST server
│
└── logs/                   ← Training logs
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
cd crop_price_prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your API keys (optional — synthetic data works without keys)
```

### 3. Train a model

```bash
# Train on wheat prices in Punjab (5 years of data)
python train.py --crop wheat --state Punjab --years 5

# Train on tomato prices in Maharashtra
python train.py --crop tomato --state Maharashtra --years 3

# All supported crops:
# wheat, rice, tomato, onion, potato, cotton, soybean, maize, barley, sugarcane
```

Training output:
```
Epoch 001/100 | train_loss=0.0312 | val_loss=0.0287 | MAE=0.0241 | MAPE=4.2% | lr=1.00e-03
Epoch 002/100 | train_loss=0.0298 | val_loss=0.0271 | MAE=0.0228 | MAPE=3.9% | lr=1.00e-03
...
Training complete! best val_loss=0.0198
Test MAPE: 3.7%
```

### 4. Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

---

## 🔌 API Endpoints

### Predict next 30 days

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"crop": "wheat", "state": "Punjab", "days_history": 90}'
```

Response:
```json
{
  "crop": "wheat",
  "state": "Punjab",
  "horizon_days": 30,
  "forecast": [
    {"date": "2024-12-01", "price_inr": 2145.50},
    {"date": "2024-12-02", "price_inr": 2152.30},
    ...
  ],
  "model_info": {"unit": "INR per quintal (100 kg)"}
}
```

### Get live mandi price

```bash
curl "http://localhost:8000/prices/live?crop=wheat&state=Punjab"
```

### Get price history (for charting)

```bash
curl "http://localhost:8000/prices/history/wheat?state=Punjab&days=90"
```

### Batch prediction (multiple crops)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"crops": ["wheat", "rice", "onion"], "state": "Punjab"}'
```

---

## 🧠 Model Architecture

```
Input (60 days × N features)
    ↓
LSTM Layer 1  (128 units, dropout=0.2)
LSTM Layer 2  (128 units, dropout=0.2)
LSTM Layer 3  (128 units)
    ↓
Layer Normalisation
    ↓
FC(128 → 64) + ReLU + Dropout(0.2)
FC(64 → 30)
    ↓
Output: 30-day price forecast (₹/quintal)
```

**Features (12 per day):**
- Modal price, min price, max price
- Market arrivals (tonnes)
- Max/min temperature, rainfall, humidity
- Month & week (sine/cosine encoded)

---

## 📊 Expected Model Performance

| Crop     | MAPE (typical) |
|----------|----------------|
| Wheat    | 3–5%           |
| Rice     | 4–6%           |
| Tomato   | 10–20%         |
| Onion    | 12–22%         |
| Cotton   | 5–8%           |
| Soybean  | 6–9%           |

> Tomato and onion have higher MAPE due to extreme price volatility.  
> Performance improves with more training data (use `--years 7` if available).

---

## 🔑 API Keys (Optional)

Without API keys, the system uses realistic **synthetic data** — perfect for  
development and testing. For production, get keys from:

| Source | Data | URL |
|--------|------|-----|
| eNAM | Live mandi prices | https://enam.gov.in |
| Agmarknet | Historical mandi prices | https://agmarknet.gov.in |
| OpenWeatherMap | Weather data | https://openweathermap.org/api |

---

## 🚀 Next Steps

1. **Disease Detection module** — Add CNN model for crop disease detection
2. **Mobile App** — React Native frontend for farmers
3. **Multi-language** — Hindi/regional language support
4. **Alerts** — Push notifications when prices are predicted to spike
5. **Government integration** — Connect to PM-KISAN and Fasal Bima portals
