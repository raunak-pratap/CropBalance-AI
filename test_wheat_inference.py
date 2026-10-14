import numpy as np
import pandas as pd
from predictor import CropPredictor

n = 120
dates = pd.date_range(
    end=pd.Timestamp.today().normalize(),
    periods=n,
)

rng = np.random.default_rng(42)
prices = 2500 + np.arange(n) * 1.5 + rng.normal(0, 40, n)

recent_df = pd.DataFrame({
    "date": dates,
    "modal_price": prices,
    "min_price": prices - 100,
    "max_price": prices + 100,
    "arrivals_tonnes": rng.uniform(100, 1000, n),
    "temp_max": rng.uniform(28, 38, n),
    "temp_min": rng.uniform(15, 25, n),
    "rainfall_mm": rng.uniform(0, 10, n),
    "humidity_pct": rng.uniform(35, 80, n),
})

predictor = CropPredictor("wheat")
result = predictor.predict(recent_df)

forecast = pd.DataFrame(result["forecast"])
values = forecast["price_inr"].to_numpy()

print("Crop:", result["crop"])
print("Forecast rows:", len(forecast))
print("First 5 predictions:")
print(forecast.head().to_string(index=False))
print("All predictions finite:", bool(np.isfinite(values).all()))
print("Generated at:", result["generated_at"])

assert len(forecast) == 30, "Expected 30 forecast days"
assert np.isfinite(values).all(), "Found non-finite predictions"
print("PASS: Inference smoke test")
