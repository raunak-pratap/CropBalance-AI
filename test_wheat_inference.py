"""Optional end-to-end smoke test for a locally trained wheat LSTM."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import PATH_CONFIG


pytestmark = pytest.mark.ml_integration


def test_wheat_inference_returns_finite_30_day_forecast():
    """Run real checkpoint inference only when explicitly requested and available."""
    if os.getenv("RUN_ML_INTEGRATION") != "1":
        pytest.skip(
            "Set RUN_ML_INTEGRATION=1 to run the trained-model inference smoke test."
        )

    project_root = Path(__file__).resolve().parent
    scaler_path = project_root / PATH_CONFIG.scalers_dir / "scalers_wheat.pkl"
    checkpoint_path = project_root / PATH_CONFIG.models_dir / "best_wheat.pt"

    missing_artifacts = [
        str(path.relative_to(project_root))
        for path in (scaler_path, checkpoint_path)
        if not path.is_file()
    ]
    if missing_artifacts:
        pytest.skip(
            "Wheat inference artifacts are unavailable: "
            + ", ".join(missing_artifacts)
        )

    # Import only after checking artifacts: collection must not load PyTorch
    # checkpoints or require local model files.
    from predictor import CropPredictor

    n = 120
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=n,
    )

    rng = np.random.default_rng(42)
    prices = 2500 + np.arange(n) * 1.5 + rng.normal(0, 40, n)

    recent_df = pd.DataFrame(
        {
            "date": dates,
            "modal_price": prices,
            "min_price": prices - 100,
            "max_price": prices + 100,
            "arrivals_tonnes": rng.uniform(100, 1000, n),
            "temp_max": rng.uniform(28, 38, n),
            "temp_min": rng.uniform(15, 25, n),
            "rainfall_mm": rng.uniform(0, 10, n),
            "humidity_pct": rng.uniform(35, 80, n),
        }
    )

    predictor = CropPredictor("wheat")
    result = predictor.predict(recent_df)

    forecast = pd.DataFrame(result["forecast"])
    values = forecast["price_inr"].to_numpy()

    assert result["crop"] == "wheat"
    assert len(forecast) == 30, "Expected 30 forecast days"
    assert np.isfinite(values).all(), "Found non-finite predictions"
