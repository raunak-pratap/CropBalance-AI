# CropBalance AI — Monthly Crop Price Forecasting Experiments

This directory contains the reproducible code for the Maharashtra monthly forecasting experiment. It is intentionally separate from CropBalance AI's existing daily LSTM and production prediction paths.

## Scope and target

- Source: Agmarknet-derived India-wide state-monthly data.
- First modelling geography: Maharashtra (state-level, not mandi-level).
- Crops: maize, onion, potato, rice, soybean, tomato, wheat.
- Target: next-month state-level weighted-average price (`price_rs_qtl`, Rs./quintal).
- Input: 12 months × 13 features.
- Chronological splits: train through 2023-12; validation 2024-01 through 2024-12; test 2025-01 onward (source currently ends 2026-10).
- Missing prices are not interpolated. Scalers are fit on training-period observations only.

**Do not describe this experiment as a Dharashiv/APMC modal-price forecast.** The target is a state-level monthly weighted average. Market-level data has a different grain and target and must be modelled separately.

## Setup

Run from the repository root with the project virtual environment active:

```bash
pip install -r preprocessing/requirements.txt
```

The cleaner expects the dataset repository to be cloned at:

```text
data/external/india-mandi-arrivals/
```

If it is not present yet:

```bash
mkdir -p data/external
git clone --depth 1 https://github.com/herrrickshaw/india-mandi-arrivals.git data/external/india-mandi-arrivals
```

## Rebuild the data pipeline

```bash
python -m unittest discover -s preprocessing/tests -v
python preprocessing/clean_india_data.py
python preprocessing/create_sequences.py
python preprocessing/validate_pipeline.py
```

Default outputs are written under `data/processed/` and `data/training/monthly_maharashtra/`. Those generated data and model artifacts should remain local and should not be committed.

## Run experiments

```bash
# Baselines
python experiments/monthly/evaluate_baselines.py

# Shared LSTM (seed 42 by default)
python experiments/monthly/train_monthly_lstm.py

# Robustness runs; each gets a separate output directory
for seed in 7 21 123 2026; do
  python experiments/monthly/train_monthly_lstm.py \
    --seed "$seed" \
    --output-dir "data/training/monthly_maharashtra/monthly_lstm_seed_${seed}"
done

# Validation ensemble and crop-aware hybrid
python experiments/monthly/evaluate_ensemble.py
python experiments/monthly/evaluate_hybrid.py

# One-time held-out test evaluation of the frozen rule
python experiments/monthly/evaluate_frozen_hybrid_test.py
```

The model-training script supports Apple MPS, CUDA and CPU selection (`--device auto|mps|cuda|cpu`). The held-out test evaluator is separate and does not train or tune models.

## Experiment results recorded so far

Validation (84 targets, 2024):

| Predictor | MAE (Rs./quintal) | RMSE | MAPE |
|---|---:|---:|---:|
| Last-month naive | 297.26 | 485.68 | 12.87% |
| Mean of five LSTMs | 306.38 | 425.30 | 12.24% |
| Median of five LSTMs | 300.50 | 422.65 | 12.03% |
| Validation-selected crop-aware hybrid | 269.57 | 412.34 | 11.14% |

The frozen routing rule uses the median of five LSTMs for onion and tomato, and last-month naive for the other five crops. The rule was selected using validation results and then evaluated once on the untouched test period.

Held-out test (154 targets, 2025-01 through 2026-10):

| Predictor | MAE (Rs./quintal) | RMSE | MAPE |
|---|---:|---:|---:|
| Last-month naive | 240.96 | 355.36 | 10.91% |
| Median LSTM ensemble | 280.25 | 421.38 | 11.20% |
| Frozen crop-aware hybrid | 221.22 | 325.90 | 9.97% |

The frozen hybrid reduced test MAE by 8.19% versus the last-month naive baseline. This is a historical held-out result, not a guarantee of future accuracy or production readiness. Do not tune the frozen strategy using the test results. Establish forward evaluation on newly collected observations before deployment.

## Reproducibility and integrity notes

- Keep the daily forecasting implementation and its trained models unchanged during this experiment.
- Do not mix state-monthly weighted-average prices with market-level modal prices in a single target series.
- Exclude missing/`NR` reported prices from metric calculations; never convert them to zero.
- Report crop-level metrics as well as aggregate metrics.
- Preserve chronological splits and training-only scaler fitting.
- The test set must remain a final evaluation set. Any further model or routing selection should use a new forward/rolling validation design, not repeated tuning against this test period.
- Before commercial use, confirm attribution and licensing requirements for the upstream dataset.
