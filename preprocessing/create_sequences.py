#!/usr/bin/env python3
"""Create leakage-safe monthly forecasting windows for CropBalance AI.

Default experiment: seven core crops in Maharashtra, 12-month lookback,
one-month-ahead target, chronological train/validation/test splits.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED = {"month", "crop", "state", "arrivals_t", "price_rs_qtl"}
DEFAULT_TRAIN_END = "2023-12"
DEFAULT_VAL_END = "2024-12"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def split_for_target(target_month: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp) -> str | None:
    if target_month <= train_end:
        return "train"
    if target_month <= val_end:
        return "val"
    return "test"


def safe_std(values: np.ndarray) -> float:
    std = float(np.std(values, ddof=0)) if values.size else 1.0
    return std if np.isfinite(std) and std > 1e-8 else 1.0


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed monthly file not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    frame["month"] = pd.to_datetime(frame["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if frame["month"].isna().any():
        raise ValueError("Input contains invalid month values")
    frame["price_rs_qtl"] = pd.to_numeric(frame["price_rs_qtl"], errors="coerce")
    frame["arrivals_t"] = pd.to_numeric(frame["arrivals_t"], errors="coerce")
    frame.loc[frame["price_rs_qtl"] <= 0, "price_rs_qtl"] = np.nan
    frame.loc[frame["arrivals_t"] < 0, "arrivals_t"] = np.nan

    key = ["month", "crop", "state"]
    if frame.duplicated(key).any():
        sample = frame.loc[frame.duplicated(key, keep=False), key].head(10)
        raise ValueError(f"Duplicate crop/state/month rows found:\n{sample.to_string(index=False)}")
    return frame.sort_values(["crop", "state", "month"]).reset_index(drop=True)


def make_series_windows(
    group: pd.DataFrame,
    crop_index: int,
    state_index: int,
    n_crops: int,
    n_states: int,
    lookback: int,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    min_train_points: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    crop = str(group["crop"].iloc[0])
    state = str(group["state"].iloc[0])
    series_key = f"{crop}|{state}"

    group = group.sort_values("month").set_index("month")
    full_months = pd.date_range(group.index.min(), group.index.max(), freq="MS")
    group = group.reindex(full_months)
    group.index.name = "month"

    prices = pd.to_numeric(group["price_rs_qtl"], errors="coerce").to_numpy(dtype=float)
    arrivals = pd.to_numeric(group["arrivals_t"], errors="coerce").to_numpy(dtype=float)
    months = pd.DatetimeIndex(group.index)

    train_mask = months <= train_end
    train_prices = prices[train_mask & np.isfinite(prices) & (prices > 0)]
    if len(train_prices) < min_train_points:
        return ({"train": [], "val": [], "test": []},
                {"train": [], "val": [], "test": []},
                {"crop": crop, "state": state, "series_key": series_key,
                 "status": "skipped_insufficient_train_prices",
                 "train_price_points": int(len(train_prices)),
                 "calendar_months": int(len(months))})

    price_mean = float(np.mean(train_prices))
    price_std = safe_std(train_prices)

    train_arrivals = arrivals[train_mask & np.isfinite(arrivals) & (arrivals >= 0)]
    arrivals_median = float(np.median(train_arrivals)) if len(train_arrivals) else 0.0
    # Use log1p to reduce the strong right skew typically seen in arrivals.
    arrivals_filled_for_fit = np.log1p(train_arrivals) if len(train_arrivals) else np.array([0.0])
    arrivals_mean = float(np.mean(arrivals_filled_for_fit))
    arrivals_std = safe_std(arrivals_filled_for_fit)

    # All transforms are fitted only on training-period observations.
    price_scaled = (prices - price_mean) / price_std
    arrivals_missing = ~np.isfinite(arrivals) | (arrivals < 0)
    arrivals_filled = np.where(arrivals_missing, arrivals_median, arrivals)
    arrivals_log = np.log1p(np.maximum(arrivals_filled, 0.0))
    arrivals_scaled = (arrivals_log - arrivals_mean) / arrivals_std

    month_number = months.month.to_numpy(dtype=float)
    month_sin = np.sin(2 * np.pi * month_number / 12.0)
    month_cos = np.cos(2 * np.pi * month_number / 12.0)

    crop_onehot = np.zeros(n_crops, dtype=np.float32)
    crop_onehot[crop_index] = 1.0
    state_onehot = np.zeros(n_states, dtype=np.float32)
    state_onehot[state_index] = 1.0
    static_onehot = np.concatenate([crop_onehot, state_onehot])

    numeric_features = np.column_stack([
        price_scaled,
        arrivals_scaled,
        arrivals_missing.astype(float),
        month_sin,
        month_cos,
    ])
    features = np.column_stack([
        numeric_features,
        np.repeat(static_onehot.reshape(1, -1), len(months), axis=0),
    ]).astype(np.float32)

    windows: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    targets: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    metadata: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for target_pos in range(lookback, len(months)):
        split = split_for_target(months[target_pos], train_end, val_end)
        if split is None:
            continue

        start_pos = target_pos - lookback
        input_prices = prices[start_pos:target_pos]
        target_price = prices[target_pos]

        # Never bridge a missing monthly target or a missing price in the
        # lookback window. Missing arrivals are handled with a train-only median
        # plus an explicit missingness feature.
        if len(input_prices) != lookback:
            continue
        if not np.isfinite(input_prices).all() or not np.isfinite(target_price) or target_price <= 0:
            continue

        x = features[start_pos:target_pos]
        if not np.isfinite(x).all():
            continue

        y_scaled = (target_price - price_mean) / price_std
        windows[split].append(x)
        targets[split].append(np.asarray([y_scaled], dtype=np.float32))
        metadata[split].append({
            "crop": crop,
            "state": state,
            "series_key": series_key,
            "target_month": months[target_pos].strftime("%Y-%m"),
            "target_price_rs_qtl": float(target_price),
            "target_price_scaled": float(y_scaled),
            "target_arrivals_t": (
                float(arrivals[target_pos])
                if np.isfinite(arrivals[target_pos]) and arrivals[target_pos] >= 0
                else np.nan
            ),
            "target_arrivals_missing": int(arrivals_missing[target_pos]),
        })

    series_info = {
        "crop": crop,
        "state": state,
        "series_key": series_key,
        "status": "included",
        "calendar_months": int(len(months)),
        "first_month": months.min().strftime("%Y-%m"),
        "last_month": months.max().strftime("%Y-%m"),
        "train_price_points": int(len(train_prices)),
        "price_missing_months": int(np.isnan(prices).sum()),
        "arrival_missing_months": int(arrivals_missing.sum()),
        "train_samples": len(windows["train"]),
        "val_samples": len(windows["val"]),
        "test_samples": len(windows["test"]),
    }

    scaler_info = {
        "crop": crop,
        "state": state,
        "price_mean_train": price_mean,
        "price_std_train": price_std,
        "arrivals_median_train": arrivals_median,
        "log1p_arrivals_mean_train": arrivals_mean,
        "log1p_arrivals_std_train": arrivals_std,
        "scaler_fit_end_month": train_end.strftime("%Y-%m"),
    }
    return ({k: windows[k] for k in windows},
            {k: targets[k] for k in targets},
            {"series": series_info, "scaler": scaler_info, "metadata": metadata})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/maharashtra_monthly.csv")
    parser.add_argument("--output-dir", default="data/training/monthly_maharashtra")
    parser.add_argument("--states", default="all", help="all or comma-separated state names")
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END, help="Inclusive YYYY-MM")
    parser.add_argument("--val-end", default=DEFAULT_VAL_END, help="Inclusive YYYY-MM")
    parser.add_argument("--min-train-points", type=int, default=24)
    args = parser.parse_args()

    if args.lookback < 1:
        raise ValueError("--lookback must be at least 1")
    train_end = pd.Period(args.train_end, freq="M").to_timestamp()
    val_end = pd.Period(args.val_end, freq="M").to_timestamp()
    if val_end <= train_end:
        raise ValueError("--val-end must be after --train-end")

    frame = load_and_validate(Path(args.input))
    if args.states.strip().lower() not in {"all", "*"}:
        wanted = {x.strip().casefold() for x in args.states.split(",") if x.strip()}
        frame = frame.loc[frame["state"].str.casefold().isin(wanted)].copy()
        if frame.empty:
            raise ValueError(f"No input rows matched states: {sorted(wanted)}")

    crops = sorted(frame["crop"].dropna().unique().tolist())
    states = sorted(frame["state"].dropna().unique().tolist())
    crop_to_index = {name: i for i, name in enumerate(crops)}
    state_to_index = {name: i for i, name in enumerate(states)}

    all_windows: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_targets: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_metadata: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    series_summaries: list[dict[str, Any]] = []
    scalers: dict[str, dict[str, Any]] = {}

    for (crop, state), group in frame.groupby(["crop", "state"], sort=True):
        windows, targets, extra = make_series_windows(
            group=group,
            crop_index=crop_to_index[crop],
            state_index=state_to_index[state],
            n_crops=len(crops),
            n_states=len(states),
            lookback=args.lookback,
            train_end=train_end,
            val_end=val_end,
            min_train_points=args.min_train_points,
        )
        series_summaries.append(extra["series"])
        if extra["series"]["status"] != "included":
            continue
        key = extra["series"]["series_key"]
        scalers[key] = extra["scaler"]
        for split in ("train", "val", "test"):
            all_windows[split].extend(windows[split])
            all_targets[split].extend(targets[split])
            all_metadata[split].extend(extra["metadata"][split])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_names = [
        "price_zscore_train_scaled",
        "log1p_arrivals_zscore_train_scaled",
        "arrivals_missing_flag",
        "month_sin",
        "month_cos",
    ] + [f"crop_is_{safe_name(crop)}" for crop in crops] + [
        f"state_is_{safe_name(state)}" for state in states
    ]

    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        if all_windows[split]:
            X = np.stack(all_windows[split]).astype(np.float32)
            y = np.stack(all_targets[split]).astype(np.float32)
        else:
            X = np.empty((0, args.lookback, len(feature_names)), dtype=np.float32)
            y = np.empty((0, 1), dtype=np.float32)

        np.save(output / f"X_{split}.npy", X)
        np.save(output / f"y_{split}.npy", y)
        pd.DataFrame(all_metadata[split]).to_csv(output / f"metadata_{split}.csv", index=False)
        split_counts[split] = int(len(X))
        print(f"{split:5} X={X.shape} y={y.shape}")

    pd.DataFrame(series_summaries).to_csv(output / "series_summary.csv", index=False)
    with open(output / "scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2, ensure_ascii=False)
    with open(output / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": feature_names,
            "lookback_months": args.lookback,
            "target": "next_month_price_rs_qtl_per_series_zscore",
            "crop_to_index": crop_to_index,
            "state_to_index": state_to_index,
            "scaling": "per crop-state series; fitted only on observations through train_end",
            "arrivals_transform": "log1p; missing arrivals imputed with training-period median and marked with arrivals_missing_flag",
            "price_missing_policy": "No price interpolation; omit windows with a missing price in the input window or target month",
        }, f, indent=2, ensure_ascii=False)

    summary = {
        "input": str(Path(args.input)),
        "output_dir": str(output),
        "lookback_months": args.lookback,
        "target_horizon_months": 1,
        "train_end_inclusive": train_end.strftime("%Y-%m"),
        "validation_end_inclusive": val_end.strftime("%Y-%m"),
        "test_start": (val_end + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        "min_train_points_per_series": args.min_train_points,
        "n_crops": len(crops),
        "n_states": len(states),
        "included_series": len(scalers),
        "split_samples": split_counts,
        "features_per_timestep": len(feature_names),
        "feature_names": feature_names,
        "leakage_controls": [
            "Scalers fitted only on rows dated on or before train_end",
            "Validation and test samples use preceding observed months only",
            "No random split",
            "Missing prices are not interpolated",
            "A sample is omitted if any input-window price or target price is missing",
        ],
    }
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not split_counts["train"]:
        raise RuntimeError("No training windows were created. Check input coverage and split dates.")
    if not split_counts["val"] or not split_counts["test"]:
        raise RuntimeError(
            "Validation or test split is empty. Check data coverage or adjust --train-end/--val-end."
        )

    print("\n--- SEQUENCE SUMMARY ---")
    print("Included crop-state series:", len(scalers))
    print("Feature count:", len(feature_names))
    print("Saved to:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
