#!/usr/bin/env python3
"""Validate outputs from clean_india_data.py and create_sequences.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/training/monthly_maharashtra")
    args = parser.parse_args()
    root = Path(args.training_dir)

    with open(root / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(root / "feature_schema.json", encoding="utf-8") as f:
        schema = json.load(f)

    expected_features = len(schema["feature_names"])
    lookback = int(schema["lookback_months"])
    boundaries = {
        "train": summary["train_end_inclusive"],
        "val": summary["validation_end_inclusive"],
    }

    for split in ("train", "val", "test"):
        X = np.load(root / f"X_{split}.npy", allow_pickle=False)
        y = np.load(root / f"y_{split}.npy", allow_pickle=False)
        metadata = pd.read_csv(root / f"metadata_{split}.csv")

        if X.ndim != 3 or X.shape[1:] != (lookback, expected_features):
            raise AssertionError(f"{split}: unexpected X shape {X.shape}")
        if y.shape != (len(X), 1):
            raise AssertionError(f"{split}: unexpected y shape {y.shape}")
        if len(metadata) != len(X):
            raise AssertionError(f"{split}: X/y/metadata row counts differ")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise AssertionError(f"{split}: arrays contain NaN or infinity")

        dates = pd.PeriodIndex(metadata["target_month"], freq="M")
        if split == "train" and (dates > pd.Period(boundaries["train"], freq="M")).any():
            raise AssertionError("Train split contains target dates after train_end")
        if split == "val":
            train_end = pd.Period(boundaries["train"], freq="M")
            val_end = pd.Period(boundaries["val"], freq="M")
            if ((dates <= train_end) | (dates > val_end)).any():
                raise AssertionError("Validation split contains out-of-range target dates")
        if split == "test" and (dates <= pd.Period(boundaries["val"], freq="M")).any():
            raise AssertionError("Test split contains target dates at or before validation end")

        print(f"{split:5}: X={X.shape}, y={y.shape}, metadata={len(metadata)}")

    print("\nValidation passed: array shapes, finite values and chronological split boundaries are consistent.")
    print("Note: this checks preprocessing outputs; it does not establish model accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
