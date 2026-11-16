"""Read-only crop readiness matrix for CropBalance AI.

The matrix separates catalog membership, provider ID mappings, stored
observations, local model artifacts, and explicit forecast validation. It
never treats a checkpoint file or an unverified price value as a validated
forecast capability.
"""
from __future__ import annotations

import csv
import os
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from services.local_mandi_service import DEFAULT_ARCHIVE_PATH
from services.reference_registry import ReferenceRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DEFAULT_SCALERS_DIR = PROJECT_ROOT / "models" / "scalers"

_REQUIRED_ARCHIVE_COLUMNS = {
    "observation_date",
    "market",
    "market_id",
    "crop",
    "reported_price",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _text(value).casefold() in {"true", "1", "yes", "y"}


def _positive_decimal(value: Any) -> Decimal | None:
    raw = _text(value).replace(",", "")
    if not raw:
        return None
    try:
        parsed = Decimal(raw)
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed <= 0:
        return None
    return parsed


def _iso_date(value: Any) -> date | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _empty_observation_stats() -> dict[str, Any]:
    return {
        "observation_rows": 0,
        "numeric_price_rows": 0,
        "semantics_verified_numeric_price_rows": 0,
        "semantics_unverified_numeric_price_rows": 0,
        "conflict_rows": 0,
        "markets": set(),
        "price_units": set(),
        "sources": set(),
        "dates": [],
    }


class CropReadinessService:
    """Build a read-only per-crop readiness report from existing local assets."""

    def __init__(
        self,
        *,
        reference_registry: ReferenceRegistry | None = None,
        archive_path: str | Path | None = None,
        models_dir: str | Path | None = None,
        scalers_dir: str | Path | None = None,
    ) -> None:
        self.reference_registry = reference_registry or ReferenceRegistry()
        configured_archive = (
            archive_path
            or os.getenv("CROPBALANCE_MANDI_ARCHIVE")
            or DEFAULT_ARCHIVE_PATH
        )
        self.archive_path = Path(configured_archive)
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.scalers_dir = Path(scalers_dir or DEFAULT_SCALERS_DIR)

    def _resolve_crop_code(self, row: dict[str, str]) -> str | None:
        # The canonical crop field takes precedence. The source commodity label
        # is a fallback for older archive snapshots that do not populate crop.
        for candidate in (row.get("crop"), row.get("commodity_name")):
            candidate = _text(candidate)
            if not candidate:
                continue
            try:
                return self.reference_registry.resolve("crops", candidate)["code"]
            except ValueError:
                continue
        return None

    def _read_archive(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if not self.archive_path.is_file():
            return {}, {
                "status": "not_available",
                "filename": self.archive_path.name,
                "rows_read": 0,
                "recognized_crop_rows": 0,
                "note": "The configured local mandi archive file was not found.",
            }

        stats_by_crop: dict[str, dict[str, Any]] = {}
        rows_read = 0
        recognized_crop_rows = 0

        try:
            with self.archive_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                headers = set(reader.fieldnames or [])
                missing = sorted(_REQUIRED_ARCHIVE_COLUMNS - headers)
                if missing:
                    return {}, {
                        "status": "invalid_schema",
                        "filename": self.archive_path.name,
                        "rows_read": 0,
                        "recognized_crop_rows": 0,
                        "missing_columns": missing,
                        "note": "The archive does not match the expected observation schema.",
                    }

                for row in reader:
                    rows_read += 1
                    crop_code = self._resolve_crop_code(row)
                    if crop_code is None:
                        continue

                    recognized_crop_rows += 1
                    stats = stats_by_crop.setdefault(
                        crop_code, _empty_observation_stats()
                    )
                    stats["observation_rows"] += 1

                    price = _positive_decimal(row.get("reported_price"))
                    if price is not None:
                        stats["numeric_price_rows"] += 1
                        if _truthy(row.get("price_semantics_verified")):
                            stats["semantics_verified_numeric_price_rows"] += 1
                        else:
                            stats["semantics_unverified_numeric_price_rows"] += 1

                    if _truthy(row.get("conflict_flag")):
                        stats["conflict_rows"] += 1

                    market_key = _text(row.get("market_id")) or _text(row.get("market"))
                    if market_key:
                        stats["markets"].add(market_key)

                    unit = _text(row.get("price_unit"))
                    if unit:
                        stats["price_units"].add(unit)

                    source = _text(row.get("data_source"))
                    if source:
                        stats["sources"].add(source)

                    observed = _iso_date(row.get("observation_date"))
                    if observed is not None:
                        stats["dates"].append(observed)

        except (OSError, UnicodeError, csv.Error):
            return {}, {
                "status": "unreadable",
                "filename": self.archive_path.name,
                "rows_read": 0,
                "recognized_crop_rows": 0,
                "note": "The configured local mandi archive could not be read.",
            }

        return stats_by_crop, {
            "status": "available",
            "filename": self.archive_path.name,
            "rows_read": rows_read,
            "recognized_crop_rows": recognized_crop_rows,
            "note": (
                "This is the configured local research archive. Its records "
                "are not automatically national coverage or model-ready targets."
            ),
        }

    def _model_artifacts(self, crop_code: str) -> dict[str, Any]:
        checkpoint = self.models_dir / f"best_{crop_code}.pt"
        scaler = self.scalers_dir / f"scalers_{crop_code}.pkl"
        checkpoint_present = checkpoint.is_file()
        scaler_present = scaler.is_file()

        if checkpoint_present and scaler_present:
            status = "complete"
        elif checkpoint_present or scaler_present:
            status = "partial"
        else:
            status = "missing"

        return {
            "status": status,
            "checkpoint_present": checkpoint_present,
            "scaler_present": scaler_present,
            "checkpoint_filename": checkpoint.name if checkpoint_present else None,
            "scaler_filename": scaler.name if scaler_present else None,
        }

    @staticmethod
    def _serialise_observations(
        stats: dict[str, Any] | None,
        archive_status: str,
    ) -> dict[str, Any]:
        stats = stats or _empty_observation_stats()
        dates = sorted(stats["dates"])
        observation_count = stats["observation_rows"]
        verified_count = stats["semantics_verified_numeric_price_rows"]

        if observation_count:
            status = "available"
        elif archive_status == "available":
            status = "no_matching_rows"
        else:
            status = "unavailable"

        return {
            "status": status,
            "observation_rows": observation_count,
            "numeric_price_rows": stats["numeric_price_rows"],
            "semantics_verified_numeric_price_rows": verified_count,
            "semantics_unverified_numeric_price_rows": (
                stats["semantics_unverified_numeric_price_rows"]
            ),
            "conflict_rows": stats["conflict_rows"],
            "market_count": len(stats["markets"]),
            "price_units": sorted(stats["price_units"], key=str.casefold),
            "sources": sorted(stats["sources"], key=str.casefold),
            "first_observation_date": dates[0].isoformat() if dates else None,
            "last_observation_date": dates[-1].isoformat() if dates else None,
            "has_verified_price_targets": verified_count > 0,
        }

    @staticmethod
    def _readiness_stage(
        *,
        source_mapped: bool,
        observations: dict[str, Any],
        model_artifacts: dict[str, Any],
        validated_model_available: bool,
    ) -> str:
        if validated_model_available:
            return "forecast_validated"
        if observations["semantics_verified_numeric_price_rows"] > 0:
            return "verified_price_observations_available"
        if observations["observation_rows"] > 0:
            return "observations_present_semantics_unverified"
        if model_artifacts["status"] != "missing":
            return "model_artifacts_present_unvalidated"
        if source_mapped:
            return "source_mapped"
        return "catalog_only"

    def report(self, *, crop: str | None = None) -> dict[str, Any]:
        """Return all crop readiness states, optionally filtered by crop alias."""
        crop_items = self.reference_registry.list_crops(language="en")
        normalized_filter = None
        if crop is not None:
            resolved = self.reference_registry.resolve("crops", crop)
            normalized_filter = resolved["code"]
            crop_items = [
                item for item in crop_items if item["code"] == normalized_filter
            ]

        archive_stats, archive = self._read_archive()
        results: list[dict[str, Any]] = []

        for item in crop_items:
            crop_code = item["code"]
            source_ids = item.get("source_ids") or {}
            model_artifacts = self._model_artifacts(crop_code)
            observations = self._serialise_observations(
                archive_stats.get(crop_code),
                archive["status"],
            )

            # A checkpoint and scaler are necessary for local inference but
            # are not evidence that forecast accuracy has been validated.
            validated_model = {
                "available": False,
                "status": "not_validated",
                "reason": (
                    "No approved temporal holdout and baseline evaluation is "
                    "registered for this crop. Artifact presence or an "
                    "inference smoke test alone does not establish forecast quality."
                ),
            }

            source_mapped = bool(source_ids)
            results.append({
                "crop_code": crop_code,
                "name": item["display_name"],
                "category": item.get("category"),
                "catalog": {
                    "present": True,
                    "status": item.get("catalog_status", "catalogued"),
                },
                "source_mapping": {
                    "status": "mapped" if source_mapped else "not_mapped",
                    "source_ids": source_ids,
                },
                "observations": observations,
                "model_artifacts": model_artifacts,
                "validated_model": validated_model,
                "readiness_stage": self._readiness_stage(
                    source_mapped=source_mapped,
                    observations=observations,
                    model_artifacts=model_artifacts,
                    validated_model_available=False,
                ),
            })

        summary = {
            "catalog_crops": len(results),
            "crops_with_source_mapping": sum(
                1 for item in results
                if item["source_mapping"]["status"] == "mapped"
            ),
            "crops_with_observations": sum(
                1 for item in results
                if item["observations"]["observation_rows"] > 0
            ),
            "crops_with_verified_price_observations": sum(
                1 for item in results
                if item["observations"]["has_verified_price_targets"]
            ),
            "crops_with_complete_model_artifacts": sum(
                1 for item in results
                if item["model_artifacts"]["status"] == "complete"
            ),
            "crops_with_validated_models": sum(
                1 for item in results if item["validated_model"]["available"]
            ),
        }

        return {
            "status": "ok",
            "report": "crop_data_readiness",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filter": {"crop": normalized_filter},
            "archive": archive,
            "summary": summary,
            "items": results,
            "interpretation": [
                "Catalog presence means the crop exists in the starter taxonomy only.",
                "A source mapping is provider-specific and does not imply observations exist.",
                "The local archive is a Dharashiv research dataset, not national market coverage.",
                "Only positive numeric prices explicitly marked price_semantics_verified count as verified price targets.",
                "Model checkpoint and scaler files indicate artifact availability, not validated forecast accuracy.",
                "The national market_price_observations database is reported separately by /market-prices/coverage.",
            ],
        }
