import csv
from pathlib import Path

import pytest

from services.crop_readiness_service import CropReadinessService


FIELDS = [
    "observation_date",
    "market",
    "market_id",
    "crop",
    "reported_price",
    "commodity_name",
    "price_semantics_verified",
    "price_unit",
    "data_source",
    "conflict_flag",
]


def observation(
    *,
    crop="soyabean",
    observation_date="2026-11-13",
    market="APMC Kalamb",
    market_id="1599",
    reported_price="7100",
    semantics_verified="False",
    price_unit="Rs./Quintal",
    data_source="agmarknet",
    conflict_flag="False",
):
    return {
        "observation_date": observation_date,
        "market": market,
        "market_id": market_id,
        "crop": crop,
        "reported_price": reported_price,
        "commodity_name": "",
        "price_semantics_verified": semantics_verified,
        "price_unit": price_unit,
        "data_source": data_source,
        "conflict_flag": conflict_flag,
    }


def write_archive(path: Path, rows, fieldnames=FIELDS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def find_item(report, crop_code):
    return next(item for item in report["items"] if item["crop_code"] == crop_code)


def test_readiness_separates_mapping_observations_artifacts_and_validation(tmp_path):
    archive = write_archive(
        tmp_path / "archive.csv",
        [
            observation(
                observation_date="2026-11-12",
                reported_price="7000",
                semantics_verified="False",
            ),
            observation(
                observation_date="2026-11-13",
                crop="soybean",
                reported_price="7100",
                semantics_verified="True",
            ),
        ],
    )
    models = tmp_path / "models"
    scalers = tmp_path / "scalers"
    models.mkdir()
    scalers.mkdir()
    (models / "best_soybean.pt").write_bytes(b"checkpoint-fixture")
    (scalers / "scalers_soybean.pkl").write_bytes(b"scaler-fixture")

    report = CropReadinessService(
        archive_path=archive,
        models_dir=models,
        scalers_dir=scalers,
    ).report(crop="soyabean")

    assert report["summary"]["catalog_crops"] == 1
    assert report["archive"]["status"] == "available"

    soybean = find_item(report, "soybean")
    assert soybean["catalog"]["present"] is True
    assert soybean["source_mapping"]["status"] == "mapped"
    assert soybean["source_mapping"]["source_ids"]["agmarknet"] == "13"

    observations = soybean["observations"]
    assert observations["observation_rows"] == 2
    assert observations["numeric_price_rows"] == 2
    assert observations["semantics_verified_numeric_price_rows"] == 1
    assert observations["semantics_unverified_numeric_price_rows"] == 1
    assert observations["market_count"] == 1
    assert observations["first_observation_date"] == "2026-11-12"
    assert observations["last_observation_date"] == "2026-11-13"
    assert observations["has_verified_price_targets"] is True

    assert soybean["model_artifacts"]["status"] == "complete"
    assert soybean["model_artifacts"]["checkpoint_present"] is True
    assert soybean["model_artifacts"]["scaler_present"] is True
    assert soybean["validated_model"]["available"] is False
    assert soybean["validated_model"]["status"] == "not_validated"
    assert soybean["readiness_stage"] == "verified_price_observations_available"


def test_unverified_local_archive_rows_are_not_reported_as_verified_targets(tmp_path):
    archive = write_archive(
        tmp_path / "archive.csv",
        [observation(semantics_verified="False")],
    )
    report = CropReadinessService(archive_path=archive).report(crop="soybean")
    soybean = find_item(report, "soybean")

    assert soybean["observations"]["observation_rows"] == 1
    assert soybean["observations"]["numeric_price_rows"] == 1
    assert soybean["observations"]["semantics_verified_numeric_price_rows"] == 0
    assert soybean["observations"]["has_verified_price_targets"] is False
    assert soybean["readiness_stage"] == "observations_present_semantics_unverified"


def test_missing_archive_is_reported_without_failing_catalog_readiness(tmp_path):
    report = CropReadinessService(
        archive_path=tmp_path / "missing.csv",
        models_dir=tmp_path / "models",
        scalers_dir=tmp_path / "scalers",
    ).report(crop="soyabean")

    soybean = find_item(report, "soybean")
    assert report["archive"]["status"] == "not_available"
    assert soybean["observations"]["status"] == "unavailable"
    assert soybean["observations"]["observation_rows"] == 0
    assert soybean["source_mapping"]["status"] == "mapped"
    assert soybean["readiness_stage"] == "source_mapped"


def test_catalog_only_crop_is_not_assumed_to_have_a_source_mapping(tmp_path):
    report = CropReadinessService(
        archive_path=tmp_path / "missing.csv",
        models_dir=tmp_path / "models",
        scalers_dir=tmp_path / "scalers",
    ).report(crop="sorghum")

    sorghum = find_item(report, "sorghum")
    assert sorghum["catalog"]["present"] is True
    assert sorghum["source_mapping"]["status"] == "not_mapped"
    assert sorghum["model_artifacts"]["status"] == "missing"
    assert sorghum["validated_model"]["available"] is False
    assert sorghum["readiness_stage"] == "catalog_only"


def test_checkpoint_without_scaler_is_partial_and_not_validated(tmp_path):
    models = tmp_path / "models"
    scalers = tmp_path / "scalers"
    models.mkdir()
    scalers.mkdir()
    (models / "best_wheat.pt").write_bytes(b"checkpoint-fixture")

    report = CropReadinessService(
        archive_path=tmp_path / "missing.csv",
        models_dir=models,
        scalers_dir=scalers,
    ).report(crop="wheat")

    wheat = find_item(report, "wheat")
    assert wheat["model_artifacts"]["status"] == "partial"
    assert wheat["model_artifacts"]["checkpoint_present"] is True
    assert wheat["model_artifacts"]["scaler_present"] is False
    assert wheat["validated_model"]["available"] is False
    assert wheat["readiness_stage"] == "model_artifacts_present_unvalidated"


def test_invalid_archive_schema_is_reported_explicitly(tmp_path):
    archive = write_archive(
        tmp_path / "bad.csv",
        [{"crop": "soyabean"}],
        fieldnames=["crop"],
    )
    report = CropReadinessService(archive_path=archive).report(crop="soybean")

    soybean = find_item(report, "soybean")
    assert report["archive"]["status"] == "invalid_schema"
    assert "observation_date" in report["archive"]["missing_columns"]
    assert soybean["observations"]["status"] == "unavailable"


def test_unknown_crop_filter_is_rejected(tmp_path):
    service = CropReadinessService(archive_path=tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="not found in the crops registry"):
        service.report(crop="made-up-crop")
