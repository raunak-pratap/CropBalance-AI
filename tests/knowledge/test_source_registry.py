import json

from pathlib import Path

import pytest

from services.knowledge_engine.schemas import (
    ApprovalStatus,
    LanguageReviewStatus,
)

from services.knowledge_engine.source_registry import (
    SourceRegistryError,
    load_source_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "manifests"
    / "soybean_sources.json"
)


def write_manifest(tmp_path, data):
    path = tmp_path / "test_manifest.json"

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def valid_manifest():
    return {
        "schema_version": 1,
        "documents": [
            {
                "document_id": "icar_nsri_soybean_2026",
                "title": "Soybean Technical Bulletin",
                "publisher": "ICAR-NSRI",
                "publication_year": 2026,
                "document_type": "technical_bulletin",
                "crop_codes": ["soybean"],
                "region_scope": ["India"],
                "source_url": "https://example.org/soybean.pdf",
                "approval_status": "pending_review",
                "language_status": "pending_review",
            }
        ],
    }


def test_loads_real_soybean_manifest():
    registry = load_source_registry(REAL_MANIFEST)

    assert registry.schema_version == 1
    assert len(registry.documents) == 1

    document = registry.get_document(
        "icar_nsri_soybean_2026"
    )

    assert document is not None
    assert document.publication_year == 2026
    assert document.document_type == "technical_bulletin"
    assert document.crop_codes == ("soybean",)
    assert document.region_scope == ("India",)


def test_document_remains_pending_approval():
    registry = load_source_registry(REAL_MANIFEST)

    document = registry.get_document(
        "icar_nsri_soybean_2026"
    )

    assert document.approval_status == ApprovalStatus.PENDING_REVIEW
    assert (
        document.language_status
        == LanguageReviewStatus.PENDING_REVIEW
    )


def test_get_document_returns_none_for_unknown_id():
    registry = load_source_registry(REAL_MANIFEST)

    assert registry.get_document("unknown_document") is None


def test_rejects_missing_manifest(tmp_path):
    with pytest.raises(SourceRegistryError):
        load_source_registry(tmp_path / "missing.json")


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_rejects_unsupported_schema_version(
    tmp_path,
    valid_manifest,
):
    valid_manifest["schema_version"] = 2

    path = write_manifest(tmp_path, valid_manifest)

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_rejects_duplicate_document_ids(
    tmp_path,
    valid_manifest,
):
    duplicate = dict(valid_manifest["documents"][0])
    valid_manifest["documents"].append(duplicate)

    path = write_manifest(tmp_path, valid_manifest)

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_rejects_missing_required_document_field(
    tmp_path,
    valid_manifest,
):
    del valid_manifest["documents"][0]["publisher"]

    path = write_manifest(tmp_path, valid_manifest)

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_rejects_invalid_approval_status(
    tmp_path,
    valid_manifest,
):
    valid_manifest["documents"][0]["approval_status"] = "verified"

    path = write_manifest(tmp_path, valid_manifest)

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_rejects_non_array_crop_codes(
    tmp_path,
    valid_manifest,
):
    valid_manifest["documents"][0]["crop_codes"] = "soybean"

    path = write_manifest(tmp_path, valid_manifest)

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)


def test_rejects_unknown_document_fields(
    tmp_path,
    valid_manifest,
):
    valid_manifest["documents"][0]["apprval_status"] = "approved"

    path = write_manifest(tmp_path, valid_manifest)

    with pytest.raises(SourceRegistryError):
        load_source_registry(path)