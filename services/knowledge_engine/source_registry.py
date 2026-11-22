import json

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from services.knowledge_engine.schemas import (
    ApprovalStatus,
    DocumentMetadata,
    LanguageReviewStatus,
)


SUPPORTED_SCHEMA_VERSION = 1


class SourceRegistryError(ValueError):
    """Raised when a source manifest is invalid."""


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Validated collection of agricultural source documents."""

    schema_version: int
    documents: tuple[DocumentMetadata, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
        ):
            raise SourceRegistryError(
                "schema_version must be an integer."
            )

        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise SourceRegistryError(
                f"Unsupported schema version: {self.schema_version}"
            )

        if not isinstance(self.documents, tuple):
            raise SourceRegistryError(
                "documents must be a tuple."
            )

        seen_ids = set()

        for document in self.documents:
            if not isinstance(document, DocumentMetadata):
                raise SourceRegistryError(
                    "All registry entries must be DocumentMetadata."
                )

            if document.document_id in seen_ids:
                raise SourceRegistryError(
                    f"Duplicate document ID: {document.document_id}"
                )

            seen_ids.add(document.document_id)

    def get_document(
        self,
        document_id: str,
    ) -> DocumentMetadata | None:
        """Find a document by its unique ID."""

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "document_id must be a non-empty string."
            )

        for document in self.documents:
            if document.document_id == document_id:
                return document

        return None


_ALLOWED_MANIFEST_FIELDS = {
    "schema_version",
    "documents",
}

_ALLOWED_DOCUMENT_FIELDS = {
    "document_id",
    "title",
    "publisher",
    "languages",
    "source_url",
    "landing_page",
    "local_filename",
    "publication_year",
    "publication_date",
    "document_type",
    "crop_codes",
    "country",
    "region_scope",
    "language_status",
    "approval_status",
}

_TUPLE_FIELDS = {
    "languages",
    "crop_codes",
    "region_scope",
}


def _parse_enum(value, enum_type, field_name):
    """Convert a manifest string into an enum value."""

    if not isinstance(value, str):
        raise SourceRegistryError(
            f"{field_name} must be a string."
        )

    try:
        return enum_type(value)
    except ValueError as exc:
        raise SourceRegistryError(
            f"Invalid {field_name}: {value}"
        ) from exc


def _parse_document(
    raw_document: dict,
    index: int,
) -> DocumentMetadata:
    """Validate and convert one manifest document."""

    location = f"documents[{index}]"

    if not isinstance(raw_document, dict):
        raise SourceRegistryError(
            f"{location} must be a JSON object."
        )

    unknown_fields = (
        set(raw_document) - _ALLOWED_DOCUMENT_FIELDS
    )

    if unknown_fields:
        raise SourceRegistryError(
            f"{location} contains unknown fields: "
            f"{sorted(unknown_fields)}"
        )

    required_fields = {
        "document_id",
        "title",
        "publisher",
    }

    missing_fields = required_fields - set(raw_document)

    if missing_fields:
        raise SourceRegistryError(
            f"{location} is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    document_data = dict(raw_document)

    # JSON arrays become immutable tuples in DocumentMetadata.
    for field_name in _TUPLE_FIELDS:
        if field_name not in document_data:
            continue

        value = document_data[field_name]

        if not isinstance(value, list):
            raise SourceRegistryError(
                f"{location}.{field_name} must be a JSON array."
            )

        document_data[field_name] = tuple(value)

    # Convert status strings into their corresponding enums.
    document_data["approval_status"] = _parse_enum(
        document_data.get(
            "approval_status",
            ApprovalStatus.PENDING_REVIEW.value,
        ),
        ApprovalStatus,
        f"{location}.approval_status",
    )

    document_data["language_status"] = _parse_enum(
        document_data.get(
            "language_status",
            LanguageReviewStatus.PENDING_REVIEW.value,
        ),
        LanguageReviewStatus,
        f"{location}.language_status",
    )

    # Convert an optional ISO date string into datetime.date.
    if "publication_date" in document_data:
        raw_date = document_data["publication_date"]

        if not isinstance(raw_date, str):
            raise SourceRegistryError(
                f"{location}.publication_date must be an ISO date string."
            )

        try:
            document_data["publication_date"] = (
                date.fromisoformat(raw_date)
            )
        except ValueError as exc:
            raise SourceRegistryError(
                f"{location}.publication_date is invalid: {raw_date}"
            ) from exc

    try:
        return DocumentMetadata(**document_data)

    except (TypeError, ValueError) as exc:
        raise SourceRegistryError(
            f"Invalid metadata in {location}: {exc}"
        ) from exc


def load_source_registry(
    manifest_path: str | Path,
) -> SourceRegistry:
    """Load and validate an agricultural source manifest."""

    path = Path(manifest_path)

    try:
        raw_content = path.read_text(encoding="utf-8")
        manifest = json.loads(raw_content)

    except (OSError, UnicodeError) as exc:
        raise SourceRegistryError(
            f"Cannot read source manifest: {path}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise SourceRegistryError(
            f"Invalid JSON in source manifest: {path}"
        ) from exc

    if not isinstance(manifest, dict):
        raise SourceRegistryError(
            "Manifest root must be a JSON object."
        )

    unknown_fields = set(manifest) - _ALLOWED_MANIFEST_FIELDS

    if unknown_fields:
        raise SourceRegistryError(
            f"Unknown manifest fields: {sorted(unknown_fields)}"
        )

    if "schema_version" not in manifest:
        raise SourceRegistryError(
            "Manifest is missing schema_version."
        )

    if "documents" not in manifest:
        raise SourceRegistryError(
            "Manifest is missing documents."
        )

    version = manifest["schema_version"]

    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SUPPORTED_SCHEMA_VERSION
    ):
        raise SourceRegistryError(
            f"Unsupported schema version: {version}"
        )

    raw_documents = manifest["documents"]

    if not isinstance(raw_documents, list):
        raise SourceRegistryError(
            "documents must be a JSON array."
        )

    documents = tuple(
        _parse_document(raw_document, index)
        for index, raw_document in enumerate(raw_documents)
    )

    return SourceRegistry(
        schema_version=version,
        documents=documents,
    )