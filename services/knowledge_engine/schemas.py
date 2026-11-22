from dataclasses import dataclass 
from datetime import date
from enum import Enum, StrEnum
from math import isfinite

class ApprovalStatus(Enum):
    """Approval state of a document or knowledge chunk."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class LanguageReviewStatus(StrEnum):
    """Review state of the source document's language metadata."""

    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    NEEDS_CORRECTION = "needs_correction"

class ContentType(StrEnum):

    """Type of content represented by a knowledge chunk. """

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    TABLE_ROW = "table_row"
    LIST_ITEM = "list_item"

class ExtractionMethod(StrEnum):

    """Method used to extract the original content"""

    PDF_TEXT = "pdf_text"
    APPLE_VISION_OCR = "apple_vision_ocr"
    MANUAL_TRANSCRIPTION = "manual_transcription"

def _require_non_empty(value: str, field_name: str) -> None:
    """Validate required string fields."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    """Validate collection containing string metadata."""

    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple. ")

    for value in values:
        _require_non_empty(value, field_name)


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata describing an original agricultural source document.
    
    A document represents the source publication, not an individual 
    extracted passage. 
    """

    document_id: str
    title: str
    publisher: str

    languages: tuple[str, ...] = ()

    source_url: str | None = None
    landing_page: str | None = None
    local_filename: str | None = None

    publication_year: int | None = None
    publication_date: date | None = None
    document_type: str | None = None

    crop_codes: tuple[str, ...] = ()
    country: str | None = None
    region_scope: tuple[str, ...] = ()

    language_status: LanguageReviewStatus = (
        LanguageReviewStatus.PENDING_REVIEW
    )

    approval_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW

    def __post_init__(self) -> None:
        _require_non_empty(self.document_id, "document_id")
        _require_non_empty(self.title, "title")
        _require_non_empty(self.publisher, "publisher")

        _validate_string_tuple(self.languages, "languages")
        _validate_string_tuple(self.crop_codes, "crop_codes")
        _validate_string_tuple(self.region_scope, "region_scope")


        if (
            self.publication_date is not None
            and not isinstance(self.publication_date, date)
        ):
            raise TypeError("publication_date must be a date.")

        if not isinstance(self.approval_status, ApprovalStatus):
            raise TypeError(
                "approval_status must be an ApprovalStatus value."
            )

        optional_strings = {
            "source_url": self.source_url,
            "landing_page": self.landing_page,
            "local_filename": self.local_filename,
            "document_type": self.document_type,
            "country": self.country,
        }

        for field_name, value in optional_strings.items():
            if value is not None:
                _require_non_empty(value, field_name)

        if self.publication_year is not None:
            if (
                isinstance(self.publication_year, bool)
                or not isinstance(self.publication_year, int)
                or self.publication_year < 1
            ):
                raise ValueError(
                    "publication_year must be a positive integer."
                )

        if self.publication_date is not None:
            if type(self.publication_date) is not date:
                raise TypeError(
                    "publication_date must be a date, not datetime."
                )

            if (
                self.publication_year is not None
                and self.publication_date.year != self.publication_year
            ):
                raise ValueError(
                    "publication_year must match publication_date.year."
                )

        if not isinstance(
            self.language_status,
            LanguageReviewStatus,
        ):
            raise TypeError(
                "language_status must be a LanguageReviewStatus value."
            )


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """
    An extracted passage from an agricultural source document.

    Page references and source observation references preserve
    traceability to the original document.
    """

    chunk_id: str
    document_id: str
    content: str

    page_start: int
    page_end: int
    language: str

    content_type: ContentType = ContentType.PARAGRAPH
    extraction_method: ExtractionMethod = ExtractionMethod.PDF_TEXT

    crop_codes: tuple[str, ...] = ()
    region_scope: tuple[str, ...] = ()

    source_observation_refs: tuple[str, ...] = ()

    approval_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW

    def __post_init__(self) -> None:
        _require_non_empty(self.chunk_id, "chunk_id")
        _require_non_empty(self.document_id, "document_id")
        _require_non_empty(self.content, "content")
        _require_non_empty(self.language, "language")

        if (
            isinstance(self.page_start, bool)
            or not isinstance(self.page_start, int)
            or self.page_start < 1
        ):
            raise ValueError("page_start must be a positive integer.")

        if (
            isinstance(self.page_end, bool)
            or not isinstance(self.page_end, int)
            or self.page_end < 1
        ):
            raise ValueError("page_end must be a positive integer.")

        if self.page_end < self.page_start:
            raise ValueError(
                "page_end cannot be smaller than page_start."
            )

        _validate_string_tuple(self.crop_codes, "crop_codes")
        _validate_string_tuple(self.region_scope, "region_scope")
        _validate_string_tuple(
            self.source_observation_refs,
            "source_observation_refs",
        )

        if not isinstance(self.content_type, ContentType):
            raise TypeError("Invalid content_type.")

        if not isinstance(self.extraction_method, ExtractionMethod):
            raise TypeError("Invalid extraction_method.")

        if not isinstance(self.approval_status, ApprovalStatus):
            raise TypeError("Invalid approval_status.")


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """
    A retrieved knowledge passage together with its source document.

    retrieval_score is a retrieval-system score or distance.
    It is not an OCR confidence score or scientific correctness score.
    """

    document: DocumentMetadata
    chunk: KnowledgeChunk

    retrieval_method: str
    retrieval_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document, DocumentMetadata):
            raise TypeError("document must be DocumentMetadata.")

        if not isinstance(self.chunk, KnowledgeChunk):
            raise TypeError("chunk must be KnowledgeChunk.")

        if self.document.document_id != self.chunk.document_id:
            raise ValueError(
                "Document and chunk IDs must match."
            )

        _require_non_empty(
            self.retrieval_method,
            "retrieval_method",
        )

        if self.retrieval_score is not None:
            if (
                isinstance(self.retrieval_score, bool)
                or not isinstance(self.retrieval_score, (int, float))
                or not isfinite(self.retrieval_score)
            ):
                raise ValueError(
                    "retrieval_score must be a finite number."
                )

    @property
    def eligible_for_farmer_response(self) -> bool:
        """
        Both the source document and its chunk must be approved.

        Retrieval similarity alone never grants approval.
        """

        return (
            self.document.approval_status == ApprovalStatus.APPROVED
            and self.chunk.approval_status == ApprovalStatus.APPROVED
        )
