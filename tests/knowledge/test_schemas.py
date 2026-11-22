
from datetime import date, datetime

import pytest

from services.knowledge_engine.schemas import (
    ApprovalStatus,
    ContentType,
    DocumentMetadata,
    EvidenceResult,
    ExtractionMethod,
    KnowledgeChunk,
    LanguageReviewStatus,
)


@pytest.fixture
def approved_document():
    return DocumentMetadata(
        document_id="icar_nsri_soybean_2026",
        title="Soybean Technical Bulletin 2026",
        publisher="ICAR-NSRI",
        source_url="https://example.org/soybean.pdf",
        landing_page="https://example.org/extension",
        local_filename="icar_nsri_soybean_2026.pdf",
        publication_year=2026,
        document_type="technical_bulletin",
        crop_codes=("soybean",),
        country="India",
        region_scope=("India",),
        approval_status=ApprovalStatus.APPROVED,
    )


@pytest.fixture
def approved_chunk():
    return KnowledgeChunk(
        chunk_id="soybean_chunk_001",
        document_id="icar_nsri_soybean_2026",
        content="Example agricultural knowledge passage.",
        page_start=18,
        page_end=18,
        language="hi-IN",
        content_type=ContentType.PARAGRAPH,
        extraction_method=ExtractionMethod.APPLE_VISION_OCR,
        crop_codes=("soybean",),
        region_scope=("India",),
        approval_status=ApprovalStatus.APPROVED,
    )


def test_document_defaults_to_pending_review():
    document = DocumentMetadata(
        document_id="test_document",
        title="Test Agricultural Document",
        publisher="Test Publisher",
    )

    assert (
        document.approval_status
        == ApprovalStatus.PENDING_REVIEW
    )


def test_document_requires_valid_identity():
    with pytest.raises(ValueError):
        DocumentMetadata(
            document_id="",
            title="Test Document",
            publisher="Test Publisher",
        )


def test_chunk_requires_valid_page_range():
    with pytest.raises(ValueError):
        KnowledgeChunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            content="Example text.",
            page_start=10,
            page_end=5,
            language="en-US",
        )


def test_chunk_defaults_to_pending_review():
    chunk = KnowledgeChunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        content="Example agricultural text.",
        page_start=1,
        page_end=1,
        language="en-US",
    )

    assert chunk.approval_status == ApprovalStatus.PENDING_REVIEW


def test_evidence_requires_matching_document_ids(
    approved_document,
    approved_chunk,
):
    invalid_chunk = KnowledgeChunk(
        chunk_id="chunk_002",
        document_id="another_document",
        content="Example text.",
        page_start=1,
        page_end=1,
        language="en-US",
    )

    with pytest.raises(ValueError):
        EvidenceResult(
            document=approved_document,
            chunk=invalid_chunk,
            retrieval_method="test",
            retrieval_score=0.5,
        )


def test_approved_evidence_is_eligible(
    approved_document,
    approved_chunk,
):
    evidence = EvidenceResult(
        document=approved_document,
        chunk=approved_chunk,
        retrieval_method="test",
        retrieval_score=0.5,
    )

    assert evidence.eligible_for_farmer_response is True


def test_pending_document_blocks_approved_chunk(
    approved_chunk,
):
    pending_document = DocumentMetadata(
        document_id="icar_nsri_soybean_2026",
        title="Soybean Bulletin",
        publisher="ICAR-NSRI",
    )

    evidence = EvidenceResult(
        document=pending_document,
        chunk=approved_chunk,
        retrieval_method="test",
    )

    assert evidence.eligible_for_farmer_response is False


def test_pending_chunk_blocks_approved_document(
    approved_document,
):
    pending_chunk = KnowledgeChunk(
        chunk_id="chunk_pending",
        document_id="icar_nsri_soybean_2026",
        content="Pending agricultural passage.",
        page_start=1,
        page_end=1,
        language="en-US",
    )

    evidence = EvidenceResult(
        document=approved_document,
        chunk=pending_chunk,
        retrieval_method="test",
    )

    assert evidence.eligible_for_farmer_response is False


def test_retrieval_score_must_be_finite(
    approved_document,
    approved_chunk,
):
    with pytest.raises(ValueError):
        EvidenceResult(
            document=approved_document,
            chunk=approved_chunk,
            retrieval_method="test",
            retrieval_score=float("nan"),
        )


def test_document_defaults_to_pending_language_review():
    document = DocumentMetadata(
        document_id="test_document",
        title="Test Document",
        publisher="Test Publisher",
    )

    assert (
        document.language_status
        == LanguageReviewStatus.PENDING_REVIEW
    )


def test_document_preserves_manifest_metadata(approved_document):
    assert approved_document.publication_year == 2026
    assert approved_document.document_type == "technical_bulletin"
    assert (
        approved_document.local_filename
        == "icar_nsri_soybean_2026.pdf"
    )
    assert approved_document.landing_page == (
        "https://example.org/extension"
    )


def test_document_rejects_invalid_language_status():
    with pytest.raises(TypeError):
        DocumentMetadata(
            document_id="test_document",
            title="Test Document",
            publisher="Test Publisher",
            language_status="verified",
        )


def test_publication_year_rejects_boolean():
    with pytest.raises(ValueError):
        DocumentMetadata(
            document_id="test_document",
            title="Test Document",
            publisher="Test Publisher",
            publication_year=True,
        )


def test_publication_year_must_match_publication_date():
    with pytest.raises(ValueError):
        DocumentMetadata(
            document_id="test_document",
            title="Test Document",
            publisher="Test Publisher",
            publication_year=2026,
            publication_date=date(2025, 4, 1),
        )


def test_publication_date_rejects_datetime():
    with pytest.raises(TypeError):
        DocumentMetadata(
            document_id="test_document",
            title="Test Document",
            publisher="Test Publisher",
            publication_date=datetime(2026, 4, 1, 10, 30),
        )


def test_chunk_page_numbers_reject_booleans():
    with pytest.raises(ValueError):
        KnowledgeChunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            content="Example text.",
            page_start=True,
            page_end=1,
            language="en-US",
        )


def test_retrieval_score_rejects_boolean(
    approved_document,
    approved_chunk,
):
    with pytest.raises(ValueError):
        EvidenceResult(
            document=approved_document,
            chunk=approved_chunk,
            retrieval_method="test",
            retrieval_score=True,
        )