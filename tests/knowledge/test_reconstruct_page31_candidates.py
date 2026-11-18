from collections import Counter

import pytest

from scripts.knowledge import reconstruct_page31_candidates as recon


def make_observation(
    text,
    x,
    y,
    height=0.01,
    confidence=0.5,
):
    """Create a synthetic OCR observation using page coordinates."""

    return {
        "sourceDocumentID": "icar_nsri_soybean_2026",
        "pdfPage": 31,
        "text": text,
        "confidence": confidence,
        "x": x,
        "yTop": y - height / 2,
        "width": 0.10,
        "height": height,
    }


@pytest.fixture
def sample_observations():
    """Minimal synthetic OCR data for reconstruction tests."""

    records = [
        # Table headers
        make_observation("क्रं.", 0.08, 0.06),
        make_observation("प्रकार", 0.17, 0.06),
        make_observation("रासायनिक नाम", 0.44, 0.06),
        make_observation("मात्रा/हेक्टे.", 0.77, 0.06),

        # Application section A
        make_observation("अ) बोवनी के", 0.12, 0.09),
        make_observation("10-12 दिन बाद", 0.12, 0.11),
        make_observation("(POE)", 0.12, 0.13),

        # Application section B
        make_observation("ब) बोवनी के", 0.12, 0.18),
        make_observation("15-20 दिन बाद", 0.12, 0.20),
        make_observation("(POE)", 0.12, 0.22),

        # Application section C
        make_observation("स) पूर्वमिश्रित", 0.12, 0.49),
        make_observation("खरपतवारनाशक", 0.12, 0.51),
        make_observation("बोवनी के", 0.12, 0.54),
        make_observation("15-20 दिन बाद", 0.12, 0.57),
        make_observation("(POE)", 0.12, 0.60),

        # Section A: wrapped chemical name
        make_observation(
            "क्लोरीम्यूरान इथाईल", 0.32, 0.10
        ),
        make_observation("36 ग्राम", 0.73, 0.10),
        make_observation(
            "25 डब्ल्यू.पी. +सर्फेक्टेन्ट", 0.32, 0.125
        ),

        # Second chemical in A
        make_observation("बेन्टाझोन 48 एस.एल.", 0.32, 0.155),
        make_observation("2.00 ली", 0.73, 0.155),

        # Section B
        make_observation(
            "इमेझेथापायर 10 एस.एल.", 0.32, 0.25
        ),
        make_observation("1.00 ली.", 0.73, 0.25),

        # Section C: quantity split over two lines
        make_observation(
            "क्विजालोफाप इथाईल +क्लोरीम्यूरान इथाईल",
            0.32,
            0.65,
        ),
        make_observation("375 मिली +", 0.73, 0.65),
        make_observation("36 ग्रा. 0.2%", 0.73, 0.675),

        # Embedded quantity that requires manual review
        make_observation(
            "फ्लूथियासेट मिथाईल O.50 ली.",
            0.32,
            0.72,
        ),
    ]

    return records


def test_identify_columns():
    assert recon.identify_column(0.08) == "SERIAL"
    assert recon.identify_column(0.17) == "APPLICATION"
    assert recon.identify_column(0.44) == "CHEMICAL"
    assert recon.identify_column(0.77) == "QUANTITY"


def test_application_sections_are_grouped(sample_observations):
    sections = recon.group_application_sections(
        sample_observations
    )

    assert len(sections) == 3

    assert "10-12 दिन बाद" in sections[0]["description"]
    assert "15-20 दिन बाद" in sections[1]["description"]
    assert "पूर्वमिश्रित" in sections[2]["description"]


def test_section_boundaries():
    sections = [
        {"section_id": "section_1"},
        {"section_id": "section_2"},
        {"section_id": "section_3"},
    ]

    assert (
        recon.find_application_section(0.10, sections)
        ["section_id"] == "section_1"
    )

    assert (
        recon.find_application_section(0.20, sections)
        ["section_id"] == "section_2"
    )

    assert (
        recon.find_application_section(0.65, sections)
        ["section_id"] == "section_3"
    )


def test_headers_are_excluded(sample_observations):
    result = recon.reconstruct_candidates(
        sample_observations
    )

    chemicals = [
        row["chemical_text_candidate"]
        for row in result["row_candidates"]
    ]

    assert len(chemicals) == 5
    assert "रासायनिक नाम" not in chemicals


def test_wrapped_chemical_name_is_preserved(sample_observations):
    result = recon.reconstruct_candidates(
        sample_observations
    )

    first_row = result["row_candidates"][0]

    assert "क्लोरीम्यूरान इथाईल" in (
        first_row["chemical_text_candidate"]
    )

    assert "25 डब्ल्यू.पी." in (
        first_row["chemical_text_candidate"]
    )

    assert any(
        "wrapped chemical-name" in flag
        for flag in first_row["review_flags"]
    )


def test_split_quantity_is_preserved(sample_observations):
    result = recon.reconstruct_candidates(
        sample_observations
    )

    row = next(
        row
        for row in result["row_candidates"]
        if row["chemical_text_candidate"].startswith(
            "क्विजालोफाप इथाईल"
        )
    )

    assert row["quantity_text_candidate"] == (
        "375 मिली + 36 ग्रा. 0.2%"
    )

    assert any(
        "continued quantity" in flag
        for flag in row["review_flags"]
    )


def test_embedded_quantity_requires_review(sample_observations):
    result = recon.reconstruct_candidates(
        sample_observations
    )

    row = next(
        row
        for row in result["row_candidates"]
        if "फ्लूथियासेट" in row["chemical_text_candidate"]
    )

    assert row["quantity_text_candidate"] == ""

    assert any(
        "embedded" in flag
        for flag in row["review_flags"]
    )


def test_no_candidate_is_automatically_approved(sample_observations):
    result = recon.reconstruct_candidates(
        sample_observations
    )

    assert result["approved_for_agricultural_advice"] is False

    assert all(
        row["review_status"] == "needs_manual_review"
        for row in result["row_candidates"]
    )


def test_all_quantities_are_assigned(sample_observations):
    result = recon.reconstruct_candidates(
        sample_observations
    )

    assert result["unassigned_quantities"] == []