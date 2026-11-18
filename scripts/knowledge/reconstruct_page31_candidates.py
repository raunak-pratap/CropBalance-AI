import json
import re
from pathlib import Path

INPUT_PATH = Path(
    "/tmp/soybean-ocr/page-31-observations.json"
)

OUTPUT_PATH = Path(
    "/tmp/soybean-ocr/page-31-table-candidates.json"
)

# Page-specific, provisional column boundaries.
def identify_column(x: float) -> str:
    if x < 0.11:
        return "SERIAL"
    if x < 0.29:
        return "APPLICATION"
    if x < 0.72:
        return "CHEMICAL"
    return "QUANTITY"


def center_y(item: dict) -> float:
    return item["yTop"] + item["height"] / 2


def group_application_sections(records: list[dict]) -> list[dict]:
    application_items = [
        item for item in records
        if identify_column(item["x"]) == "APPLICATION"
        and item["yTop"] < 0.85
    ]

    application_items.sort(key=center_y)

    sections = []

    markers = ("अ)", "ब)", "स)")

    for item in application_items:
        text = item["text"].strip()

        if text.startswith(markers):
            sections.append({
                "section_id": f"section_{len(sections) + 1}",
                "start_y": center_y(item),
                "observations": [item],
                "labels": [text],
            })

        elif sections:
            sections[-1]["observations"].append(item)
            sections[-1]["labels"].append(text)

    for index, section in enumerate(sections):
        if index + 1 < len(sections):
            section["end_y"] = sections[index + 1]["start_y"]
        else:
            section["end_y"] = 0.85

        section["description"] = " ".join(section["labels"])

    return sections


# Manually calibrated using the original ICAR PDF page 31.
# These coordinates are specific to this page and layout.
PAGE_31_SECTION_BOUNDARIES = [
    0.1615,  # Section A ends; section B begins
    0.470,  # Section B ends; section C begins
]


def find_application_section(
    y: float,
    sections: list[dict],
):
    if len(sections) != 3:
        raise ValueError(
            f"Expected 3 sections, found {len(sections)}"
        )

    if y < PAGE_31_SECTION_BOUNDARIES[0]:
        return sections[0]

    if y < PAGE_31_SECTION_BOUNDARIES[1]:
        return sections[1]

    if y < 0.85:
        return sections[2]

    return None

HEADER_TEXTS = {
    "क्रं.",
    "प्रकार",
    "रासायनिक नाम",
    "मात्रा/हेक्टे.",
}

def reconstruct_candidates(records: list[dict]) -> dict:
    chemicals = []
    quantities = []

    for index, item in enumerate(records):
        if item["yTop"] >= 0.85:
            continue

        if item["text"].strip() in HEADER_TEXTS:
            continue

        column = identify_column(item["x"])


        prepared = {
            **item,
            "observation_index": index,
            "y_center": center_y(item),
        }

        if column == "CHEMICAL":
            chemicals.append(prepared)

        elif column == "QUANTITY":
            quantities.append(prepared)

    chemicals.sort(key=lambda item: item["y_center"])
    quantities.sort(key=lambda item: item["y_center"])

    used_quantities = set()
    candidates = []

    # Match each chemical to its nearest unused quantity.
    for chemical in chemicals:
        nearest = None
        nearest_distance = float("inf")

        for quantity in quantities:
            q_index = quantity["observation_index"]

            if q_index in used_quantities:
                continue

            distance = abs(
                chemical["y_center"] - quantity["y_center"]
            )

            if distance < nearest_distance:
                nearest = quantity
                nearest_distance = distance

        matched_quantity = None

        # A small distance tolerance for same-row matching.
        if nearest is not None and nearest_distance <= 0.016:
            matched_quantity = nearest
            used_quantities.add(
                nearest["observation_index"]
            )

        # If there is no quantity, this may be a wrapped
        # continuation of the preceding chemical name.
        if (
            matched_quantity is None
            and candidates
            and chemical["y_center"]
            - candidates[-1]["last_y"] <= 0.035
        ):
            candidates[-1]["chemical_fragments"].append(
                chemical["text"]
            )

            candidates[-1][
                "chemical_observation_indices"
            ].append(chemical["observation_index"])

            candidates[-1]["last_y"] = chemical["y_center"]

            candidates[-1]["review_flags"].append(
                "Possible wrapped chemical-name continuation"
            )

            continue

        quantity_fragments = []
        quantity_indices = []

        if matched_quantity is not None:
            quantity_fragments.append(
                matched_quantity["text"]
            )
            quantity_indices.append(
                matched_quantity["observation_index"]
            )
   

        candidates.append({
            "quantity_y_centers": (
            [matched_quantity["y_center"]]
            if matched_quantity is not None
            else []
            ),
            "sourceDocumentID": chemical["sourceDocumentID"],
            "pdfPage": chemical["pdfPage"],
            "chemical_fragments": [chemical["text"]],
            "chemical_observation_indices": [
                chemical["observation_index"]
            ],
            "quantity_fragments": quantity_fragments,
            "quantity_observation_indices": quantity_indices,
            "last_y": chemical["y_center"],
            "review_flags": [],
        })

    # Attach a second quantity fragment when the previous
    # quantity explicitly ends with a plus sign.
    unassigned_quantities = []

    for quantity in quantities:
        index = quantity["observation_index"]

        if index in used_quantities:
            continue

        text = quantity["text"].strip()
        quantity_y = quantity["y_center"]

        possible_continuations = []

        for candidate in candidates:
            fragments = candidate["quantity_fragments"]
            y_positions = candidate["quantity_y_centers"]

            if not fragments or not y_positions:
                continue

            last_fragment = fragments[-1].rstrip()
            last_y = y_positions[-1]

            distance = quantity_y - last_y

            if (
                last_fragment.endswith("+")
                and 0 < distance <= 0.040
            ):
                possible_continuations.append(
                    (distance, candidate)
                )

        if possible_continuations:
            distance, previous = min(
                possible_continuations,
                key=lambda item: item[0],
            )

            previous["quantity_fragments"].append(text)
            previous["quantity_observation_indices"].append(index)
            previous["quantity_y_centers"].append(quantity_y)

            previous["review_flags"].append(
                "Possible continued quantity expression"
            )

        else:
            unassigned_quantities.append({
                "observation_index": index,
                "text": text,
                "y": quantity_y,
                "review_status": "needs_manual_review",
            })

            # Build application sections.
    sections = group_application_sections(records)

    # Create final reconstruction candidates.
    output_rows = []

    for number, candidate in enumerate(candidates, start=1):

        section = find_application_section(
            candidate["last_y"],
            sections,
        )

        chemical_text = " ".join(
            candidate["chemical_fragments"]
        )

        quantity_text = " ".join(
            candidate["quantity_fragments"]
        )

        # Detect quantities that OCR may have merged
        # into the chemical-name observation.
        if re.search(
            r"[0-9O][0-9O.,-]*\s*(ली|मिली|ग्रा|ग्राम)",
            chemical_text,
        ):
            candidate["review_flags"].append(
                "Possible quantity embedded in chemical text"
            )

        if not quantity_text:
            candidate["review_flags"].append(
                "No separate quantity was matched"
            )

        output_rows.append({
            "row_candidate_id": f"page31_row_{number:02}",

            "sourceDocumentID": candidate["sourceDocumentID"],
            "pdfPage": candidate["pdfPage"],

            "chemical_text_candidate": chemical_text,
            "quantity_text_candidate": quantity_text,

            "chemical_observation_indices":
                candidate["chemical_observation_indices"],

            "quantity_observation_indices":
                candidate["quantity_observation_indices"],

            "application_section": (
                section["section_id"]
                if section else None
            ),

            "application_description": (
                section["description"]
                if section else None
            ),

            "review_status": "needs_manual_review",

            "review_flags": sorted(
                set(candidate["review_flags"])
            ),
        })

    return {
        "sourceDocumentID": "icar_nsri_soybean_2026",
        "pdfPage": 31,
        "reconstruction_method": "coordinate_based_prototype",
        "approved_for_agricultural_advice": False,

        "application_sections": sections,
        "row_candidates": output_rows,
        "unassigned_quantities": unassigned_quantities,
    }

def main():
    records = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    result = reconstruct_candidates(records)


    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("===== TABLE RECONSTRUCTION =====")
    print("Candidate rows:", len(result["row_candidates"]))
    print(
        "Application sections:",
        len(result["application_sections"])
    )
    print(
        "Unassigned quantities:",
        len(result["unassigned_quantities"])
    )

    for row in result["row_candidates"]:
        print("\n", row["row_candidate_id"])
        print("Chemical:", row["chemical_text_candidate"])
        print("Quantity:", row["quantity_text_candidate"])
        print("Section:", row["application_description"])
        print("Review flags:", row["review_flags"])

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()