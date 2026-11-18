import json
from pathlib import Path
from statistics import mean

INPUT_PATH = Path(
    "/tmp/soybean-ocr/page-31-observations.json"
)

# Provisional column boundaries for PDF page 31.
def identify_column(x: float) -> str:
    if x < 0.11:
        return "SERIAL"

    if x < 0.29:
        return "APPLICATION"

    if x < 0.72:
        return "CHEMICAL"

    return "QUANTITY"


def group_visual_lines(records, tolerance=0.014):
    """
    Group OCR observations that appear on approximately
    the same horizontal line.

    yTop uses a normalized top-origin coordinate.
    """

    prepared = []

    for record in records:
        item = dict(record)

        item["y_center"] = (
            item["yTop"] + item["height"] / 2
        )

        item["column"] = identify_column(item["x"])

        prepared.append(item)

    prepared.sort(key=lambda item: item["y_center"])

    groups = []

    for item in prepared:
        if not groups:
            groups.append([item])
            continue

        current_center = mean(
            entry["y_center"] for entry in groups[-1]
        )

        if abs(item["y_center"] - current_center) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])

    return groups


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    records = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    # Exclude the footer for this initial table experiment.
    table_records = [
        item for item in records
        if item["yTop"] < 0.85
    ]

    groups = group_visual_lines(table_records)

    print("=" * 90)
    print("SOYBEAN TABLE — VISUAL LINE RECONSTRUCTION")
    print("=" * 90)

    print("Original observations:", len(records))
    print("Table observations:", len(table_records))
    print("Visual line groups:", len(groups))

    for number, group in enumerate(groups, start=1):
        group.sort(key=lambda item: item["x"])

        print(f"\nLINE {number:02d}")

        for item in group:
            print(
                f"  {item['column']:<12} "
                f"x={item['x']:.3f} "
                f"y={item['y_center']:.3f} "
                f"confidence={item['confidence']:.2f} "
                f"| {item['text']}"
            )


if __name__ == "__main__":
    main()