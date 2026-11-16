
from pathlib import Path
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "knowledge" / "raw"


def inspect_pdf(pdf_path: Path) -> dict:
    """Extract text and basic quality information from one PDF."""

    reader = PdfReader(str(pdf_path))

    page_results = []
    total_characters = 0
    non_empty_pages = 0

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        characters = len(page_text)
        total_characters += characters

        if characters > 0:
            non_empty_pages += 1

        page_results.append({
            "page": page_number,
            "characters": characters,
            "text_preview": page_text[:150]
        })

    return {
        "filename": pdf_path.name,
        "total_pages": len(reader.pages),
        "non_empty_pages": non_empty_pages,
        "empty_pages": len(reader.pages) - non_empty_pages,
        "total_characters": total_characters,
        "pages": page_results,
    }


def main() -> None:

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {RAW_DIR}"
        )

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found.")
        return

    for pdf_path in pdf_files:

        print("\n" + "=" * 60)
        print("INSPECTING:", pdf_path.name)

        try:
            result = inspect_pdf(pdf_path)

            print("Total pages:", result["total_pages"])
            print("Non-empty pages:", result["non_empty_pages"])
            print("Empty pages:", result["empty_pages"])
            print("Extracted characters:", result["total_characters"])

            if result["non_empty_pages"] == 0:
                print("WARNING: No extractable text found.")
                print("This PDF may require a separate OCR workflow.")
                continue

            print("\nFIRST PAGE PREVIEW:")
            print(result["pages"][0]["text_preview"])

        except Exception as exc:
            print("PDF INSPECTION FAILED:", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
