from pathlib import Path

from pypdf import PdfReader


DATA_DIR = Path("data")


def extract_pdf_text(filepath: Path) -> str:
    reader = PdfReader(str(filepath))
    text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text.append(extracted)
    return "\n".join(text)


def extract_txt_file(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def _find_first_matching_file(patterns: list[str], data_dir: Path) -> Path | None:
    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_documents(data_dir: str = "data") -> list[dict[str, str]]:
    base_dir = Path(data_dir)
    documents: list[dict[str, str]] = []

    resume_path = _find_first_matching_file(["*.pdf", "*.PDF"], base_dir)
    if resume_path:
        documents.append(
            {
                "source": resume_path.name,
                "section": "Resume",
                "content": extract_pdf_text(resume_path),
            }
        )

    projects_path = _find_first_matching_file(
        ["github_projects.txt", "*github*.txt", "*.txt"],
        base_dir,
    )
    if projects_path:
        documents.append(
            {
                "source": projects_path.name,
                "section": "GitHub Projects",
                "content": extract_txt_file(projects_path),
            }
        )

    return [doc for doc in documents if doc["content"].strip()]


def load_all_documents(data_dir: str = "data") -> str:
    all_sections = []
    for document in load_documents(data_dir):
        all_sections.append(f"=== {document['section'].upper()} ({document['source']}) ===")
        all_sections.append(document["content"])
    return "\n\n".join(all_sections)


if __name__ == "__main__":
    text = load_all_documents()
    print(f"\nTotal characters extracted: {len(text)}")
    print("\nFirst 500 characters preview:")
    print(text[:500])

    combined_path = DATA_DIR / "combined_text.txt"
    combined_path.write_text(text, encoding="utf-8")
    print(f"\nSaved to {combined_path}")
