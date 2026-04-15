from __future__ import annotations

from pathlib import Path

from app.config import DATA_DIR, RAW_DATA_DIR

try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    from PyPDF2 import PdfReader


def extract_pdf_text(filepath: Path) -> str:
    reader = PdfReader(str(filepath))
    text: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text.append(extracted)
    return "\n".join(text)


def extract_txt_file(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def _candidate_directories(data_dir: Path | None = None) -> list[Path]:
    directories = []
    if data_dir is not None:
        directories.append(data_dir)
    directories.extend([RAW_DATA_DIR, DATA_DIR])
    unique: list[Path] = []
    seen = set()
    for directory in directories:
        resolved = directory.resolve()
        if resolved not in seen and directory.exists():
            seen.add(resolved)
            unique.append(directory)
    return unique


def _find_first_matching_file(patterns: list[str], data_dir: Path | None = None) -> Path | None:
    for directory in _candidate_directories(data_dir):
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    return None


def load_documents(data_dir: Path | None = None) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    resume_path = _find_first_matching_file(["*.pdf", "*.PDF"], data_dir)
    if resume_path:
        documents.append({"source": resume_path.name, "section": "Resume", "content": extract_pdf_text(resume_path)})
    projects_path = _find_first_matching_file(["github_projects.txt", "*github*.txt", "*.txt"], data_dir)
    if projects_path:
        documents.append(
            {"source": projects_path.name, "section": "GitHub Projects", "content": extract_txt_file(projects_path)}
        )
    return [document for document in documents if document["content"].strip()]


def load_all_documents(data_dir: Path | None = None) -> str:
    parts: list[str] = []
    for document in load_documents(data_dir):
        parts.append(f"=== {document['section'].upper()} ({document['source']}) ===")
        parts.append(document["content"])
    return "\n\n".join(parts)
