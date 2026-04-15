from __future__ import annotations

import pickle
from dataclasses import dataclass

from app.config import VECTOR_STORE_DIR, VECTOR_STORE_FILE, get_settings
from app.services.documents import load_documents
from app.services.text_utils import (
    build_term_frequencies,
    cosine_similarity,
    exact_phrase_score,
    keyword_overlap_score,
    split_into_chunks,
)


@dataclass
class Document:
    page_content: str
    metadata: dict


class LocalVectorStore:
    def __init__(self, records: list[dict]):
        self.records = records

    def _score_record(self, query: str, record: dict) -> float:
        query_vector = build_term_frequencies(query)
        similarity = cosine_similarity(query_vector, record["term_frequencies"])
        keyword_score = keyword_overlap_score(query, record["content"])
        phrase_score = exact_phrase_score(query, record["content"])

        section = str(record["section"]).lower()
        lowered_query = query.lower()
        section_boost = 0.0
        if "project" in lowered_query and "project" in section:
            section_boost = 0.12
        elif any(word in lowered_query for word in ["experience", "work", "employment"]) and "resume" in section:
            section_boost = 0.08

        return similarity * 0.5 + keyword_score * 0.3 + phrase_score * 0.15 + section_boost

    def search(self, query: str, k: int | None = None) -> list[Document]:
        limit = k or get_settings().retriever_top_k
        scored_records = [(self._score_record(query, record), record) for record in self.records]
        ranked_records = sorted(scored_records, key=lambda item: item[0], reverse=True)
        top_records = [record for score, record in ranked_records[:limit] if score > 0]
        return [
            Document(
                page_content=record["content"],
                metadata={
                    "source": record["source"],
                    "section": record["section"],
                    "chunk_id": record["chunk_id"],
                    "score": round(self._score_record(query, record), 4),
                },
            )
            for record in top_records
        ]

    def save_local(self, filepath=VECTOR_STORE_FILE) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as handle:
            pickle.dump(self.records, handle)


def build_vector_store() -> LocalVectorStore:
    records: list[dict] = []
    for document in load_documents():
        for index, chunk in enumerate(split_into_chunks(document["content"]), start=1):
            records.append(
                {
                    "content": chunk,
                    "source": document["source"],
                    "section": document["section"],
                    "chunk_id": index,
                    "term_frequencies": build_term_frequencies(chunk),
                }
            )
    if not records:
        raise FileNotFoundError("No source documents were found in the data folder. Add your resume PDF and project text file first.")
    store = LocalVectorStore(records)
    store.save_local()
    return store


def load_vector_store() -> LocalVectorStore:
    if not VECTOR_STORE_FILE.exists():
        return build_vector_store()
    with VECTOR_STORE_FILE.open("rb") as handle:
        records = pickle.load(handle)
    return LocalVectorStore(records)
