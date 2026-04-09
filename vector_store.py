import pickle
from dataclasses import dataclass
from pathlib import Path

from embeddings import (
    build_term_frequencies,
    cosine_similarity,
    exact_phrase_score,
    keyword_overlap_score,
    split_into_chunks,
)
from extract_text import BASE_DIR, load_documents


VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_FILE = VECTOR_STORE_DIR / "chunk_index.pkl"


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
        section_boost = 0.0
        lowered_query = query.lower()
        if "project" in lowered_query and "project" in section:
            section_boost = 0.12
        elif any(word in lowered_query for word in ["experience", "work", "employment"]) and "resume" in section:
            section_boost = 0.08

        return (
            similarity * 0.5
            + keyword_score * 0.3
            + phrase_score * 0.15
            + section_boost
        )

    def search(self, query: str, k: int = 6) -> list[Document]:
        scored_records = []
        for record in self.records:
            score = self._score_record(query, record)
            scored_records.append((score, record))

        ranked_records = sorted(scored_records, key=lambda item: item[0], reverse=True)
        top_records = [record for score, record in ranked_records[:k] if score > 0]

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

    def as_retriever(self, search_kwargs: dict | None = None):
        search_kwargs = search_kwargs or {}
        store = self

        class Retriever:
            def invoke(self, query: str):
                return store.search(query, k=search_kwargs.get("k", 6))

        return Retriever()

    def save_local(self, filepath: Path = VECTOR_STORE_FILE) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as file:
            pickle.dump(self.records, file)


def build_vector_store() -> LocalVectorStore:
    documents = load_documents()
    records: list[dict] = []

    for document in documents:
        chunks = split_into_chunks(document["content"])
        for index, chunk in enumerate(chunks, start=1):
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
        raise FileNotFoundError(
            "No source documents were found in the data folder. Add your resume PDF and project text file first."
        )

    vector_store = LocalVectorStore(records)
    vector_store.save_local()
    return vector_store


def load_vector_store() -> LocalVectorStore:
    if not VECTOR_STORE_FILE.exists():
        return build_vector_store()

    with VECTOR_STORE_FILE.open("rb") as file:
        records = pickle.load(file)
    return LocalVectorStore(records)


def test_retrieval(query: str) -> None:
    store = load_vector_store()
    results = store.search(query, k=3)

    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, doc in enumerate(results, start=1):
        print(
            f"\nResult {i} [{doc.metadata['section']} - {doc.metadata['source']}] "
            f"(score={doc.metadata['score']}) :"
        )
        print(doc.page_content[:500])


if __name__ == "__main__":
    build_vector_store()
    test_retrieval("What projects are mentioned?")
