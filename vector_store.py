import pickle
from dataclasses import dataclass
from pathlib import Path

from embeddings import build_term_frequencies, cosine_similarity, split_into_chunks
from extract_text import load_documents


VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_FILE = VECTOR_STORE_DIR / "chunk_index.pkl"


@dataclass
class Document:
    page_content: str
    metadata: dict


class LocalVectorStore:
    def __init__(self, records: list[dict]):
        self.records = records

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        query_vector = build_term_frequencies(query)
        scored_records = []

        for record in self.records:
            score = cosine_similarity(query_vector, record["term_frequencies"])
            scored_records.append((score, record))

        ranked_records = sorted(scored_records, key=lambda item: item[0], reverse=True)
        top_records = [record for score, record in ranked_records[:k] if score > 0]

        return [
            Document(
                page_content=record["content"],
                metadata={
                    "source": record["source"],
                    "section": record["section"],
                    "score": round(record["score"], 4) if "score" in record else None,
                },
            )
            for record in top_records
        ]

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
    results = store.similarity_search(query, k=3)

    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i} [{doc.metadata['section']} - {doc.metadata['source']}] :")
        print(doc.page_content[:500])


if __name__ == "__main__":
    build_vector_store()
    test_retrieval("What projects are mentioned?")
