from __future__ import annotations

from app.components.hybrid_retriever import Document, LocalVectorStore


def search_vector_store(store: LocalVectorStore, queries: list[str], k: int = 5) -> list[Document]:
    results: list[Document] = []
    seen = set()
    for query in queries:
        for result in store.search(query, k=k):
            key = (result.metadata.get("source"), result.metadata.get("section"), result.metadata.get("chunk_id"))
            if key not in seen:
                seen.add(key)
                results.append(result)
    return results
