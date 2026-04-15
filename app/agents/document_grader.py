from __future__ import annotations

from app.components.hybrid_retriever import Document


class DocumentGrader:
    def keep(self, documents: list[Document]) -> list[Document]:
        return [document for document in documents if float(document.metadata.get("score") or 0.0) >= 0.05]
