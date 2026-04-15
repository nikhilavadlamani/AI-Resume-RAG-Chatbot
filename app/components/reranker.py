from __future__ import annotations

from app.components.hybrid_retriever import Document
from app.config import get_settings
from app.services.text_utils import keyword_overlap_score


class SimpleReranker:
    def rerank(self, question: str, candidates: list[Document]) -> list[Document]:
        ranked = sorted(
            candidates,
            key=lambda doc: (
                float(doc.metadata.get("score") or 0.0),
                keyword_overlap_score(question, doc.page_content),
            ),
            reverse=True,
        )
        return ranked[: get_settings().reranker_top_k]
