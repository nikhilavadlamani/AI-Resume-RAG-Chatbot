from __future__ import annotations

from app.agents.query_decomposer import QueryDecomposer
from app.services.text_utils import extract_keywords


class QueryRewriter:
    def __init__(self):
        self.decomposer = QueryDecomposer()

    def rewrite(self, question: str, route: str) -> list[str]:
        keywords = extract_keywords(question)
        keyword_query = " ".join(keywords[:8]).strip()
        variants = [question]
        variants.extend(self.decomposer.decompose(question))
        if keyword_query and keyword_query.lower() != question.lower():
            variants.append(keyword_query)
        if route != "general":
            variants.append(f"{route} {question}")
        deduped: list[str] = []
        seen = set()
        for variant in variants:
            normalized = variant.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(variant.strip())
        return deduped
