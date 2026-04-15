from __future__ import annotations


class QueryDecomposer:
    def decompose(self, question: str) -> list[str]:
        lowered = question.lower()
        if " and " in lowered:
            return [part.strip() for part in question.split(" and ") if len(part.strip()) > 3]
        return []
