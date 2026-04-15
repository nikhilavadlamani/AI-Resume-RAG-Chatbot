from __future__ import annotations


class OutputFilter:
    def filter_answer(self, answer: str) -> str:
        return answer.strip()
