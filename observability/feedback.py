from __future__ import annotations


class FeedbackStore:
    def __init__(self):
        self.records: list[dict[str, str]] = []

    def add(self, question: str, verdict: str) -> None:
        self.records.append({"question": question, "verdict": verdict})
