from __future__ import annotations

from collections import OrderedDict

from app.models import ChatResponse


class SemanticCache:
    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._store: OrderedDict[str, ChatResponse] = OrderedDict()

    def _normalize(self, question: str) -> str:
        return " ".join(question.lower().split())

    def get(self, question: str) -> ChatResponse | None:
        key = self._normalize(question)
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
        return cached

    def set(self, question: str, response: ChatResponse) -> None:
        key = self._normalize(question)
        self._store[key] = response
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def size(self) -> int:
        return len(self._store)
