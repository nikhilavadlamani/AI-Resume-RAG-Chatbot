from __future__ import annotations

from collections import defaultdict, deque


class ConversationStore:
    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self._messages: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=window_size))

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        self._messages[conversation_id].append({"role": role, "content": content})

    def history(self, conversation_id: str) -> list[dict[str, str]]:
        return list(self._messages.get(conversation_id, ()))
