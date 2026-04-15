from __future__ import annotations


class ContentFilter:
    def filter_context(self, context: str) -> str:
        return context.strip()
