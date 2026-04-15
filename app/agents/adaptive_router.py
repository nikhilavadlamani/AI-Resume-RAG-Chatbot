from __future__ import annotations

from app.services.query_router import QueryRouter


class AdaptiveRouter:
    def __init__(self):
        self.router = QueryRouter()

    def select_route(self, question: str) -> str:
        return self.router.route(question)
