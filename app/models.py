from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source: str
    section: str
    chunk_id: int | None = None
    score: float | None = None
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]
    route: str
    sources: list[SourceDocument]
    conversation_id: str | None = None
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    cache_entries: int
