from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.components.hybrid_retriever import build_vector_store
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.conversation import ConversationStore
from app.services.rag_pipeline import create_rag_chain
from app.services.semantic_cache import SemanticCache


settings = get_settings()
app = FastAPI(title=settings.app_name)
pipeline = create_rag_chain()
conversation_store = ConversationStore()
semantic_cache = SemanticCache(max_size=settings.semantic_cache_size)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents_indexed=len(pipeline.vector_store.records),
        cache_entries=semantic_cache.size(),
    )


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        cached = semantic_cache.get(payload.question)
        if cached is not None:
            return cached.model_copy(update={"conversation_id": payload.conversation_id, "cached": True})

        response = pipeline.ask_as_response(payload.question, conversation_id=payload.conversation_id)
        if payload.conversation_id:
            conversation_store.add_turn(payload.conversation_id, "user", payload.question)
            conversation_store.add_turn(payload.conversation_id, "assistant", response.answer)
        semantic_cache.set(payload.question, response)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/index/rebuild")
def rebuild_index() -> dict[str, str]:
    global pipeline
    build_vector_store()
    pipeline = create_rag_chain()
    return {"status": "rebuilt"}
