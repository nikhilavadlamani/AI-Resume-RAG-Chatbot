# Architecture

This project now follows a layered production-style layout:

- `app/`: FastAPI backend, shared config, schemas, retrieval components, agents, services, prompts, and security guards.
- `frontend/`: Streamlit UI that talks to the backend API.
- `observability/`: lightweight tracing, feedback capture, and cost tracking placeholders.
- `evaluation/`: reserved space for offline and online evaluation assets.
- `data/raw`, `data/processed`, `data/index_config`: source docs, processed artifacts, and indexing metadata.
- `scripts/`: seed, migrate, and healthcheck entrypoints.
- `tests/`: retrieval, cache, and routing tests.

Request flow:

1. Frontend sends a chat request to FastAPI.
2. Input guard validates the question.
3. Adaptive router classifies intent.
4. Query rewriter creates search variants.
5. Hybrid retriever searches the local vector store.
6. Reranker and document grader trim candidates.
7. RAG pipeline produces a deterministic or LLM-backed answer.
8. Output filter returns the final response and sources.
