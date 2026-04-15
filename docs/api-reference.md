# API Reference

## `GET /health`

Returns service status, indexed document count, and semantic cache size.

## `POST /api/v1/chat`

Request body:

```json
{
  "question": "What are my top skills?",
  "conversation_id": "optional-session-id"
}
```

Response body includes:

- `answer`
- `confidence`
- `route`
- `sources`
- `conversation_id`
- `cached`

## `POST /api/v1/index/rebuild`

Rebuilds the local retrieval index from the current documents in `data/` or `data/raw/`.
