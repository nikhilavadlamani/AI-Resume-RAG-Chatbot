from app.models import ChatResponse, SourceDocument
from app.services.semantic_cache import SemanticCache


def test_semantic_cache_normalizes_questions():
    cache = SemanticCache(max_size=2)
    response = ChatResponse(
        answer="Test",
        confidence="high",
        route="general",
        sources=[SourceDocument(source="resume.pdf", section="Resume", content="sample")],
    )

    cache.set("What are my skills?", response)

    assert cache.get("  what are my skills? ") is not None


def test_semantic_cache_respects_size_limit():
    cache = SemanticCache(max_size=1)
    response = ChatResponse(
        answer="Test",
        confidence="high",
        route="general",
        sources=[SourceDocument(source="resume.pdf", section="Resume", content="sample")],
    )

    cache.set("one", response)
    cache.set("two", response)

    assert cache.get("one") is None
    assert cache.get("two") is not None
