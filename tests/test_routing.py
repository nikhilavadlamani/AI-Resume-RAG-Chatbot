from app.agents.adaptive_router import AdaptiveRouter


def test_router_detects_skills_queries():
    router = AdaptiveRouter()
    assert router.select_route("What are my technical skills?") == "skills"


def test_router_detects_recruiter_reply_queries():
    router = AdaptiveRouter()
    assert router.select_route("Write a response to a recruiter") == "recruiter_reply"
