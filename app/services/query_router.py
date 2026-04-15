from __future__ import annotations


class QueryRouter:
    def route(self, question: str) -> str:
        lowered = question.lower()
        if any(phrase in lowered for phrase in ["tell me about yourself", "summarize my profile", "profile summary"]):
            return "profile_summary"
        if any(phrase in lowered for phrase in ["strength", "weakness"]):
            return "strengths_weaknesses"
        if any(phrase in lowered for phrase in ["recruiter", "write a response", "reply to recruiter"]):
            return "recruiter_reply"
        if any(phrase in lowered for phrase in ["good fit", "fit for", "match my profile", "match score"]):
            return "job_fit"
        if any(phrase in lowered for phrase in ["improve my resume", "project suggestions", "projects to improve"]):
            return "resume_improvement"
        if any(phrase in lowered for phrase in ["simpler terms", "simple terms", "explain that project"]):
            return "project_simplify"
        if any(word in lowered for word in ["skill", "technology", "tool", "language", "stack"]):
            return "skills"
        if any(word in lowered for word in ["project", "build", "portfolio", "github"]):
            return "projects"
        if any(word in lowered for word in ["experience", "work", "role", "company", "job", "years"]):
            return "experience"
        if any(word in lowered for word in ["education", "degree", "university", "college"]):
            return "education"
        if any(word in lowered for word in ["email", "phone", "contact", "linkedin"]):
            return "contact"
        return "general"
