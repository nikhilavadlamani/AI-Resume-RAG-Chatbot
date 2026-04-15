from __future__ import annotations


class InputGuard:
    blocked_terms = {"ignore previous instructions", "system prompt", "reveal prompt"}

    def validate(self, question: str) -> str:
        cleaned = " ".join(question.split())
        lowered = cleaned.lower()
        for term in self.blocked_terms:
            if term in lowered:
                raise ValueError("The question contains unsupported prompt-injection language.")
        if len(cleaned) > 4000:
            raise ValueError("The question is too long.")
        return cleaned
