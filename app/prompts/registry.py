from __future__ import annotations

from app.prompts.templates import SYSTEM_PROMPT


PROMPT_REGISTRY = {
    "resume_assistant:v1": SYSTEM_PROMPT,
}


def get_prompt(name: str = "resume_assistant:v1") -> str:
    return PROMPT_REGISTRY[name]
