from __future__ import annotations


class CostTracker:
    def estimate(self, used_llm: bool) -> dict[str, float]:
        return {"llm_calls": 1.0 if used_llm else 0.0, "estimated_cost_usd": 0.0}
