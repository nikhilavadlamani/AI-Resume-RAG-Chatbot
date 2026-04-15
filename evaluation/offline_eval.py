import json
from pathlib import Path

from app.agents.adaptive_router import AdaptiveRouter


if __name__ == "__main__":
    dataset = json.loads(Path("evaluation/golden_dataset.json").read_text(encoding="utf-8"))
    router = AdaptiveRouter()
    results = []
    for item in dataset:
        predicted = router.select_route(item["question"])
        results.append({"question": item["question"], "predicted": predicted, "expected": item["expected_route"]})
    print(results)
