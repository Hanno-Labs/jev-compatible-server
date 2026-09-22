from __future__ import annotations

from typing import Any

import pytest

from jev_compatible_server.decision_native import DecisionNativeRuntime
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase


def _request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "model": "decision-1.0-eos-0.8b",
            "state": {"request": "Please help me."},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Pick a route",
                    "criteria": {"self": "Self service", "human": "Human support"},
                },
                "risk": {
                    "type": "score",
                    "instructions": "Rate risk",
                    "criteria": ["low", "high"],
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            },
        }
    )


def _native_result() -> dict[str, Any]:
    return {
        "model": "Eos",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "human",
                "probabilities": {"self": 0.2, "human": 0.8},
                "confidence": 0.8,
            },
            "risk": {
                "type": "score",
                "score": 0.9,
                "probabilities": {"0": 0.1, "1": 0.9},
                "confidence": 0.9,
                "legend": {"0": "low", "1": "high"},
            },
            "urgent": {"type": "noul", "noul": 0.7},
        },
        "usage": {"input_tokens": 34, "scored_questions": 3},
    }


class FakeAgent:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.received: list[dict[str, Any]] = []

    def decide_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.received = payloads
        return self.results


def test_native_eos_preserves_typed_probabilities_and_batch_request() -> None:
    agent = FakeAgent([_native_result()])
    runtime = DecisionNativeRuntime(
        "llm-semantic-router/Decision-1.0-Eos-0.8B",
        config={
            "decision": {
                "revision": "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
                "public_model_name": "decision-1.0-eos-0.8b",
            }
        },
        agent=agent,
    )

    result = runtime.decide(_request())

    assert len(agent.received) == 1
    assert "model" not in agent.received[0]
    assert agent.received[0]["state"] == {"request": "Please help me."}
    assert result.model == "decision-1.0-eos-0.8b"
    assert result.usage.input_tokens == 34
    assert result.usage.output_tokens == 0
    answers = result.model_dump(mode="json")["answers"]
    assert answers["route"]["probabilities"] == {"self": 0.2, "human": 0.8}
    assert answers["risk"]["legend"] == ["low", "high"]


def test_native_eos_rejects_misaligned_native_distribution() -> None:
    bad = _native_result()
    bad["answers"]["route"]["probabilities"] = {"human": 1.0}
    runtime = DecisionNativeRuntime(
        "llm-semantic-router/Decision-1.0-Eos-0.8B",
        config={"decision.revision": "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd"},
        agent=FakeAgent([bad]),
    )
    with pytest.raises(RuntimeErrorBase, match="misaligned"):
        runtime.decide(_request())
