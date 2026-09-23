from __future__ import annotations

from typing import Any

import pytest
from jev_compatible_server.native_systemone import (
    DjevHTTPRuntime,
    DjevThinkingRuntime,
    NativeSystemOneHTTPRuntime,
    OpenJevThinkingHTTPRuntime,
    SystemOneOpenHTTPRuntime,
)
from jev_compatible_server.protocol import DecisionRequest
from jev_compatible_server.runtime import RuntimeErrorBase


def _request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"body": "The card was retained."},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Select a team.",
                    "criteria": {"cash": "ATM", "card": "Card support"},
                },
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity.",
                    "criteria": ["low", "high"],
                },
            },
        }
    )


def test_native_systemone_preserves_native_typed_distribution() -> None:
    received: dict[str, Any] = {}

    def post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        received.update(url=url, body=body, timeout=timeout)
        return {
            "model": "native-alias",
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "card",
                    "probabilities": {"cash": 0.2, "card": 0.8},
                    "confidence": 0.8,
                },
                "urgent": {"type": "noul", "noul": 0.7},
                "severity": {
                    "type": "score",
                    "score": 0.9,
                    "probabilities": {"0": 0.1, "1": 0.9},
                    "confidence": 0.9,
                    "legend": {"0": "low", "1": "high"},
                },
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }

    runtime = NativeSystemOneHTTPRuntime(
        "public/model",
        config={
            "decision": {
                "endpoint": "http://native:8091/v1/systemone",
                "native_model": "winnow-latest",
                "request_fields": {"think": 32},
            }
        },
        post_json=post,
    )
    response = runtime.decide(_request())

    assert received["url"] == "http://native:8091/v1/systemone"
    assert received["body"]["model"] == "winnow-latest"
    assert received["body"]["think"] == 32
    assert response.model == "public/model"
    answer_payload = response.model_dump(mode="json")["answers"]
    assert answer_payload["route"]["probabilities"] == {"cash": 0.2, "card": 0.8}
    assert answer_payload["severity"]["legend"] == ["low", "high"]


def test_native_systemone_rejects_misaligned_distribution() -> None:
    runtime = NativeSystemOneHTTPRuntime(
        "public/model",
        config={"decision.endpoint": "http://native/v1/systemone"},
        post_json=lambda *_: {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "cash",
                    "probabilities": {"cash": 1.0},
                    "confidence": 1.0,
                },
                "urgent": {"type": "noul", "noul": 0.5},
                "severity": {
                    "type": "score",
                    "score": 0.0,
                    "probabilities": {"0": 1.0, "1": 0.0},
                    "confidence": 1.0,
                },
            }
        },
    )
    with pytest.raises(RuntimeErrorBase, match="misaligned"):
        runtime.decide(_request())


def test_system_one_open_maps_list_shaped_decide_contract() -> None:
    captured: dict[str, Any] = {}

    def post(url: str, body: dict[str, Any], _: float) -> dict[str, Any]:
        captured.update(url=url, body=body)
        return {
            "model": "e2b-full",
            "answers": [
                {
                    "id": "route", "type": "choice", "choice": "card",
                    "probabilities": {"cash": 0.2, "card": 0.8}, "confidence": 0.6,
                },
                {"id": "urgent", "type": "noul", "noul": 0.7, "confidence": 0.4},
                {
                    "id": "severity", "type": "score", "score": 0.9,
                    "probabilities": {"0": 0.1, "1": 0.9}, "confidence": 0.8,
                },
            ],
        }

    runtime = SystemOneOpenHTTPRuntime(
        "mithalouni/system-one-open",
        config={"decision.endpoint": "https://example.test/decide"},
        post_json=post,
    )
    response = runtime.decide(_request())

    assert captured["url"] == "https://example.test/decide"
    assert "model" not in captured["body"]
    questions = captured["body"]["questions"]
    assert questions[0]["options"] == {"cash": "ATM", "card": "Card support"}
    assert questions[1]["id"] == "urgent" and "criteria" not in questions[1]
    assert questions[2]["levels"] == ["low", "high"]
    assert response.model == "mithalouni/system-one-open"
    assert response.answers["route"].probabilities["card"] == 0.8
    assert response.answers["severity"].legend == ["low", "high"]
    assert response.answers["urgent"].noul == 0.7
    assert response.usage.input_tokens == 0


def test_djev_uses_its_distinct_request_shape_and_options() -> None:
    captured: dict[str, Any] = {}

    def post(_: str, body: dict[str, Any], __: float) -> dict[str, Any]:
        captured.update(body)
        return {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "cash",
                    "probabilities": {"cash": 0.5, "card": 0.5},
                    "confidence": 0.0,
                },
                "urgent": {"type": "noul", "noul": 0.5},
                "severity": {
                    "type": "score",
                    "score": 0.5,
                    "probabilities": {"0": 0.5, "1": 0.5},
                    "confidence": 0.0,
                    "legend": {"0": "low", "1": "high"},
                },
            }
        }

    runtime = DjevHTTPRuntime(
        "google/diffusiongemma-26B-A4B-it",
        config={
            "decision": {
                "endpoint": "http://djev:8000/v1/request",
                "options": {
                    "isolation": "independent",
                    "score_mode": "independent_levels",
                    "seed": 0,
                },
            }
        },
        post_json=post,
    )
    runtime.decide(_request())
    assert captured["model"] == "djev"
    assert captured["options"]["score_mode"] == "independent_levels"
    assert "isolation" not in captured


def test_djev_clamps_only_model_facing_criterion_descriptions() -> None:
    request = _request()
    request.questions["route"].criteria["cash"] = "é" * 501
    request.questions["severity"].criteria[0] = "z" * 501
    runtime = DjevHTTPRuntime(
        "djev",
        config={"decision": {"endpoint": "http://djev:8000/v1/request"}},
    )

    body = runtime._body(request)

    assert body["questions"]["route"]["criteria"] == {
        "cash": "é" * 500,
        "card": "Card support",
    }
    assert body["questions"]["severity"]["criteria"] == ["z" * 500, "high"]
    assert request.questions["route"].criteria["cash"] == "é" * 501
    assert request.questions["severity"].criteria[0] == "z" * 501


def test_djev_thinking_recombines_wide_choices_through_shared_anchor() -> None:
    calls: list[list[str]] = []

    def post(_: str, body: dict[str, Any], __: float) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for name, question in body["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.5}
                continue
            keys = list(question["criteria"])
            calls.append(keys)
            assert len(keys) <= 26
            weights = {key: int(key[1:]) + 1 for key in keys}
            total = sum(weights.values())
            probabilities = {key: value / total for key, value in weights.items()}
            winner = max(keys, key=probabilities.__getitem__)
            answers[name] = {
                "type": "choice",
                "choice": winner,
                "probabilities": probabilities,
                "confidence": probabilities[winner],
            }
        return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}

    request = DecisionRequest.model_validate(
        {
            "state": "A frozen example",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "Choose one.",
                    "criteria": {f"c{i}": f"Option {i}" for i in range(55)},
                },
                "yes": {"type": "noul", "instructions": "Is it yes?"},
            },
        }
    )
    runtime = DjevThinkingRuntime(
        "djev-thinking",
        config={
            "decision": {
                "endpoint": "http://djev-thinking:8011/v1/systemone",
                "request_fields": {"think": 0},
            }
        },
        post_json=post,
    )

    response = runtime.decide(request)

    assert [len(keys) for keys in calls] == [26, 26, 5]
    assert all(keys[0] == "c0" for keys in calls)
    assert response.answers["wide"].choice == "c54"
    assert response.answers["wide"].probabilities["c54"] == pytest.approx(55 / sum(range(1, 56)))
    assert response.answers["yes"].noul == 0.5
    assert response.usage.input_tokens == 4
    assert len(request.questions["wide"].criteria) == 55


def test_djev_thinking_forwards_its_published_thought_budget() -> None:
    captured: dict[str, Any] = {}

    def post(_: str, body: dict[str, Any], __: float) -> dict[str, Any]:
        captured.update(body)
        return {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "card",
                    "probabilities": {"cash": 0.2, "card": 0.8},
                    "confidence": 0.8,
                },
                "urgent": {"type": "noul", "noul": 0.3},
                "severity": {
                    "type": "score",
                    "score": 0.8,
                    "probabilities": {"0": 0.2, "1": 0.8},
                    "confidence": 0.8,
                    "legend": {"0": "low", "1": "high"},
                },
            }
        }

    runtime = DjevThinkingRuntime(
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        config={
            "decision": {
                "endpoint": "http://djev-thinking:8011/v1/systemone",
                "request_fields": {"think": 64},
            }
        },
        post_json=post,
    )

    response = runtime.decide(_request())

    assert captured["think"] == 64
    assert response.answers["route"].type == "choice"


def test_djev_thinking_requires_its_published_thought_budget() -> None:
    with pytest.raises(RuntimeErrorBase, match="requires"):
        DjevThinkingRuntime(
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            config={"decision.endpoint": "http://djev-thinking:8011/v1/systemone"},
        )


def test_openjev_thinking_requires_its_published_think_budget() -> None:
    with pytest.raises(RuntimeErrorBase, match="requires"):
        OpenJevThinkingHTTPRuntime(
            "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
            config={"decision.endpoint": "http://openjev:8080/v1/systemone"},
        )
