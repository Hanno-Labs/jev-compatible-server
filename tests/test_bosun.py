import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from jev_compatible_server.bosun import BosunDecisionBackend
from jev_compatible_server.protocol import (
    ChoiceAnswer,
    DecisionRequest,
    NoulAnswer,
    ScoreAnswer,
)
from jev_compatible_server.runtime import RuntimeErrorBase


def request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"ticket": "duplicate charge"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a route.",
                    "criteria": {"billing": "payment issue", "other": None},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity.",
                    "criteria": ["low", "high"],
                },
                "confirmed": {
                    "type": "noul",
                    "instructions": "Was it confirmed?",
                    "criteria": {
                        "true": "the bank confirmed it",
                        "false": "no confirmation",
                    },
                },
            },
        }
    )


class FakeBosun:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> dict[str, list[float]]:
        self.calls.append(kwargs)
        count = len(kwargs["candidates"])
        return {"probabilities": [float(index + 1) for index in range(count)]}


def backend() -> BosunDecisionBackend:
    value = object.__new__(BosunDecisionBackend)
    value.model_name = "Hanno-Labs/bosun-v3.1-0.6b"
    value.metadata = {"readout": "bosun_decision_tokens"}
    value._seed = 0
    value._model = FakeBosun()
    return value


def test_bosun_maps_all_three_question_types() -> None:
    value = backend()
    response = value.decide(request())

    choice = response.answers["route"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice == "other"
    assert choice.probabilities == pytest.approx(
        {"billing": 1.0 / 3.0, "other": 2.0 / 3.0}
    )

    score = response.answers["severity"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == pytest.approx(2.0 / 3.0)
    assert score.legend == ["low", "high"]

    noul = response.answers["confirmed"]
    assert isinstance(noul, NoulAnswer)
    assert noul.noul == pytest.approx(1.0 / 3.0)

    calls = value._model.calls
    assert [call["decision_type"] for call in calls] == [
        "choice",
        "score",
        "noul",
    ]
    assert calls[0]["candidates"] == [
        {"id": "billing", "label": "billing", "description": "payment issue"},
        {"id": "other", "label": "other", "description": ""},
    ]
    assert calls[1]["candidates"] == [
        {"id": "0", "label": "low", "description": ""},
        {"id": "1", "label": "high", "description": ""},
    ]
    assert calls[2]["candidates"] == [
        {"id": "true", "label": "true", "description": "the bank confirmed it"},
        {"id": "false", "label": "false", "description": "no confirmation"},
    ]
    assert all(len(call["row_id"]) == 64 for call in calls)


def test_bosun_loader_enables_remote_code_and_pins_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    class FakeModel:
        def to(self, device: str) -> None:
            received["device"] = device

        def eval(self) -> None:
            received["eval"] = True

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> FakeModel:
            received["model_id"] = model_id
            received["kwargs"] = kwargs
            return FakeModel()

    torch = ModuleType("torch")
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.float16 = object()  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    BosunDecisionBackend(
        "Hanno-Labs/bosun-v3.1-1.7b",
        config={
            "decision": {
                "readout": "bosun_decision_tokens",
                "loader": {
                    "revision": "abc123",
                    "dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                },
            }
        },
    )

    kwargs = received["kwargs"]
    assert isinstance(kwargs, dict)
    assert received["model_id"] == "Hanno-Labs/bosun-v3.1-1.7b"
    assert kwargs["trust_remote_code"] is True
    assert kwargs["revision"] == "abc123"
    assert kwargs["dtype"] is torch.bfloat16
    assert kwargs["attn_implementation"] == "sdpa"
    assert "device" not in received
    assert received["eval"] is True


def test_bosun_rejects_invalid_probabilities() -> None:
    value = backend()
    value._model.predict = lambda **_kwargs: {
        "probabilities": [float("nan"), 1.0]
    }

    with pytest.raises(RuntimeErrorBase, match="invalid probability"):
        value.decide(
            request().model_copy(
                update={"questions": {"route": request().questions["route"]}}
            )
        )
