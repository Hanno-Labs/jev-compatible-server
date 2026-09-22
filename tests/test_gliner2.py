from __future__ import annotations

from jev_compatible_server.gliner2 import GLiNER2Runtime
from jev_compatible_server.protocol import DecisionRequest


class _FakeSchema:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.kind = ""
        self.instruction: str | None = None

    def single(
        self,
        _: str,
        labels: dict[str, str | None],
        **kwargs: object,
    ) -> _FakeSchema:
        self.labels = list(labels)
        self.kind = "single"
        self.instruction = kwargs.get("instruction") if isinstance(
            kwargs.get("instruction"), str
        ) else None
        return self

    def ordinal(self, _: str, labels: list[str], **kwargs: object) -> _FakeSchema:
        self.labels = labels
        self.kind = "ordinal"
        self.instruction = kwargs.get("instruction") if isinstance(
            kwargs.get("instruction"), str
        ) else None
        return self


class _FakeScores:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities

    def probability(self, task: str, label: str) -> float:
        assert task == "decision"
        return self._probabilities[label]


class _FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, _FakeSchema]] = []

    def score(self, state: str, schema: _FakeSchema) -> _FakeScores:
        self.calls.append((state, schema))
        probabilities = {
            "cash": 0.2,
            "card": 0.8,
            "low": 0.1,
            "high": 0.9,
            "yes": 0.25,
            "no": 0.75,
        }
        return _FakeScores({label: probabilities[label] for label in schema.labels})


def test_gliner2_uses_native_single_label_probabilities_for_all_types() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": {"body": "The card was retained."},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Select a team.",
                    "criteria": {"cash": "ATM", "card": "Card support"},
                },
                "valid": {
                    "type": "noul",
                    "instructions": "Is it valid?",
                    "criteria": {"true": "valid", "false": "not valid"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate it.",
                    "criteria": ["low", "high"],
                },
            },
        }
    )
    classifier = _FakeClassifier()

    response = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        classifier=classifier,
        schema_factory=_FakeSchema,
    ).decide(request)

    assert response.model == "fastino/gliner2.5-base-v1"
    answers = response.model_dump(mode="json")["answers"]
    assert answers["route"] == {
        "type": "choice",
        "choice": "card",
        "probabilities": {"cash": 0.2, "card": 0.8},
        "confidence": 0.8,
    }
    assert answers["valid"] == {"type": "noul", "noul": 0.25}
    assert answers["severity"] == {
        "type": "score",
        "score": 0.9,
        "probabilities": {"0": 0.1, "1": 0.9},
        "confidence": 0.9,
        "legend": ["low", "high"],
    }
    assert [schema.kind for _, schema in classifier.calls] == [
        "single",
        "single",
        "ordinal",
    ]
    assert classifier.calls[0][0] == '{"body":"The card was retained."}'


def test_gliner2_marks_only_unexpressible_questions_unsupported() -> None:
    request = DecisionRequest.model_validate(
        {
            "state": "The card was retained.",
            "questions": {
                "valid": {"type": "noul", "instructions": "Is it valid?"},
                "invalid": {
                    "type": "choice",
                    "instructions": "Select a team.",
                    "criteria": {"[L]": None, "card": None},
                },
            },
        }
    )

    response = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        classifier=_FakeClassifier(),
        schema_factory=_FakeSchema,
    ).decide(request)

    assert response.answers["valid"].type == "noul"
    assert response.answers["invalid"].type == "unsupported"
    assert response.answers["invalid"].supported_types == ["choice", "score", "noul"]
