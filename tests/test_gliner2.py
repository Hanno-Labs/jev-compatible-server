from __future__ import annotations

from types import SimpleNamespace

import jev_compatible_server.gliner2 as gliner2_module
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
        instruction = kwargs.get("instruction")
        self.instruction = instruction if isinstance(instruction, str) else None
        return self

    def ordinal(self, _: str, labels: list[str], **kwargs: object) -> _FakeSchema:
        self.labels = labels
        self.kind = "ordinal"
        instruction = kwargs.get("instruction")
        self.instruction = instruction if isinstance(instruction, str) else None
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
        self.batch_calls: list[tuple[list[str], _FakeSchema, object]] = []

    @staticmethod
    def _scores(schema: _FakeSchema) -> _FakeScores:
        probabilities = {
            "cash": 0.2,
            "card": 0.8,
            "low": 0.1,
            "high": 0.9,
            "yes": 0.25,
            "no": 0.75,
        }
        return _FakeScores({label: probabilities[label] for label in schema.labels})

    @staticmethod
    def compile_schema(schema: _FakeSchema) -> object:
        return SimpleNamespace(
            fingerprint=f"{schema.kind}:{schema.instruction}:{schema.labels}"
        )

    def score(self, state: str, schema: _FakeSchema) -> _FakeScores:
        self.calls.append((state, schema))
        return self._scores(schema)

    def batch_score(
        self,
        states: list[str],
        schema: _FakeSchema,
        *,
        config: object,
    ) -> list[_FakeScores]:
        self.batch_calls.append((states, schema, config))
        return [self._scores(schema) for _ in states]


def test_gliner2_uses_native_single_label_probabilities_for_all_types(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gliner2_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(ClassificationConfig=lambda **kwargs: kwargs),
    )
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
    assert [schema.kind for _, schema, _ in classifier.batch_calls] == [
        "single",
        "single",
        "ordinal",
    ]
    assert classifier.batch_calls[0][0] == ['{"body":"The card was retained."}']
    assert classifier.calls == []


def test_gliner2_marks_only_unexpressible_questions_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        gliner2_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(ClassificationConfig=lambda **kwargs: kwargs),
    )
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

    classifier = _FakeClassifier()
    response = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        classifier=classifier,
        schema_factory=_FakeSchema,
    ).decide(request)

    assert response.answers["valid"].type == "noul"
    assert response.answers["invalid"].type == "unsupported"
    assert response.answers["invalid"].supported_types == ["choice", "score", "noul"]
    assert len(classifier.batch_calls) == 1


def test_gliner2_batches_repeated_schemas_once(monkeypatch) -> None:
    monkeypatch.setattr(
        gliner2_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(ClassificationConfig=lambda **kwargs: kwargs),
    )
    requests = [
        DecisionRequest.model_validate(
            {
                "state": state,
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Select a team.",
                        "criteria": {"cash": "ATM", "card": "Card support"},
                    }
                },
            }
        )
        for state in ("first", "second")
    ]
    classifier = _FakeClassifier()

    responses = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        config={"decision": {"batch_size": 16}},
        classifier=classifier,
        schema_factory=_FakeSchema,
    ).decide_batch(requests)

    assert len(classifier.batch_calls) == 1
    states, _, config = classifier.batch_calls[0]
    assert states == ["first", "second"]
    assert config == {"batch_size": 16}
    assert [response.answers["route"].choice for response in responses] == ["card", "card"]


def test_gliner2_groups_mixed_schemas_and_preserves_response_order(monkeypatch) -> None:
    monkeypatch.setattr(
        gliner2_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(ClassificationConfig=lambda **kwargs: kwargs),
    )
    requests = [
        DecisionRequest.model_validate(
            {
                "state": "first",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Select a team.",
                        "criteria": {"cash": "ATM", "card": "Card support"},
                    },
                    "valid": {"type": "noul", "instructions": "Is it valid?"},
                },
            }
        ),
        DecisionRequest.model_validate(
            {
                "state": "second",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Select a team.",
                        "criteria": {"cash": "ATM", "card": "Card support"},
                    }
                },
            }
        ),
    ]
    classifier = _FakeClassifier()

    responses = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        classifier=classifier,
        schema_factory=_FakeSchema,
    ).decide_batch(requests)

    assert [len(states) for states, _, _ in classifier.batch_calls] == [2, 1]
    assert responses[0].answers["route"].choice == "card"
    assert responses[0].answers["valid"].noul == 0.25
    assert responses[1].answers["route"].choice == "card"


def test_gliner2_auto_selects_cuda_and_honors_explicit_dtype(monkeypatch) -> None:
    created: list[_FakeClassifier] = []
    cuda_available = True

    class _LoadingClassifier(_FakeClassifier):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _LoadingClassifier:
            assert model_id == "fastino/gliner2.5-base-v1"
            instance = cls()
            instance.load_options = kwargs
            instance.to_calls: list[dict[str, object]] = []
            created.append(instance)
            return instance

        def to(self, **kwargs: object) -> _LoadingClassifier:
            self.to_calls.append(kwargs)
            return self

    def import_module(name: str) -> object:
        if name == "gliner2.classification":
            return SimpleNamespace(Classifier=_LoadingClassifier)
        if name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: cuda_available)
            )
        raise AssertionError(name)

    monkeypatch.setattr(gliner2_module.importlib, "import_module", import_module)

    runtime = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        config={
            "decision": {
                "gliner2_load_options": {
                    "revision": "pinned",
                    "dtype": "bfloat16",
                }
            }
        },
    )

    assert runtime._classifier_instance() is created[0]
    assert created[0].load_options == {"revision": "pinned", "dtype": "bfloat16"}
    assert created[0].to_calls == [{"device": "cuda", "dtype": "bfloat16"}]

    cuda_available = False
    cpu_runtime = GLiNER2Runtime(
        "fastino/gliner2.5-base-v1",
        config={"decision": {"gliner2_load_options": {"revision": "pinned"}}},
    )

    assert cpu_runtime._classifier_instance() is created[1]
    assert created[1].load_options == {"revision": "pinned"}
    assert created[1].to_calls == [{"device": "cpu"}]
