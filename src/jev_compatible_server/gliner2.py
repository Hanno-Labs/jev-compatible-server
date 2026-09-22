"""Native GLiNER2 single-label classifier readout for typed decisions."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .protocol import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    QuestionType,
    ScoreAnswer,
    ScoreQuestion,
    UnsupportedAnswer,
    Usage,
)
from .runtime import DecisionRuntime, RuntimeErrorBase

_STRUCTURAL_TOKENS = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "(",
    ")",
)
_SUPPORTED_TYPES: list[QuestionType] = ["choice", "score", "noul"]


class _UnexpressibleQuestion(ValueError):
    """A Jev value that GLiNER2's public classification schema cannot encode."""


def _decision_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    nested = config.get("decision")
    if isinstance(nested, Mapping) and key in nested:
        default = nested[key]
    return config.get(f"decision.{key}", default)


def _render_content(value: Any, field: str, *, schema_value: bool = True) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise _UnexpressibleQuestion(
                f"GLiNER2 requires JSON-serializable {field}"
            ) from exc
    if schema_value and any(token in rendered for token in _STRUCTURAL_TOKENS):
        raise _UnexpressibleQuestion(
            f"GLiNER2 cannot encode {field} containing a reserved schema token"
        )
    return rendered


def _instruction(value: Any) -> str | None:
    rendered = _render_content(value, "question instructions")
    return rendered if rendered.strip() else None


def _description(value: Any | None, field: str) -> str | None:
    if value is None:
        return None
    rendered = _render_content(value, field)
    return rendered if rendered.strip() else None


def _finite_distribution(scores: Any, labels: Sequence[str]) -> dict[str, float]:
    probabilities = {
        label: float(scores.probability("decision", label)) for label in labels
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in probabilities.values()
    ):
        raise RuntimeErrorBase("GLiNER2 returned an invalid classification probability")
    if not math.isclose(
        sum(probabilities.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise RuntimeErrorBase("GLiNER2 single-label probabilities do not sum to one")
    return probabilities


class GLiNER2Runtime(DecisionRuntime):
    """Adapt GLiNER2's public constrained single-label classifier.

    ``Classifier.score`` exposes the model's raw per-label logits and its
    public ``ClassificationScores.probability`` applies the schema's native
    softmax for exclusive tasks.  Each Jev question is therefore a separate
    single-label task: Choice labels are the supplied candidates, Score labels
    are the ordered legend, and Noul uses the explicit ``yes``/``no`` pair.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        classifier: Any | None = None,
        schema_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.model_name = model_id
        self._classifier = classifier
        self._schema_factory = schema_factory

    def _classifier_instance(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            module: Any = importlib.import_module("gliner2.classification")
            classifier_type = module.Classifier
        except (ImportError, AttributeError) as exc:
            raise RuntimeErrorBase(
                "GLiNER2 requires the published gliner2 classification package"
            ) from exc
        load_options = _decision_value(self.config, "gliner2_load_options", {})
        if not isinstance(load_options, Mapping):
            raise RuntimeErrorBase("decision.gliner2_load_options must be an object")
        self._classifier = classifier_type.from_pretrained(
            self.model_name, **dict(load_options)
        )
        device = load_options.get("device")
        if device is None:
            try:
                torch_module = importlib.import_module("torch")
                device = "cuda" if torch_module.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        to_options: dict[str, Any] = {"device": device}
        if "dtype" in load_options:
            to_options["dtype"] = load_options["dtype"]
        self._classifier.to(**to_options)
        return self._classifier

    def _new_schema(self) -> Any:
        if self._schema_factory is not None:
            return self._schema_factory()
        try:
            module: Any = importlib.import_module("gliner2.classification")
            schema_type = module.ClassificationSchema
        except (ImportError, AttributeError) as exc:
            raise RuntimeErrorBase(
                "GLiNER2 requires the published gliner2 classification package"
            ) from exc
        return schema_type()

    def _schema_for_question(self, question: Question) -> tuple[Any, list[str]]:
        schema = self._new_schema()
        instruction = _instruction(question.instructions)
        kwargs: dict[str, Any] = {}
        if instruction is not None:
            kwargs["instruction"] = instruction
        if isinstance(question, ChoiceQuestion):
            labels = {
                label: _description(description, f"description for {label!r}")
                for label, description in question.criteria.items()
            }
            for label in labels:
                _render_content(label, "choice label")
            schema.single("decision", labels, **kwargs)
            return schema, list(labels)
        if isinstance(question, ScoreQuestion):
            score_labels = [
                _render_content(level, "score level") for level in question.criteria
            ]
            schema.ordinal("decision", score_labels, **kwargs)
            return schema, score_labels
        if isinstance(question, NoulQuestion):
            criteria = question.criteria
            labels = {
                "yes": _description(
                    criteria.true if criteria is not None else None,
                    "noul true criterion",
                ),
                "no": _description(
                    criteria.false if criteria is not None else None,
                    "noul false criterion",
                ),
            }
            schema.single("decision", labels, **kwargs)
            return schema, list(labels)
        raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")

    @staticmethod
    def _unsupported(question: Question) -> UnsupportedAnswer:
        return UnsupportedAnswer(
            type="unsupported",
            question_type=question.type,
            supported_types=_SUPPORTED_TYPES,
        )

    def _answer_from_scores(
        self,
        question: Question,
        labels: Sequence[str],
        scores: Any,
    ) -> Answer:
        probabilities = _finite_distribution(scores, labels)
        if isinstance(question, ChoiceQuestion):
            choice = max(probabilities, key=probabilities.__getitem__)
            return ChoiceAnswer(
                type="choice",
                choice=choice,
                probabilities=probabilities,
                confidence=probabilities[choice],
            )
        if isinstance(question, ScoreQuestion):
            indexed = {
                str(index): probabilities[label] for index, label in enumerate(labels)
            }
            return ScoreAnswer(
                type="score",
                score=sum(index * indexed[str(index)] for index in range(len(labels))),
                probabilities=indexed,
                confidence=max(indexed.values()),
                legend=question.criteria,
            )
        if isinstance(question, NoulQuestion):
            return NoulAnswer(type="noul", noul=probabilities["yes"])
        raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        answers_by_request: list[dict[str, Answer]] = [{} for _ in requests]
        grouped: dict[str, list[tuple[int, str, str, Question, list[str], Any]]] = {}
        classifier: Any | None = None
        for request_index, request in enumerate(requests):
            try:
                state = _render_content(request.state, "state", schema_value=False)
            except _UnexpressibleQuestion:
                for name, question in request.questions.items():
                    answers_by_request[request_index][name] = self._unsupported(question)
            else:
                for name, question in request.questions.items():
                    try:
                        schema, labels = self._schema_for_question(question)
                        if classifier is None:
                            classifier = self._classifier_instance()
                        fingerprint = classifier.compile_schema(schema).fingerprint
                        grouped.setdefault(fingerprint, []).append(
                            (request_index, name, state, question, labels, schema)
                        )
                    except _UnexpressibleQuestion:
                        answers_by_request[request_index][name] = self._unsupported(question)

        if grouped:
            module: Any = importlib.import_module("gliner2.classification")
            batch_size = _decision_value(self.config, "batch_size", 8)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                raise RuntimeErrorBase("decision.batch_size must be a positive integer")
            batch_config = module.ClassificationConfig(batch_size=batch_size)
            assert classifier is not None
            for items in grouped.values():
                scores = classifier.batch_score(
                    [item[2] for item in items],
                    items[0][5],
                    config=batch_config,
                )
                for (request_index, name, _, question, labels, _), score in zip(
                    items, scores, strict=True
                ):
                    answers_by_request[request_index][name] = self._answer_from_scores(
                        question, labels, score
                    )

        return [
            DecisionResponse(model=self.model_name, answers=answers, usage=Usage())
            for answers in answers_by_request
        ]
