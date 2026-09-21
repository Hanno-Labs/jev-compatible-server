"""Runtime interface and shared decision post-processing."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from .protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    QuestionType,
    ScoreAnswer,
    ScoreQuestion,
    UnsupportedAnswer,
    Usage,
)


class RuntimeErrorBase(RuntimeError):
    """A user-facing runtime configuration or inference error."""


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise RuntimeErrorBase("cannot normalize an empty score vector")
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


class DecisionRuntime(ABC):
    """Backend contract. Implementations may process the whole batch at once."""

    model_name: str

    @abstractmethod
    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        """Evaluate requests, preserving input order."""

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide_batch([request])[0]


ALL_QUESTION_TYPES: tuple[QuestionType, ...] = ("choice", "score", "noul")


def configured_question_types(config: dict[str, Any]) -> tuple[QuestionType, ...]:
    """Read a model's declared capability boundary from decision metadata."""

    nested = config.get("decision", {})
    nested_types = nested.get("question_types") if isinstance(nested, dict) else None
    value = config.get("decision.question_types", nested_types)
    if value is None:
        return ALL_QUESTION_TYPES
    if not isinstance(value, list) or not value:
        raise RuntimeErrorBase("decision.question_types must be a non-empty array")
    if any(item not in ALL_QUESTION_TYPES for item in value):
        raise RuntimeErrorBase(
            "decision.question_types may contain only choice, score, and noul"
        )
    result: list[QuestionType] = []
    for item in value:
        typed_item = cast(QuestionType, item)
        if typed_item not in result:
            result.append(typed_item)
    return tuple(result)


class QuestionTypeRuntime(DecisionRuntime):
    """Return explicit per-question unsupported results around any backend."""

    def __init__(
        self,
        runtime: DecisionRuntime,
        supported_types: Sequence[QuestionType],
    ) -> None:
        self.runtime = runtime
        self.model_name = runtime.model_name
        self.supported_types = tuple(supported_types)
        self._supported = frozenset(supported_types)

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        filtered: list[DecisionRequest] = []
        filtered_indices: list[int] = []
        for index, request in enumerate(requests):
            questions = {
                name: question
                for name, question in request.questions.items()
                if question.type in self._supported
            }
            if questions:
                filtered.append(request.model_copy(update={"questions": questions}))
                filtered_indices.append(index)

        inferred: dict[int, DecisionResponse] = {}
        if filtered:
            responses = self.runtime.decide_batch(filtered)
            if len(responses) != len(filtered):
                raise RuntimeErrorBase("runtime returned the wrong batch length")
            inferred = dict(zip(filtered_indices, responses, strict=True))

        results: list[DecisionResponse] = []
        for index, request in enumerate(requests):
            response = inferred.get(index)
            supported_answers = response.answers if response is not None else {}
            answers: dict[str, Any] = {}
            for name, question in request.questions.items():
                if question.type in self._supported:
                    answer = supported_answers.get(name)
                    if answer is None:
                        raise RuntimeErrorBase(
                            f"runtime did not return answer for question {name!r}"
                        )
                    answers[name] = answer
                else:
                    answers[name] = UnsupportedAnswer(
                        type="unsupported",
                        question_type=question.type,
                        supported_types=list(self.supported_types),
                    )
            results.append(
                DecisionResponse(
                    model=response.model if response is not None else self.model_name,
                    answers=answers,
                    usage=response.usage if response is not None else Usage(),
                )
            )
        return results


def apply_question_type_support(
    runtime: DecisionRuntime, config: dict[str, Any]
) -> DecisionRuntime:
    supported = configured_question_types(config)
    if supported == ALL_QUESTION_TYPES:
        return runtime
    return QuestionTypeRuntime(runtime, supported)


class TokenLogitRuntime(DecisionRuntime):
    """Shared formatter for backends that expose next-token logits."""

    def __init__(self, *, model_name: str, config: dict[str, Any] | None = None):
        self.model_name = model_name
        self.config = config or {}

    def _metadata(self, name: str, legacy_name: str) -> Any:
        value = self.config.get(name, self.config.get(legacy_name))
        if isinstance(value, dict) and name.startswith("decision."):
            return value
        return value

    def _prompt(self, request: DecisionRequest, question_name: str, question: Any) -> str:
        template = self._metadata("decision.prompt_template", "prompt_template")
        if not isinstance(template, str):
            raise RuntimeErrorBase(
                "model metadata must define a string decision prompt_template"
            )
        return template.format(
            state=request.state,
            question_name=question_name,
            instructions=question.instructions,
            criteria=question.criteria if hasattr(question, "criteria") else "",
        )

    def _token_id(self, label: str) -> int:
        tokens = self._metadata("decision.tokens", "decision_tokens")
        if isinstance(tokens, str):
            import json

            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError as exc:
                raise RuntimeErrorBase("decision.tokens is not valid JSON") from exc
        if not isinstance(tokens, dict) or label not in tokens:
            raise RuntimeErrorBase(f"missing decision token mapping for {label!r}")
        token_id = tokens[label]
        if not isinstance(token_id, int):
            raise RuntimeErrorBase(f"decision token for {label!r} is not an integer")
        return token_id

    def _answer(self, question: Any, logits: dict[int, float]) -> Any:
        labels = self._labels_for_question(question)
        return self.answer_from_label_scores(
            question, {label: logits[self._token_id(label)] for label in labels}
        )

    @staticmethod
    def _labels_for_question(question: Any) -> list[str]:
        if isinstance(question, ChoiceQuestion):
            return list(question.criteria)
        if isinstance(question, ScoreQuestion):
            return [str(index) for index in range(len(question.criteria))]
        if isinstance(question, NoulQuestion):
            return ["true", "false"]
        raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")

    def answer_from_label_scores(self, question: Any, scores: dict[str, float]) -> Any:
        if isinstance(question, ChoiceQuestion):
            labels = list(question.criteria)
            probabilities = softmax([scores[label] for label in labels])
            distribution = dict(zip(labels, probabilities, strict=True))
            choice = max(distribution, key=distribution.__getitem__)
            return ChoiceAnswer(
                type="choice",
                choice=choice,
                probabilities=distribution,
                confidence=distribution[choice],
            )
        if isinstance(question, ScoreQuestion):
            labels = [str(index) for index in range(len(question.criteria))]
            probabilities = softmax([scores[label] for label in labels])
            distribution = dict(zip(labels, probabilities, strict=True))
            score = sum(index * probability for index, probability in enumerate(probabilities))
            return ScoreAnswer(
                type="score",
                score=score,
                probabilities=distribution,
                confidence=max(probabilities),
                legend=question.criteria,
            )
        if isinstance(question, NoulQuestion):
            yes = scores["true"]
            no = scores["false"]
            return NoulAnswer(type="noul", noul=1.0 / (1.0 + math.exp(no - yes)))
        raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")
