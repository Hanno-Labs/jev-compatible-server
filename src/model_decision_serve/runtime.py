"""Runtime interface and shared decision post-processing."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from .protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
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
