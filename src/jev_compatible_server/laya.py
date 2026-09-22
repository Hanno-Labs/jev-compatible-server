"""Adapter for the published convaiinnovations/laya checkpoint."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .protocol import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from .runtime import DecisionRuntime, RuntimeErrorBase


class LayaBackend(DecisionRuntime):
    """Run Laya's checkpoint-native RLAgent API behind the JEV wire contract."""

    def __init__(self, model_id: str, *, config: dict[str, Any] | None = None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase("LayaBackend requires huggingface_hub") from exc

        self.config = config or {}
        self.model_name = str(self.config.get("model", model_id))
        revision = self.config.get("revision")
        download_kwargs: dict[str, Any] = {}
        if isinstance(revision, str):
            download_kwargs["revision"] = revision
        model_dir = Path(snapshot_download(model_id, **download_kwargs))
        # The published checkpoint ships its inference modules beside the weights.
        source_dir = str(model_dir)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        try:
            from rl_agent_api import RLAgent  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - model-owned optional code
            raise RuntimeErrorBase(
                "Laya checkpoint does not contain rl_agent_api.py"
            ) from exc
        try:
            self._agent = RLAgent(str(model_dir))
        except Exception as exc:  # pragma: no cover - model-owned loading errors
            raise RuntimeErrorBase(f"failed to load Laya checkpoint {model_id!r}: {exc}") from exc

    @staticmethod
    def _question_payload(question: Any) -> dict[str, Any]:
        if hasattr(question, "model_dump"):
            payload = question.model_dump(mode="json")
            if isinstance(payload, dict) and all(
                isinstance(key, str) for key in payload
            ):
                return cast(dict[str, Any], payload)
        raise RuntimeErrorBase("Laya requires pydantic decision questions")

    @staticmethod
    def _answer(value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("type"), str):
            raise RuntimeErrorBase("Laya returned an invalid answer")
        kind = value["type"]
        if kind == "choice":
            probabilities = value.get("probabilities")
            choice = value.get("choice")
            confidence = value.get("confidence")
            if (
                not isinstance(probabilities, dict)
                or not isinstance(choice, str)
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
            ):
                raise RuntimeErrorBase("Laya returned an invalid choice answer")
            return ChoiceAnswer(
                type="choice",
                choice=choice,
                probabilities={str(key): float(prob) for key, prob in probabilities.items()},
                confidence=float(confidence),
            )
        if kind == "score":
            probabilities = value.get("probabilities")
            legend = value.get("legend")
            if not isinstance(probabilities, dict):
                raise RuntimeErrorBase("Laya returned an invalid score answer")
            ordered_legend: list[Any] | None = None
            if isinstance(legend, dict):
                ordered_legend = [legend[key] for key in sorted(legend, key=lambda item: int(item))]
            elif isinstance(legend, list):
                ordered_legend = legend
            return ScoreAnswer(
                type="score",
                score=float(value["score"]),
                probabilities={str(key): float(prob) for key, prob in probabilities.items()},
                confidence=float(value["confidence"]),
                legend=ordered_legend,
            )
        if kind == "noul":
            return NoulAnswer(type="noul", noul=float(value["noul"]))
        raise RuntimeErrorBase(f"Laya returned unsupported answer type: {kind!r}")

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        responses: list[DecisionResponse] = []
        for request in requests:
            questions = {
                name: self._question_payload(question)
                for name, question in request.questions.items()
            }
            try:
                result = self._agent.system_one(request.state, questions)
            except Exception as exc:  # pragma: no cover - model-owned inference errors
                raise RuntimeErrorBase(f"Laya inference failed: {exc}") from exc
            if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
                raise RuntimeErrorBase("Laya returned no answers")
            answers = {
                name: self._answer(result["answers"][name])
                for name in questions
                if name in result["answers"]
            }
            if len(answers) != len(questions):
                raise RuntimeErrorBase("Laya omitted one or more requested answers")
            usage_value = result.get("usage")
            usage = Usage(
                input_tokens=int(usage_value.get("input_tokens", 0))
                if isinstance(usage_value, dict)
                else 0,
                output_tokens=int(usage_value.get("output_tokens", 0))
                if isinstance(usage_value, dict)
                else 0,
            )
            responses.append(DecisionResponse(model=self.model_name, answers=answers, usage=usage))
        return responses
