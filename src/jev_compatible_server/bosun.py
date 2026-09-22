"""Jev protocol adapter for Bosun's native Transformers decision readout."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .encoder_decoder import decision_metadata, render_content
from .protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)
from .runtime import DecisionRuntime, RuntimeErrorBase


def _loader_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = metadata.get("loader", {})
    if not isinstance(value, dict):
        raise RuntimeErrorBase("decision.loader must be an object")
    return value


def _candidate(
    candidate_id: str,
    label: str,
    description: Any | None = None,
) -> dict[str, str]:
    return {
        "id": candidate_id,
        "label": label,
        "description": "" if description is None else render_content(description),
    }


def _candidates(question: Question) -> list[dict[str, str]]:
    if isinstance(question, ChoiceQuestion):
        return [
            _candidate(candidate_id, candidate_id, description)
            for candidate_id, description in question.criteria.items()
        ]
    if isinstance(question, ScoreQuestion):
        return [
            _candidate(str(index), render_content(level))
            for index, level in enumerate(question.criteria)
        ]
    if isinstance(question, NoulQuestion):
        if question.criteria is None:
            return [
                _candidate("true", "true", "yes"),
                _candidate("false", "false", "no"),
            ]
        return [
            _candidate("true", "true", question.criteria.true),
            _candidate("false", "false", question.criteria.false),
        ]
    raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")


def _row_id(request: DecisionRequest, question_name: str) -> str:
    try:
        encoded = json.dumps(
            {
                "question_name": question_name,
                "request": request.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise RuntimeErrorBase("Bosun request must be JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _probabilities(result: Any, candidate_count: int) -> list[float]:
    if not isinstance(result, Mapping):
        raise RuntimeErrorBase("Bosun predict() returned a non-object result")
    raw = result.get("probabilities")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise RuntimeErrorBase("Bosun predict() did not return probabilities")
    try:
        probabilities = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise RuntimeErrorBase("Bosun probabilities must be numeric") from exc
    if (
        len(probabilities) != candidate_count
        or not all(math.isfinite(value) and value >= 0 for value in probabilities)
    ):
        raise RuntimeErrorBase("Bosun returned an invalid probability distribution")
    total = math.fsum(probabilities)
    if not math.isfinite(total) or total <= 0:
        raise RuntimeErrorBase("Bosun returned an invalid probability distribution")
    return [value / total for value in probabilities]


class BosunDecisionBackend(DecisionRuntime):
    """Translate Jev requests to Bosun's public ``model.predict`` contract."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "BosunDecisionBackend requires transformers and torch"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.metadata = decision_metadata(config or {})
        if self.metadata.get("readout") != "bosun_decision_tokens":
            raise RuntimeErrorBase(
                "BosunDecisionBackend requires decision.readout=bosun_decision_tokens"
            )
        seed = self.metadata.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise RuntimeErrorBase("decision.seed must be an integer")
        self._seed = seed

        loader = _loader_config(self.metadata)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        revision = loader.get("revision")
        if revision is not None:
            if not isinstance(revision, str):
                raise RuntimeErrorBase("decision.loader.revision must be a string")
            kwargs["revision"] = revision
        dtype = loader.get("dtype")
        if dtype in {"bf16", "bfloat16"}:
            kwargs["dtype"] = torch.bfloat16
        elif dtype in {"fp16", "float16"}:
            kwargs["dtype"] = torch.float16
        elif dtype is not None:
            raise RuntimeErrorBase("decision.loader.dtype must be bfloat16 or float16")
        device_map = loader.get("device_map")
        if device_map is not None:
            if not isinstance(device_map, str):
                raise RuntimeErrorBase("decision.loader.device_map must be a string")
            kwargs["device_map"] = device_map
        attention = loader.get("attn_implementation")
        if attention is not None:
            if not isinstance(attention, str):
                raise RuntimeErrorBase(
                    "decision.loader.attn_implementation must be a string"
                )
            kwargs["attn_implementation"] = attention

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map is None:
            target = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if target != "auto":
                self._model.to(target)
        self._model.eval()

    def _answer(
        self,
        request: DecisionRequest,
        question_name: str,
        question: Question,
    ) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
        candidates = _candidates(question)
        result = self._model.predict(
            state=request.state,
            instructions=render_content(question.instructions),
            candidates=candidates,
            decision_type=question.type,
            seed=self._seed,
            row_id=_row_id(request, question_name),
        )
        values = _probabilities(result, len(candidates))
        probabilities = {
            candidate["id"]: probability
            for candidate, probability in zip(candidates, values, strict=True)
        }
        if isinstance(question, ChoiceQuestion):
            choice = max(probabilities, key=probabilities.__getitem__)
            return ChoiceAnswer(
                type="choice",
                choice=choice,
                probabilities=probabilities,
                confidence=probabilities[choice],
            )
        if isinstance(question, ScoreQuestion):
            return ScoreAnswer(
                type="score",
                score=math.fsum(
                    index * probabilities[str(index)]
                    for index in range(len(question.criteria))
                ),
                probabilities=probabilities,
                confidence=max(probabilities.values()),
                legend=question.criteria,
            )
        if isinstance(question, NoulQuestion):
            return NoulAnswer(type="noul", noul=probabilities["true"])
        raise RuntimeErrorBase(
            f"unsupported question type: {type(question).__name__}"
        )

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        return [
            DecisionResponse(
                model=self.model_name,
                answers={
                    question_name: self._answer(request, question_name, question)
                    for question_name, question in request.questions.items()
                },
                usage=Usage(),
            )
            for request in requests
        ]
