"""Certo's published per-option decision head behind the JEV wire contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .encoder_decoder import decision_metadata
from .protocol import (
    Answer,
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
from .runtime import DecisionRuntime, RuntimeErrorBase


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeErrorBase("Certo content must be JSON serializable") from exc


def certo_options(question: ChoiceQuestion | ScoreQuestion | NoulQuestion) -> list[tuple[str, str]]:
    """Map typed criteria to the option IDs and descriptions Certo was trained on."""

    if isinstance(question, ChoiceQuestion):
        return [
            (key, key if description is None else _content(description))
            for key, description in question.criteria.items()
        ]
    if isinstance(question, ScoreQuestion):
        return [(str(index), _content(description)) for index, description in enumerate(question.criteria)]
    if question.criteria is None:
        instruction = _content(question.instructions)
        return [(label, f"{instruction} — {label}") for label in ("true", "false")]
    return [
        ("true", _content(question.criteria.true)),
        ("false", _content(question.criteria.false)),
    ]


def certo_answer(
    question: ChoiceQuestion | ScoreQuestion | NoulQuestion,
    probabilities: Mapping[str, float],
) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    labels = [key for key, _ in certo_options(question)]
    if set(probabilities) != set(labels):
        raise RuntimeErrorBase("Certo probabilities do not match presented options")
    values = [probabilities[key] for key in labels]
    if any(not math.isfinite(value) or value < 0.0 for value in values) or not math.isclose(
        sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise RuntimeErrorBase("Certo produced an invalid probability distribution")
    if isinstance(question, NoulQuestion):
        return NoulAnswer(type="noul", noul=probabilities["true"])
    winner = max(labels, key=probabilities.__getitem__)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            type="choice", choice=winner, probabilities=dict(probabilities),
            confidence=probabilities[winner],
        )
    return ScoreAnswer(
        type="score",
        score=sum(index * probabilities[str(index)] for index in range(len(labels))),
        probabilities=dict(probabilities),
        confidence=probabilities[winner],
        legend=question.criteria,
    )


class CertoDecisionBackend(DecisionRuntime):
    """Load the pinned custom checkpoint and score runtime options independently."""

    def __init__(self, model_id: str, *, config: dict[str, Any] | None = None) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            from .certo_model import CertoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase("Certo requires torch, transformers, and huggingface_hub") from exc

        self.config = config or {}
        metadata = decision_metadata(self.config)
        if metadata.get("readout") != "certo_decision":
            raise RuntimeErrorBase("Certo requires decision.readout=certo_decision")
        revision = metadata.get("revision")
        backbone_revision = metadata.get("backbone_revision")
        if not isinstance(revision, str) or not isinstance(backbone_revision, str):
            raise RuntimeErrorBase("Certo requires pinned checkpoint and backbone revisions")
        self.model_name = str(self.config.get("model", model_id))
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        snapshot = Path(snapshot_download(model_id, revision=revision))
        with (snapshot / "certo_config.json").open() as handle:
            checkpoint_config = json.load(handle)
        if not isinstance(checkpoint_config, dict) or checkpoint_config.get("kind") != "generic":
            raise RuntimeErrorBase("Certo checkpoint lacks the published generic architecture")
        backbone = checkpoint_config.get("backbone")
        heads = checkpoint_config.get("heads", 8)
        temperature = checkpoint_config.get("temperature", 1.0)
        if not isinstance(backbone, str) or not isinstance(heads, int) or isinstance(heads, bool):
            raise RuntimeErrorBase("Certo checkpoint has invalid backbone or attention heads")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise RuntimeErrorBase("Certo checkpoint has invalid calibration temperature")
        self._temperature = float(temperature)
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot)
        self._model = CertoModel(backbone, backbone_revision, heads)
        state = torch.load(snapshot / "model.pt", map_location="cpu", weights_only=True)
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()
        self._max_state_len = self._positive_int(metadata.get("max_state_len", 256), "max_state_len")
        self._max_option_len = self._positive_int(metadata.get("max_option_len", 64), "max_option_len")
        self._option_batch_size = self._positive_int(
            metadata.get("option_batch_size", 16), "option_batch_size"
        )

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeErrorBase(f"Certo {name} must be a positive integer")
        return value

    def _probabilities(self, state: str, options: list[tuple[str, str]]) -> tuple[dict[str, float], int]:
        torch = self._torch
        state_encoding = self._tokenizer(
            [state], padding="max_length", truncation=True,
            max_length=self._max_state_len, return_tensors="pt",
        )
        state_ids = state_encoding["input_ids"].to(self._device)
        state_mask = state_encoding["attention_mask"].to(self._device)
        input_tokens = int(state_mask.sum().item())
        all_logits: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(options), self._option_batch_size):
                chunk = options[start : start + self._option_batch_size]
                encoded = self._tokenizer(
                    [description for _, description in chunk],
                    padding="max_length", truncation=True,
                    max_length=self._max_option_len, return_tensors="pt",
                )
                option_ids = encoded["input_ids"].unsqueeze(0).to(self._device)
                option_mask = encoded["attention_mask"].unsqueeze(0).to(self._device)
                input_tokens += int(option_mask.sum().item())
                valid = torch.ones(1, len(chunk), dtype=torch.bool, device=self._device)
                logits = self._model(state_ids, state_mask, option_ids, option_mask, valid)
                all_logits.extend(float(value) for value in logits[0].float().cpu().tolist())
        if not all(math.isfinite(value) for value in all_logits):
            raise RuntimeErrorBase("Certo produced non-finite option logits")
        logits_tensor = torch.tensor(all_logits, dtype=torch.float32)
        values = torch.softmax(logits_tensor / self._temperature, dim=0).tolist()
        return dict(zip((key for key, _ in options), values, strict=True)), input_tokens

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        responses: list[DecisionResponse] = []
        for request in requests:
            state = _content(request.state)
            answers: dict[str, Answer] = {}
            input_tokens = 0
            for name, question in request.questions.items():
                options = certo_options(question)
                probabilities, used = self._probabilities(state, options)
                answers[name] = certo_answer(question, probabilities)
                input_tokens += used
            responses.append(
                DecisionResponse(
                    model=self.model_name, answers=answers,
                    usage=Usage(input_tokens=input_tokens),
                )
            )
        return responses
