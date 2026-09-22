"""Exact option-logprob readout used by ``us/jev-local``.

The upstream implementation at
``us/jev-local@56bfc2a96543f2fc6a4d4227460a2c17553c6e24`` uses Qwen3.5-9B,
a direct teacher-forced forward pass for each option, and no generation. This
module keeps that contract separate from the service's other causal-option
profiles so its prompt, token boundary, and calibration remain auditable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

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
from .runtime import DecisionRuntime, RuntimeErrorBase, softmax

UPSTREAM_SOURCE_REVISION = "56bfc2a96543f2fc6a4d4227460a2c17553c6e24"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CANDIDATE_BATCH_SIZE = 8

# The plain-prompt values are the upstream shipped defaults for Qwen3.5-9B.
DEFAULT_TEMPERATURES: dict[str, float] = {
    "choice": 0.5,
    "noul": 0.25,
    "score": 0.25,
}


def _as_text(value: Any) -> str:
    """Preserve upstream's ``str(...)`` rendering for structured wire values."""

    return str(value)


def render_prefix(
    state: Any,
    instructions: Any,
    options: list[str] | None,
    option_help: dict[str, str] | None,
    levels: list[Any] | None,
    noul_help: dict[str, str] | None = None,
) -> str:
    """Render the upstream plain prompt byte-for-byte in structure."""

    lines = [f"State: {_as_text(state)}", f"Question: {_as_text(instructions)}"]
    if option_help:
        lines.extend(f"{key} means: {value}" for key, value in option_help.items())
    if options is not None:
        lines.append("Answer with exactly one of: " + " | ".join(options))
    if noul_help:
        lines.extend(
            (f"Yes means: {noul_help['true']}", f"No means: {noul_help['false']}")
        )
    elif levels is not None:
        lines.append(
            "Levels: "
            + ", ".join(
                f"{index} = {_as_text(level)}" for index, level in enumerate(levels)
            )
        )
    lines.append("Answer:")
    return "\n".join(lines)


def answer_from_mean_logprobabilities(
    question: ChoiceQuestion | ScoreQuestion | NoulQuestion,
    means: Sequence[float],
    *,
    temperature: float,
) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    """Apply the upstream probability conversion without rounding distributions."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeErrorBase(
            "jev-local temperature must be a finite positive number"
        )
    if not means or not all(math.isfinite(value) for value in means):
        raise RuntimeErrorBase(
            "jev-local returned invalid option mean log-probabilities"
        )

    if isinstance(question, ChoiceQuestion):
        labels = list(question.criteria)
        if len(labels) != len(means):
            raise RuntimeErrorBase(
                "jev-local choice scorer returned the wrong option count"
            )
        probabilities = softmax([value / temperature for value in means])
        distribution = dict(zip(labels, probabilities, strict=True))
        choice = max(distribution, key=distribution.__getitem__)
        return ChoiceAnswer(
            type="choice",
            choice=choice,
            probabilities=distribution,
            confidence=distribution[choice],
        )

    if isinstance(question, ScoreQuestion):
        if len(question.criteria) != len(means):
            raise RuntimeErrorBase(
                "jev-local score scorer returned the wrong level count"
            )
        probabilities = softmax([value / temperature for value in means])
        distribution = {str(index): probability for index, probability in enumerate(probabilities)}
        return ScoreAnswer(
            type="score",
            score=sum(index * probability for index, probability in enumerate(probabilities)),
            probabilities=distribution,
            confidence=max(probabilities),
            legend=question.criteria,
        )

    if isinstance(question, NoulQuestion):
        if len(means) != 2:
            raise RuntimeErrorBase("jev-local noul scorer requires Yes and No means")
        return NoulAnswer(
            type="noul",
            noul=softmax([value / temperature for value in means])[0],
        )

    raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")


class JevLocalOptionsBackend(DecisionRuntime):
    """Qwen causal-LM backend that reproduces the public jev-local scorer."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "JevLocalOptionsBackend requires transformers and torch"
            ) from exc

        metadata = (config or {}).get("decision", {})
        if not isinstance(metadata, dict):
            raise RuntimeErrorBase("decision metadata must be an object")
        loader = metadata.get("loader", {})
        if not isinstance(loader, dict):
            raise RuntimeErrorBase("decision.loader must be an object")
        base_model = loader.get("model", DEFAULT_MODEL)
        revision = loader.get("revision", DEFAULT_MODEL_REVISION)
        if not isinstance(base_model, str) or not isinstance(revision, str):
            raise RuntimeErrorBase("jev-local loader model and revision must be strings")

        self.model_name = model_id
        self._torch = torch
        self._head_temperatures = self._configured_temperatures(metadata)
        self._chat = metadata.get("chat", False)
        if not isinstance(self._chat, bool):
            raise RuntimeErrorBase("jev-local chat must be boolean")
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": True,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(base_model, **load_kwargs)
        model_kwargs: dict[str, Any] = {
            **load_kwargs,
            "dtype": "auto",
            "device_map": "auto",
        }
        if device != "auto":
            model_kwargs.pop("device_map")
        self._model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if device != "auto":
            self._model.to(device)
        self._model.eval()

    @staticmethod
    def _configured_temperatures(metadata: dict[str, Any]) -> dict[str, float]:
        configured = metadata.get("temperatures", {})
        if not isinstance(configured, dict):
            raise RuntimeErrorBase("jev-local temperatures must be an object")
        values = dict(DEFAULT_TEMPERATURES)
        for head, value in configured.items():
            if (
                head not in values
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise RuntimeErrorBase(
                    "jev-local temperatures must define numeric choice, noul, score values"
                )
            if not math.isfinite(float(value)) or value <= 0:
                raise RuntimeErrorBase(
                    "jev-local temperatures must be finite and positive"
                )
            values[head] = float(value)
        return values

    def _maybe_wrap_chat(self, body: str) -> str:
        if not self._chat or getattr(self._tokenizer, "chat_template", None) is None:
            return body
        try:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(rendered, str):
            raise RuntimeErrorBase("jev-local chat template did not return text")
        return rendered

    def _candidate_means(self, prefix: str, candidates: list[str]) -> list[float]:
        if not candidates:
            raise RuntimeErrorBase("jev-local has no candidates to score")
        prefix_ids = self._tokenizer(
            prefix, return_tensors="pt"
        ).input_ids[0].tolist()
        if len(prefix_ids) > 4096:
            raise RuntimeErrorBase(
                f"jev-local prefix too long: {len(prefix_ids)} tokens"
            )
        encoded_candidates: list[tuple[list[int], int]] = []
        for candidate in candidates:
            encoded = self._tokenizer(prefix + " " + candidate, return_tensors="pt")
            full_ids = encoded.input_ids[0].tolist()
            if len(full_ids) > 4096:
                raise RuntimeErrorBase(
                    f"jev-local input too long: {len(full_ids)} tokens"
                )
            common = 0
            while (
                common < len(prefix_ids)
                and common < len(full_ids)
                and prefix_ids[common] == full_ids[common]
            ):
                common += 1
            if common == 0:
                raise RuntimeErrorBase("jev-local candidate shares no token prefix")
            if common == len(full_ids):
                raise RuntimeErrorBase(
                    f"jev-local candidate {candidate!r} scores zero tokens"
                )
            encoded_candidates.append((full_ids, common))

        means: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(encoded_candidates), CANDIDATE_BATCH_SIZE):
                chunk = encoded_candidates[start : start + CANDIDATE_BATCH_SIZE]
                max_length = max(len(full_ids) for full_ids, _ in chunk)
                tokens = self._torch.zeros(
                    (len(chunk), max_length), dtype=self._torch.long
                )
                attention = self._torch.zeros_like(tokens)
                for row, (full_ids, _) in enumerate(chunk):
                    tokens[row, : len(full_ids)] = self._torch.tensor(full_ids)
                    attention[row, : len(full_ids)] = 1
                tokens = tokens.to(self._model.device)
                attention = attention.to(self._model.device)
                output = self._model(tokens, attention_mask=attention)
                for row, (full_ids, common) in enumerate(chunk):
                    candidate_logits = output.logits[
                        row, common - 1 : len(full_ids) - 1, :
                    ]
                    log_probabilities = self._torch.log_softmax(
                        candidate_logits, dim=-1
                    )
                    scored_tokens = tokens[row, common : len(full_ids)]
                    terms = log_probabilities.gather(
                        dim=-1, index=scored_tokens.unsqueeze(-1)
                    ).squeeze(-1).tolist()
                    means.append(sum(terms) / len(terms))
        return means

    def _score_question(
        self,
        request: DecisionRequest,
        question: ChoiceQuestion | ScoreQuestion | NoulQuestion,
    ) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
        if isinstance(question, ChoiceQuestion):
            candidates = list(question.criteria)
            help_text = {
                key: _as_text(value)
                for key, value in question.criteria.items()
                if value is not None
            }
            prefix = render_prefix(
                request.state, question.instructions, candidates, help_text or None, None
            )
        elif isinstance(question, ScoreQuestion):
            candidates = [str(index) for index in range(len(question.criteria))]
            prefix = render_prefix(
                request.state, question.instructions, candidates, None, question.criteria
            )
        else:
            candidates = ["Yes", "No"]
            help_text = None
            if question.criteria is not None:
                help_text = {
                    "true": _as_text(question.criteria.true),
                    "false": _as_text(question.criteria.false),
                }
            prefix = render_prefix(
                request.state, question.instructions, candidates, None, None, help_text
            )
        means = self._candidate_means(self._maybe_wrap_chat(prefix), candidates)
        return answer_from_mean_logprobabilities(
            question, means, temperature=self._head_temperatures[question.type]
        )

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        responses: list[DecisionResponse] = []
        for request in requests:
            answers: dict[str, Answer] = {
                name: self._score_question(request, question)
                for name, question in request.questions.items()
            }
            responses.append(
                DecisionResponse(model=self.model_name, answers=answers, usage=Usage())
            )
        return responses
