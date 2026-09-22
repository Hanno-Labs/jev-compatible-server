"""Configuration-driven LoRA plus calibrated custom-head decision readouts.

The published Open-Jev and SmallJev checkpoints are not ordinary generation
checkpoints: each combines a pinned base model, a PEFT adapter, a separately
saved head, and a calibration artifact.  This module owns that composition
without importing an author's serving package or branching on a model name.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .encoder_decoder import (
    _mapping,
    _template,
    aggregate_margin_answers,
    compile_margin_tasks,
    decision_metadata,
)
from .hidden_state_probe import render_probe_task
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
from .runtime import DecisionRuntime, RuntimeErrorBase, softmax


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeErrorBase(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeErrorBase(f"{name} must be a positive finite number")
    return result


def calibration_temperature(payload: Mapping[str, Any], field: str) -> float:
    """Extract the one saved temperature that calibrates every candidate logit."""

    if not isinstance(field, str) or not field:
        raise RuntimeErrorBase("decision.artifacts.calibration.field must be a string")
    if field not in payload:
        raise RuntimeErrorBase(f"calibration artifact is missing field: {field}")
    return _positive_number(payload[field], f"calibration artifact field {field!r}")


def custom_head_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = decision_metadata(config)
    if metadata.get("readout") not in {"openjev_scalar_head", "semantic_option_head"}:
        raise RuntimeErrorBase(
            "custom-head backend requires decision.readout=openjev_scalar_head "
            "or semantic_option_head"
        )
    _mapping(metadata.get("loader"), "decision.loader")
    _mapping(metadata.get("artifacts"), "decision.artifacts")
    _mapping(metadata.get("input"), "decision.input")
    return metadata


def render_custom_head_task(task: Any, metadata: Mapping[str, Any]) -> str:
    """Render a candidate exactly once from service-owned input metadata."""

    return render_probe_task(task, metadata)


def _openjev_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeErrorBase("Open-Jev content must be JSON serializable") from exc


@dataclass(frozen=True)
class OpenJevTask:
    question_id: str
    question: ChoiceQuestion | ScoreQuestion | NoulQuestion
    labels: tuple[str, ...]
    prompts: tuple[str, ...]


def compile_openjev_tasks(request: DecisionRequest) -> list[OpenJevTask]:
    """Match Open-Jev's isolated candidate and single-Noul prompt contract."""

    state = _openjev_content(request.state)
    tasks: list[OpenJevTask] = []
    for name, question in request.questions.items():
        instruction = _openjev_content(question.instructions)
        if isinstance(question, NoulQuestion) and question.criteria is not None:
            instruction += (
                f"\nYes means: {_openjev_content(question.criteria.true)}"
                f"\nNo means: {_openjev_content(question.criteria.false)}"
            )
        prefix = f"Context:\n{state}\n\nQuestion: {instruction}\n"
        if isinstance(question, NoulQuestion):
            tasks.append(
                OpenJevTask(
                    name,
                    question,
                    ("false", "true"),
                    (prefix + "Is the answer to this question yes? Answer Yes or No.",),
                )
            )
            continue
        if isinstance(question, ChoiceQuestion):
            labels = tuple(question.criteria)
            options = tuple(
                key if value is None else f"{key}: {_openjev_content(value)}"
                for key, value in question.criteria.items()
            )
        else:
            if len(question.criteria) > 10:
                raise RuntimeErrorBase("Open-Jev Score supports at most 10 levels")
            labels = tuple(str(index) for index in range(len(question.criteria)))
            options = tuple(_openjev_content(value) for value in question.criteria)
        prompts = tuple(
            prefix
            + f"Proposed answer: {option}\n"
            + "Is this proposed answer correct? Answer Yes or No."
            for option in options
        )
        tasks.append(OpenJevTask(name, question, labels, prompts))
    return tasks


def format_openjev_answers(
    tasks: Sequence[OpenJevTask], scores: Sequence[float], temperature: float
) -> dict[str, Any]:
    """Apply the checkpoint temperature before each typed Open-Jev readout."""

    temperature = _positive_number(temperature, "Open-Jev temperature")
    answers: dict[str, Any] = {}
    offset = 0
    for task in tasks:
        size = len(task.prompts)
        values = scores[offset : offset + size]
        if len(values) != size or not all(math.isfinite(value) for value in values):
            raise RuntimeErrorBase("Open-Jev scorer returned invalid candidate logits")
        offset += size
        if isinstance(task.question, NoulQuestion):
            probability = softmax([0.0, values[0] / temperature])[1]
            answers[task.question_id] = NoulAnswer(type="noul", noul=probability)
            continue
        probabilities = softmax([value / temperature for value in values])
        distribution = dict(zip(task.labels, probabilities, strict=True))
        confidence = max(probabilities)
        if isinstance(task.question, ChoiceQuestion):
            answers[task.question_id] = ChoiceAnswer(
                type="choice",
                choice=max(distribution, key=distribution.__getitem__),
                probabilities=distribution,
                confidence=confidence,
            )
        else:
            answers[task.question_id] = ScoreAnswer(
                type="score",
                score=math.fsum(index * probability for index, probability in enumerate(probabilities)),
                probabilities=distribution,
                confidence=confidence,
                legend=task.question.criteria,
            )
    if offset != len(scores):
        raise RuntimeErrorBase("Open-Jev scorer returned extra candidate logits")
    return answers


class ConfiguredCustomHeadBackend(DecisionRuntime):
    """Score independently rendered candidates through a LoRA and saved head.

    ``decision.head.kind=linear`` implements Open-Jev's saved ``nn.Linear``
    state dict.  ``semantic_layers`` evaluates a named sequence of linear
    tensor layers, allowing a SmallJev-style semantic OptionScorerHead to be
    expressed by checkpoint keys rather than a copied upstream Python class.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from peft import PeftModel
            from transformers import (
                AutoModel,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "custom-head backends require transformers, torch, peft, and "
                "huggingface-hub"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = custom_head_metadata(self.config)
        self._torch = torch
        loader = _mapping(self.metadata["loader"], "decision.loader")
        base_model = loader.get("base_model", model_id)
        if not isinstance(base_model, str):
            raise RuntimeErrorBase("decision.loader.base_model must be a string")
        revision = loader.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise RuntimeErrorBase("decision.loader.revision must be a string")
        trust_remote_code = loader.get("trust_remote_code", False)
        if not isinstance(trust_remote_code, bool):
            raise RuntimeErrorBase("decision.loader.trust_remote_code must be boolean")
        tokenizer_id = loader.get("tokenizer", base_model)
        if not isinstance(tokenizer_id, str):
            raise RuntimeErrorBase("decision.loader.tokenizer must be a string")
        requested_device = self.metadata.get("device", device)
        target = self._resolve_device(requested_device)
        dtype = self._resolve_dtype(self.metadata.get("dtype"), target)
        common: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            common["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **common)
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise RuntimeErrorBase("tokenizer must define a pad token or an EOS token")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        model_kwargs = dict(common)
        model_kwargs["dtype"] = dtype
        attention = loader.get("attn_implementation", "sdpa")
        if attention is not None:
            if not isinstance(attention, str):
                raise RuntimeErrorBase("decision.loader.attn_implementation must be a string or null")
            model_kwargs["attn_implementation"] = attention
        model_class = loader.get("model_class", "auto")
        if model_class == "auto":
            loaded_model = AutoModel.from_pretrained(base_model, **model_kwargs)
        elif model_class == "image_text_to_text":
            loaded_model = AutoModelForImageTextToText.from_pretrained(
                base_model, **model_kwargs
            )
        else:
            raise RuntimeErrorBase(
                "decision.loader.model_class must be auto or image_text_to_text"
            )
        backbone_path = loader.get("backbone_path")
        self._model = self._resolve_backbone(loaded_model, backbone_path)
        adapter = _mapping(loader.get("adapter"), "decision.loader.adapter")
        adapter_repo = adapter.get("repo")
        if not isinstance(adapter_repo, str):
            raise RuntimeErrorBase("decision.loader.adapter.repo must be a string")
        adapter_kwargs: dict[str, Any] = {}
        adapter_revision = adapter.get("revision")
        if adapter_revision is not None:
            if not isinstance(adapter_revision, str):
                raise RuntimeErrorBase("decision.loader.adapter.revision must be a string")
            adapter_kwargs["revision"] = adapter_revision
        adapter_subfolder = adapter.get("subfolder")
        if adapter_subfolder is not None:
            if not isinstance(adapter_subfolder, str):
                raise RuntimeErrorBase("decision.loader.adapter.subfolder must be a string")
            adapter_kwargs["subfolder"] = adapter_subfolder
        self._model = PeftModel.from_pretrained(self._model, adapter_repo, **adapter_kwargs)
        self._model.to(target)
        self._model.eval()
        self._device = next(self._model.parameters()).device

        input_config = _mapping(self.metadata["input"], "decision.input")
        _template(input_config.get("template"), "decision.input.template")
        self._max_length = self._positive_int(input_config.get("max_length"), "decision.input.max_length")
        self._batch_size = self._positive_int(self.metadata.get("batch_size", 8), "decision.batch_size")
        chat = _mapping(input_config.get("chat_template", {}), "decision.input.chat_template")
        self._add_generation_prompt = chat.get("add_generation_prompt", True)
        self._enable_thinking = chat.get("enable_thinking", False)
        if not isinstance(self._add_generation_prompt, bool) or not isinstance(self._enable_thinking, bool):
            raise RuntimeErrorBase("decision.input.chat_template flags must be boolean")

        artifacts = _mapping(self.metadata["artifacts"], "decision.artifacts")
        head_config = _mapping(artifacts.get("head"), "decision.artifacts.head")
        self._head_state = self._download_torch_state(hf_hub_download, head_config, "head")
        self._head = self._build_head(head_config)
        calibration = _mapping(artifacts.get("calibration"), "decision.artifacts.calibration")
        calibration_payload = self._download_json(hf_hub_download, calibration, "calibration")
        field = calibration.get("field", "temperature")
        self._temperature = calibration_temperature(calibration_payload, field)

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeErrorBase(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _resolve_backbone(model: Any, path: Any) -> Any:
        if path is None:
            return model
        if not isinstance(path, str) or not path:
            raise RuntimeErrorBase("decision.loader.backbone_path must be a non-empty string")
        value = model
        for part in path.split("."):
            if not hasattr(value, part):
                raise RuntimeErrorBase(
                    f"decision.loader.backbone_path is missing component: {part}"
                )
            value = getattr(value, part)
        return value

    def _resolve_device(self, requested: Any) -> str:
        torch = self._torch
        target = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
        if not isinstance(target, str):
            raise RuntimeErrorBase("decision.device must be a string")
        if target.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeErrorBase("CUDA was requested, but no CUDA device is available")
        return target

    def _resolve_dtype(self, value: Any, device: str) -> Any:
        torch = self._torch
        if value is None or value == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        mapping = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
        if value not in mapping:
            raise RuntimeErrorBase(f"unsupported decision dtype: {value!r}")
        return mapping[value]

    @staticmethod
    def _artifact_download_kwargs(config: Mapping[str, Any], name: str) -> dict[str, str]:
        repo = config.get("repo")
        file = config.get("file")
        if not isinstance(repo, str) or not isinstance(file, str):
            raise RuntimeErrorBase(f"decision.artifacts.{name} requires string repo and file")
        result = {"repo_id": repo, "filename": file}
        revision = config.get("revision")
        if revision is not None:
            if not isinstance(revision, str):
                raise RuntimeErrorBase(f"decision.artifacts.{name}.revision must be a string")
            result["revision"] = revision
        return result

    def _download_torch_state(self, download: Any, config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        path = download(**self._artifact_download_kwargs(config, name))
        payload = self._torch.load(path, map_location="cpu", weights_only=True)
        state_key = config.get("state_key")
        if state_key is not None:
            if not isinstance(state_key, str) or not isinstance(payload, Mapping) or state_key not in payload:
                raise RuntimeErrorBase(f"decision.artifacts.{name}.state_key is missing from artifact")
            payload = payload[state_key]
        if not isinstance(payload, Mapping):
            raise RuntimeErrorBase(f"decision {name} artifact must be a tensor state mapping")
        return payload

    def _download_json(self, download: Any, config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        path = download(**self._artifact_download_kwargs(config, name))
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeErrorBase(f"decision {name} artifact is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeErrorBase(f"decision {name} artifact must be a JSON object")
        return payload

    def _build_head(self, config: Mapping[str, Any]) -> Any:
        kind = config.get("kind")
        hidden_size = getattr(self._model.config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise RuntimeErrorBase("base model config must expose a positive hidden_size")
        torch = self._torch
        if kind == "linear":
            head = torch.nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)
            try:
                head.load_state_dict(self._head_state, strict=True)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeErrorBase("linear head artifact does not match hidden_size -> 1") from exc
            return head.to(self._device).eval()
        if kind != "semantic_layers":
            raise RuntimeErrorBase("decision.artifacts.head.kind must be linear or semantic_layers")
        layers = config.get("layers")
        if not isinstance(layers, list) or not layers:
            raise RuntimeErrorBase("semantic_layers head requires a non-empty layers array")
        parsed: list[tuple[Any, Any | None, str]] = []
        for index, raw in enumerate(layers):
            layer = _mapping(raw, f"decision.artifacts.head.layers[{index}]")
            weight_key = layer.get("weight")
            bias_key = layer.get("bias")
            activation = layer.get("activation", "identity")
            if not isinstance(weight_key, str) or weight_key not in self._head_state:
                raise RuntimeErrorBase(f"semantic head layer {index} is missing its weight tensor")
            if bias_key is not None and (not isinstance(bias_key, str) or bias_key not in self._head_state):
                raise RuntimeErrorBase(f"semantic head layer {index} is missing its bias tensor")
            if activation not in {"identity", "gelu", "relu", "silu", "tanh"}:
                raise RuntimeErrorBase(f"semantic head layer {index} has unsupported activation")
            parsed.append((self._head_state[weight_key].to(self._device), self._head_state[bias_key].to(self._device) if isinstance(bias_key, str) else None, activation))
        return parsed

    def _apply_head(self, hidden: Any) -> Any:
        torch = self._torch
        if not isinstance(self._head, list):
            return self._head(hidden.float()).squeeze(-1)
        values = hidden.float()
        for weight, bias, activation in self._head:
            values = torch.nn.functional.linear(values, weight.float(), None if bias is None else bias.float())
            values = {"identity": lambda x: x, "gelu": torch.nn.functional.gelu, "relu": torch.relu, "silu": torch.nn.functional.silu, "tanh": torch.tanh}[activation](values)
        if values.ndim != 1:
            if values.ndim != 2 or values.shape[1] != 1:
                raise RuntimeErrorBase("semantic OptionScorerHead must produce one scalar per candidate")
            values = values[:, 0]
        return values

    def _score_texts(self, texts: Sequence[str]) -> tuple[list[float], list[int]]:
        torch = self._torch
        scores: list[float] = []
        input_tokens: list[int] = []
        for start in range(0, len(texts), self._batch_size):
            messages = list(texts[start : start + self._batch_size])
            rendered = [self._tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=self._add_generation_prompt, enable_thinking=self._enable_thinking) for text in messages]
            encoded = self._tokenizer(rendered, return_tensors="pt", padding=True, truncation=False)
            mask = encoded.get("attention_mask")
            if mask is None:
                raise RuntimeErrorBase("tokenizer output has no attention_mask")
            if int(mask.sum(dim=1).max().item()) > self._max_length:
                raise RuntimeErrorBase(f"input length exceeds configured max_length={self._max_length}; no silent truncation")
            input_tokens.extend(int(value) for value in mask.sum(dim=1).tolist())
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self._model(**encoded, use_cache=False, return_dict=True)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeErrorBase("base model output has no last_hidden_state")
            positions = encoded["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            values = self._apply_head(hidden[rows, positions])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeErrorBase("custom head produced non-finite candidate logits")
            # Preserve native head logits here.  Each typed readout applies the
            # checkpoint's saved temperature exactly once when it constructs
            # the final probability distribution.
            scores.extend(float(value) for value in values.cpu().tolist())
        return scores, input_tokens

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        compiled = [compile_margin_tasks(request, self.metadata) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        scores, token_counts = self._score_texts([render_custom_head_task(task, self.metadata) for task in tasks])
        responses: list[DecisionResponse] = []
        offset = 0
        for request, request_tasks in zip(requests, compiled, strict=True):
            end = offset + len(request_tasks)
            responses.append(DecisionResponse(model=self.model_name, answers=aggregate_margin_answers(request, request_tasks, scores[offset:end], self.metadata), usage=Usage(input_tokens=sum(token_counts[offset:end]))))
            offset = end
        return responses


class OpenJevScalarHeadBackend(ConfiguredCustomHeadBackend):
    """Faithful Open-Jev LoRA, scalar-head, and temperature composition."""

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        compiled = [compile_openjev_tasks(request) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        prompts = [prompt for task in tasks for prompt in task.prompts]
        scores, token_counts = self._score_texts(prompts)
        responses: list[DecisionResponse] = []
        task_offset = 0
        score_offset = 0
        for request, request_tasks in zip(requests, compiled, strict=True):
            task_end = task_offset + len(request_tasks)
            request_tasks = tasks[task_offset:task_end]
            count = sum(len(task.prompts) for task in request_tasks)
            score_end = score_offset + count
            responses.append(
                DecisionResponse(
                    model=self.model_name,
                    answers=format_openjev_answers(
                        request_tasks, scores[score_offset:score_end], self._temperature
                    ),
                    usage=Usage(input_tokens=sum(token_counts[score_offset:score_end])),
                )
            )
            task_offset = task_end
            score_offset = score_end
        return responses


def build_smalljev_semantic_ids(
    tokenizer: Any,
    state: str,
    question: str,
    options: Sequence[str],
    *,
    max_length: int = 1024,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Exact public SmallJev span construction, including its state-only trim."""

    if len(options) > 26:
        raise RuntimeErrorBase("SmallJev semantic Choice supports at most 26 options")
    current_state = state
    for _ in range(4):
        head = tokenizer(
            f"State: {current_state}\nQuestion: {question}\nOptions:",
            add_special_tokens=True,
        )["input_ids"]
        chunks = [
            (
                tokenizer(f"\n{chr(ord('A') + index)}.", add_special_tokens=False)[
                    "input_ids"
                ],
                tokenizer(f" {option}", add_special_tokens=False)["input_ids"],
            )
            for index, option in enumerate(options)
        ]
        tail = tokenizer("\nAnswer with a single letter:", add_special_tokens=False)[
            "input_ids"
        ]
        total = len(head) + sum(len(marker) + len(text) for marker, text in chunks) + len(tail)
        if total <= max_length or len(current_state) < 100:
            break
        current_state = current_state[: max(50, len(current_state) - int((total - max_length) * 1.5))]
    input_ids = list(head)
    spans: list[tuple[int, int]] = []
    for marker, text in chunks:
        input_ids.extend(marker)
        spans.append((len(input_ids), len(input_ids) + len(text)))
        input_ids.extend(text)
    input_ids.extend(tail)
    return input_ids, spans


class SmallJevSemanticBackend(DecisionRuntime):
    """Faithful published SmallJev semantic-v9 Choice scorer.

    The public semantic runtime does not apply a saved calibration artifact and
    only uses ``OptionScorerHead`` for Choice.  Noul and Score use a separate
    LM-verbalizer path, so this backend deliberately exposes Choice only.
    ``QuestionTypeRuntime`` supplies explicit unsupported responses for the
    remaining wire types when the registry declares that boundary.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "SmallJev semantic backend requires transformers, torch, peft, "
                "and huggingface-hub"
            ) from exc
        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = decision_metadata(self.config)
        if self.metadata.get("readout") != "semantic_option_head":
            raise RuntimeErrorBase(
                "SmallJevSemanticBackend requires decision.readout=semantic_option_head"
            )
        self._torch = torch
        loader = _mapping(self.metadata.get("loader"), "decision.loader")
        base_model = loader.get("base_model", model_id)
        revision = loader.get("revision")
        if not isinstance(base_model, str) or not isinstance(revision, str):
            raise RuntimeErrorBase(
                "SmallJev loader requires pinned string base_model and revision"
            )
        tokenizer_id = loader.get("tokenizer", base_model)
        if not isinstance(tokenizer_id, str):
            raise RuntimeErrorBase("decision.loader.tokenizer must be a string")
        requested = self.metadata.get("device", device)
        target = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
        if not isinstance(target, str) or (target.startswith("cuda") and not torch.cuda.is_available()):
            raise RuntimeErrorBase("requested SmallJev device is unavailable")
        dtype = torch.bfloat16 if target.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=revision,
            dtype=dtype,
            attn_implementation="eager",
        )
        adapter = _mapping(loader.get("adapter"), "decision.loader.adapter")
        adapter_repo = adapter.get("repo")
        adapter_revision = adapter.get("revision")
        adapter_subfolder = adapter.get("subfolder")
        if not isinstance(adapter_repo, str) or not isinstance(adapter_revision, str) or not isinstance(adapter_subfolder, str):
            raise RuntimeErrorBase(
                "SmallJev adapter requires pinned repo, revision, and subfolder strings"
            )
        self._model = PeftModel.from_pretrained(
            self._model,
            adapter_repo,
            revision=adapter_revision,
            subfolder=adapter_subfolder,
        ).to(target).eval()
        self._device = next(self._model.parameters()).device
        artifacts = _mapping(self.metadata.get("artifacts"), "decision.artifacts")
        head = _mapping(artifacts.get("head"), "decision.artifacts.head")
        repo = head.get("repo")
        file = head.get("file")
        head_revision = head.get("revision")
        if not isinstance(repo, str) or not isinstance(file, str) or not isinstance(head_revision, str):
            raise RuntimeErrorBase("SmallJev head requires pinned repo, file, and revision")
        blob = torch.load(
            hf_hub_download(repo, file, revision=head_revision),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(blob, Mapping) or not isinstance(blob.get("hidden_size"), int):
            raise RuntimeErrorBase("SmallJev OptionScorerHead artifact is malformed")
        state = blob.get("state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeErrorBase("SmallJev OptionScorerHead lacks state_dict")
        linear_state = {
            name: state.get(name, state.get(f"scorer.{name}"))
            for name in ("weight", "bias")
        }
        if any(value is None for value in linear_state.values()):
            raise RuntimeErrorBase(
                "SmallJev OptionScorerHead lacks scorer.weight or scorer.bias"
            )
        self._head = torch.nn.Linear(int(blob["hidden_size"]), 1)
        try:
            self._head.load_state_dict(linear_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeErrorBase("SmallJev OptionScorerHead state is incompatible") from exc
        self._head.to(self._device).eval()
        self._max_length = 1024

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        torch = self._torch
        responses: list[DecisionResponse] = []
        for request in requests:
            answers: dict[str, Any] = {}
            input_tokens = 0
            state = _openjev_content(request.state)
            for name, question in request.questions.items():
                if not isinstance(question, ChoiceQuestion):
                    raise RuntimeErrorBase(
                        "SmallJev semantic-v9 faithfully supports Choice only; "
                        "declare decision.question_types=[\"choice\"]"
                    )
                labels = list(question.criteria)
                options = [
                    key if value is None else f"{key}: {_openjev_content(value)}"
                    for key, value in question.criteria.items()
                ]
                ids, spans = build_smalljev_semantic_ids(
                    self._tokenizer,
                    state,
                    _openjev_content(question.instructions),
                    options,
                    max_length=self._max_length,
                )
                input_tokens += len(ids)
                tensor = torch.tensor([ids], device=self._device)
                with torch.inference_mode():
                    output = self._model(
                        input_ids=tensor,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                hidden = output.hidden_states[-1][0]
                representations = [
                    hidden[-1, :] if end <= start else hidden[start:end, :].float().mean(0)
                    for start, end in spans
                ]
                with torch.inference_mode():
                    logits = self._head(torch.stack(representations).float()).squeeze(-1)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeErrorBase("SmallJev OptionScorerHead produced non-finite logits")
                probabilities = softmax([float(value) for value in logits.cpu().tolist()])
                distribution = dict(zip(labels, probabilities, strict=True))
                answers[name] = ChoiceAnswer(
                    type="choice",
                    choice=max(distribution, key=distribution.__getitem__),
                    probabilities=distribution,
                    confidence=max(probabilities),
                )
            responses.append(
                DecisionResponse(
                    model=self.model_name,
                    answers=answers,
                    usage=Usage(input_tokens=input_tokens),
                )
            )
        return responses
