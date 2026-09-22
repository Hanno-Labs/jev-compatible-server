"""Generic final-hidden-state probe readout for Transformers checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .encoder_decoder import (
    MarginTask,
    _mapping,
    _template,
    aggregate_margin_answers,
    compile_margin_tasks,
    decision_metadata,
    render_content,
)
from .protocol import DecisionRequest, DecisionResponse, NoulAnswer, Usage
from .runtime import DecisionRuntime, RuntimeErrorBase


def _state_path(state: Any, path: str) -> Any:
    value = state
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise RuntimeErrorBase(f"decision input state path is missing: {path}")
        value = value[part]
    return value


def render_probe_input(request: DecisionRequest, metadata: Mapping[str, Any]) -> str:
    """Render a shared-state probe input entirely from declarative metadata."""

    input_config = _mapping(metadata.get("input"), "decision.input")
    template = _template(input_config.get("template"), "decision.input.template")
    fields = _mapping(input_config.get("fields", {}), "decision.input.fields")
    values: dict[str, str] = {"state": render_content(request.state)}
    for name, path in fields.items():
        if not isinstance(name, str) or not isinstance(path, str):
            raise RuntimeErrorBase("decision.input.fields must map strings to state paths")
        values[name] = render_content(_state_path(request.state, path))
    try:
        return template.format_map(values)
    except KeyError as exc:
        raise RuntimeErrorBase(
            f"decision.input.template references an undeclared field: {exc.args[0]}"
        ) from exc


def render_probe_task(task: MarginTask, metadata: Mapping[str, Any]) -> str:
    """Render one candidate as an external answer for a verifier probe."""

    input_config = _mapping(metadata.get("input"), "decision.input")
    template = _template(input_config.get("template"), "decision.input.template")
    try:
        return template.format_map(
            {
                "query": task.query,
                "instructions": task.instruction,
                "candidate": task.document,
            }
        )
    except KeyError as exc:
        raise RuntimeErrorBase(
            f"decision.input.template references an undeclared candidate field: "
            f"{exc.args[0]}"
        ) from exc


class HiddenStateProbeBackend(DecisionRuntime):
    """Apply a configured linear or RBF probe to one hidden state per request."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import numpy as np
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "HiddenStateProbeBackend requires the 'transformers' optional dependency"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = decision_metadata(self.config)
        if self.metadata.get("readout") != "hidden_state_probe":
            raise RuntimeErrorBase(
                "HiddenStateProbeBackend requires decision.readout=hidden_state_probe"
            )
        self._np = np
        self._torch = torch

        loader = _mapping(self.metadata.get("loader", {}), "decision.loader")
        revision = loader.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise RuntimeErrorBase("decision.loader.revision must be a string")
        trust_remote_code = loader.get("trust_remote_code", False)
        if not isinstance(trust_remote_code, bool):
            raise RuntimeErrorBase("decision.loader.trust_remote_code must be boolean")
        tokenizer_id = loader.get("tokenizer", model_id)
        if not isinstance(tokenizer_id, str):
            raise RuntimeErrorBase("decision.loader.tokenizer must be a string")
        common_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            common_kwargs["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **common_kwargs)
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise RuntimeErrorBase("tokenizer must define a pad token or an EOS token")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        requested_device = self.metadata.get("device", device)
        target = (
            "cuda"
            if requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested_device == "auto"
            else requested_device
        )
        if not isinstance(target, str):
            raise RuntimeErrorBase("decision.device must be a string")
        if target.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeErrorBase("CUDA was requested, but no CUDA device is available")
        dtype = self._resolve_dtype(self.metadata.get("dtype"), target)
        model_kwargs = dict(common_kwargs)
        model_kwargs["dtype"] = dtype
        model_kwargs["low_cpu_mem_usage"] = True
        attention = loader.get("attn_implementation", "sdpa")
        if attention is not None:
            if not isinstance(attention, str):
                raise RuntimeErrorBase(
                    "decision.loader.attn_implementation must be a string or null"
                )
            model_kwargs["attn_implementation"] = attention
        device_map = loader.get("device_map")
        if device_map is not None:
            if not isinstance(device_map, str):
                raise RuntimeErrorBase("decision.loader.device_map must be a string")
            model_kwargs["device_map"] = device_map
        self._model = AutoModel.from_pretrained(model_id, **model_kwargs)
        if device_map is None:
            self._model.to(target)
        self._model.eval()

        input_config = _mapping(self.metadata.get("input"), "decision.input")
        self._input_mode = input_config.get("mode", "shared_state")
        if self._input_mode not in {"shared_state", "candidate_tasks"}:
            raise RuntimeErrorBase(
                "decision.input.mode must be shared_state or candidate_tasks"
            )
        max_length = input_config.get("max_length", 2048)
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise RuntimeErrorBase("decision.input.max_length must be a positive integer")
        self._max_length = max_length
        batch_size = self.metadata.get("batch_size", 8)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise RuntimeErrorBase("decision.batch_size must be a positive integer")
        self._batch_size = batch_size
        hidden = _mapping(self.metadata.get("hidden", {}), "decision.hidden")
        if hidden.get("token", "last_non_pad") != "last_non_pad":
            raise RuntimeErrorBase(
                "hidden_state_probe currently supports only hidden.token=last_non_pad"
            )

        probe = _mapping(self.metadata.get("probe"), "decision.probe")
        self._probe_kind = probe.get("kind")
        if self._probe_kind not in {"linear", "rbf"}:
            raise RuntimeErrorBase("decision.probe.kind must be linear or rbf")
        probe_file = probe.get("file")
        if not isinstance(probe_file, str):
            raise RuntimeErrorBase("decision.probe.file must be a string")
        probe_repo = probe.get("repo", model_id)
        if probe_repo == "auto":
            probe_repo = model_id
        if not isinstance(probe_repo, str):
            raise RuntimeErrorBase("decision.probe.repo must be a string")
        probe_revision = probe.get("revision", revision)
        if probe_revision is not None and not isinstance(probe_revision, str):
            raise RuntimeErrorBase("decision.probe.revision must be a string")
        local_probe = hf_hub_download(
            repo_id=probe_repo,
            filename=probe_file,
            revision=probe_revision,
        )
        payload = np.load(local_probe, allow_pickle=False)
        required = (
            ("w", "mu", "sd")
            if self._probe_kind == "linear"
            else ("anchors", "alpha", "mu", "sd", "med", "gamma")
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeErrorBase(
                f"decision probe is missing arrays: {', '.join(missing)}"
            )
        self._probe = {
            name: torch.from_numpy(
                np.asarray(payload[name], dtype=np.float32).copy()
            )
            for name in required
        }
        if bool((self._probe["sd"] == 0).any()):
            raise RuntimeErrorBase("decision probe sd contains zero")
        self._calibration: dict[str, float] | None
        self._output = _mapping(self.metadata.get("output", {}), "decision.output")
        transform = self._output.get("transform")
        if self._input_mode == "shared_state" and transform not in {
            "sigmoid",
            "probe_calibrated_sigmoid",
        }:
            raise RuntimeErrorBase(
                "shared_state probes require decision.output.transform="
                "sigmoid or probe_calibrated_sigmoid"
            )
        if transform == "probe_calibrated_sigmoid":
            calibration = ("s_mean", "s_std", "cal_A", "cal_B")
            absent = [name for name in calibration if name not in payload]
            if absent:
                raise RuntimeErrorBase(
                    f"decision probe is missing calibration arrays: {', '.join(absent)}"
                )
            self._calibration = {
                name: float(payload[name]) for name in calibration
            }
            if self._calibration["s_std"] == 0:
                raise RuntimeErrorBase("decision probe calibration s_std is zero")
        else:
            self._calibration = None

    def _resolve_dtype(self, value: Any, device: str) -> Any:
        torch = self._torch
        if value is None or value == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        if value in {"bfloat16", "bf16", "torch.bfloat16"}:
            return torch.bfloat16
        if value in {"float16", "fp16", "torch.float16"}:
            return torch.float16
        if value in {"float32", "fp32", "torch.float32"}:
            return torch.float32
        raise RuntimeErrorBase(f"unsupported decision dtype: {value!r}")

    def _input_device(self) -> Any:
        embeddings = self._model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
        return next(self._model.parameters()).device

    def _raw_scores(self, hidden: Any) -> Any:
        torch = self._torch
        probe = {name: value.to(hidden.device) for name, value in self._probe.items()}
        standardized = (hidden.float() - probe["mu"]) / probe["sd"]
        if self._probe_kind == "linear":
            scores = standardized @ probe["w"]
        else:
            anchors = probe["anchors"]
            squared = (
                (anchors * anchors).sum(dim=1).unsqueeze(0)
                + (standardized * standardized).sum(dim=1).unsqueeze(1)
                - 2.0 * (standardized @ anchors.T)
            )
            denominator = probe["med"] * probe["gamma"]
            if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
                raise RuntimeErrorBase("decision RBF probe has an invalid denominator")
            kernel = torch.exp(-torch.clamp(squared, min=0.0) / denominator)
            scores = kernel @ probe["alpha"]
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeErrorBase("decision probe produced non-finite scores")
        return scores

    def _probabilities(self, scores: Any) -> Any:
        torch = self._torch
        if self._calibration is not None:
            values = (
                self._calibration["cal_A"]
                * (scores - self._calibration["s_mean"])
                / self._calibration["s_std"]
                + self._calibration["cal_B"]
            )
        else:
            scale = self._output.get("scale", 1.0)
            bias = self._output.get("bias", 0.0)
            if not isinstance(scale, int | float) or not isinstance(bias, int | float):
                raise RuntimeErrorBase("decision.output scale and bias must be numeric")
            if not math.isfinite(float(scale)) or not math.isfinite(float(bias)):
                raise RuntimeErrorBase("decision.output scale and bias must be finite")
            values = scores * float(scale) + float(bias)
        return torch.sigmoid(values)

    def _score_texts(self, texts: Sequence[str]) -> tuple[list[float], list[int]]:
        scores: list[float] = []
        input_tokens: list[int] = []
        torch = self._torch
        device = self._input_device()
        for start in range(0, len(texts), self._batch_size):
            encoded = self._tokenizer(
                list(texts[start : start + self._batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                output = self._model(**encoded)
            last_hidden = getattr(output, "last_hidden_state", None)
            if last_hidden is None:
                raise RuntimeErrorBase("model output has no last_hidden_state")
            mask = encoded.get("attention_mask")
            if mask is None:
                raise RuntimeErrorBase("tokenizer output has no attention_mask")
            positions = mask.sum(dim=1) - 1
            rows = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            selected = last_hidden[rows, positions.to(last_hidden.device)]
            scores.extend(
                float(value)
                for value in self._raw_scores(selected).cpu().tolist()
            )
            input_tokens.extend(int(value) for value in mask.sum(dim=1).cpu().tolist())
        return scores, input_tokens

    def _decide_candidate_tasks(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        compiled = [compile_margin_tasks(request, self.metadata) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        scores, input_tokens = self._score_texts(
            [render_probe_task(task, self.metadata) for task in tasks]
        )
        responses: list[DecisionResponse] = []
        offset = 0
        for request, request_tasks in zip(requests, compiled, strict=True):
            end = offset + len(request_tasks)
            responses.append(
                DecisionResponse(
                    model=self.model_name,
                    answers=aggregate_margin_answers(
                        request,
                        request_tasks,
                        scores[offset:end],
                        self.metadata,
                    ),
                    usage=Usage(input_tokens=sum(input_tokens[offset:end])),
                )
            )
            offset = end
        return responses

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        if self._input_mode == "candidate_tasks":
            return self._decide_candidate_tasks(requests)
        texts = [render_probe_input(request, self.metadata) for request in requests]
        scores, input_tokens = self._score_texts(texts)
        probabilities = [
            float(value)
            for value in self._probabilities(
                self._torch.tensor(scores, dtype=self._torch.float32)
            ).tolist()
        ]

        responses: list[DecisionResponse] = []
        for request, probability, tokens in zip(
            requests, probabilities, input_tokens, strict=True
        ):
            responses.append(
                DecisionResponse(
                    model=self.model_name,
                    answers={
                        name: NoulAnswer(type="noul", noul=probability)
                        for name in request.questions
                    },
                    usage=Usage(input_tokens=tokens),
                )
            )
        return responses
