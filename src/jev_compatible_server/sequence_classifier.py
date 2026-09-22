"""Generic per-candidate sequence-classification decision readout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .encoder_decoder import (
    _mapping,
    _template,
    aggregate_margin_answers,
    compile_margin_tasks,
    decision_metadata,
)
from .protocol import DecisionRequest, DecisionResponse, Usage
from .runtime import DecisionRuntime, RuntimeErrorBase


class SequenceClassifierMarginBackend(DecisionRuntime):
    """Score each candidate with a scalar sequence-classification head."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "SequenceClassifierMarginBackend requires transformers and torch"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = decision_metadata(self.config)
        if self.metadata.get("readout") != "sequence_classifier_margin":
            raise RuntimeErrorBase(
                "SequenceClassifierMarginBackend requires "
                "decision.readout=sequence_classifier_margin"
            )
        self._torch = torch
        self._batch_size = self._positive_int("batch_size", 32)
        input_config = _mapping(self.metadata.get("input"), "decision.input")
        self._max_length = self._positive_int_from(input_config, "max_length", 512)
        self._state_template = _template(
            input_config.get("state_template"), "decision.input.state_template"
        )
        self._tail_template = _template(
            input_config.get("tail_template"), "decision.input.tail_template"
        )
        truncation = input_config.get("truncation", "preserve_tail")
        if truncation != "preserve_tail":
            raise RuntimeErrorBase(
                "sequence_classifier_margin currently supports preserve_tail truncation"
            )

        loader = _mapping(self.metadata.get("loader", {}), "decision.loader")
        base_model = loader.get("base_model", model_id)
        if not isinstance(base_model, str):
            raise RuntimeErrorBase("decision.loader.base_model must be a string")
        tokenizer_id = loader.get("tokenizer", base_model)
        if not isinstance(tokenizer_id, str):
            raise RuntimeErrorBase("decision.loader.tokenizer must be a string")
        trust_remote_code = loader.get("trust_remote_code", False)
        if not isinstance(trust_remote_code, bool):
            raise RuntimeErrorBase("decision.loader.trust_remote_code must be boolean")
        revision = loader.get("revision")
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if isinstance(revision, str):
            load_kwargs["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **load_kwargs)
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
        num_labels = loader.get("num_labels", 1)
        if not isinstance(num_labels, int) or isinstance(num_labels, bool) or num_labels <= 0:
            raise RuntimeErrorBase("decision.loader.num_labels must be a positive integer")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            dtype=dtype,
            **load_kwargs,
        )
        self._set_padding_ids()
        adapter = loader.get("adapter")
        if adapter == "auto":
            adapter = model_id
        if adapter is not None:
            if not isinstance(adapter, str):
                raise RuntimeErrorBase("decision.loader.adapter must be a string")
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeErrorBase("sequence-classifier adapters require peft") from exc
            self._model = PeftModel.from_pretrained(self._model, adapter)
        self._model.to(target)
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def _set_padding_ids(self) -> None:
        pad_token_id = self._tokenizer.pad_token_id
        eos_token_id = self._tokenizer.eos_token_id
        self._model.config.pad_token_id = pad_token_id
        self._model.config.eos_token_id = eos_token_id
        get_text_config = getattr(self._model.config, "get_text_config", None)
        if callable(get_text_config):
            text_config = get_text_config()
            text_config.pad_token_id = pad_token_id
            text_config.eos_token_id = eos_token_id

    def _positive_int(self, name: str, default: int) -> int:
        return self._positive_int_from(self.metadata, name, default)

    @staticmethod
    def _positive_int_from(values: Mapping[str, Any], name: str, default: int) -> int:
        value = values.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeErrorBase(f"decision {name} must be a positive integer")
        return value

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

    def _encode(self, task: Any) -> list[int]:
        state_ids = list(
            self._tokenizer(
                self._state_template.format(state=task.query),
                add_special_tokens=False,
            )["input_ids"]
        )
        tail_ids = list(
            self._tokenizer(
                self._tail_template.format(
                    instructions=task.instruction,
                    option=task.document,
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        if len(tail_ids) >= self._max_length:
            return tail_ids[-self._max_length :]
        return state_ids[: self._max_length - len(tail_ids)] + tail_ids

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        torch = self._torch
        compiled = [compile_margin_tasks(request, self.metadata) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        rows = [self._encode(task) for task in tasks]
        margins: list[float] = []
        for start in range(0, len(rows), self._batch_size):
            batch = self._tokenizer.pad(
                {"input_ids": rows[start : start + self._batch_size]},
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self._device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self._model(**batch)
            logits = output.logits.float()
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeErrorBase(
                    "sequence_classifier_margin requires exactly one scalar logit"
                )
            values = logits[:, 0]
            if not torch.isfinite(values).all():
                raise RuntimeErrorBase("model produced non-finite candidate logits")
            margins.extend(float(value) for value in values.cpu().tolist())

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
                        margins[offset:end],
                        self.metadata,
                    ),
                    usage=Usage(
                        input_tokens=sum(len(row) for row in rows[offset:end])
                    ),
                )
            )
            offset = end
        return responses
