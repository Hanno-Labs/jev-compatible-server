"""Built-in llama.cpp and Transformers adapters.

Both adapters consume the same model metadata shape. A model can therefore be
added by publishing metadata and weights, without adding a service branch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .protocol import DecisionRequest, DecisionResponse, NoulQuestion, Usage
from .runtime import RuntimeErrorBase, TokenLogitRuntime


def load_decision_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeErrorBase("decision config must be a JSON object")
    return value


class LlamaBackend(TokenLogitRuntime):
    """llama-cpp-python adapter for GGUF token-logit decision models."""

    def __init__(
        self,
        model_path: str,
        *,
        config: dict[str, Any] | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "LlamaBackend requires the 'llama' optional dependency"
            ) from exc
        self._llama: Any = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            verbose=False,
        )
        metadata = getattr(self._llama, "metadata", {})
        if callable(metadata):
            metadata = metadata()
        merged_config = dict(metadata) if isinstance(metadata, dict) else {}
        merged_config.update(config or {})
        super().__init__(model_name=str(merged_config.get("model", model_path)), config=merged_config)

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        results: list[DecisionResponse] = []
        for request in requests:
            answers: dict[str, Any] = {}
            input_tokens = 0
            for name, question in request.questions.items():
                prompt = self._prompt(request, name, question)
                encoded = self._llama.tokenize(prompt.encode("utf-8"), add_bos=True)
                self._llama.reset()
                self._llama.eval(encoded)
                input_tokens += len(encoded)
                row = self._llama.scores[-1]
                logits = {self._token_id(label): float(row[self._token_id(label)]) for label in self._labels(question)}
                answers[name] = self._answer(question, logits)
            results.append(DecisionResponse(model=self.model_name, answers=answers, usage=Usage(input_tokens=input_tokens)))
        return results

    @staticmethod
    def _labels(question: Any) -> list[str]:
        if hasattr(question, "criteria") and isinstance(question.criteria, dict):
            return list(question.criteria) 
        if hasattr(question, "criteria") and isinstance(question.criteria, list):
            return [str(index) for index in range(len(question.criteria))]
        return ["true", "false"]


class TransformersBackend(TokenLogitRuntime):
    """Transformers adapter for causal token-logit decision models.

    Scalar/pointer models can use the same service by providing a future
    readout adapter; this built-in path intentionally handles causal logits.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "TransformersBackend requires the 'transformers' optional dependency"
            ) from exc
        super().__init__(model_name=str((config or {}).get("model", model_id)), config=config)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        model_metadata = self._model.config.to_dict()
        merged_config = dict(model_metadata)
        merged_config.update(config or {})
        self.config = merged_config
        if device != "auto":
            self._model.to(device)
        elif torch.cuda.is_available():
            self._model.to("cuda")
        self._model.eval()

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        prompts: list[tuple[int, str, Any]] = []
        for request_index, request in enumerate(requests):
            for name, question in request.questions.items():
                prompts.append((request_index, name, question))
        encoded = self._tokenizer(
            [self._prompt(requests[index], name, question) for index, name, question in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            output = self._model(**encoded)
        logits = output.logits[:, -1, :]
        answers: list[dict[str, Any]] = [{} for _ in requests]
        for row, (request_index, name, question) in enumerate(prompts):
            labels = LlamaBackend._labels(question)
            values = {self._token_id(label): float(logits[row, self._token_id(label)].item()) for label in labels}
            answers[request_index][name] = self._answer(question, values)
        return [
            DecisionResponse(model=self.model_name, answers=answer, usage=Usage())
            for answer in answers
        ]


class PointerTransformersBackend(TokenLogitRuntime):
    """Owned implementation of the Qwen + LoRA + pointer-head pattern.

    The model repository is only a source of weights. Packing delimiters,
    masking, LoRA location, and pointer tensors come from service-owned
    registry metadata, so this does not import a model author's serving code.
    """

    def __init__(self, model_id: str, *, config: dict[str, Any] | None = None, device: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase("PointerTransformersBackend requires transformers and torch") from exc
        super().__init__(model_name=str((config or {}).get("model", model_id)), config=config)
        self._torch = torch
        backbone = self.config.get("decision.backbone") or self.config.get("backbone") or model_id
        self._tokenizer = AutoTokenizer.from_pretrained(backbone)
        load_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
        # Pointer models provide a custom 4-D block-causal mask; eager attention
        # preserves that mask contract across Transformers releases.
        load_kwargs["attn_implementation"] = "eager"
        full_model = AutoModelForCausalLM.from_pretrained(backbone, **load_kwargs)
        self._model = getattr(full_model, "model", full_model)
        adapter = self.config.get("decision.adapter") or self.config.get("adapter")
        if adapter == "auto":
            adapter = model_id
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeErrorBase("pointer-head LoRA models require peft") from exc
            self._model = PeftModel.from_pretrained(self._model, adapter)
        target = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if target != "auto":
            self._model.to(target)
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._load_pointer_head()

    def _load_pointer_head(self) -> None:
        import torch

        head_path = self.config.get("decision.head_path") or self.config.get("head_path")
        if not isinstance(head_path, str):
            raise RuntimeErrorBase("pointer_head requires decision.head_path")
        if not __import__("os").path.exists(head_path):
            repo = self.config.get("decision.head_repo") or self.config.get("head_repo") or self.config.get("model")
            if not isinstance(repo, str):
                raise RuntimeErrorBase("relative pointer head path requires decision.head_repo")
            try:
                from huggingface_hub import hf_hub_download

                head_path = hf_hub_download(repo_id=repo, filename=head_path)
            except Exception as exc:  # pragma: no cover - network/model dependent
                raise RuntimeErrorBase(f"could not resolve pointer head {head_path!r} from {repo!r}") from exc
        payload = torch.load(head_path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state, dict) and isinstance(state.get("head"), dict):
            state = state["head"]
        if not isinstance(state, dict):
            raise RuntimeErrorBase("pointer head checkpoint must contain a state dict")
        q_weight = self._find_tensor(state, ("q.weight", "query.weight"))
        k_weight = self._find_tensor(state, ("k.weight", "key.weight"))
        q_bias = self._find_tensor(state, ("q.bias", "query.bias"), required=False)
        k_bias = self._find_tensor(state, ("k.bias", "key.bias"), required=False)
        if q_weight is None or k_weight is None:
            raise RuntimeErrorBase("pointer head checkpoint needs q/query and k/key weights")
        self._q_weight = q_weight.to(self._device)
        self._k_weight = k_weight.to(self._device)
        self._q_bias = q_bias.to(self._device) if q_bias is not None else None
        self._k_bias = k_bias.to(self._device) if k_bias is not None else None

    @staticmethod
    def _find_tensor(
        state: dict[str, Any],
        suffixes: tuple[str, ...],
        *,
        required: bool = True,
    ) -> Any | None:
        for key, value in state.items():
            if isinstance(key, str) and any(key.endswith(suffix) for suffix in suffixes):
                return value
        if required:
            return None
        return None

    def _packing(self) -> dict[str, Any]:
        packing = self.config.get("decision.packing", self.config.get("packing", {}))
        if not isinstance(packing, dict):
            raise RuntimeErrorBase("decision.packing must be an object")
        required = ("state_token", "question_token", "option_start", "option_end", "decision_token")
        missing = [key for key in required if not isinstance(packing.get(key), str)]
        if missing:
            raise RuntimeErrorBase(f"pointer_head packing metadata missing: {', '.join(missing)}")
        return packing

    def _encode(self, request: DecisionRequest, question: Any) -> dict[str, Any]:
        packing = self._packing()
        tok = self._tokenizer

        def user(value: Any) -> list[int]:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            return cast(list[int], tok(text, add_special_tokens=False).input_ids)

        def special(name: str) -> int:
            token = packing[name]
            token_id = tok.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                raise RuntimeErrorBase(f"packing token is not in tokenizer vocabulary: {token}")
            return int(token_id)

        state = [special("state_token"), *user(request.state)]
        instruction = [special("question_token"), *user(question.instructions)]
        if isinstance(question, NoulQuestion):
            criteria: list[Any] = ["false", "true"]
        elif isinstance(question.criteria, dict):
            criteria = [f"{key}: {value}" if value else key for key, value in question.criteria.items()]
        else:
            criteria = list(question.criteria)
        spans = [[special("option_start"), *user(value), special("option_end")] for value in criteria]
        ids = state + instruction
        seg = [0] * len(state) + [1] * len(instruction)
        opt = [-1] * len(ids)
        ends: list[int] = []
        for option_index, span in enumerate(spans):
            start = len(ids)
            ids.extend(span)
            seg.extend([1] * len(span))
            opt.extend([option_index] * len(span))
            ends.append(start + len(span) - 1)
        decision_index = len(ids)
        ids.append(special("decision_token")); seg.append(1); opt.append(-2)
        return {"ids": ids, "seg": seg, "opt": opt, "ends": ends, "decision": decision_index}

    def _hidden_batch(self, encodings: list[dict[str, Any]]) -> Any:
        torch = self._torch
        length = max(len(item["ids"]) for item in encodings)
        pad_id = self._tokenizer.pad_token_id or 0
        ids = torch.full((len(encodings), length), pad_id, dtype=torch.long, device=self._device)
        mask = torch.full((len(encodings), 1, length, length), torch.finfo(torch.float32).min, device=self._device)
        positions = torch.zeros((len(encodings), length), dtype=torch.long, device=self._device)
        for row, item in enumerate(encodings):
            size = len(item["ids"]); ids[row, :size] = torch.tensor(item["ids"], device=self._device)
            positions[row, :size] = torch.arange(size, device=self._device)
            seg = torch.tensor(item["seg"], device=self._device)
            opt = torch.tensor(item["opt"], device=self._device)
            causal = torch.tril(torch.ones((size, size), dtype=torch.bool, device=self._device))
            allowed = causal & ((seg[None, :] == 0) | (seg[None, :] == seg[:, None]))
            option_keys = opt[None, :] >= 0
            decide = opt[:, None] == -2
            allowed &= (~option_keys | decide | (opt[None, :] == opt[:, None]))
            mask[row, 0, :size, :size] = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        with torch.inference_mode():
            return self._model(input_ids=ids, position_ids=positions, attention_mask=mask).last_hidden_state.float()

    @staticmethod
    def _encoding_batches(encodings: list[dict[str, Any]]) -> list[list[int]]:
        order = sorted(range(len(encodings)), key=lambda index: len(encodings[index]["ids"]))
        batches: list[list[int]] = []
        batch: list[int] = []
        for index in order:
            length = len(encodings[index]["ids"])
            if batch:
                smallest = len(encodings[batch[0]]["ids"])
                if (
                    len(batch) >= 8
                    or length > 2 * smallest
                    or (len(batch) + 1) * length * length > 1_000_000_000
                ):
                    batches.append(batch)
                    batch = []
            batch.append(index)
        if batch:
            batches.append(batch)
        return batches

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        flattened: list[tuple[int, str, Any, dict[str, Any]]] = []
        for request_index, request in enumerate(requests):
            for name, question in request.questions.items():
                flattened.append((request_index, name, question, self._encode(request, question)))
        answers: list[dict[str, Any]] = [{} for _ in requests]
        encodings = [item[3] for item in flattened]
        for batch in self._encoding_batches(encodings):
            hidden = self._hidden_batch([encodings[index] for index in batch])
            for row, index in enumerate(batch):
                request_index, name, question, encoding = flattened[index]
                query = hidden[row, encoding["decision"]]
                options = hidden[row, encoding["ends"]]
                q = self._q_weight @ query + (self._q_bias if self._q_bias is not None else 0)
                k = options @ self._k_weight.T
                scores = (k @ q) / math.sqrt(q.shape[-1])
                labels = ["false", "true"] if isinstance(question, NoulQuestion) else self._labels_for_question(question)
                answers[request_index][name] = self.answer_from_label_scores(
                    question, {label: float(value) for label, value in zip(labels, scores.tolist(), strict=True)}
                )
        return [DecisionResponse(model=self.model_name, answers=answer, usage=Usage()) for answer in answers]
