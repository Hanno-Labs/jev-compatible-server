"""Generic encoder-decoder margin readout configured by model recipes."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from .runtime import DecisionRuntime, RuntimeErrorBase, softmax


@dataclass(frozen=True)
class MarginTask:
    question_id: str
    option_id: str
    instruction: str
    query: str
    document: str


@dataclass(frozen=True)
class EncodedDocument:
    hidden: Any
    mask: Any

    @property
    def nbytes(self) -> int:
        return int(
            self.hidden.numel() * self.hidden.element_size()
            + self.mask.numel() * self.mask.element_size()
        )


def decision_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read nested decision metadata while retaining dotted-key compatibility."""

    nested = config.get("decision", {})
    result = dict(nested) if isinstance(nested, dict) else {}
    for key, value in config.items():
        if key.startswith("decision."):
            result[key.removeprefix("decision.")] = value
    return result


def render_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeErrorBase("decision content must be JSON serializable") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeErrorBase(f"{name} must be an object")
    return value


def _template(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeErrorBase(f"{name} must be a string")
    return value


def compile_margin_tasks(
    request: DecisionRequest, metadata: Mapping[str, Any]
) -> list[MarginTask]:
    """Compile Jev questions into generic query/document relevance tasks."""

    adapters = _mapping(metadata.get("instruction_adapters"), "decision.instruction_adapters")
    candidates = _mapping(metadata.get("candidates"), "decision.candidates")
    choice_config = _mapping(candidates.get("choice"), "decision.candidates.choice")
    score_config = _mapping(candidates.get("score"), "decision.candidates.score")
    noul_config = _mapping(candidates.get("noul"), "decision.candidates.noul")
    query = render_content(request.state)
    tasks: list[MarginTask] = []

    for question_id, question in request.questions.items():
        adapter = _template(adapters.get(question.type), f"decision.instruction_adapters.{question.type}")
        original_instruction = render_content(question.instructions)
        separator = metadata.get("instruction_adapter_separator", "\n\n")
        if not isinstance(separator, str):
            raise RuntimeErrorBase("decision.instruction_adapter_separator must be a string")
        instruction = (
            f"{original_instruction}{separator}{adapter}"
            if adapter
            else original_instruction
        )
        documents: list[tuple[str, str]]
        if isinstance(question, ChoiceQuestion):
            populated = _template(
                choice_config.get("criterion_template"),
                "decision.candidates.choice.criterion_template",
            )
            empty = _template(
                choice_config.get("empty_criterion_template"),
                "decision.candidates.choice.empty_criterion_template",
            )
            documents = []
            for key, criterion in question.criteria.items():
                template = empty if criterion is None else populated
                documents.append(
                    (
                        key,
                        template.format(key=key, criterion=render_content(criterion)),
                    )
                )
        elif isinstance(question, ScoreQuestion):
            template = _template(
                score_config.get("criterion_template"),
                "decision.candidates.score.criterion_template",
            )
            documents = [
                (str(index), template.format(criterion=render_content(criterion)))
                for index, criterion in enumerate(question.criteria)
            ]
        elif isinstance(question, NoulQuestion) and question.criteria is None:
            no_criteria_options = noul_config.get("no_criteria_options")
            if isinstance(no_criteria_options, dict):
                documents = [
                    (str(key), render_content(value))
                    for key, value in no_criteria_options.items()
                ]
            else:
                template = _template(
                    noul_config.get("no_criteria_template"),
                    "decision.candidates.noul.no_criteria_template",
                )
                documents = [("", template.format(state=query))]
        elif isinstance(question, NoulQuestion):
            template = _template(
                noul_config.get("criterion_template"),
                "decision.candidates.noul.criterion_template",
            )
            if question.criteria is None:  # pragma: no cover - narrowed above
                raise RuntimeErrorBase("noul criteria unexpectedly missing")
            documents = [
                ("true", template.format(criterion=render_content(question.criteria.true))),
                ("false", template.format(criterion=render_content(question.criteria.false))),
            ]
        else:  # pragma: no cover - exhaustive protocol union
            raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")
        tasks.extend(
            MarginTask(question_id, option_id, instruction, query, document)
            for option_id, document in documents
        )
    return tasks


def normalized_entropy_confidence(probabilities: Sequence[float]) -> float:
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def aggregate_margin_answers(
    request: DecisionRequest,
    tasks: Sequence[MarginTask],
    margins: Sequence[float],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if len(tasks) != len(margins) or not all(math.isfinite(value) for value in margins):
        raise RuntimeErrorBase("encoder-decoder backend returned invalid margins")
    aggregation = _mapping(metadata.get("aggregation"), "decision.aggregation")
    choice_temperature = float(aggregation.get("choice_temperature", 1.0))
    score_temperature = float(aggregation.get("score_temperature", 1.0))
    noul_a = float(aggregation.get("noul_a", 1.0))
    noul_b = float(aggregation.get("noul_b", 0.0))
    if (
        choice_temperature <= 0
        or score_temperature <= 0
        or noul_a <= 0
        or not all(
            math.isfinite(value)
            for value in (choice_temperature, score_temperature, noul_a, noul_b)
        )
    ):
        raise RuntimeErrorBase("decision aggregation parameters are invalid")
    confidence_method = aggregation.get("confidence", "normalized_entropy")
    if confidence_method not in {"normalized_entropy", "max_probability"}:
        raise RuntimeErrorBase(
            "decision aggregation confidence must be normalized_entropy or max_probability"
        )

    grouped: dict[str, list[tuple[str, float]]] = {
        question_id: [] for question_id in request.questions
    }
    for task, margin in zip(tasks, margins, strict=True):
        grouped[task.question_id].append((task.option_id, margin))

    answers: dict[str, Any] = {}
    for question_id, question in request.questions.items():
        candidates = grouped[question_id]
        if isinstance(question, NoulQuestion):
            noul_mode = aggregation.get("noul_mode", "single_margin")
            if noul_mode == "true_false_softmax":
                by_label = dict(candidates)
                if len(candidates) != 2 or set(by_label) != {"true", "false"}:
                    raise RuntimeErrorBase(
                        "true_false_softmax noul requires true and false margins"
                    )
                margin = by_label["true"] - by_label["false"]
            elif noul_mode != "single_margin":
                raise RuntimeErrorBase(
                    "decision aggregation noul_mode must be single_margin or true_false_softmax"
                )
            elif question.criteria is None:
                if len(candidates) != 1 or candidates[0][0] != "":
                    raise RuntimeErrorBase(
                        "noul without criteria requires exactly one state margin"
                    )
                margin = candidates[0][1]
            else:
                by_label = dict(candidates)
                if len(candidates) != 2 or set(by_label) != {"true", "false"}:
                    raise RuntimeErrorBase(
                        "noul criteria require exactly one true and one false margin"
                    )
                margin = by_label["true"] - by_label["false"]
            answers[question_id] = NoulAnswer(
                type="noul", noul=_sigmoid(noul_a * margin + noul_b)
            )
            continue

        temperature = (
            choice_temperature if isinstance(question, ChoiceQuestion) else score_temperature
        )
        probabilities = softmax([margin / temperature for _, margin in candidates])
        distribution = {
            option_id: probability
            for (option_id, _), probability in zip(candidates, probabilities, strict=True)
        }
        confidence = (
            normalized_entropy_confidence(probabilities)
            if confidence_method == "normalized_entropy"
            else max(probabilities)
        )
        if isinstance(question, ChoiceQuestion):
            choice = max(distribution, key=distribution.__getitem__)
            answers[question_id] = ChoiceAnswer(
                type="choice",
                choice=choice,
                probabilities=distribution,
                confidence=confidence,
            )
        elif isinstance(question, ScoreQuestion):
            answers[question_id] = ScoreAnswer(
                type="score",
                score=math.fsum(
                    index * probability
                    for index, probability in enumerate(probabilities)
                ),
                probabilities=distribution,
                confidence=confidence,
                legend=question.criteria,
            )
        else:  # pragma: no cover - exhaustive protocol union
            raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")
    return answers


class EncoderDecoderMarginBackend(DecisionRuntime):
    """Encoder-decoder reranker whose entire behavior comes from model metadata."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "EncoderDecoderMarginBackend requires transformers and torch"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = decision_metadata(self.config)
        if self.metadata.get("readout") != "encoder_decoder_margin":
            raise RuntimeErrorBase(
                "EncoderDecoderMarginBackend requires decision.readout=encoder_decoder_margin"
            )
        self._torch = torch
        self._batch_size = self._positive_int("batch_size", 4)
        limits = _mapping(self.metadata.get("limits"), "decision.limits")
        self._query_max_length = self._positive_int_from(limits, "query", 512)
        self._document_max_length = self._positive_int_from(limits, "document", 1024)
        self._decoder_max_length = self._positive_int_from(limits, "decoder", 2048)

        revision = self.metadata.get("revision")
        load_kwargs: dict[str, Any] = {}
        if isinstance(revision, str):
            load_kwargs["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
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
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, dtype=dtype, **load_kwargs
        )
        for parameter in self._model.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)
        self._model.to(target)
        self._model.eval()
        self._device = next(self._model.parameters()).device

        labels = _mapping(self.metadata.get("labels"), "decision.labels")
        self._positive_token = self._answer_token_id(
            _template(labels.get("positive"), "decision.labels.positive")
        )
        self._negative_token = self._answer_token_id(
            _template(labels.get("negative"), "decision.labels.negative")
        )

        encoder = _mapping(self.metadata.get("encoder"), "decision.encoder")
        self._document_template = _template(
            encoder.get("document_template"), "decision.encoder.document_template"
        )
        pooling = encoder.get("pooling", "mean_chunks")
        if pooling not in {"mean_chunks", "none"}:
            raise RuntimeErrorBase("decision.encoder.pooling must be mean_chunks or none")
        self._chunk_size = (
            self._positive_int_from(encoder, "chunk_size", 4)
            if pooling == "mean_chunks"
            else None
        )
        decoder = _mapping(self.metadata.get("decoder"), "decision.decoder")
        self._decoder_template = _template(
            decoder.get("template"), "decision.decoder.template"
        )
        self._pad_to_multiple_of = self._positive_int_from(
            decoder, "pad_to_multiple_of", 8
        )

        cache = _mapping(self.metadata.get("cache", {}), "decision.cache")
        max_mib = cache.get("max_mib", 256)
        if not isinstance(max_mib, (int, float)) or max_mib < 0:
            raise RuntimeErrorBase("decision.cache.max_mib must be non-negative")
        self._cache_max_bytes = int(float(max_mib) * 1024 * 1024)
        self._cache_bytes = 0
        self._document_cache: OrderedDict[str, EncodedDocument] = OrderedDict()

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

    def _answer_token_id(self, answer: str) -> int:
        token_ids = self._tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise RuntimeErrorBase(
                f"decision label {answer!r} must tokenize to exactly one token"
            )
        return int(token_ids[0])

    def _decoder_ids(self, task: MarginTask) -> list[int]:
        query_ids = self._tokenizer(
            task.query,
            add_special_tokens=False,
            truncation=True,
            max_length=self._query_max_length,
        )["input_ids"]
        truncated_query = self._tokenizer.decode(
            query_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        text = self._decoder_template.format(
            query=truncated_query,
            instruction=task.instruction,
        )
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _pool_chunks(self, hidden: Any, mask: Any) -> tuple[Any, Any]:
        if self._chunk_size is None:
            return hidden, mask
        torch = self._torch
        chunk_size = self._chunk_size
        batch_size, sequence_length, hidden_size = hidden.shape
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = num_chunks * chunk_size
        pad_length = padded_length - sequence_length
        if pad_length:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad_length))
            mask = torch.nn.functional.pad(mask, (0, pad_length))
        hidden = hidden.view(batch_size, num_chunks, chunk_size, hidden_size)
        chunk_mask = mask.view(batch_size, num_chunks, chunk_size)
        expanded_mask = chunk_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * expanded_mask).sum(dim=2)
        pooled = pooled / chunk_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        pooled_mask = (chunk_mask.sum(dim=2) > 0).to(mask.dtype)
        return pooled, pooled_mask

    def _pad(self, rows: Sequence[list[int]], *, decoder: bool = False) -> dict[str, Any]:
        batch = self._tokenizer.pad(
            {"input_ids": list(rows)},
            padding=True,
            pad_to_multiple_of=self._pad_to_multiple_of if decoder else None,
            return_tensors="pt",
        )
        return {key: value.to(self._device) for key, value in batch.items()}

    def _get_encoder(self) -> Any:
        if hasattr(self._model, "get_encoder"):
            return self._model.get_encoder()
        encoder = getattr(self._model, "encoder", None)
        if encoder is None:
            raise RuntimeErrorBase("conditional-generation model has no encoder")
        return encoder

    def _cache_put(self, key: str, value: EncodedDocument) -> None:
        if self._cache_max_bytes == 0 or value.nbytes > self._cache_max_bytes:
            return
        previous = self._document_cache.pop(key, None)
        if previous is not None:
            self._cache_bytes -= previous.nbytes
        self._document_cache[key] = value
        self._cache_bytes += value.nbytes
        while self._cache_bytes > self._cache_max_bytes:
            _, evicted = self._document_cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes

    def _encode_documents(
        self, documents: Mapping[str, list[int]]
    ) -> dict[str, EncodedDocument]:
        torch = self._torch
        result: dict[str, EncodedDocument] = {}
        missing: list[tuple[str, list[int]]] = []
        for document, ids in documents.items():
            cached = self._document_cache.get(document)
            if cached is None:
                missing.append((document, ids))
            else:
                self._document_cache.move_to_end(document)
                result[document] = cached
        encoder = self._get_encoder()
        for start in range(0, len(missing), self._batch_size):
            chunk = missing[start : start + self._batch_size]
            batch = self._pad([ids for _, ids in chunk])
            with torch.inference_mode():
                hidden = encoder(**batch, return_dict=True).last_hidden_state
            mask = batch["attention_mask"]
            hidden, mask = self._pool_chunks(hidden, mask)
            for row, (document, _) in enumerate(chunk):
                length = int(mask[row].sum().item())
                encoded = EncodedDocument(
                    hidden[row, :length].detach().clone(),
                    mask[row, :length].detach().clone(),
                )
                result[document] = encoded
                self._cache_put(document, encoded)
        return result

    def _score(
        self,
        tasks: Sequence[MarginTask],
        decoder_rows: Sequence[list[int]],
        documents: Mapping[str, EncodedDocument],
    ) -> list[float]:
        torch = self._torch
        try:
            from transformers.modeling_outputs import BaseModelOutput
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase("transformers modeling outputs are unavailable") from exc
        margins: list[float] = []
        for start in range(0, len(tasks), self._batch_size):
            task_chunk = tasks[start : start + self._batch_size]
            decoder_chunk = decoder_rows[start : start + self._batch_size]
            decoder_batch = self._pad(decoder_chunk, decoder=True)
            encoded = [documents[task.document] for task in task_chunk]
            hidden = torch.nn.utils.rnn.pad_sequence(
                [item.hidden for item in encoded], batch_first=True
            )
            mask = torch.nn.utils.rnn.pad_sequence(
                [item.mask for item in encoded], batch_first=True
            )
            with torch.inference_mode():
                output = self._model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=mask,
                    decoder_input_ids=decoder_batch["input_ids"],
                    decoder_attention_mask=decoder_batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
            sequence_lengths = decoder_batch["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(output.logits.shape[0], device=self._device)
            last_logits = output.logits[batch_indices, sequence_lengths]
            yes = last_logits[:, self._positive_token].float()
            no = last_logits[:, self._negative_token].float()
            values = yes - no
            if not torch.isfinite(values).all():
                raise RuntimeErrorBase("model produced non-finite decision label logits")
            margins.extend(float(value) for value in values.cpu().tolist())
        return margins

    def decide_batch(
        self, requests: Sequence[DecisionRequest]
    ) -> list[DecisionResponse]:
        compiled = [compile_margin_tasks(request, self.metadata) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        if not tasks:
            return []

        document_rows: dict[str, list[int]] = {}
        decoder_rows: list[list[int]] = []
        for task in tasks:
            query_ids = self._tokenizer(task.query, add_special_tokens=False)["input_ids"]
            if len(query_ids) > self._query_max_length:
                raise RuntimeErrorBase(
                    f"state exceeds {self._query_max_length} query tokens"
                )
            if task.document not in document_rows:
                document_text = self._document_template.format(document=task.document)
                document_ids = list(
                    self._tokenizer(document_text, add_special_tokens=False)["input_ids"]
                )
                if len(document_ids) > self._document_max_length:
                    raise RuntimeErrorBase(
                        f"document exceeds {self._document_max_length} tokens"
                    )
                document_rows[task.document] = document_ids
            decoder_ids = self._decoder_ids(task)
            padded_length = (
                (len(decoder_ids) + self._pad_to_multiple_of - 1)
                // self._pad_to_multiple_of
                * self._pad_to_multiple_of
            )
            if padded_length > self._decoder_max_length:
                raise RuntimeErrorBase(
                    f"decoder template exceeds {self._decoder_max_length} tokens"
                )
            decoder_rows.append(decoder_ids)

        encoded = self._encode_documents(document_rows)
        margins = self._score(tasks, decoder_rows, encoded)
        responses: list[DecisionResponse] = []
        offset = 0
        for request, request_tasks in zip(requests, compiled, strict=True):
            end = offset + len(request_tasks)
            request_margins = margins[offset:end]
            input_tokens = sum(
                len(document_rows[task.document]) + len(decoder_rows[index])
                for index, task in enumerate(request_tasks, start=offset)
            )
            responses.append(
                DecisionResponse(
                    model=self.model_name,
                    answers=aggregate_margin_answers(
                        request,
                        request_tasks,
                        request_margins,
                        self.metadata,
                    ),
                    usage=Usage(input_tokens=input_tokens),
                )
            )
            offset = end
        return responses
