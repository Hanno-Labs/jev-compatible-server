"""Classifier readouts for NLI and GLiClass decision checkpoints.

Both adapters retain the server's single candidate representation: Jev question
candidates are compiled once, then projected into the checkpoint's native
classifier output space before the shared answer aggregation runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .encoder_decoder import (
    MarginTask,
    _mapping,
    _template,
    aggregate_margin_answers,
    compile_margin_tasks,
    decision_metadata,
    normalized_entropy_confidence,
    render_content,
)
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

VERDICT_ABSTENTION_ID = "__insufficient_evidence__"
VERDICT_ABSTENTION_LABEL = "insufficient evidence"
VERDICT_MAX_SUBSTANTIVE_CANDIDATES = 24


def _positive_int(value: Any, name: str, default: int) -> int:
    resolved = default if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved <= 0:
        raise RuntimeErrorBase(f"{name} must be a positive integer")
    return resolved


def _device(torch: Any, requested: Any, default: str) -> str:
    value = default if requested is None else requested
    if not isinstance(value, str):
        raise RuntimeErrorBase("decision.device must be a string")
    target = "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value
    if target.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeErrorBase("CUDA was requested, but no CUDA device is available")
    return target


def _loader_kwargs(loader: Mapping[str, Any]) -> dict[str, Any]:
    trust_remote_code = loader.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise RuntimeErrorBase("decision.loader.trust_remote_code must be boolean")
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    revision = loader.get("revision")
    if revision is not None:
        if not isinstance(revision, str):
            raise RuntimeErrorBase("decision.loader.revision must be a string")
        kwargs["revision"] = revision
    return kwargs


def project_entailment_logits(
    logits: Any, label2id: Mapping[str, Any], entailment_label: str = "entailment"
) -> Any:
    """Select one NLI entailment logit per candidate without model-name heuristics."""

    if not isinstance(entailment_label, str) or not entailment_label:
        raise RuntimeErrorBase("decision.entailment_label must be a non-empty string")
    matches = [
        value
        for label, value in label2id.items()
        if isinstance(label, str) and label.casefold() == entailment_label.casefold()
    ]
    if len(matches) != 1 or not isinstance(matches[0], int) or isinstance(matches[0], bool):
        raise RuntimeErrorBase(
            f"NLI model must define exactly one {entailment_label!r} label in label2id"
        )
    index = matches[0]
    if getattr(logits, "ndim", None) != 2 or index < 0 or index >= logits.shape[1]:
        raise RuntimeErrorBase("NLI model output does not contain the configured entailment logit")
    values = logits[:, index]
    if not bool(values.isfinite().all()):
        raise RuntimeErrorBase("NLI model produced non-finite entailment logits")
    return values


class NLIEntailmentBackend(DecisionRuntime):
    """Project each candidate pair through an NLI head's entailment dimension."""

    def __init__(
        self, model_id: str, *, config: dict[str, Any] | None = None, device: str = "auto"
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase("NLIEntailmentBackend requires transformers and torch") from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = decision_metadata(self.config)
        if self.metadata.get("readout") != "nli_entailment":
            raise RuntimeErrorBase("NLIEntailmentBackend requires decision.readout=nli_entailment")
        self._torch = torch
        self._batch_size = _positive_int(self.metadata.get("batch_size"), "decision.batch_size", 32)
        input_config = _mapping(self.metadata.get("input"), "decision.input")
        self._max_length = _positive_int(input_config.get("max_length"), "decision.input.max_length", 512)
        self._premise_template = _template(input_config.get("premise_template"), "decision.input.premise_template")
        self._hypothesis_template = _template(input_config.get("hypothesis_template"), "decision.input.hypothesis_template")

        loader = _mapping(self.metadata.get("loader", {}), "decision.loader")
        base_model = loader.get("base_model", model_id)
        tokenizer_id = loader.get("tokenizer", base_model)
        if not isinstance(base_model, str) or not isinstance(tokenizer_id, str):
            raise RuntimeErrorBase("decision.loader base_model and tokenizer must be strings")
        kwargs = _loader_kwargs(loader)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(base_model, **kwargs)
        self._model.to(_device(torch, self.metadata.get("device"), device))
        self._model.eval()
        self._input_device = next(self._model.parameters()).device
        self._label2id = self._model.config.label2id
        self._entailment_label = self.metadata.get("entailment_label", "entailment")

    def _pairs(self, tasks: Sequence[MarginTask]) -> tuple[list[str], list[str]]:
        premises: list[str] = []
        hypotheses: list[str] = []
        for task in tasks:
            try:
                premises.append(self._premise_template.format(state=task.query))
                hypotheses.append(
                    self._hypothesis_template.format(
                        instructions=task.instruction, candidate=task.document
                    )
                )
            except KeyError as exc:
                raise RuntimeErrorBase(
                    f"NLI input template references an unknown field: {exc.args[0]}"
                ) from exc
        return premises, hypotheses

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        compiled = [compile_margin_tasks(request, self.metadata) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        premises, hypotheses = self._pairs(tasks)
        margins: list[float] = []
        token_counts: list[int] = []
        for start in range(0, len(tasks), self._batch_size):
            encoded = self._tokenizer(
                premises[start : start + self._batch_size],
                hypotheses[start : start + self._batch_size],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self._input_device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                output = self._model(**encoded)
            margins.extend(
                float(value)
                for value in project_entailment_logits(
                    output.logits.float(), self._label2id, self._entailment_label
                ).cpu().tolist()
            )
            token_counts.extend(int(value) for value in encoded["attention_mask"].sum(dim=1).cpu().tolist())

        responses: list[DecisionResponse] = []
        offset = 0
        for request, request_tasks in zip(requests, compiled, strict=True):
            end = offset + len(request_tasks)
            responses.append(DecisionResponse(
                model=self.model_name,
                answers=aggregate_margin_answers(request, request_tasks, margins[offset:end], self.metadata),
                usage=Usage(input_tokens=sum(token_counts[offset:end])),
            ))
            offset = end
        return responses


@dataclass(frozen=True)
class RLCDTemperatureCalibrator:
    """Safe JSON-only temperature calibrator emitted by OpenJev Verdict."""

    temperature: float
    per_k: Mapping[str, float]

    def temperature_for(self, candidates: int) -> float:
        value = self.per_k.get(str(candidates), self.temperature)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeErrorBase("Verdict calibrator temperature must be finite and positive")
        return value


def load_rlcd_calibrator(path: str | Path) -> RLCDTemperatureCalibrator:
    """Load the public ``rlcd-calibrator-v1`` JSON artifact, never pickle data."""

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeErrorBase("unable to read Verdict calibrator JSON") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != "rlcd-calibrator-v1":
        raise RuntimeErrorBase("unsupported Verdict calibrator format; expected rlcd-calibrator-v1 JSON")
    temperature = payload.get("temperature")
    raw_per_k = payload.get("per_k", {})
    if not isinstance(temperature, int | float) or isinstance(temperature, bool):
        raise RuntimeErrorBase("Verdict calibrator temperature must be numeric")
    if not isinstance(raw_per_k, dict):
        raise RuntimeErrorBase("Verdict calibrator per_k must be an object")
    per_k: dict[str, float] = {}
    for count, value in raw_per_k.items():
        if not isinstance(count, str) or not isinstance(value, int | float) or isinstance(value, bool):
            raise RuntimeErrorBase("Verdict calibrator per_k must map strings to numbers")
        try:
            cardinality = int(count)
        except ValueError as exc:
            raise RuntimeErrorBase("Verdict calibrator per_k keys must be positive integers") from exc
        if cardinality <= 0 or str(cardinality) != count:
            raise RuntimeErrorBase("Verdict calibrator per_k keys must be positive integers")
        per_k[count] = float(value)
    calibrator = RLCDTemperatureCalibrator(float(temperature), per_k)
    calibrator.temperature_for(1)
    for count in per_k:
        calibrator.temperature_for(int(count))
    return calibrator


def render_gliclass_prompt(question: str, context: str, labels: Sequence[str]) -> str:
    """Render Verdict's GLiClass marker contract in one model input string."""

    if not labels:
        raise RuntimeErrorBase("GLiClass requires at least one candidate label")
    return "".join(f"<<LABEL>>{label}" for label in labels) + f"<<SEP>>Question: {question}\n\nContext:\n{context}"


def compile_verdict_tasks(request: DecisionRequest) -> list[MarginTask]:
    """Compile Jev fields into Verdict's labels plus its explicit abstention route.

    Verdict's published GLiClass contract has 25 slots: at most 24 substantive
    labels and one final ``insufficient evidence`` label.  This intentionally
    does not reuse generic candidate templates because Verdict's label wording
    is part of the checkpoint's input contract.
    """

    query = render_content(request.state)
    tasks: list[MarginTask] = []
    for question_id, question in request.questions.items():
        instruction = render_content(question.instructions)
        substantive: list[tuple[str, str]]
        if isinstance(question, ChoiceQuestion):
            substantive = [
                (option_id, f"It is {render_content(criterion)}")
                for option_id, criterion in question.criteria.items()
            ]
        elif isinstance(question, ScoreQuestion):
            substantive = [
                (str(index), f"{render_content(criterion)} (Value: {index})")
                for index, criterion in enumerate(question.criteria)
            ]
        elif isinstance(question, NoulQuestion):
            if question.criteria is None:
                proposition = instruction
                substantive = [
                    ("true", f"true: {proposition}"),
                    ("false", f"false: not {proposition}"),
                ]
            else:
                substantive = [
                    ("true", f"true: {render_content(question.criteria.true)}"),
                    ("false", f"false: not {render_content(question.criteria.false)}"),
                ]
        else:  # pragma: no cover - exhaustive protocol union
            raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")
        if len(substantive) > VERDICT_MAX_SUBSTANTIVE_CANDIDATES:
            raise RuntimeErrorBase(
                "Verdict supports at most 24 substantive candidates plus abstention"
            )
        tasks.extend(
            MarginTask(question_id, option_id, instruction, query, document)
            for option_id, document in substantive
        )
        tasks.append(
            MarginTask(
                question_id,
                VERDICT_ABSTENTION_ID,
                instruction,
                query,
                VERDICT_ABSTENTION_LABEL,
            )
        )
    return tasks


def aggregate_verdict_answers(
    request: DecisionRequest,
    tasks: Sequence[MarginTask],
    margins: Sequence[float],
) -> dict[str, Any]:
    """Map Verdict's calibrated full distributions to the Jev response types.

    Choice and score responses retain the explicit abstention probability under
    ``__insufficient_evidence__``.  Jev's current Noul wire type has no field
    for that mass, so it exposes Verdict's published conditional probability
    ``P(true | sufficient evidence)`` instead.
    """

    if len(tasks) != len(margins) or not all(math.isfinite(value) for value in margins):
        raise RuntimeErrorBase("Verdict returned invalid distribution logits")
    grouped: dict[str, list[tuple[str, float]]] = {
        question_id: [] for question_id in request.questions
    }
    for task, margin in zip(tasks, margins, strict=True):
        grouped[task.question_id].append((task.option_id, margin))

    answers: dict[str, Any] = {}
    for question_id, question in request.questions.items():
        candidates = grouped[question_id]
        option_ids = [option_id for option_id, _ in candidates]
        if option_ids.count(VERDICT_ABSTENTION_ID) != 1:
            raise RuntimeErrorBase("Verdict field must include exactly one abstention candidate")
        probabilities = softmax([margin for _, margin in candidates])
        distribution = dict(zip(option_ids, probabilities, strict=True))
        confidence = normalized_entropy_confidence(probabilities)
        if isinstance(question, ChoiceQuestion):
            answers[question_id] = ChoiceAnswer(
                type="choice",
                choice=max(distribution, key=distribution.__getitem__),
                probabilities=distribution,
                confidence=confidence,
            )
        elif isinstance(question, ScoreQuestion):
            substantive = probabilities[:-1]
            substantive_mass = math.fsum(substantive)
            if substantive_mass <= 0:
                raise RuntimeErrorBase("Verdict score distribution has no substantive mass")
            answers[question_id] = ScoreAnswer(
                type="score",
                score=math.fsum(
                    index * (probability / substantive_mass)
                    for index, probability in enumerate(substantive)
                ),
                probabilities=distribution,
                confidence=confidence,
                legend=question.criteria,
            )
        elif isinstance(question, NoulQuestion):
            by_id = distribution
            true_false_mass = by_id["true"] + by_id["false"]
            if true_false_mass <= 0:
                raise RuntimeErrorBase("Verdict noul distribution has no true/false mass")
            answers[question_id] = NoulAnswer(
                type="noul", noul=by_id["true"] / true_false_mass
            )
        else:  # pragma: no cover - exhaustive protocol union
            raise RuntimeErrorBase(f"unsupported question type: {type(question).__name__}")
    return answers


class GLiClassCalibratedBackend(DecisionRuntime):
    """One-pass GLiClass distribution head with optional RLCD temperature scaling."""

    def __init__(
        self, model_id: str, *, config: dict[str, Any] | None = None, device: str = "auto"
    ) -> None:
        try:
            import torch
            from gliclass import GLiClassModel
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "GLiClassCalibratedBackend requires transformers, torch, huggingface-hub, and gliclass"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        self.metadata = decision_metadata(self.config)
        if self.metadata.get("readout") != "gliclass_calibrated":
            raise RuntimeErrorBase("GLiClassCalibratedBackend requires decision.readout=gliclass_calibrated")
        self._torch = torch
        loader = _mapping(self.metadata.get("loader", {}), "decision.loader")
        base_model = loader.get("base_model", model_id)
        tokenizer_id = loader.get("tokenizer", base_model)
        if not isinstance(base_model, str) or not isinstance(tokenizer_id, str):
            raise RuntimeErrorBase("decision.loader base_model and tokenizer must be strings")
        kwargs = _loader_kwargs(loader)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
        self._model = GLiClassModel.from_pretrained(base_model, **kwargs)
        self._model.to(_device(torch, self.metadata.get("device"), device))
        self._model.eval()
        self._input_device = next(self._model.parameters()).device
        input_config = _mapping(self.metadata.get("input"), "decision.input")
        self._max_length = _positive_int(input_config.get("max_length"), "decision.input.max_length", 1024)
        self._max_candidates = _positive_int(input_config.get("max_candidates"), "decision.input.max_candidates", 25)
        calibrator_config = self.metadata.get("calibrator")
        self._calibrator: RLCDTemperatureCalibrator | None = None
        if calibrator_config is not None:
            calibrator_spec = _mapping(calibrator_config, "decision.calibrator")
            filename = calibrator_spec.get("file", "calibrator.json")
            repo = calibrator_spec.get("repo", base_model)
            if not isinstance(filename, str) or not isinstance(repo, str):
                raise RuntimeErrorBase("decision.calibrator file and repo must be strings")
            revision = calibrator_spec.get("revision", kwargs.get("revision"))
            if revision is not None and not isinstance(revision, str):
                raise RuntimeErrorBase("decision.calibrator.revision must be a string")
            local_file = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
            self._calibrator = load_rlcd_calibrator(local_file)

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        questions: list[tuple[DecisionRequest, str, Any, list[MarginTask]]] = []
        for request in requests:
            by_question: dict[str, list[MarginTask]] = {}
            for task in compile_verdict_tasks(request):
                by_question.setdefault(task.question_id, []).append(task)
            for question_id, question in request.questions.items():
                tasks = by_question[question_id]
                if len(tasks) > self._max_candidates:
                    raise RuntimeErrorBase(
                        f"GLiClass candidate count {len(tasks)} exceeds configured maximum {self._max_candidates}"
                    )
                questions.append((request, question_id, question, tasks))

        prompts = [
            render_gliclass_prompt(tasks[0].instruction, tasks[0].query, [task.document for task in tasks])
            for _, _, _, tasks in questions
        ]
        encoded = self._tokenizer(prompts, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt")
        encoded = {name: value.to(self._input_device) for name, value in encoded.items()}
        with self._torch.inference_mode():
            output = self._model(**encoded)
        logits = output.logits.float()
        if logits.ndim != 2 or logits.shape[0] != len(questions):
            raise RuntimeErrorBase("GLiClass model returned an invalid distribution-head shape")

        answers_by_request: list[dict[str, Any]] = [{} for _ in requests]
        token_counts = [int(value) for value in encoded["attention_mask"].sum(dim=1).cpu().tolist()]
        request_indices = {id(request): index for index, request in enumerate(requests)}
        for row, (request, question_id, question, tasks) in enumerate(questions):
            if logits.shape[1] < len(tasks):
                raise RuntimeErrorBase("GLiClass distribution head has fewer logits than candidates")
            margins = logits[row, : len(tasks)]
            if not bool(margins.isfinite().all()):
                raise RuntimeErrorBase("GLiClass model produced non-finite candidate logits")
            if self._calibrator is not None:
                margins = margins / self._calibrator.temperature_for(len(tasks))
            single_request = request.model_copy(
                update={"questions": {question_id: question}}
            )
            answers_by_request[request_indices[id(request)]].update(
                aggregate_verdict_answers(single_request, tasks, margins.cpu().tolist())
            )

        usage_by_request = [0] * len(requests)
        for tokens, (request, _, _, _) in zip(token_counts, questions, strict=True):
            usage_by_request[request_indices[id(request)]] += tokens
        return [
            DecisionResponse(model=self.model_name, answers=answers_by_request[index], usage=Usage(input_tokens=usage_by_request[index]))
            for index in range(len(requests))
        ]
