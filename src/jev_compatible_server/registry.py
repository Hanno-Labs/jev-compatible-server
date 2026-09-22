"""Configurable model registry for models with stale or incomplete metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .backends import (
    LlamaBackend,
    PointerTransformersBackend,
    TransformersBackend,
    load_decision_config,
)
from .bosun import BosunDecisionBackend
from .causal_options import CausalOptionsBackend
from .classifier_adapters import GLiClassCalibratedBackend, NLIEntailmentBackend
from .cross_encoder import CrossEncoderBackend
from .custom_heads import OpenJevScalarHeadBackend, SmallJevSemanticBackend
from .encoder_decoder import EncoderDecoderMarginBackend, decision_metadata
from .gliner2 import GLiNER2Runtime
from .hidden_state_probe import HiddenStateProbeBackend
from .jev_local_options import JevLocalOptionsBackend
from .laya import LayaBackend
from .native_systemone import (
    DjevHTTPRuntime,
    DjevThinkingRuntime,
    JeffHTTPRuntime,
    OpenJevThinkingHTTPRuntime,
    WinnowHTTPRuntime,
)
from .protocol import DecisionRequest, DecisionResponse, UnsupportedAnswer, Usage
from .runtime import (
    DecisionRuntime,
    RuntimeErrorBase,
    apply_question_type_support,
    configured_question_types,
)
from .sequence_classifier import SequenceClassifierMarginBackend


class RegistryModel(BaseModel):
    """One deployable model and its service-owned decision metadata."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["llama", "transformers", "mlx"]
    model: str
    recipe: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    config_path: str | None = None
    enabled: bool = True
    description: str | None = None
    support_status: Literal["supported", "pending"] = "supported"

    def resolved_config(
        self, recipe_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        resolved = _merge_config({}, recipe_config or {})
        file_config = load_decision_config(self.config_path) if self.config_path else {}
        # Registry values win over model-published metadata and file defaults.
        resolved = _merge_config(resolved, file_config)
        return _merge_config(resolved, self.config)


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str | None = None
    recipes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    models: dict[str, RegistryModel] = Field(min_length=1)


def _merge_config(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _merge_config(existing, value)
        else:
            result[key] = value
    return result


class ModelRegistry:
    def __init__(self, definition: RegistryFile):
        self.definition = definition

    @classmethod
    def from_file(cls, path: str | Path) -> ModelRegistry:
        return cls.from_json(Path(path).read_text())

    @classmethod
    def from_builtin(cls) -> ModelRegistry:
        resource = resources.files("jev_compatible_server").joinpath("public-models.json")
        try:
            value = resource.read_text()
        except FileNotFoundError:
            source_path = Path(__file__).parents[2] / "configs" / "public-models.json"
            value = source_path.read_text()
        return cls.from_json(value)

    @classmethod
    def from_json(cls, value: str) -> ModelRegistry:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise RuntimeErrorBase("model registry must be a JSON object")
        try:
            definition = RegistryFile.model_validate(raw)
        except ValueError as exc:
            raise RuntimeErrorBase(f"invalid model registry: {exc}") from exc
        if definition.default is not None and definition.default not in definition.models:
            raise RuntimeErrorBase(f"registry default is not registered: {definition.default}")
        for name, entry in definition.models.items():
            if entry.recipe is not None and entry.recipe not in definition.recipes:
                raise RuntimeErrorBase(
                    f"model {name!r} references unregistered recipe: {entry.recipe}"
                )
        return cls(definition)

    def resolve(self, requested: str | None) -> tuple[str, RegistryModel]:
        name = requested or self.definition.default
        if name is None:
            raise RuntimeErrorBase("request must specify model or registry needs a default")
        entry = self.definition.models.get(name)
        if entry is None:
            raise RuntimeErrorBase(f"model is not registered: {name}")
        if not entry.enabled:
            raise RuntimeErrorBase(f"model is disabled: {name}")
        return name, entry


def build_transformers_runtime(
    model_id: str,
    config: dict[str, Any],
    *,
    batch_size_override: int | None = None,
) -> DecisionRuntime:
    """Select a Transformers readout from metadata, never from a model name."""

    effective_config = dict(config)
    if batch_size_override is not None:
        if isinstance(batch_size_override, bool) or batch_size_override <= 0:
            raise RuntimeErrorBase("model batch size override must be a positive integer")
        # Dotted metadata has highest precedence in decision_metadata(), so this
        # overrides both nested and dotted registry recipe values without
        # mutating the registry's reusable configuration.
        effective_config["decision.batch_size"] = batch_size_override

    readout = decision_metadata(effective_config).get("readout")
    if readout == "pointer_head":
        return PointerTransformersBackend(model_id, config=effective_config)
    if readout == "encoder_decoder_margin":
        return EncoderDecoderMarginBackend(model_id, config=effective_config)
    if readout == "sequence_classifier_margin":
        return SequenceClassifierMarginBackend(model_id, config=effective_config)
    if readout == "cross_encoder_margin":
        return CrossEncoderBackend(model_id, config=effective_config)
    if readout == "causal_options":
        return CausalOptionsBackend(model_id, config=effective_config)
    if readout == "bosun_decision_tokens":
        return BosunDecisionBackend(model_id, config=effective_config)
    if readout == "nli_entailment":
        return NLIEntailmentBackend(model_id, config=effective_config)
    if readout == "gliclass_calibrated":
        return GLiClassCalibratedBackend(model_id, config=effective_config)
    if readout == "openjev_scalar_head":
        return OpenJevScalarHeadBackend(model_id, config=effective_config)
    if readout == "semantic_option_head":
        return SmallJevSemanticBackend(model_id, config=effective_config)
    if readout == "hidden_state_probe":
        return HiddenStateProbeBackend(model_id, config=effective_config)
    if readout == "laya_native":
        return LayaBackend(model_id, config=effective_config)
    if readout == "diffusion_structured_read":
        return DjevHTTPRuntime(model_id, config=effective_config)
    if readout == "diffusion_thinking_read":
        native_contract = decision_metadata(effective_config).get("native_contract")
        if native_contract == "djev":
            return DjevThinkingRuntime(model_id, config=effective_config)
        if native_contract == "openjev":
            return OpenJevThinkingHTTPRuntime(model_id, config=effective_config)
        raise RuntimeErrorBase(
            "diffusion_thinking_read requires decision.native_contract of 'djev' or 'openjev'"
        )
    if readout == "gliformer_native":
        return JeffHTTPRuntime(model_id, config=effective_config)
    if readout == "gliner2_multilabel":
        return GLiNER2Runtime(model_id, config=effective_config)
    if readout == "jev_local_options":
        return JevLocalOptionsBackend(model_id, config=effective_config)
    return TransformersBackend(model_id, config=effective_config)


class RegistryRuntime(DecisionRuntime):
    """Cached registry dispatch, optionally pinned to one eagerly loaded model."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        pinned_model: str | None = None,
        model_batch_size: int | None = None,
    ) -> None:
        self.registry = registry
        self.model_batch_size = model_batch_size
        self._runtimes: dict[str, DecisionRuntime] = {}
        self.pinned_model: str | None = None
        if pinned_model is None:
            self.model_name = registry.definition.default or "registry"
            return
        name, entry = registry.resolve(pinned_model)
        self.pinned_model = name
        self.model_name = name
        self._runtime(name, entry)

    def _runtime(self, name: str, entry: RegistryModel) -> DecisionRuntime:
        cached = self._runtimes.get(name)
        if cached is not None:
            return cached
        recipe_config = (
            self.registry.definition.recipes.get(entry.recipe, {})
            if entry.recipe is not None
            else {}
        )
        config = entry.resolved_config(recipe_config)
        runtime: DecisionRuntime
        if entry.backend == "llama":
            if decision_metadata(config).get("readout") == "winnow_shared_branch":
                runtime = WinnowHTTPRuntime(entry.model, config=config)
            else:
                runtime = LlamaBackend(entry.model, config=config)
        elif entry.backend == "mlx":
            raise RuntimeErrorBase(
                "the MLX backend is registered but not installed in this service image"
            )
        else:
            runtime = build_transformers_runtime(
                entry.model,
                config,
                batch_size_override=self.model_batch_size,
            )
        runtime = apply_question_type_support(runtime, config)
        self._runtimes[name] = runtime
        return runtime

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        grouped: dict[str, list[tuple[int, DecisionRequest]]] = defaultdict(list)
        entries: dict[str, RegistryModel] = {}
        for index, request in enumerate(requests):
            requested_model = request.model
            if self.pinned_model is not None:
                if requested_model not in (None, self.pinned_model):
                    raise RuntimeErrorBase(
                        f"server is pinned to model {self.pinned_model!r}; "
                        f"request selected {requested_model!r}"
                    )
                requested_model = self.pinned_model
            name, entry = self.registry.resolve(requested_model)
            grouped[name].append((index, request))
            entries[name] = entry

        results: list[DecisionResponse | None] = [None] * len(requests)
        for name, indexed_requests in grouped.items():
            entry = entries[name]
            recipe_config = (
                self.registry.definition.recipes.get(entry.recipe, {})
                if entry.recipe is not None
                else {}
            )
            config = entry.resolved_config(recipe_config)
            supported_types = configured_question_types(config)
            supported_set = frozenset(supported_types)
            if not any(
                question.type in supported_set
                for _, request in indexed_requests
                for question in request.questions.values()
            ):
                for index, request in indexed_requests:
                    results[index] = DecisionResponse(
                        model=entry.model,
                        answers={
                            question_name: UnsupportedAnswer(
                                type="unsupported",
                                question_type=question.type,
                                supported_types=list(supported_types),
                            )
                            for question_name, question in request.questions.items()
                        },
                        usage=Usage(),
                    )
                continue
            runtime = self._runtime(name, entries[name])
            responses = runtime.decide_batch([request for _, request in indexed_requests])
            for (index, _), response in zip(indexed_requests, responses, strict=True):
                results[index] = response
        return [result for result in results if result is not None]
