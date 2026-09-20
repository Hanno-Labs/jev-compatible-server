"""Configurable model registry for models with stale or incomplete metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .backends import (
    LlamaBackend,
    PointerTransformersBackend,
    TransformersBackend,
    load_decision_config,
)
from .protocol import DecisionRequest, DecisionResponse
from .runtime import DecisionRuntime, RuntimeErrorBase


class RegistryModel(BaseModel):
    """One deployable model and its service-owned decision metadata."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["llama", "transformers", "mlx"]
    model: str
    config: dict[str, Any] = Field(default_factory=dict)
    config_path: str | None = None
    enabled: bool = True
    description: str | None = None
    support_status: Literal["supported", "pending"] = "supported"

    def resolved_config(self) -> dict[str, Any]:
        file_config = load_decision_config(self.config_path) if self.config_path else {}
        # Registry values win over model-published metadata and file defaults.
        file_config.update(self.config)
        return file_config


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str | None = None
    models: dict[str, RegistryModel] = Field(min_length=1)


class ModelRegistry:
    def __init__(self, definition: RegistryFile):
        self.definition = definition

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRegistry":
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise RuntimeErrorBase("model registry must be a JSON object")
        try:
            definition = RegistryFile.model_validate(raw)
        except ValueError as exc:
            raise RuntimeErrorBase(f"invalid model registry: {exc}") from exc
        if definition.default is not None and definition.default not in definition.models:
            raise RuntimeErrorBase(f"registry default is not registered: {definition.default}")
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


class RegistryRuntime(DecisionRuntime):
    """Lazy, cached runtime dispatch with per-model microbatching."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.model_name = registry.definition.default or "registry"
        self._runtimes: dict[str, DecisionRuntime] = {}

    def _runtime(self, name: str, entry: RegistryModel) -> DecisionRuntime:
        cached = self._runtimes.get(name)
        if cached is not None:
            return cached
        config = entry.resolved_config()
        if entry.backend == "llama":
            runtime = LlamaBackend(entry.model, config=config)
        elif entry.backend == "mlx":
            raise RuntimeErrorBase(
                "the MLX backend is registered but not installed in this service image"
            )
        elif config.get("decision.readout", config.get("readout")) == "pointer_head":
            runtime = PointerTransformersBackend(entry.model, config=config)
        else:
            runtime = TransformersBackend(entry.model, config=config)
        self._runtimes[name] = runtime
        return runtime

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        grouped: dict[str, list[tuple[int, DecisionRequest]]] = defaultdict(list)
        entries: dict[str, RegistryModel] = {}
        for index, request in enumerate(requests):
            name, entry = self.registry.resolve(request.model)
            grouped[name].append((index, request))
            entries[name] = entry

        results: list[DecisionResponse | None] = [None] * len(requests)
        for name, indexed_requests in grouped.items():
            runtime = self._runtime(name, entries[name])
            responses = runtime.decide_batch([request for _, request in indexed_requests])
            for (index, _), response in zip(indexed_requests, responses, strict=True):
                results[index] = response
        return [result for result in results if result is not None]
