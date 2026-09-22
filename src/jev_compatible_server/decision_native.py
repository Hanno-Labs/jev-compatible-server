"""Checkpoint-native Decision 1.0 inference behind the typed JEV contract."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .native_systemone import NativeSystemOneHTTPRuntime, _decision_value
from .protocol import DecisionRequest, DecisionResponse
from .runtime import RuntimeErrorBase


class DecisionNativeRuntime(NativeSystemOneHTTPRuntime):
    """Use a published Decision checkpoint's own head and probability readout.

    The native HTTP adapter also owns the strict typed-response decoder; only
    that decoder is inherited here.  No HTTP endpoint is contacted.
    """

    def __init__(
        self,
        model_id: str,
        *,
        config: Mapping[str, Any] | None = None,
        agent: Any = None,
    ) -> None:
        self.config = dict(config or {})
        self.model_name = str(_decision_value(self.config, "public_model_name", model_id))
        revision = _decision_value(self.config, "revision")
        if not isinstance(revision, str) or not revision:
            raise RuntimeErrorBase("decision_native requires a pinned decision.revision")
        batch_size = _decision_value(self.config, "batch_size", 8)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise RuntimeErrorBase("decision.batch_size must be a positive integer")
        max_length = _decision_value(self.config, "max_length", 16_384)
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise RuntimeErrorBase("decision.max_length must be a positive integer")
        self._agent = agent or self._load_agent(model_id, revision, batch_size, max_length)

    @staticmethod
    def _load_agent(model_id: str, revision: str, batch_size: int, max_length: int) -> Any:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase("decision_native requires huggingface_hub") from exc
        snapshot = Path(snapshot_download(model_id, revision=revision))
        if not (snapshot / "decision" / "__init__.py").is_file():
            raise RuntimeErrorBase("Decision checkpoint is missing its native decision package")
        source_dir = str(snapshot)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        try:
            module = importlib.import_module("decision")
            model_class = module.DecisionModel
            return model_class.from_pretrained(
                source_dir,
                device="cuda:0",
                batch_size=batch_size,
                max_length=max_length,
            )
        except Exception as exc:  # pragma: no cover - model-owned loading errors
            raise RuntimeErrorBase(f"failed to load native Decision checkpoint {model_id!r}: {exc}") from exc

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        if not requests:
            return []
        payloads = [
            request.model_dump(mode="json", exclude={"model"}, exclude_none=True)
            for request in requests
        ]
        try:
            raw = self._agent.decide_batch(payloads)
        except Exception as exc:  # pragma: no cover - model-owned inference errors
            raise RuntimeErrorBase(f"native Decision inference failed: {exc}") from exc
        if not isinstance(raw, list) or len(raw) != len(requests):
            raise RuntimeErrorBase("native Decision returned the wrong batch length")
        for result in raw:
            if not isinstance(result, dict):
                raise RuntimeErrorBase("native Decision returned a non-object response")
        return [self._decode(request, result) for request, result in zip(requests, raw, strict=True)]
