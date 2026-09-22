"""Sentence-Transformers CrossEncoder adapter for open rerankers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .encoder_decoder import aggregate_margin_answers, compile_margin_tasks, decision_metadata
from .protocol import DecisionRequest, DecisionResponse, Usage
from .runtime import DecisionRuntime, RuntimeErrorBase


class CrossEncoderBackend(DecisionRuntime):
    """Expose a public CrossEncoder/reranker as typed candidate probabilities."""

    def __init__(
        self,
        model_id: str,
        *,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeErrorBase(
                "CrossEncoderBackend requires sentence-transformers and torch"
            ) from exc

        self.model_name = str((config or {}).get("model", model_id))
        self.config = config or {}
        metadata = decision_metadata(self.config)
        if metadata.get("readout") != "cross_encoder_margin":
            raise RuntimeErrorBase(
                "CrossEncoderBackend requires decision.readout=cross_encoder_margin"
            )
        self._batch_size = int(metadata.get("batch_size", 32))
        target = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self._model = CrossEncoder(
            model_id,
            max_length=int(metadata.get("max_length", 512)),
            device=target,
        )

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        compiled = [compile_margin_tasks(request, decision_metadata(self.config)) for request in requests]
        tasks = [task for request_tasks in compiled for task in request_tasks]
        pairs = [(task.query, f"{task.instruction}\n\n{task.document}") for task in tasks]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        margins = [float(value) for value in scores]
        responses: list[DecisionResponse] = []
        offset = 0
        metadata = decision_metadata(self.config)
        for request, request_tasks in zip(requests, compiled, strict=True):
            end = offset + len(request_tasks)
            responses.append(
                DecisionResponse(
                    model=self.model_name,
                    answers=aggregate_margin_answers(
                        request, request_tasks, margins[offset:end], metadata
                    ),
                    usage=Usage(),
                )
            )
            offset = end
        return responses
