"""Capability boundary for GLiNER2's public multilabel classifier API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .protocol import DecisionRequest, DecisionResponse, UnsupportedAnswer, Usage
from .runtime import DecisionRuntime


class GLiNER2Runtime(DecisionRuntime):
    """Return explicit non-support for a model without a categorical readout.

    GLiNER2's public ``classify_text`` API reports independent multilabel
    confidences.  They are not a posterior over dynamically supplied criteria,
    so applying softmax or complement arithmetic here would measure a service
    invented mapping rather than GLiNER2.
    """

    def __init__(self, model_id: str, *, config: dict[str, Any] | None = None) -> None:
        del config
        self.model_name = model_id

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResponse]:
        return [
            DecisionResponse(
                model=self.model_name,
                answers={
                    name: UnsupportedAnswer(
                        type="unsupported",
                        question_type=question.type,
                        supported_types=[],
                    )
                    for name, question in request.questions.items()
                },
                usage=Usage(),
            )
            for request in requests
        ]
