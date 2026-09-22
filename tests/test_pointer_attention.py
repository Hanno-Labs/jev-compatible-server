"""Pointer-head mask dtype and geometry checks."""

from collections.abc import Iterator
from types import SimpleNamespace

import torch
from jev_compatible_server.backends import PointerTransformersBackend


class _CaptureModel:
    def __init__(self, dtype: torch.dtype) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.mask: torch.Tensor | None = None

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self.weight

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.mask = attention_mask
        return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 2)))


def _capture_mask(implementation: str) -> torch.Tensor:
    backend = object.__new__(PointerTransformersBackend)
    backend._torch = torch
    backend._device = torch.device("cpu")
    backend._tokenizer = SimpleNamespace(pad_token_id=0)
    backend._attention_impl = implementation
    model = _CaptureModel(torch.bfloat16)
    backend._model = model
    backend._hidden_batch(
        [
            {
                "ids": [1, 2, 3, 4, 5, 6],
                "seg": [0, 1, 1, 1, 1, 1],
                "opt": [-1, 0, 0, 1, 1, -2],
                "ends": [2, 4],
                "decision": 5,
            }
        ]
    )
    assert model.mask is not None
    return model.mask


def test_pointer_sdpa_mask_uses_model_dtype_without_changing_geometry() -> None:
    eager = _capture_mask("eager")
    sdpa = _capture_mask("sdpa")
    assert eager.dtype == torch.float32
    assert sdpa.dtype == torch.bfloat16
    assert torch.equal(eager == 0, sdpa == 0)
    assert sdpa[0, 0, 4, 2] < 0
    assert sdpa[0, 0, 5, 4] == 0
