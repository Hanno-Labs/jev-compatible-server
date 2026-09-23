"""Certo's published ModernBERT option-query architecture.

Imported only when the optional Transformers runtime loads Certo. The module
layout and parameter names match the checkpoint's ``model.pt`` state dict.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class CertoModel(nn.Module):  # type: ignore[misc]
    def __init__(self, backbone: str, revision: str, heads: int) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone, revision=revision)
        hidden = self.bert.config.hidden_size
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _enc(self, ids: Any, mask: Any) -> Any:
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state

    @staticmethod
    def _pool(hidden: Any, mask: Any) -> Any:
        expanded = mask.unsqueeze(-1).float()
        return (hidden * expanded).sum(1) / expanded.sum(1).clamp_min(1.0)

    def forward(
        self, state_ids: Any, state_mask: Any, option_ids: Any, option_mask: Any, valid: Any
    ) -> Any:
        batch, options, option_length = option_ids.shape
        state_hidden = self._enc(state_ids, state_mask)
        option_hidden = self._enc(
            option_ids.reshape(batch * options, option_length),
            option_mask.reshape(batch * options, option_length),
        )
        option_vectors = self._pool(
            option_hidden, option_mask.reshape(batch * options, option_length)
        ).reshape(batch, options, -1)
        context, _ = self.cross(
            option_vectors, state_hidden, state_hidden,
            key_padding_mask=(state_mask == 0),
        )
        combined = self.norm(option_vectors + context)
        logits = self.score(torch.cat([option_vectors, combined], -1)).squeeze(-1)
        return logits.masked_fill(~valid, -1e9)
