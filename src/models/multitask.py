"""Shared-encoder multi-task model for topic and sentiment classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel


class SharedEncoderMTL(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        *,
        num_topics: int = 13,
        num_sentiments: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.topic_head = nn.Linear(hidden_size, num_topics)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiments)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        *,
        task: str,
    ) -> torch.Tensor:
        encoder_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        if task == "topic":
            return self.topic_head(pooled)
        if task == "sentiment":
            return self.sentiment_head(pooled)
        raise ValueError(f"Unknown task: {task}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
