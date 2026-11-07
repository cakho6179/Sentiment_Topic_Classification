"""Feature-fusion architecture: text encoder + brand embedding."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.preprocessing.word_seg import segment_vietnamese


class BrandFusionClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        *,
        num_brands: int,
        num_labels: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = int(self.encoder.config.hidden_size)
        self.pooling = pooling
        self.brand_embedding = nn.Embedding(num_brands, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        brand_id: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        text_repr = self._pool(outputs.last_hidden_state, attention_mask)
        brand_repr = self.brand_embedding(brand_id)
        fusion = torch.cat(
            [
                text_repr,
                brand_repr,
                torch.abs(text_repr - brand_repr),
                text_repr * brand_repr,
            ],
            dim=-1,
        )
        logits = self.classifier(self.dropout(fusion))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class BrandFusionDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        brands: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        brand_to_id: dict[str, int],
        *,
        max_length: int = 256,
        segment: bool = False,
    ) -> None:
        self.texts = ["" if text is None else str(text) for text in texts]
        self.brands = [str(brand) for brand in brands]
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.brand_to_id = brand_to_id
        self.max_length = max_length
        self.segment = segment

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text = self.texts[index].replace("[SEP]", "__SEP__")
        if self.segment:
            text = segment_vietnamese(text)
            text = text.replace("__SEP__", self.tokenizer.sep_token)
        else:
            text = text.replace("__SEP__", self.tokenizer.sep_token)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        item = {key: value for key, value in encoded.items()}
        item["brand_id"] = self.brand_to_id.get(self.brands[index], self.brand_to_id["[UNK]"])
        item["labels"] = self.labels[index]
        return item


class FusionCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        input_ids = []
        attention_mask = []
        token_type_ids = []
        brand_ids = []
        labels = []
        has_token_types = any("token_type_ids" in feature for feature in features)
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            token_type_ids.append(feature.get("token_type_ids", [0] * len(feature["input_ids"])) + [0] * pad)
            brand_ids.append(feature["brand_id"])
            labels.append(feature["labels"])
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "brand_id": torch.tensor(brand_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if has_token_types:
            batch["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
        return batch
