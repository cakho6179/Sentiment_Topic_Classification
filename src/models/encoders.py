"""Encoder loading, tokenization, and dataset wrappers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.preprocessing.word_seg import segment_vietnamese
from src.training.metrics import compute_metrics


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_id: str
    segment: bool = False
    family: str = "xlmr"


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "visobert": EncoderSpec("visobert", "uitnlp/visobert", segment=False, family="xlmr"),
    "phobert": EncoderSpec("phobert", "vinai/phobert-base-v2", segment=True, family="roberta"),
    "xlmr": EncoderSpec("xlmr", "xlm-roberta-base", segment=False, family="xlmr"),
    "bert_tiny": EncoderSpec("bert_tiny", "prajjwal1/bert-tiny", segment=False, family="bert"),
    "tiny_bert": EncoderSpec("tiny_bert", "hf-internal-testing/tiny-random-BertModel", segment=False, family="bert"),
}


def resolve_spec(name_or_path: str) -> EncoderSpec:
    if name_or_path in ENCODER_SPECS:
        return ENCODER_SPECS[name_or_path]
    path = Path(name_or_path)
    if path.exists():
        return EncoderSpec(name_or_path, str(path), segment=False, family="auto")
    return EncoderSpec(name_or_path, name_or_path, segment=False, family="auto")


def load_tokenizer(spec: EncoderSpec):
    return AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True)


def load_sequence_classifier(spec: EncoderSpec, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )


class EncoderDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        *,
        max_length: int = 256,
        segment: bool = False,
    ) -> None:
        self.texts = ["" if text is None else str(text) for text in texts]
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segment = segment

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text = self.texts[index]
        if self.segment:
            text = text.replace("[SEP]", "__SEP__")
            text = segment_vietnamese(text)
            text = text.replace("__SEP__", self.tokenizer.sep_token)
        else:
            text = text.replace("[SEP]", self.tokenizer.sep_token)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        item = {key: value for key, value in encoded.items()}
        item["labels"] = self.labels[index]
        return item


def build_training_arguments(
    *,
    output_dir: Path,
    learning_rate: float,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 20,
) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": "linear",
        "logging_steps": logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "fp16": fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


class WeightedTrainer(Trainer):
    """Trainer with optional class-weighted cross entropy."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(label_names: Sequence[str]):
    def _compute(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        result = compute_metrics(labels, predictions, label_names=label_names)
        return {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
        }

    return _compute


def make_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    args: TrainingArguments,
    label_names: Sequence[str],
    class_weights: torch.Tensor | None = None,
) -> Trainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": make_compute_metrics(label_names),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    if class_weights is not None:
        return WeightedTrainer(class_weights=class_weights, **kwargs)
    return Trainer(**kwargs)
