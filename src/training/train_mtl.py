"""Train a shared-encoder multi-task model for topic and sentiment."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME, TOPIC_IDS
from src.evaluation.bootstrap import bootstrap_macro_f1
from src.models.encoders import EncoderDataset, load_tokenizer, resolve_spec
from src.models.multitask import SharedEncoderMTL, move_batch_to_device
from src.models.tfidf import build_variant
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


TOPIC_LABELS = [f"{tid}:{TOPIC_ID_TO_NAME[tid]}" for tid in TOPIC_IDS]
SENTIMENT_LABELS = [SENTIMENT_ID_TO_NAME[i] for i in range(3)]


def _autocast(enabled: bool):
    if not enabled:
        import contextlib

        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def _make_grad_scaler(enabled: bool):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.amp.GradScaler(device, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _next_batch(iterator: Iterable):
    try:
        return next(iterator)
    except StopIteration:
        return None


def compute_class_weights(y: np.ndarray, num_labels: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(y) / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_loader(
    frame: pd.DataFrame,
    *,
    task: str,
    variant: str,
    tokenizer,
    segment: bool,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> tuple[DataLoader, np.ndarray]:
    if task == "topic":
        label_column = "topic_index"
    elif task == "sentiment":
        label_column = "sentiment_id"
    else:
        raise ValueError(task)
    frame = frame.reset_index(drop=True)
    texts = build_variant(frame, task, variant).fillna("").astype(str).tolist()
    labels = frame[label_column].astype(int).to_numpy()
    dataset = EncoderDataset(texts, labels, tokenizer, max_length=max_length, segment=segment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        num_workers=0,
    )
    return loader, labels


class MultiTaskTrainer:
    def __init__(
        self,
        *,
        model: SharedEncoderMTL,
        tokenizer,
        topic_train_loader: DataLoader,
        topic_val_loader: DataLoader,
        topic_test_loader: DataLoader,
        sentiment_train_loader: DataLoader,
        sentiment_val_loader: DataLoader,
        sentiment_test_loader: DataLoader,
        config: dict[str, Any],
        output_dir: Path,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.topic_train_loader = topic_train_loader
        self.topic_val_loader = topic_val_loader
        self.topic_test_loader = topic_test_loader
        self.sentiment_train_loader = sentiment_train_loader
        self.sentiment_val_loader = sentiment_val_loader
        self.sentiment_test_loader = sentiment_test_loader
        self.config = config
        self.output_dir = output_dir
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fp16 = bool(config.get("fp16", torch.cuda.is_available()))
        self.scaler = _make_grad_scaler(self.fp16)

        params = list(self.model.parameters())
        self.uncertainty = bool(config.get("uncertainty_weighting", False))
        self.log_sigma = None
        if self.uncertainty:
            self.log_sigma = torch.nn.Parameter(torch.zeros(2, device=self.device))
            params.append(self.log_sigma)

        self.optimizer = AdamW(
            params,
            lr=float(config.get("learning_rate", 2e-5)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        self.steps_per_epoch = max(len(topic_train_loader), len(sentiment_train_loader))
        self.epochs = int(config.get("num_train_epochs", 10))
        self.total_steps = max(1, self.steps_per_epoch * self.epochs)
        self.warmup_steps = int(float(config.get("warmup_ratio", 0.1)) * self.total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        self.topic_class_weights = None
        self.sentiment_class_weights = None
        if bool(config.get("class_weights", False)):
            self.topic_class_weights = compute_class_weights(
                topic_train_loader.dataset.labels, 13
            ).to(self.device)
            self.sentiment_class_weights = compute_class_weights(
                sentiment_train_loader.dataset.labels, 3
            ).to(self.device)

    def _loss(self, task: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if task == "topic":
            weight = self.topic_class_weights
        else:
            weight = self.sentiment_class_weights
        return F.cross_entropy(logits, labels, weight=weight)

    def _combine_losses(self, topic_loss: torch.Tensor | None, sentiment_loss: torch.Tensor | None) -> torch.Tensor:
        if self.uncertainty and self.log_sigma is not None:
            losses = []
            if topic_loss is not None:
                losses.append(torch.exp(-self.log_sigma[0]) * topic_loss + self.log_sigma[0])
            if sentiment_loss is not None:
                losses.append(torch.exp(-self.log_sigma[1]) * sentiment_loss + self.log_sigma[1])
            return sum(losses)
        lambda_topic = float(self.config.get("lambda_topic", 1.0))
        lambda_sentiment = float(self.config.get("lambda_sentiment", 1.0))
        loss = 0.0
        if topic_loss is not None:
            loss = loss + lambda_topic * topic_loss
        if sentiment_loss is not None:
            loss = loss + lambda_sentiment * sentiment_loss
        return loss

    def _train_step(self, topic_batch: dict | None, sentiment_batch: dict | None) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        topic_loss = None
        sentiment_loss = None
        with _autocast(self.fp16):
            if topic_batch is not None:
                topic_labels = topic_batch.pop("labels").to(self.device)
                topic_batch = move_batch_to_device(topic_batch, self.device)
                topic_logits = self.model(task="topic", **topic_batch)
                topic_loss = self._loss("topic", topic_logits, topic_labels)
            if sentiment_batch is not None:
                sentiment_labels = sentiment_batch.pop("labels").to(self.device)
                sentiment_batch = move_batch_to_device(sentiment_batch, self.device)
                sentiment_logits = self.model(task="sentiment", **sentiment_batch)
                sentiment_loss = self._loss("sentiment", sentiment_logits, sentiment_labels)
            loss = self._combine_losses(topic_loss, sentiment_loss)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("max_grad_norm", 1.0)))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return {
            "loss": float(loss.detach().cpu()),
            "topic_loss": float(topic_loss.detach().cpu()) if topic_loss is not None else float("nan"),
            "sentiment_loss": float(sentiment_loss.detach().cpu()) if sentiment_loss is not None else float("nan"),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, task: str) -> dict[str, Any]:
        self.model.eval()
        all_logits = []
        all_labels = []
        for batch in loader:
            labels = batch.pop("labels").to(self.device)
            batch = move_batch_to_device(batch, self.device)
            with _autocast(self.fp16):
                logits = self.model(task=task, **batch)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=-1)
        names = TOPIC_LABELS if task == "topic" else SENTIMENT_LABELS
        metrics = compute_metrics(labels, preds, y_prob=probs, label_names=names)
        return {"metrics": metrics, "probs": probs, "preds": preds, "labels": labels}

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        best_score = -math.inf
        best_state = None
        best_epoch = -1
        patience = int(self.config.get("early_stopping_patience", 3))
        epochs_without_improvement = 0
        train_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            topic_iter = iter(self.topic_train_loader)
            sentiment_iter = iter(self.sentiment_train_loader)
            step = 0
            running = {"loss": 0.0, "topic_loss": 0.0, "sentiment_loss": 0.0, "n": 0}
            for _ in range(self.steps_per_epoch):
                topic_batch = _next_batch(topic_iter)
                sentiment_batch = _next_batch(sentiment_iter)
                if topic_batch is None and sentiment_batch is None:
                    break
                stats = self._train_step(topic_batch, sentiment_batch)
                running["loss"] += stats["loss"]
                if not math.isnan(stats["topic_loss"]):
                    running["topic_loss"] += stats["topic_loss"]
                if not math.isnan(stats["sentiment_loss"]):
                    running["sentiment_loss"] += stats["sentiment_loss"]
                running["n"] += 1
                step += 1

            topic_val = self.evaluate(self.topic_val_loader, "topic")
            sentiment_val = self.evaluate(self.sentiment_val_loader, "sentiment")
            topic_f1 = topic_val["metrics"]["macro_f1"]
            sentiment_f1 = sentiment_val["metrics"]["macro_f1"]
            score = 0.5 * (topic_f1 + sentiment_f1)
            entry = {
                "epoch": epoch,
                "steps": step,
                "train_loss": running["loss"] / max(1, running["n"]),
                "train_topic_loss": running["topic_loss"] / max(1, running["n"]),
                "train_sentiment_loss": running["sentiment_loss"] / max(1, running["n"]),
                "val_topic_macro_f1": topic_f1,
                "val_sentiment_macro_f1": sentiment_f1,
                "val_avg_macro_f1": score,
            }
            if self.log_sigma is not None:
                entry["log_sigma_topic"] = float(self.log_sigma[0].detach().cpu())
                entry["log_sigma_sentiment"] = float(self.log_sigma[1].detach().cpu())
            history.append(entry)
            print(json.dumps(entry, ensure_ascii=False))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        train_seconds = time.perf_counter() - train_start
        if best_state is not None:
            self.model.load_state_dict(best_state)
        topic_val = self.evaluate(self.topic_val_loader, "topic")
        sentiment_val = self.evaluate(self.sentiment_val_loader, "sentiment")
        topic_test = self.evaluate(self.topic_test_loader, "topic")
        sentiment_test = self.evaluate(self.sentiment_test_loader, "sentiment")
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "train_seconds": train_seconds,
            "topic_val": topic_val,
            "sentiment_val": sentiment_val,
            "topic_test": topic_test,
            "sentiment_test": sentiment_test,
        }


def _load_frames(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    topics = pd.read_csv(processed_dir / "comments.csv")
    sentiment = pd.read_csv(processed_dir / "comment_brands.csv")
    for column in ["target", "context", "brands"]:
        if column in topics.columns:
            topics[column] = topics[column].fillna("").astype(str)
    for column in ["target", "context", "brand"]:
        if column in sentiment.columns:
            sentiment[column] = sentiment[column].fillna("").astype(str)
    return topics, sentiment


def run_mtl(
    *,
    encoder_name: str,
    processed_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    topics, sentiment = _load_frames(processed_dir)
    if smoke:
        topics = topics.groupby("split", group_keys=False).head(int(config.get("smoke_train_size", 64)))
        sentiment = sentiment.groupby("split", group_keys=False).head(int(config.get("smoke_train_size", 64)))

    spec = resolve_spec(encoder_name)
    tokenizer = load_tokenizer(spec)
    max_length = int(config.get("max_length", 256))
    batch_size = int(config.get("batch_size", 16))
    eval_batch_size = int(config.get("eval_batch_size", 32))

    topic_train, topic_train_labels = build_loader(
        topics[topics["split"] == "train"],
        task="topic",
        variant=str(config.get("topic_variant", "target_context")),
        tokenizer=tokenizer,
        segment=spec.segment,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    topic_val, _ = build_loader(
        topics[topics["split"] == "val"],
        task="topic",
        variant=str(config.get("topic_variant", "target_context")),
        tokenizer=tokenizer,
        segment=spec.segment,
        max_length=max_length,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    topic_test, _ = build_loader(
        topics[topics["split"] == "test"],
        task="topic",
        variant=str(config.get("topic_variant", "target_context")),
        tokenizer=tokenizer,
        segment=spec.segment,
        max_length=max_length,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    sentiment_train, sentiment_train_labels = build_loader(
        sentiment[sentiment["split"] == "train"],
        task="sentiment",
        variant=str(config.get("sentiment_variant", "full")),
        tokenizer=tokenizer,
        segment=spec.segment,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    sentiment_val, _ = build_loader(
        sentiment[sentiment["split"] == "val"],
        task="sentiment",
        variant=str(config.get("sentiment_variant", "full")),
        tokenizer=tokenizer,
        segment=spec.segment,
        max_length=max_length,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    sentiment_test, _ = build_loader(
        sentiment[sentiment["split"] == "test"],
        task="sentiment",
        variant=str(config.get("sentiment_variant", "full")),
        tokenizer=tokenizer,
        segment=spec.segment,
        max_length=max_length,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    model = SharedEncoderMTL(
        spec.hf_id,
        num_topics=13,
        num_sentiments=3,
        dropout=float(config.get("dropout", 0.1)),
        pooling=str(config.get("pooling", "mean")),
    )
    if bool(config.get("uncertainty_weighting", False)):
        suffix = "_uncertainty"
    else:
        suffix = f"_lt{config.get('lambda_topic', 1.0)}_ls{config.get('lambda_sentiment', 1.0)}"
    experiment_id = f"mtl_{encoder_name}{suffix}_seed{seed}"
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        topic_train_loader=topic_train,
        topic_val_loader=topic_val,
        topic_test_loader=topic_test,
        sentiment_train_loader=sentiment_train,
        sentiment_val_loader=sentiment_val,
        sentiment_test_loader=sentiment_test,
        config=config,
        output_dir=output_dir,
        seed=seed,
    )
    result = trainer.fit()

    topic_test_frame = topics[topics["split"] == "test"].reset_index(drop=True)
    sentiment_test_frame = sentiment[sentiment["split"] == "test"].reset_index(drop=True)
    topic_probs = result["topic_test"]["probs"]
    sentiment_probs = result["sentiment_test"]["probs"]
    topic_preds = result["topic_test"]["preds"]
    sentiment_preds = result["sentiment_test"]["preds"]
    topic_true = result["topic_test"]["labels"]
    sentiment_true = result["sentiment_test"]["labels"]

    topic_metrics = result["topic_test"]["metrics"]
    sentiment_metrics = result["sentiment_test"]["metrics"]
    topic_metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        topic_true, topic_preds, labels=list(range(13)), seed=seed
    )
    sentiment_metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        sentiment_true, sentiment_preds, labels=list(range(3)), seed=seed
    )

    topic_predictions = topic_test_frame[["comment_id", "topic_id", "brands"]].copy()
    topic_predictions["split"] = "test"
    topic_predictions["y_true"] = topic_true
    topic_predictions["y_pred"] = topic_preds
    topic_predictions["confidence"] = topic_probs.max(axis=1)
    topic_predictions["task"] = "topic"
    topic_predictions["model"] = "mtl"
    topic_predictions["seed"] = seed
    for index in range(13):
        topic_predictions[f"prob_{index}"] = topic_probs[:, index]

    sentiment_predictions = sentiment_test_frame[["comment_id", "brand", "topic_id"]].copy()
    sentiment_predictions["split"] = "test"
    sentiment_predictions["y_true"] = sentiment_true
    sentiment_predictions["y_pred"] = sentiment_preds
    sentiment_predictions["confidence"] = sentiment_probs.max(axis=1)
    sentiment_predictions["task"] = "sentiment"
    sentiment_predictions["model"] = "mtl"
    sentiment_predictions["seed"] = seed
    for index in range(3):
        sentiment_predictions[f"prob_{index}"] = sentiment_probs[:, index]

    topic_val_frame = topics[topics["split"] == "val"].reset_index(drop=True)
    sentiment_val_frame = sentiment[sentiment["split"] == "val"].reset_index(drop=True)
    topic_val_probs = result["topic_val"]["probs"]
    sentiment_val_probs = result["sentiment_val"]["probs"]
    topic_val_preds = result["topic_val"]["preds"]
    sentiment_val_preds = result["sentiment_val"]["preds"]
    topic_val_true = result["topic_val"]["labels"]
    sentiment_val_true = result["sentiment_val"]["labels"]

    topic_val_predictions = topic_val_frame[["comment_id", "topic_id", "brands"]].copy()
    topic_val_predictions["split"] = "val"
    topic_val_predictions["y_true"] = topic_val_true
    topic_val_predictions["y_pred"] = topic_val_preds
    topic_val_predictions["confidence"] = topic_val_probs.max(axis=1)
    topic_val_predictions["task"] = "topic"
    topic_val_predictions["model"] = "mtl"
    topic_val_predictions["seed"] = seed
    for index in range(13):
        topic_val_predictions[f"prob_{index}"] = topic_val_probs[:, index]

    sentiment_val_predictions = sentiment_val_frame[["comment_id", "brand", "topic_id"]].copy()
    sentiment_val_predictions["split"] = "val"
    sentiment_val_predictions["y_true"] = sentiment_val_true
    sentiment_val_predictions["y_pred"] = sentiment_val_preds
    sentiment_val_predictions["confidence"] = sentiment_val_probs.max(axis=1)
    sentiment_val_predictions["task"] = "sentiment"
    sentiment_val_predictions["model"] = "mtl"
    sentiment_val_predictions["seed"] = seed
    for index in range(3):
        sentiment_val_predictions[f"prob_{index}"] = sentiment_val_probs[:, index]

    ensure_dir(output_dir)
    topic_predictions.to_csv(output_dir / f"{experiment_id}_topic_predictions.csv", index=False, encoding="utf-8")
    sentiment_predictions.to_csv(output_dir / f"{experiment_id}_sentiment_predictions.csv", index=False, encoding="utf-8")
    topic_val_predictions.to_csv(
        output_dir / f"{experiment_id}_topic_val_predictions.csv", index=False, encoding="utf-8"
    )
    sentiment_val_predictions.to_csv(
        output_dir / f"{experiment_id}_sentiment_val_predictions.csv", index=False, encoding="utf-8"
    )
    save_json(
        output_dir / f"{experiment_id}_metrics.json",
        {
            "experiment_id": experiment_id,
            "model": "mtl",
            "encoder": encoder_name,
            "hf_id": spec.hf_id,
            "seed": seed,
            "config": config,
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "train_seconds": result["train_seconds"],
            "history": result["history"],
            "topic_val": result["topic_val"]["metrics"],
            "sentiment_val": result["sentiment_val"]["metrics"],
            "topic_test": topic_metrics,
            "sentiment_test": sentiment_metrics,
            "topic_label_names": TOPIC_LABELS,
            "sentiment_label_names": SENTIMENT_LABELS,
            "n_topic_train": int(len(topic_train_labels)),
            "n_sentiment_train": int(len(sentiment_train_labels)),
        },
    )

    if bool(config.get("save_model", False)):
        model_dir = ensure_dir(output_dir / "models") / experiment_id
        model.encoder.save_pretrained(str(model_dir / "encoder"))
        tokenizer.save_pretrained(str(model_dir / "encoder"))
        torch.save(
            {
                "topic_head": model.topic_head.state_dict(),
                "sentiment_head": model.sentiment_head.state_dict(),
                "config": config,
                "topic_label_names": TOPIC_LABELS,
                "sentiment_label_names": SENTIMENT_LABELS,
            },
            model_dir / "heads.pt",
        )
        save_json(
            model_dir / "model_metadata.json",
            {
                "experiment_id": experiment_id,
                "encoder": encoder_name,
                "hf_id": spec.hf_id,
                "task": "both",
                "topic_variant": str(config.get("topic_variant", "target_context")),
                "sentiment_variant": str(config.get("sentiment_variant", "full")),
                "max_length": max_length,
                "segment": spec.segment,
            },
        )

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment_id": experiment_id,
        "encoder": encoder_name,
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "topic_test_macro_f1": topic_metrics["macro_f1"],
        "topic_test_accuracy": topic_metrics["accuracy"],
        "topic_test_macro_f1_ci": topic_metrics["bootstrap_macro_f1"],
        "sentiment_test_macro_f1": sentiment_metrics["macro_f1"],
        "sentiment_test_accuracy": sentiment_metrics["accuracy"],
        "sentiment_test_macro_f1_ci": sentiment_metrics["bootstrap_macro_f1"],
        "train_seconds": result["train_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="visobert")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda-topic", type=float, default=1.0)
    parser.add_argument("--lambda-sentiment", type=float, default=1.0)
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = {
        "max_length": 256,
        "num_train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "lambda_topic": args.lambda_topic,
        "lambda_sentiment": args.lambda_sentiment,
        "uncertainty_weighting": args.uncertainty,
        "save_model": args.save_model,
        "fp16": torch.cuda.is_available(),
        "topic_variant": "target_context",
        "sentiment_variant": "full",
        "early_stopping_patience": 3,
        "smoke_train_size": 64,
    }
    result = run_mtl(
        encoder_name=args.encoder,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        config=config,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
