"""Fine-tune encoder models for sentiment and topic classification."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME, TOPIC_IDS
from src.evaluation.bootstrap import bootstrap_macro_f1
from src.models.encoders import (
    ENCODER_SPECS,
    EncoderDataset,
    build_training_arguments,
    load_sequence_classifier,
    load_tokenizer,
    make_trainer,
    resolve_spec,
)
from src.models.tfidf import build_variant
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


SENTIMENT_VARIANTS = [
    "full",
    "brand_target",
    "target_context",
    "target_only",
    "brand_only",
    "context_only",
    "full_masked_brand",
    "full_target_marked",
]
TOPIC_VARIANTS = [
    "target_context",
    "target_only",
    "context_only",
    "target_context_marked",
    "target_context_masked_brand",
    "target_masked_brand",
    "context_masked_brand",
]


def load_data(task: str, processed_dir: Path) -> pd.DataFrame:
    if task == "sentiment":
        frame = pd.read_csv(processed_dir / "comment_brands.csv")
        required = {"comment_id", "brand", "sentiment_id", "target", "context", "split", "topic_id"}
    elif task == "topic":
        frame = pd.read_csv(processed_dir / "comments.csv")
        required = {"comment_id", "topic_id", "topic_index", "target", "context", "brands", "split"}
    else:
        raise ValueError(task)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for {task}: {sorted(missing)}")
    return frame


def target_column(task: str) -> str:
    return "sentiment_id" if task == "sentiment" else "topic_index"


def label_names(task: str) -> list[str]:
    if task == "sentiment":
        return [SENTIMENT_ID_TO_NAME[i] for i in range(3)]
    return [f"{tid}:{TOPIC_ID_TO_NAME[tid]}" for tid in TOPIC_IDS]


def id_columns(task: str) -> list[str]:
    return ["comment_id", "brand"] if task == "sentiment" else ["comment_id"]


def compute_class_weights(y: np.ndarray, num_labels: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(y) / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_experiment(
    *,
    task: str,
    variant: str,
    encoder_name: str,
    processed_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    frame = load_data(task, processed_dir)
    y_col = target_column(task)
    frame = frame.dropna(subset=[y_col]).reset_index(drop=True)
    frame[y_col] = frame[y_col].astype(int)
    for column in ["target", "context", "brand", "brands"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)

    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if smoke:
        train = train.head(int(config.get("smoke_train_size", 64)))
        val = val.head(int(config.get("smoke_eval_size", 32)))
        test = test.head(int(config.get("smoke_eval_size", 32)))
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("Empty split")

    train_text = build_variant(train, task, variant).fillna("").astype(str).tolist()
    val_text = build_variant(val, task, variant).fillna("").astype(str).tolist()
    test_text = build_variant(test, task, variant).fillna("").astype(str).tolist()

    y_train = train[y_col].to_numpy()
    y_val = val[y_col].to_numpy()
    y_test = test[y_col].to_numpy()

    spec = resolve_spec(encoder_name)
    tokenizer = load_tokenizer(spec)
    num_labels = len(label_names(task))
    model = load_sequence_classifier(spec, num_labels=num_labels)

    max_length = int(config.get("max_length", 256))
    train_dataset = EncoderDataset(
        train_text, y_train, tokenizer, max_length=max_length, segment=spec.segment
    )
    val_dataset = EncoderDataset(
        val_text, y_val, tokenizer, max_length=max_length, segment=spec.segment
    )
    test_dataset = EncoderDataset(
        test_text, y_test, tokenizer, max_length=max_length, segment=spec.segment
    )

    experiment_id = f"{encoder_name}_{task}_{variant}_seed{seed}"
    model_dir = ensure_dir(output_dir / "models") / experiment_id
    args = build_training_arguments(
        output_dir=model_dir,
        learning_rate=float(config.get("learning_rate", 2e-5)),
        num_train_epochs=int(config.get("num_train_epochs", 10)),
        per_device_train_batch_size=int(config.get("batch_size", 16)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 32)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        seed=seed,
        fp16=bool(config.get("fp16", torch.cuda.is_available())),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        logging_steps=int(config.get("logging_steps", 20)),
    )

    class_weights = None
    if bool(config.get("class_weights", False)):
        class_weights = compute_class_weights(y_train, num_labels)

    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
        label_names=label_names(task),
        class_weights=class_weights,
    )

    train_start = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_start

    val_predictions = trainer.predict(val_dataset)
    test_predictions = trainer.predict(test_dataset)

    val_logits = val_predictions.predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    test_logits = test_predictions.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]

    val_prob = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    test_prob = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    val_pred = val_prob.argmax(axis=-1)
    test_pred = test_prob.argmax(axis=-1)

    names = label_names(task)
    val_metrics = compute_metrics(y_val, val_pred, y_prob=val_prob, label_names=names)
    test_metrics = compute_metrics(y_test, test_pred, y_prob=test_prob, label_names=names)
    test_metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        y_test, test_pred, labels=list(range(num_labels)), seed=seed
    )

    predictions = test[id_columns(task)].copy()
    predictions["split"] = "test"
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    predictions["confidence"] = test_prob.max(axis=1)
    predictions["topic_id"] = test["topic_id"].to_numpy()
    if task == "topic":
        predictions["brands"] = test["brands"].to_numpy()
    predictions["task"] = task
    predictions["variant"] = variant
    predictions["model"] = encoder_name
    predictions["seed"] = seed
    for index, name in enumerate(names):
        predictions[f"prob_{index}"] = test_prob[:, index]

    val_frame = val[id_columns(task)].copy()
    val_frame["split"] = "val"
    val_frame["y_true"] = y_val
    val_frame["y_pred"] = val_pred
    val_frame["confidence"] = val_prob.max(axis=1)
    val_frame["topic_id"] = val["topic_id"].to_numpy()
    if task == "topic":
        val_frame["brands"] = val["brands"].to_numpy()
    val_frame["task"] = task
    val_frame["variant"] = variant
    val_frame["model"] = encoder_name
    val_frame["seed"] = seed
    for index, name in enumerate(names):
        val_frame[f"prob_{index}"] = val_prob[:, index]

    ensure_dir(output_dir)
    predictions.to_csv(output_dir / f"{experiment_id}_predictions.csv", index=False, encoding="utf-8")
    val_frame.to_csv(output_dir / f"{experiment_id}_val_predictions.csv", index=False, encoding="utf-8")
    save_json(
        output_dir / f"{experiment_id}_metrics.json",
        {
            "experiment_id": experiment_id,
            "task": task,
            "variant": variant,
            "encoder": encoder_name,
            "hf_id": spec.hf_id,
            "seed": seed,
            "model": "encoder",
            "config": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "train_seconds": float(train_seconds),
            "train_loss": float(train_result.training_loss),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
            "val": val_metrics,
            "test": test_metrics,
            "label_names": names,
            "log_history": trainer.state.log_history,
        },
    )

    if bool(config.get("save_model", False)):
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        save_json(
            model_dir / "model_metadata.json",
            {
                "experiment_id": experiment_id,
                "task": task,
                "variant": variant,
                "encoder": encoder_name,
                "hf_id": spec.hf_id,
                "label_names": names,
                "max_length": max_length,
                "segment": spec.segment,
            },
        )

    result = {
        "experiment_id": experiment_id,
        "task": task,
        "variant": variant,
        "encoder": encoder_name,
        "seed": seed,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1_ci": test_metrics["bootstrap_macro_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "train_seconds": float(train_seconds),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    del trainer, model, tokenizer, train_dataset, val_dataset, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "topic"], required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="visobert")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = {
        "max_length": args.max_length,
        "num_train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "class_weights": args.class_weights,
        "save_model": args.save_model,
        "fp16": torch.cuda.is_available(),
        "smoke_train_size": 64,
        "smoke_eval_size": 32,
    }
    result = run_experiment(
        task=args.task,
        variant=args.variant,
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
