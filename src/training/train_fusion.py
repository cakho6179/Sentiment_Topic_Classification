"""Train the feature-fusion architecture (text encoder + brand embedding)."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import EarlyStoppingCallback, Trainer

from src.data.labels import SENTIMENT_ID_TO_NAME
from src.evaluation.bootstrap import bootstrap_macro_f1
from src.models.encoders import build_training_arguments, load_tokenizer, make_compute_metrics, resolve_spec
from src.models.fusion import BrandFusionClassifier, BrandFusionDataset, FusionCollator
from src.models.tfidf import build_variant
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


LABEL_NAMES = [SENTIMENT_ID_TO_NAME[i] for i in range(3)]


def run_fusion(
    *,
    encoder_name: str,
    text_variant: str,
    processed_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    frame = pd.read_csv(processed_dir / "comment_brands.csv")
    for column in ["target", "context", "brand"]:
        frame[column] = frame[column].fillna("").astype(str)
    frame["sentiment_id"] = frame["sentiment_id"].astype(int)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if smoke:
        train = train.head(int(config.get("smoke_train_size", 64)))
        val = val.head(int(config.get("smoke_eval_size", 32)))
        test = test.head(int(config.get("smoke_eval_size", 32)))

    brands = sorted(train["brand"].unique().tolist())
    brand_to_id = {brand: idx for idx, brand in enumerate(brands)}
    brand_to_id["[UNK]"] = len(brand_to_id)

    spec = resolve_spec(encoder_name)
    tokenizer = load_tokenizer(spec)
    max_length = int(config.get("max_length", 256))

    def texts(subset: pd.DataFrame) -> list[str]:
        return build_variant(subset, "sentiment", text_variant).fillna("").astype(str).tolist()

    train_dataset = BrandFusionDataset(
        texts(train), train["brand"].tolist(), train["sentiment_id"].tolist(), tokenizer, brand_to_id,
        max_length=max_length, segment=spec.segment,
    )
    val_dataset = BrandFusionDataset(
        texts(val), val["brand"].tolist(), val["sentiment_id"].tolist(), tokenizer, brand_to_id,
        max_length=max_length, segment=spec.segment,
    )
    test_dataset = BrandFusionDataset(
        texts(test), test["brand"].tolist(), test["sentiment_id"].tolist(), tokenizer, brand_to_id,
        max_length=max_length, segment=spec.segment,
    )

    experiment_id = f"fusion_{encoder_name}_{text_variant}_seed{seed}"
    model_dir = ensure_dir(output_dir / "models") / experiment_id
    model = BrandFusionClassifier(
        spec.hf_id,
        num_brands=len(brand_to_id),
        num_labels=3,
        dropout=float(config.get("dropout", 0.1)),
        pooling=str(config.get("pooling", "mean")),
    )
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
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": FusionCollator(tokenizer),
        "compute_metrics": make_compute_metrics(LABEL_NAMES),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    start = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - start

    val_predictions = trainer.predict(val_dataset)
    test_predictions = trainer.predict(test_dataset)
    val_logits = val_predictions.predictions
    test_logits = test_predictions.predictions
    val_prob = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    test_prob = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    y_val = val["sentiment_id"].to_numpy()
    y_test = test["sentiment_id"].to_numpy()
    val_pred = val_prob.argmax(axis=-1)
    test_pred = test_prob.argmax(axis=-1)
    val_metrics = compute_metrics(y_val, val_pred, y_prob=val_prob, label_names=LABEL_NAMES)
    test_metrics = compute_metrics(y_test, test_pred, y_prob=test_prob, label_names=LABEL_NAMES)
    test_metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        y_test, test_pred, labels=[0, 1, 2], seed=seed
    )

    predictions = test[["comment_id", "brand", "topic_id"]].copy()
    predictions["split"] = "test"
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    predictions["confidence"] = test_prob.max(axis=1)
    predictions["task"] = "sentiment"
    predictions["model"] = "fusion"
    predictions["seed"] = seed
    for index in range(3):
        predictions[f"prob_{index}"] = test_prob[:, index]

    val_predictions = val[["comment_id", "brand", "topic_id"]].copy()
    val_predictions["split"] = "val"
    val_predictions["y_true"] = y_val
    val_predictions["y_pred"] = val_pred
    val_predictions["confidence"] = val_prob.max(axis=1)
    val_predictions["task"] = "sentiment"
    val_predictions["model"] = "fusion"
    val_predictions["seed"] = seed
    for index in range(3):
        val_predictions[f"prob_{index}"] = val_prob[:, index]

    ensure_dir(output_dir)
    predictions.to_csv(output_dir / f"{experiment_id}_predictions.csv", index=False, encoding="utf-8")
    val_predictions.to_csv(
        output_dir / f"{experiment_id}_val_predictions.csv", index=False, encoding="utf-8"
    )
    save_json(
        output_dir / f"{experiment_id}_metrics.json",
        {
            "experiment_id": experiment_id,
            "model": "fusion",
            "encoder": encoder_name,
            "hf_id": spec.hf_id,
            "text_variant": text_variant,
            "seed": seed,
            "config": config,
            "train_seconds": train_seconds,
            "train_loss": float(train_result.training_loss),
            "val": val_metrics,
            "test": test_metrics,
            "label_names": LABEL_NAMES,
            "brand_to_id": brand_to_id,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
    )

    if bool(config.get("save_model", False)):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "encoder_name": encoder_name,
                "hf_id": spec.hf_id,
                "num_brands": len(brand_to_id),
                "num_labels": 3,
                "brand_to_id": brand_to_id,
                "text_variant": text_variant,
                "max_length": max_length,
                "segment": spec.segment,
                "dropout": float(config.get("dropout", 0.1)),
                "pooling": str(config.get("pooling", "mean")),
            },
            model_dir / "fusion_model.pt",
        )
        tokenizer.save_pretrained(str(model_dir))

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "experiment_id": experiment_id,
        "model": "fusion",
        "encoder": encoder_name,
        "seed": seed,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1_ci": test_metrics["bootstrap_macro_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "train_seconds": train_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="visobert")
    parser.add_argument("--text-variant", type=str, default="target_context")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = {
        "max_length": 256,
        "num_train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "save_model": args.save_model,
        "fp16": torch.cuda.is_available(),
        "smoke_train_size": 64,
        "smoke_eval_size": 32,
    }
    result = run_fusion(
        encoder_name=args.encoder,
        text_variant=args.text_variant,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        config=config,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
