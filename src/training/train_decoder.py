"""Fine-tune an open-weight decoder with LoRA/QLoRA for both tasks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME, TOPIC_IDS
from src.evaluation.bootstrap import bootstrap_macro_f1
from src.models.decoder import (
    DecoderCollator,
    DecoderDataset,
    DecoderExample,
    build_chat_prompt,
    generate_predictions,
    label_to_answer,
)
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


TOPIC_LABELS = [f"{tid}:{TOPIC_ID_TO_NAME[tid]}" for tid in TOPIC_IDS]
SENTIMENT_LABELS = [SENTIMENT_ID_TO_NAME[i] for i in range(3)]


def _autocast(enabled: bool):
    if not enabled:
        import contextlib

        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_decoder_model(
    model_name: str,
    *,
    load_in_4bit: bool,
    device: torch.device,
):
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as causal_error:
        print(f"AutoModelForCausalLM failed ({causal_error}); trying image-text model")
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_lora(model, *, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_examples(
    frame: pd.DataFrame,
    *,
    task: str,
    tokenizer,
    split: str,
) -> list[DecoderExample]:
    examples: list[DecoderExample] = []
    label_column = "sentiment_id" if task == "sentiment" else "topic_index"
    for row in frame[frame["split"] == split].itertuples(index=False):
        label = int(getattr(row, label_column))
        prompt = build_chat_prompt(
            tokenizer,
            task=task,
            target=str(getattr(row, "target")),
            context=str(getattr(row, "context", "") or ""),
            brand=str(getattr(row, "brand", "")) if task == "sentiment" else None,
        )
        answer = label_to_answer(task, label)
        meta = {
            "comment_id": str(getattr(row, "comment_id")),
            "brand": str(getattr(row, "brand", "")) if task == "sentiment" else "",
            "topic_id": str(getattr(row, "topic_id", "")),
            "y_true": label,
            "split": split,
        }
        examples.append(DecoderExample(task=task, prompt=prompt, answer=answer, meta=meta))
    return examples


def _fallback_label(examples: list[DecoderExample]) -> int:
    labels = np.array([int(example.meta["y_true"]) for example in examples], dtype=int)
    if len(labels) == 0:
        return 0
    return int(np.bincount(labels).argmax())


def predictions_to_metrics(
    predictions: list[dict[str, Any]],
    *,
    task: str,
    fallback: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.DataFrame(predictions)
    y_true = frame["y_true"].astype(int).to_numpy()
    parsed = frame["parsed_label"].tolist()
    y_pred = np.array([fallback if value is None else int(value) for value in parsed], dtype=int)
    names = SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS
    metrics = compute_metrics(y_true, y_pred, label_names=names)
    metrics["invalid_rate"] = float(frame["invalid"].mean()) if len(frame) else 0.0
    metrics["fallback_label"] = int(fallback)
    metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        y_true, y_pred, labels=list(range(len(names))), seed=seed
    )
    frame["y_pred"] = y_pred
    frame["confidence"] = 1.0
    return metrics, frame


def train_decoder(
    *,
    model_name: str,
    processed_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    topics = pd.read_csv(processed_dir / "comments.csv")
    sentiment = pd.read_csv(processed_dir / "comment_brands.csv")
    for column in ["target", "context", "brands"]:
        if column in topics.columns:
            topics[column] = topics[column].fillna("").astype(str)
    for column in ["target", "context", "brand"]:
        if column in sentiment.columns:
            sentiment[column] = sentiment[column].fillna("").astype(str)

    tokenizer = load_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_decoder_model(model_name, load_in_4bit=bool(config.get("load_in_4bit", False)), device=device)

    if bool(config.get("load_in_4bit", False)):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    model = apply_lora(
        model,
        r=int(config.get("lora_r", 16)),
        alpha=int(config.get("lora_alpha", 32)),
        dropout=float(config.get("lora_dropout", 0.05)),
    )
    if not bool(config.get("load_in_4bit", False)):
        model.to(device)

    topic_train = build_examples(topics, task="topic", tokenizer=tokenizer, split="train")
    topic_val = build_examples(topics, task="topic", tokenizer=tokenizer, split="val")
    topic_test = build_examples(topics, task="topic", tokenizer=tokenizer, split="test")
    sentiment_train = build_examples(sentiment, task="sentiment", tokenizer=tokenizer, split="train")
    sentiment_val = build_examples(sentiment, task="sentiment", tokenizer=tokenizer, split="val")
    sentiment_test = build_examples(sentiment, task="sentiment", tokenizer=tokenizer, split="test")

    if smoke:
        topic_train = topic_train[:16]
        sentiment_train = sentiment_train[:16]
        topic_val = topic_val[:8]
        sentiment_val = sentiment_val[:8]
        topic_test = topic_test[:8]
        sentiment_test = sentiment_test[:8]

    train_examples = topic_train + sentiment_train
    random.Random(seed).shuffle(train_examples)
    train_dataset = DecoderDataset(
        train_examples, tokenizer, max_length=int(config.get("max_length", 512))
    )
    collator = DecoderCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        if bool(config.get("load_in_4bit", False)):
            from bitsandbytes.optim import PagedAdamW8bit

            optimizer = PagedAdamW8bit(trainable, lr=float(config.get("learning_rate", 2e-4)))
        else:
            optimizer = AdamW(trainable, lr=float(config.get("learning_rate", 2e-4)))
    except Exception:
        optimizer = AdamW(trainable, lr=float(config.get("learning_rate", 2e-4)))

    grad_accum = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("num_train_epochs", 3))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(float(config.get("warmup_ratio", 0.05)) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = _grad_scaler(bool(config.get("fp16", torch.cuda.is_available())))

    experiment_id = f"decoder_{_safe_name(model_name)}_seed{seed}"
    best_dir = output_dir / "models" / experiment_id / "best_adapter"
    ensure_dir(best_dir.parent)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = -1
    patience = int(config.get("early_stopping_patience", 2))
    epochs_without_improvement = 0
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(bool(config.get("fp16", torch.cuda.is_available()))):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(config.get("max_grad_norm", 0.3)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                running_steps += 1
            running_loss += float(loss.detach().cpu()) * grad_accum
            if step % int(config.get("logging_steps", 10)) == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / step:.4f}")

        topic_val_predictions = generate_predictions(
            model,
            tokenizer,
            topic_val,
            batch_size=int(config.get("eval_batch_size", 8)),
            max_new_tokens=int(config.get("max_new_tokens", 8)),
            max_length=int(config.get("max_length", 512)),
        )
        sentiment_val_predictions = generate_predictions(
            model,
            tokenizer,
            sentiment_val,
            batch_size=int(config.get("eval_batch_size", 8)),
            max_new_tokens=int(config.get("max_new_tokens", 8)),
            max_length=int(config.get("max_length", 512)),
        )
        topic_fallback = _fallback_label(topic_train)
        sentiment_fallback = _fallback_label(sentiment_train)
        topic_val_metrics, _ = predictions_to_metrics(
            topic_val_predictions, task="topic", fallback=topic_fallback, seed=seed
        )
        sentiment_val_metrics, _ = predictions_to_metrics(
            sentiment_val_predictions, task="sentiment", fallback=sentiment_fallback, seed=seed
        )
        score = 0.5 * (topic_val_metrics["macro_f1"] + sentiment_val_metrics["macro_f1"])
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "optimizer_steps": running_steps,
            "val_topic_macro_f1": topic_val_metrics["macro_f1"],
            "val_sentiment_macro_f1": sentiment_val_metrics["macro_f1"],
            "val_avg_macro_f1": score,
            "val_topic_invalid_rate": topic_val_metrics["invalid_rate"],
            "val_sentiment_invalid_rate": sentiment_val_metrics["invalid_rate"],
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_seconds = time.perf_counter() - train_start
    if best_dir.exists():
        try:
            model.load_adapter(str(best_dir), adapter_name="best")
            model.set_adapter("best")
        except Exception as error:
            print(f"Could not reload best adapter: {error}")

    topic_test_predictions = generate_predictions(
        model,
        tokenizer,
        topic_test,
        batch_size=int(config.get("eval_batch_size", 8)),
        max_new_tokens=int(config.get("max_new_tokens", 8)),
        max_length=int(config.get("max_length", 512)),
    )
    sentiment_test_predictions = generate_predictions(
        model,
        tokenizer,
        sentiment_test,
        batch_size=int(config.get("eval_batch_size", 8)),
        max_new_tokens=int(config.get("max_new_tokens", 8)),
        max_length=int(config.get("max_length", 512)),
    )
    topic_val_predictions = generate_predictions(
        model,
        tokenizer,
        topic_val,
        batch_size=int(config.get("eval_batch_size", 8)),
        max_new_tokens=int(config.get("max_new_tokens", 8)),
        max_length=int(config.get("max_length", 512)),
    )
    sentiment_val_predictions = generate_predictions(
        model,
        tokenizer,
        sentiment_val,
        batch_size=int(config.get("eval_batch_size", 8)),
        max_new_tokens=int(config.get("max_new_tokens", 8)),
        max_length=int(config.get("max_length", 512)),
    )
    topic_fallback = _fallback_label(topic_train)
    sentiment_fallback = _fallback_label(sentiment_train)
    topic_metrics, topic_frame = predictions_to_metrics(
        topic_test_predictions, task="topic", fallback=topic_fallback, seed=seed
    )
    sentiment_metrics, sentiment_frame = predictions_to_metrics(
        sentiment_test_predictions, task="sentiment", fallback=sentiment_fallback, seed=seed
    )
    topic_val_metrics, topic_val_frame = predictions_to_metrics(
        topic_val_predictions, task="topic", fallback=topic_fallback, seed=seed
    )
    sentiment_val_metrics, sentiment_val_frame = predictions_to_metrics(
        sentiment_val_predictions, task="sentiment", fallback=sentiment_fallback, seed=seed
    )

    topic_frame["model"] = model_name
    topic_frame["seed"] = seed
    sentiment_frame["model"] = model_name
    sentiment_frame["seed"] = seed
    topic_val_frame["model"] = model_name
    topic_val_frame["seed"] = seed
    sentiment_val_frame["model"] = model_name
    sentiment_val_frame["seed"] = seed
    ensure_dir(output_dir)
    topic_frame.to_csv(output_dir / f"{experiment_id}_topic_predictions.csv", index=False, encoding="utf-8")
    sentiment_frame.to_csv(output_dir / f"{experiment_id}_sentiment_predictions.csv", index=False, encoding="utf-8")
    topic_val_frame.to_csv(
        output_dir / f"{experiment_id}_topic_val_predictions.csv", index=False, encoding="utf-8"
    )
    sentiment_val_frame.to_csv(
        output_dir / f"{experiment_id}_sentiment_val_predictions.csv", index=False, encoding="utf-8"
    )
    save_json(
        output_dir / f"{experiment_id}_metrics.json",
        {
            "experiment_id": experiment_id,
            "model": "decoder",
            "model_name": model_name,
            "seed": seed,
            "config": config,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "train_seconds": train_seconds,
            "history": history,
            "topic_val": topic_val_metrics,
            "sentiment_val": sentiment_val_metrics,
            "topic_test": topic_metrics,
            "sentiment_test": sentiment_metrics,
            "topic_label_names": TOPIC_LABELS,
            "sentiment_label_names": SENTIMENT_LABELS,
            "n_topic_train": int(len(topic_train)),
            "n_sentiment_train": int(len(sentiment_train)),
        },
    )

    if bool(config.get("save_model", False)):
        final_dir = output_dir / "models" / experiment_id / "final_adapter"
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        save_json(
            final_dir / "model_metadata.json",
            {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "task": "both",
                "max_length": int(config.get("max_length", 512)),
                "load_in_4bit": bool(config.get("load_in_4bit", False)),
                "best_dir": str(best_dir),
            },
        )

    del model, tokenizer, train_loader, train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "topic_test_macro_f1": topic_metrics["macro_f1"],
        "topic_test_accuracy": topic_metrics["accuracy"],
        "topic_test_macro_f1_ci": topic_metrics["bootstrap_macro_f1"],
        "topic_invalid_rate": topic_metrics["invalid_rate"],
        "sentiment_test_macro_f1": sentiment_metrics["macro_f1"],
        "sentiment_test_accuracy": sentiment_metrics["accuracy"],
        "sentiment_test_macro_f1_ci": sentiment_metrics["bootstrap_macro_f1"],
        "sentiment_invalid_rate": sentiment_metrics["invalid_rate"],
        "train_seconds": train_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = {
        "load_in_4bit": args.load_in_4bit,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "max_length": args.max_length,
        "max_new_tokens": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.05,
        "max_grad_norm": 0.3,
        "logging_steps": 10,
        "save_model": args.save_model,
        "fp16": torch.cuda.is_available(),
        "early_stopping_patience": 2,
    }
    result = train_decoder(
        model_name=args.model,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        config=config,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
