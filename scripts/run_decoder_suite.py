"""Run decoder LoRA/QLoRA experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.training.train_decoder import train_decoder
from src.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3.5-4B"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    config: dict[str, Any] = {
        "load_in_4bit": args.load_in_4bit,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
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

    rows: list[dict[str, Any]] = []
    summary_path = output_dir / "decoder_summary.jsonl"
    for model_name in args.models:
        for seed in args.seeds:
            safe = model_name.replace("/", "_")
            experiment_id = f"decoder_{safe}_seed{seed}"
            metrics_path = output_dir / f"{experiment_id}_metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"skip existing {experiment_id}")
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model": "decoder",
                        "model_name": model_name,
                        "seed": seed,
                        "topic_test_macro_f1": payload["topic_test"]["macro_f1"],
                        "sentiment_test_macro_f1": payload["sentiment_test"]["macro_f1"],
                    }
                )
                continue
            print(f"running {experiment_id}")
            result = train_decoder(
                model_name=model_name,
                processed_dir=processed_dir,
                output_dir=output_dir,
                config=config,
                seed=seed,
                smoke=args.smoke,
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model": "decoder",
                    "model_name": model_name,
                    "seed": seed,
                    "topic_test_macro_f1": result["topic_test_macro_f1"],
                    "sentiment_test_macro_f1": result["sentiment_test_macro_f1"],
                }
            )
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    frame = pd.DataFrame(rows).sort_values(["model_name", "seed"])
    frame.to_csv(output_dir / "decoder_summary.csv", index=False, encoding="utf-8")
    save_json(output_dir / "decoder_summary.json", rows)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
