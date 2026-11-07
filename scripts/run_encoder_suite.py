"""Run encoder fine-tuning experiments with resumable output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.training.train_encoder import (
    SENTIMENT_VARIANTS,
    TOPIC_VARIANTS,
    run_experiment,
)
from src.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=["visobert", "phobert"])
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "topic"])
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    config: dict[str, Any] = {
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

    rows: list[dict[str, Any]] = []
    summary_path = output_dir / "encoder_summary.jsonl"
    for encoder in args.encoders:
        for task in args.tasks:
            default_variants = SENTIMENT_VARIANTS if task == "sentiment" else TOPIC_VARIANTS
            if args.variants:
                variants = [variant for variant in args.variants if variant in default_variants]
            else:
                variants = default_variants
            for variant in variants:
                for seed in args.seeds:
                    experiment_id = f"{encoder}_{task}_{variant}_seed{seed}"
                    metrics_path = output_dir / f"{experiment_id}_metrics.json"
                    if metrics_path.exists() and not args.force:
                        print(f"skip existing {experiment_id}")
                        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                        rows.append(
                            {
                                "experiment_id": experiment_id,
                                "model": "encoder",
                                "encoder": encoder,
                                "task": task,
                                "variant": variant,
                                "seed": seed,
                                "test_macro_f1": payload["test"]["macro_f1"],
                                "test_accuracy": payload["test"]["accuracy"],
                            }
                        )
                        continue
                    print(f"running {experiment_id}")
                    result = run_experiment(
                        task=task,
                        variant=variant,
                        encoder_name=encoder,
                        processed_dir=processed_dir,
                        output_dir=output_dir,
                        config=config,
                        seed=seed,
                        smoke=args.smoke,
                    )
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "model": "encoder",
                            "encoder": encoder,
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "test_macro_f1": result["test_macro_f1"],
                            "test_accuracy": result["test_accuracy"],
                        }
                    )
                    with summary_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    frame = pd.DataFrame(rows).sort_values(["task", "encoder", "variant", "seed"])
    frame.to_csv(output_dir / "encoder_summary.csv", index=False, encoding="utf-8")
    save_json(output_dir / "encoder_summary.json", rows)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
