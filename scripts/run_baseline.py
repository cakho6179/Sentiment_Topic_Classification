"""Run the TF-IDF + Linear SVM baseline suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.train_tfidf import SENTIMENT_VARIANTS, TOPIC_VARIANTS, run_experiment
from src.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    reports_dir = ensure_dir(Path(args.reports_dir))
    ensure_dir(output_dir)

    config: dict[str, Any] = {
        "C_grid": [0.1, 0.3, 1.0, 3.0, 10.0],
        "class_weight_grid": [None, "balanced"],
        "word_ngrams": [1, 2],
        "char_ngrams": [3, 5],
        "max_features_word": 50000,
        "max_features_char": 100000,
    }

    rows: list[dict[str, Any]] = []
    for task, variants in [("sentiment", SENTIMENT_VARIANTS), ("topic", TOPIC_VARIANTS)]:
        for variant in variants:
            for seed in args.seeds:
                experiment_id = f"tfidf_{task}_{variant}_seed{seed}"
                metrics_path = output_dir / f"{experiment_id}_metrics.json"
                if metrics_path.exists() and not args.force:
                    print(f"skip existing {experiment_id}")
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "task": task,
                            "model": "tfidf",
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
                    processed_dir=processed_dir,
                    output_dir=output_dir,
                    config=config,
                    seed=seed,
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "task": task,
                        "model": "tfidf",
                        "variant": variant,
                        "seed": seed,
                        "test_macro_f1": result["test_macro_f1"],
                        "test_accuracy": result["test_accuracy"],
                    }
                )

    frame = pd.DataFrame(rows).sort_values(["task", "model", "variant", "seed"])
    frame.to_csv(output_dir / "tfidf_summary.csv", index=False, encoding="utf-8")
    save_json(output_dir / "tfidf_summary.json", rows)
    frame.to_markdown(reports_dir / "tfidf_summary.md", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
