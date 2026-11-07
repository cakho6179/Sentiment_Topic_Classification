"""Tune and evaluate a primary + verifier hybrid on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME, TOPIC_IDS
from src.evaluation.bootstrap import bootstrap_macro_f1, mcnemar_test
from src.evaluation.hybrid import apply_hybrid, merge_predictions, tune_hybrid
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json


def _label_names(task: str) -> list[str]:
    if task == "sentiment":
        return [SENTIMENT_ID_TO_NAME[i] for i in range(3)]
    return [f"{tid}:{TOPIC_ID_TO_NAME[tid]}" for tid in TOPIC_IDS]


def _find_split_file(experiments_dir: Path, experiment_id: str, task: str, split: str) -> Path:
    candidates = [
        experiments_dir / f"{experiment_id}_{task}_{split}_predictions.csv",
        experiments_dir / f"{experiment_id}_{split}_predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {split} predictions for {experiment_id}: {candidates}")


def _evaluate(frame: pd.DataFrame, pred: np.ndarray, task: str, seed: int) -> dict[str, Any]:
    y_true = frame["y_true"].astype(int).to_numpy()
    names = _label_names(task)
    metrics = compute_metrics(y_true, pred, label_names=names)
    metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        y_true, pred, labels=list(range(len(names))), seed=seed
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "topic"], required=True)
    parser.add_argument("--primary", required=True, help="Primary experiment id")
    parser.add_argument("--verifier", required=True, help="Verifier experiment id")
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    reports_dir = ensure_dir(Path(args.reports_dir))
    primary_test = pd.read_csv(_find_split_file(experiments_dir, args.primary, args.task, "test"))
    primary_val = pd.read_csv(_find_split_file(experiments_dir, args.primary, args.task, "val"))
    verifier_test = pd.read_csv(_find_split_file(experiments_dir, args.verifier, args.task, "test"))
    verifier_val = pd.read_csv(_find_split_file(experiments_dir, args.verifier, args.task, "val"))

    tuned = tune_hybrid(primary_val, verifier_val, task=args.task, labels=list(range(len(_label_names(args.task)))))
    best = tuned["best"]
    merged_test = merge_predictions(primary_test, verifier_test, task=args.task)
    hybrid_pred = apply_hybrid(merged_test, threshold=best["threshold"], margin=best["margin"])
    primary_pred = merged_test["primary_pred"].to_numpy()
    verifier_pred = merged_test["verifier_pred"].to_numpy()

    primary_metrics = _evaluate(merged_test, primary_pred, args.task, args.seed)
    verifier_metrics = _evaluate(merged_test, verifier_pred, args.task, args.seed)
    hybrid_metrics = _evaluate(merged_test, hybrid_pred, args.task, args.seed)
    hybrid_metrics["intervention_rate"] = float((hybrid_pred != primary_pred).mean())
    hybrid_metrics["mcnemar_vs_primary"] = mcnemar_test(
        merged_test["y_true"].to_numpy(), primary_pred, hybrid_pred
    )
    hybrid_metrics["mcnemar_vs_verifier"] = mcnemar_test(
        merged_test["y_true"].to_numpy(), verifier_pred, hybrid_pred
    )

    output = merged_test.copy()
    output["hybrid_pred"] = hybrid_pred
    output.to_csv(reports_dir / f"hybrid_{args.task}_test_predictions.csv", index=False, encoding="utf-8")
    payload = {
        "task": args.task,
        "primary": args.primary,
        "verifier": args.verifier,
        "best": best,
        "primary_test": primary_metrics,
        "verifier_test": verifier_metrics,
        "hybrid_test": hybrid_metrics,
        "grid": tuned["grid"],
    }
    save_json(reports_dir / f"hybrid_{args.task}.json", payload)

    lines = [f"# Hybrid {args.task}", ""]
    lines.append(
        f"- Primary: `{args.primary}` (Macro-F1 {primary_metrics['macro_f1']:.4f})"
    )
    lines.append(
        f"- Verifier: `{args.verifier}` (Macro-F1 {verifier_metrics['macro_f1']:.4f})"
    )
    lines.append(
        f"- Hybrid: threshold={best['threshold']}, margin={best['margin']}, "
        f"val Macro-F1={best['val_macro_f1']:.4f}, intervention={best['intervention_rate']:.4f}"
    )
    lines.append(
        f"- Hybrid test Macro-F1: **{hybrid_metrics['macro_f1']:.4f}** "
        f"(CI {hybrid_metrics['bootstrap_macro_f1']['ci_lower']:.4f}–"
        f"{hybrid_metrics['bootstrap_macro_f1']['ci_upper']:.4f})"
    )
    lines.append("")
    lines.append("| Model | Macro-F1 | Accuracy | Weighted-F1 |")
    lines.append("|---|---|---|---|")
    for name, metrics in [
        ("primary", primary_metrics),
        ("verifier", verifier_metrics),
        ("hybrid", hybrid_metrics),
    ]:
        lines.append(
            f"| {name} | {metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} | "
            f"{metrics['weighted_f1']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"McNemar hybrid vs primary: p={hybrid_metrics['mcnemar_vs_primary']['p_value']:.4f}, "
        f"b={hybrid_metrics['mcnemar_vs_primary']['b']}, c={hybrid_metrics['mcnemar_vs_primary']['c']}"
    )
    (reports_dir / f"hybrid_{args.task}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
