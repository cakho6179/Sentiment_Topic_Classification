"""Aggregate every metrics JSON into a single benchmark table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, save_json


def _get(payload: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _row_from_payload(payload: dict[str, Any], metrics_path: Path) -> list[dict[str, Any]]:
    model = payload.get("model", "unknown")
    base = {
        "experiment_id": payload.get("experiment_id", metrics_path.stem),
        "model": model,
        "encoder": payload.get("encoder"),
        "model_name": payload.get("model_name"),
        "seed": payload.get("seed"),
        "n_train": payload.get("n_train") or payload.get("n_topic_train"),
        "n_test": payload.get("n_test"),
        "train_seconds": payload.get("train_seconds"),
        "metrics_file": metrics_path.name,
    }
    rows: list[dict[str, Any]] = []
    if model in {"tfidf", "tfidf_linear_svm", "encoder"} and payload.get("task"):
        rows.append(
            {
                **base,
                "task": payload["task"],
                "variant": payload.get("variant"),
                "test_macro_f1": _get(payload, "test", "macro_f1"),
                "test_accuracy": _get(payload, "test", "accuracy"),
                "test_weighted_f1": _get(payload, "test", "weighted_f1"),
                "val_macro_f1": _get(payload, "val", "macro_f1"),
                "val_accuracy": _get(payload, "val", "accuracy"),
                "ci_lower": _get(payload, "test", "bootstrap_macro_f1", "ci_lower"),
                "ci_upper": _get(payload, "test", "bootstrap_macro_f1", "ci_upper"),
                "invalid_rate": None,
            }
        )
    elif model == "fusion":
        rows.append(
            {
                **base,
                "task": "sentiment",
                "variant": payload.get("text_variant"),
                "test_macro_f1": _get(payload, "test", "macro_f1"),
                "test_accuracy": _get(payload, "test", "accuracy"),
                "test_weighted_f1": _get(payload, "test", "weighted_f1"),
                "val_macro_f1": _get(payload, "val", "macro_f1"),
                "val_accuracy": _get(payload, "val", "accuracy"),
                "ci_lower": _get(payload, "test", "bootstrap_macro_f1", "ci_lower"),
                "ci_upper": _get(payload, "test", "bootstrap_macro_f1", "ci_upper"),
                "invalid_rate": None,
            }
        )
    elif model in {"mtl", "decoder"}:
        for task in ["topic", "sentiment"]:
            rows.append(
                {
                    **base,
                    "task": task,
                    "variant": (
                        "target_context+full" if model == "mtl" else "prompt"
                    ),
                    "test_macro_f1": _get(payload, f"{task}_test", "macro_f1"),
                    "test_accuracy": _get(payload, f"{task}_test", "accuracy"),
                    "test_weighted_f1": _get(payload, f"{task}_test", "weighted_f1"),
                    "val_macro_f1": _get(payload, f"{task}_val", "macro_f1"),
                    "val_accuracy": _get(payload, f"{task}_val", "accuracy"),
                    "ci_lower": _get(payload, f"{task}_test", "bootstrap_macro_f1", "ci_lower"),
                    "ci_upper": _get(payload, f"{task}_test", "bootstrap_macro_f1", "ci_upper"),
                    "invalid_rate": _get(payload, f"{task}_test", "invalid_rate"),
                }
            )
    return rows


def to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    reports_dir = ensure_dir(Path(args.reports_dir))
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(experiments_dir.glob("*_metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"skip unreadable {metrics_path}: {error}")
            continue
        rows.extend(_row_from_payload(payload, metrics_path))

    if not rows:
        raise SystemExit("No metrics found")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["task", "test_macro_f1"], ascending=[True, False])
    frame.to_csv(reports_dir / "benchmark.csv", index=False, encoding="utf-8")
    save_json(reports_dir / "benchmark.json", rows)

    lines = ["# Benchmark tổng hợp", ""]
    lines.append(f"Số experiment: **{len(frame)}**.")
    lines.append("")
    for task in ["sentiment", "topic"]:
        subset = frame[frame["task"] == task].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("test_macro_f1", ascending=False)
        lines.append(f"## {task.capitalize()}")
        lines.append("")
        lines.append("### Top 15")
        lines.append("")
        columns = [
            "experiment_id",
            "model",
            "encoder",
            "model_name",
            "variant",
            "seed",
            "test_macro_f1",
            "test_accuracy",
            "val_macro_f1",
            "ci_lower",
            "ci_upper",
            "invalid_rate",
        ]
        lines.append(to_markdown(subset[columns].head(15)))
        lines.append("")
        lines.append("### Theo model")
        lines.append("")
        grouped = (
            subset.groupby("model", dropna=False)["test_macro_f1"]
            .agg(["count", "mean", "std", "max"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        lines.append(to_markdown(grouped))
        lines.append("")
    (reports_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nWrote {reports_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
