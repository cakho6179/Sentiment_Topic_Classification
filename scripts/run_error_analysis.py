"""Error analysis for the best sentiment and topic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME, TOPIC_IDS
from src.evaluation.error_analysis import build_error_table, summarize_errors
from src.utils.io import ensure_dir, save_json


def _best_row(benchmark: pd.DataFrame, task: str, allow_perfect: bool = False) -> pd.Series:
    subset = benchmark[benchmark["task"] == task].dropna(subset=["test_macro_f1"])
    if subset.empty:
        raise SystemExit(f"No benchmark row for {task}")
    if not allow_perfect:
        non_perfect = subset[subset["test_macro_f1"] < 0.999]
        if not non_perfect.empty:
            subset = non_perfect
    return subset.sort_values(["test_macro_f1", "val_macro_f1"], ascending=False).iloc[0]


def _prediction_path(experiments_dir: Path, row: pd.Series, task: str) -> Path:
    experiment_id = str(row["experiment_id"])
    model = str(row["model"])
    if model in {"mtl", "decoder"}:
        path = experiments_dir / f"{experiment_id}_{task}_predictions.csv"
    else:
        path = experiments_dir / f"{experiment_id}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _label_names(task: str) -> list[str]:
    if task == "sentiment":
        return [SENTIMENT_ID_TO_NAME[i] for i in range(3)]
    return [f"{tid}:{TOPIC_ID_TO_NAME[tid]}" for tid in TOPIC_IDS]


def _enrich(frame: pd.DataFrame, task: str, processed_dir: Path) -> pd.DataFrame:
    comments = pd.read_csv(processed_dir / "comments.csv")
    pairs = pd.read_csv(processed_dir / "comment_brands.csv")
    for column in ["target", "context", "brands", "brand"]:
        if column in comments.columns:
            comments[column] = comments[column].fillna("").astype(str)
        if column in pairs.columns:
            pairs[column] = pairs[column].fillna("").astype(str)
    if task == "sentiment":
        merge_columns = [
            "comment_id",
            "brand",
            "target",
            "context",
            "target_type",
            "channel",
            "brand_in_text",
            "target_len",
            "context_len",
        ]
        merge_columns = [column for column in merge_columns if column in pairs.columns]
        frame = frame.merge(
            pairs[merge_columns].drop_duplicates(["comment_id", "brand"]),
            on=["comment_id", "brand"],
            how="left",
            suffixes=("", "_data"),
        )
    else:
        merge_columns = [
            "comment_id",
            "target",
            "context",
            "target_type",
            "channel",
            "brands",
            "n_brands",
            "brand_in_text",
            "target_len",
            "context_len",
        ]
        merge_columns = [column for column in merge_columns if column in comments.columns]
        frame = frame.merge(
            comments[merge_columns].drop_duplicates("comment_id"),
            on="comment_id",
            how="left",
            suffixes=("", "_data"),
        )
    if "topic_id" not in frame.columns:
        frame["topic_id"] = "unknown"
    if "context" in frame.columns:
        frame["context_present"] = frame["context"].astype(str).str.strip() != ""
    else:
        frame["context_present"] = False
    if "n_brands" in frame.columns:
        frame["multi_brand"] = frame["n_brands"].astype(float).fillna(1) > 1
    elif "brands" in frame.columns:
        frame["multi_brand"] = frame["brands"].astype(str).str.contains("\\|", regex=True)
    else:
        frame["multi_brand"] = False
    if "brand_in_text" not in frame.columns:
        frame["brand_in_text"] = False
    if "confidence" in frame.columns:
        frame["confidence_bin"] = pd.cut(
            frame["confidence"].astype(float),
            bins=[-0.01, 0.4, 0.6, 0.8, 1.01],
            labels=["<0.4", "0.4-0.6", "0.6-0.8", ">=0.8"],
        ).astype(str)
    return frame


def _confusion_table(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    labels = list(range(3 if task == "sentiment" else 13))
    names = _label_names(task)
    matrix = confusion_matrix(frame["y_true"].astype(int), frame["y_pred"].astype(int), labels=labels)
    return pd.DataFrame(matrix, index=[f"true_{name}" for name in names], columns=[f"pred_{name}" for name in names])


def _markdown(frame: pd.DataFrame) -> str:
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
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--benchmark", type=str, default="reports/benchmark.csv")
    parser.add_argument("--sentiment-predictions", type=str, default=None)
    parser.add_argument("--topic-predictions", type=str, default=None)
    parser.add_argument("--allow-perfect", action="store_true")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    processed_dir = Path(args.processed_dir)
    reports_dir = ensure_dir(Path(args.reports_dir))
    benchmark = pd.read_csv(args.benchmark)

    selected: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for task, explicit in [
        ("sentiment", args.sentiment_predictions),
        ("topic", args.topic_predictions),
    ]:
        row = _best_row(benchmark, task, allow_perfect=args.allow_perfect)
        path = Path(explicit) if explicit else _prediction_path(experiments_dir, row, task)
        frame = pd.read_csv(path)
        frame = _enrich(frame, task, processed_dir)
        frames[task] = frame
        selected[task] = {
            "experiment_id": str(row["experiment_id"]),
            "model": str(row["model"]),
            "encoder": None if pd.isna(row.get("encoder")) else str(row.get("encoder")),
            "variant": None if pd.isna(row.get("variant")) else str(row.get("variant")),
            "test_macro_f1": float(row["test_macro_f1"]),
            "predictions_path": str(path),
        }

    sections = ["# Phân tích lỗi", ""]
    sections.append("## Model được chọn")
    sections.append("")
    sections.append(_markdown(pd.DataFrame(selected).T.reset_index().rename(columns={"index": "task"})))
    sections.append("")

    rows: list[dict[str, Any]] = []
    for task, frame in frames.items():
        sections.append(f"## {task.capitalize()}")
        sections.append("")
        accuracy = float((frame["y_true"] == frame["y_pred"]).mean())
        sections.append(f"- Accuracy: **{accuracy:.4f}**")
        sections.append(f"- Số mẫu: **{len(frame)}**")
        if "confidence" in frame.columns:
            sections.append(f"- Confidence trung bình: **{frame['confidence'].mean():.4f}**")
        sections.append("")
        sections.append("### Confusion matrix")
        sections.append("")
        sections.append(_markdown(_confusion_table(frame, task)))
        sections.append("")
        summaries = summarize_errors(
            frame,
            true_column="y_true",
            pred_column="y_pred",
            group_columns=["brand", "topic_id", "context_present", "multi_brand", "brand_in_text", "confidence_bin"],
        )
        for item in summaries:
            item["task"] = task
        rows.extend(summaries)
        summary_frame = pd.DataFrame(summaries)
        if not summary_frame.empty:
            sections.append("### Nhóm lỗi")
            sections.append("")
            sections.append(_markdown(summary_frame))
            sections.append("")
        errors = build_error_table(
            frame,
            id_columns=["comment_id", "brand", "topic_id"],
            true_column="y_true",
            pred_column="y_pred",
            confidence_column="confidence",
            text_column="target",
            max_rows=100,
        )
        errors.to_csv(reports_dir / f"error_examples_{task}.csv", index=False, encoding="utf-8")
        sections.append(f"Đã lưu tối đa 100 ví dụ lỗi vào `reports/error_examples_{task}.csv`.")
        sections.append("")

    pd.DataFrame(rows).to_csv(reports_dir / "error_summary.csv", index=False, encoding="utf-8")
    save_json(reports_dir / "error_analysis.json", {"selected": selected, "groups": rows})
    (reports_dir / "error_analysis.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections[:80]))
    print(f"\nWrote {reports_dir / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
