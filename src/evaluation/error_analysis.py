"""Turn prediction tables into a compact error-analysis artifact."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_error_table(
    frame: pd.DataFrame,
    *,
    id_columns: list[str],
    true_column: str,
    pred_column: str,
    confidence_column: str | None = None,
    text_column: str | None = None,
    max_rows: int = 200,
) -> pd.DataFrame:
    errors = frame[frame[true_column] != frame[pred_column]].copy()
    if confidence_column and confidence_column in errors.columns:
        errors = errors.sort_values(confidence_column, ascending=False)
    columns = [c for c in id_columns if c in errors.columns]
    columns += [true_column, pred_column]
    if confidence_column and confidence_column in errors.columns:
        columns.append(confidence_column)
    if text_column and text_column in errors.columns:
        columns.append(text_column)
    return errors[columns].head(max_rows).reset_index(drop=True)


def summarize_errors(
    frame: pd.DataFrame,
    *,
    true_column: str,
    pred_column: str,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_column in group_columns:
        if group_column not in frame.columns:
            continue
        grouped = frame.groupby(group_column)
        for value, group in grouped:
            total = len(group)
            correct = int((group[true_column] == group[pred_column]).sum())
            rows.append(
                {
                    "group_column": group_column,
                    "group": str(value),
                    "n": int(total),
                    "correct": correct,
                    "accuracy": float(correct / total) if total else 0.0,
                    "errors": int(total - correct),
                }
            )
    return rows
