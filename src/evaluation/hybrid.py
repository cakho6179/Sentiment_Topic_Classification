"""Hybrid verifier: combine a primary classifier with a second model."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.training.metrics import macro_f1


def merge_predictions(
    primary: pd.DataFrame,
    verifier: pd.DataFrame,
    *,
    task: str,
) -> pd.DataFrame:
    keys = ["comment_id", "brand"] if task == "sentiment" else ["comment_id"]
    primary_columns = keys + ["y_true", "y_pred", "confidence"]
    verifier_columns = keys + ["y_pred", "confidence"]
    left = primary[primary_columns].rename(
        columns={"y_pred": "primary_pred", "confidence": "primary_confidence"}
    )
    right = verifier[verifier_columns].rename(
        columns={"y_pred": "verifier_pred", "confidence": "verifier_confidence"}
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    return merged


def apply_hybrid(
    merged: pd.DataFrame,
    *,
    threshold: float,
    margin: float,
) -> np.ndarray:
    primary = merged["primary_pred"].to_numpy()
    verifier = merged["verifier_pred"].to_numpy()
    primary_conf = merged["primary_confidence"].to_numpy()
    verifier_conf = merged["verifier_confidence"].to_numpy()
    use_verifier = (primary_conf < threshold) & (verifier_conf >= primary_conf + margin)
    return np.where(use_verifier, verifier, primary)


def tune_hybrid(
    primary_val: pd.DataFrame,
    verifier_val: pd.DataFrame,
    *,
    task: str,
    thresholds: Sequence[float] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    margins: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3),
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    merged = merge_predictions(primary_val, verifier_val, task=task)
    y_true = merged["y_true"].to_numpy()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        for margin in margins:
            pred = apply_hybrid(merged, threshold=threshold, margin=margin)
            score = macro_f1(y_true, pred, labels=labels)
            intervention = float((pred != merged["primary_pred"].to_numpy()).mean())
            row = {
                "threshold": float(threshold),
                "margin": float(margin),
                "val_macro_f1": float(score),
                "intervention_rate": intervention,
            }
            rows.append(row)
            key = (row["val_macro_f1"], -row["intervention_rate"])
            if best is None or key > best["key"]:
                best = {**row, "key": key}
    if best is None:
        raise RuntimeError("No hybrid candidate")
    return {"best": best, "grid": rows, "merged_val": merged}
