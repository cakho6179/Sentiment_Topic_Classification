"""Classification metrics used by every experiment."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray | None = None,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(label_names))) if label_names is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for idx, label in enumerate(labels):
        per_class.append(
            {
                "label": int(label),
                "name": label_names[idx] if label_names is not None and idx < len(label_names) else str(label),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [int(x) for x in labels],
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            result["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
        except Exception:
            result["log_loss"] = None
        result["ece"] = float(expected_calibration_error(y_true, y_prob))
        result["mean_confidence"] = float(np.max(y_prob, axis=1).mean())
    return result


def expected_calibration_error(y_true: Sequence[int], y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidence = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan
