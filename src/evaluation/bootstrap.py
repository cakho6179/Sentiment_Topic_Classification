"""Bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest

from src.training.metrics import macro_f1


def bootstrap_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(macro_f1(y_true[idx], y_pred[idx], labels=labels))
    scores = np.asarray(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "ci_lower": float(np.quantile(scores, 0.025)),
        "ci_upper": float(np.quantile(scores, 0.975)),
        "n_boot": int(n_boot),
    }


def mcnemar_test(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
) -> dict[str, Any]:
    """Exact McNemar test on discordant pairs (no continuity correction)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0, "statistic": 0.0}
    result = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided")
    return {
        "b": b,
        "c": c,
        "p_value": float(result.pvalue),
        "statistic": float(min(b, c)),
    }
