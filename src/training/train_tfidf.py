"""Train and evaluate TF-IDF + Linear SVM baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME
from src.evaluation.bootstrap import bootstrap_macro_f1
from src.models.tfidf import (
    CalibratedLinearSVM,
    LinearSVMClassifier,
    build_variant,
    build_vectorizer,
    save_model,
)
from src.training.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


SENTIMENT_VARIANTS = [
    "full",
    "brand_target",
    "target_context",
    "target_only",
    "brand_only",
    "context_only",
    "full_masked_brand",
    "full_target_marked",
]
TOPIC_VARIANTS = [
    "target_context",
    "target_only",
    "context_only",
    "target_context_marked",
    "target_context_masked_brand",
    "target_masked_brand",
    "context_masked_brand",
]


def load_data(task: str, processed_dir: Path) -> pd.DataFrame:
    if task == "sentiment":
        frame = pd.read_csv(processed_dir / "comment_brands.csv")
        required = {"comment_id", "brand", "sentiment_id", "target", "context", "split", "topic_id"}
    elif task == "topic":
        frame = pd.read_csv(processed_dir / "comments.csv")
        required = {"comment_id", "topic_id", "topic_index", "target", "context", "brands", "split"}
    else:
        raise ValueError(task)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for {task}: {sorted(missing)}")
    return frame


def target_column(task: str) -> str:
    return "sentiment_id" if task == "sentiment" else "topic_index"


def label_names(task: str) -> list[str]:
    if task == "sentiment":
        return [SENTIMENT_ID_TO_NAME[i] for i in range(3)]
    return [f"{tid}:{TOPIC_ID_TO_NAME[tid]}" for tid in [f"T{i:02d}" for i in range(1, 14)]]


def id_columns(task: str) -> list[str]:
    return ["comment_id", "brand"] if task == "sentiment" else ["comment_id"]


def run_experiment(
    *,
    task: str,
    variant: str,
    processed_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    frame = load_data(task, processed_dir)
    y_col = target_column(task)
    frame = frame.dropna(subset=[y_col]).reset_index(drop=True)
    frame[y_col] = frame[y_col].astype(int)

    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("Empty split")

    train_text = build_variant(train, task, variant).fillna("").astype(str)
    val_text = build_variant(val, task, variant).fillna("").astype(str)
    test_text = build_variant(test, task, variant).fillna("").astype(str)

    vectorizer = build_vectorizer(
        word_ngrams=tuple(config.get("word_ngrams", [1, 2])),
        char_ngrams=tuple(config.get("char_ngrams", [3, 5])),
        max_features_word=int(config.get("max_features_word", 50000)),
        max_features_char=int(config.get("max_features_char", 100000)),
    )
    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)
    x_test = vectorizer.transform(test_text)

    y_train = train[y_col].to_numpy()
    y_val = val[y_col].to_numpy()
    y_test = test[y_col].to_numpy()

    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for c_value in config.get("C_grid", [0.1, 0.3, 1.0, 3.0, 10.0]):
        for class_weight in config.get("class_weight_grid", [None, "balanced"]):
            model = LinearSVMClassifier(C=float(c_value), class_weight=class_weight, seed=seed)
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
            val_metrics = compute_metrics(y_val, val_pred, label_names=label_names(task))
            row = {
                "C": float(c_value),
                "class_weight": class_weight,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
            grid_rows.append(row)
            key = (row["val_macro_f1"], -abs(np.log10(max(row["C"], 1e-12))), 1 if class_weight is None else 0)
            if best is None or key > best["key"]:
                best = {**row, "key": key}
    if best is None:
        raise RuntimeError("Hyperparameter grid produced no model")

    best_c = float(best["C"])
    best_class_weight = best["class_weight"]

    uncalibrated = LinearSVMClassifier(C=best_c, class_weight=best_class_weight, seed=seed)
    uncalibrated.fit(x_train, y_train)
    test_pred_uncal = uncalibrated.predict(x_test)
    test_metrics_uncal = compute_metrics(y_test, test_pred_uncal, label_names=label_names(task))

    calibrated = CalibratedLinearSVM(C=best_c, class_weight=best_class_weight, seed=seed)
    calibrated.fit(x_train, y_train)
    test_prob = calibrated.predict_proba(x_test)
    test_pred = calibrated.predict(x_test)
    test_metrics = compute_metrics(y_test, test_pred, y_prob=test_prob, label_names=label_names(task))
    test_metrics["bootstrap_macro_f1"] = bootstrap_macro_f1(
        y_test, test_pred, labels=list(range(len(label_names(task)))), seed=seed
    )

    val_prob = calibrated.predict_proba(x_val)
    val_metrics = compute_metrics(y_val, calibrated.predict(x_val), y_prob=val_prob, label_names=label_names(task))

    predictions = test[id_columns(task)].copy()
    predictions["split"] = "test"
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    predictions["confidence"] = test_prob.max(axis=1)
    predictions["topic_id"] = test["topic_id"].to_numpy()
    if task == "topic":
        predictions["brands"] = test["brands"].to_numpy()
    predictions["variant"] = variant
    predictions["task"] = task
    predictions["seed"] = seed

    val_predictions = val[id_columns(task)].copy()
    val_predictions["split"] = "val"
    val_predictions["y_true"] = y_val
    val_predictions["y_pred"] = val_prob.argmax(axis=1)
    val_predictions["confidence"] = val_prob.max(axis=1)
    val_predictions["topic_id"] = val["topic_id"].to_numpy()
    if task == "topic":
        val_predictions["brands"] = val["brands"].to_numpy()
    val_predictions["variant"] = variant
    val_predictions["task"] = task
    val_predictions["seed"] = seed

    experiment_id = f"tfidf_{task}_{variant}_seed{seed}"
    ensure_dir(output_dir)
    predictions.to_csv(output_dir / f"{experiment_id}_predictions.csv", index=False, encoding="utf-8")
    val_predictions.to_csv(output_dir / f"{experiment_id}_val_predictions.csv", index=False, encoding="utf-8")
    save_json(output_dir / f"{experiment_id}_metrics.json", {
        "experiment_id": experiment_id,
        "task": task,
        "variant": variant,
        "seed": seed,
        "model": "tfidf_linear_svm",
        "best_hyperparameters": {"C": best_c, "class_weight": best_class_weight},
        "grid": grid_rows,
        "val": val_metrics,
        "test": test_metrics,
        "test_uncalibrated": test_metrics_uncal,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "label_names": label_names(task),
    })

    model_payload = {
        "model": calibrated,
        "vectorizer": vectorizer,
        "task": task,
        "variant": variant,
        "label_names": label_names(task),
        "best_hyperparameters": {"C": best_c, "class_weight": best_class_weight},
        "seed": seed,
    }
    save_model(output_dir / f"{experiment_id}.joblib", model_payload)
    return {
        "experiment_id": experiment_id,
        "task": task,
        "variant": variant,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "best_hyperparameters": {"C": best_c, "class_weight": best_class_weight},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "topic"], required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C-grid", type=str, default="0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--class-weight-grid", type=str, default="none,balanced")
    args = parser.parse_args()

    config = {
        "C_grid": [float(x) for x in args.C_grid.split(",")],
        "class_weight_grid": [None if x.strip().lower() == "none" else x.strip() for x in args.class_weight_grid.split(",")],
    }
    result = run_experiment(
        task=args.task,
        variant=args.variant,
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        config=config,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
