"""TF-IDF + Linear SVM baselines and input ablations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.preprocessing.text import build_sentiment_text, build_topic_text


def _mask_brands(text: str, brands: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    for brand in sorted([b for b in str(brands).split("|") if b], key=len, reverse=True):
        text = re.sub(re.escape(brand), "[BRAND]", text, flags=re.I)
    return text


def build_variant(frame: pd.DataFrame, task: str, variant: str) -> pd.Series:
    """Build a text column for a named input ablation."""
    if task == "sentiment":
        if variant == "full":
            return frame["text_sentiment"]
        if variant == "brand_target":
            return frame.apply(
                lambda r: build_sentiment_text(r["brand"], r["target"], r["context"], include_context=False),
                axis=1,
            )
        if variant == "target_context":
            return frame["target"].fillna("") + " [SEP] " + frame["context"].fillna("")
        if variant == "target_only":
            return frame["target"].fillna("")
        if variant == "context_only":
            return frame["context"].fillna("")
        if variant == "brand_only":
            return frame["brand"].astype(str)
        if variant == "full_masked_brand":
            return frame.apply(lambda r: _mask_brands(r["text_sentiment"], r["brand"]), axis=1)
        if variant == "full_target_marked":
            return frame.apply(
                lambda r: (
                    f"Brand: {r['brand']} [SEP] Target: <TARGET> {r['target']} </TARGET>"
                    + (f" [SEP] Context: {r['context']}" if str(r["context"]).strip() else "")
                ),
                axis=1,
            )
        if variant == "full_no_context":
            return frame.apply(
                lambda r: build_sentiment_text(r["brand"], r["target"], r["context"], include_context=False),
                axis=1,
            )
    elif task == "topic":
        if variant == "target_context":
            return frame.apply(lambda r: build_topic_text(r["target"], r["context"], mode="target_context"), axis=1)
        if variant == "target_context_marked":
            return frame.apply(
                lambda r: (
                    f"Target: <TARGET> {r['target']} </TARGET>"
                    + (f" [SEP] Context: {r['context']}" if str(r["context"]).strip() else "")
                ),
                axis=1,
            )
        if variant == "target_only":
            return frame["target"].fillna("")
        if variant == "context_only":
            return frame["context"].fillna("")
        if variant == "target_context_masked_brand":
            return frame.apply(
                lambda r: _mask_brands(build_topic_text(r["target"], r["context"], mode="target_context"), r["brands"]),
                axis=1,
            )
        if variant == "target_masked_brand":
            return frame.apply(lambda r: _mask_brands(r["target"], r["brands"]), axis=1)
        if variant == "context_masked_brand":
            return frame.apply(lambda r: _mask_brands(r["context"], r["brands"]), axis=1)
    raise ValueError(f"Unknown task/variant: {task}/{variant}")


def build_vectorizer(
    word_ngrams: tuple[int, int] = (1, 2),
    char_ngrams: tuple[int, int] = (3, 5),
    max_features_word: int = 50000,
    max_features_char: int = 100000,
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_ngrams,
                    min_df=1,
                    max_features=max_features_word,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngrams,
                    min_df=1,
                    max_features=max_features_char,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ]
    )


class LinearSVMClassifier:
    """LinearSVC with softmax-transformed decision scores as confidence."""

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", seed: int = 42) -> None:
        self.C = C
        self.class_weight = class_weight
        self.seed = seed
        self.estimator = LinearSVC(C=C, class_weight=class_weight, max_iter=10000, random_state=seed)
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> "LinearSVMClassifier":
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def decision_function(self, X) -> np.ndarray:
        return self.estimator.decision_function(X)

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


class CalibratedLinearSVM:
    """Calibrated LinearSVC for probability reporting."""

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", seed: int = 42) -> None:
        base = LinearSVC(C=C, class_weight=class_weight, max_iter=10000, random_state=seed)
        self.estimator = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=1)
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> "CalibratedLinearSVM":
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)


def save_model(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)
