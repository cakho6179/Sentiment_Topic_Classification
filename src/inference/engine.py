"""Unified inference engine for sentiment and topic classification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.labels import SENTIMENT_ID_TO_NAME, TOPIC_ID_TO_NAME, TOPIC_IDS
from src.models.tfidf import build_variant
from src.preprocessing.text import build_sentiment_text, build_topic_text, normalize_text
from src.preprocessing.word_seg import segment_vietnamese


def _mask_brands(text: str, brands: Sequence[str]) -> str:
    for brand in sorted([b for b in brands if b], key=len, reverse=True):
        text = re.sub(re.escape(brand), "[BRAND]", text, flags=re.I)
    return text


def _text_for_variant(
    *,
    task: str,
    variant: str,
    brand: str,
    target: str,
    context: str,
    brands: Sequence[str] | None = None,
) -> str:
    brand = normalize_text(brand)
    target = normalize_text(target)
    context = normalize_text(context)
    brands = list(brands or ([brand] if brand else []))
    if task == "sentiment":
        if variant == "full":
            return build_sentiment_text(brand, target, context, include_context=True)
        if variant in {"brand_target", "full_no_context"}:
            return build_sentiment_text(brand, target, context, include_context=False)
        if variant == "target_context":
            return f"Target: {target} [SEP] Context: {context}" if context else f"Target: {target}"
        if variant == "target_only":
            return target
        if variant == "brand_only":
            return brand
        if variant == "context_only":
            return context
        if variant == "full_masked_brand":
            return _mask_brands(build_sentiment_text(brand, target, context), brands)
        if variant == "full_target_marked":
            return (
                f"Brand: {brand} [SEP] Target: <TARGET> {target} </TARGET>"
                + (f" [SEP] Context: {context}" if context else "")
            )
        raise ValueError(f"Unknown sentiment variant: {variant}")
    if task == "topic":
        if variant == "target_context":
            return build_topic_text(target, context, mode="target_context")
        if variant == "target_context_marked":
            return (
                f"Target: <TARGET> {target} </TARGET>"
                + (f" [SEP] Context: {context}" if context else "")
            )
        if variant == "target_only":
            return build_topic_text(target, context, mode="target")
        if variant == "context_only":
            return build_topic_text(target, context, mode="context")
        if variant == "target_context_masked_brand":
            return _mask_brands(build_topic_text(target, context, mode="target_context"), brands)
        if variant == "target_masked_brand":
            return _mask_brands(target, brands)
        if variant == "context_masked_brand":
            return _mask_brands(context, brands)
        raise ValueError(f"Unknown topic variant: {variant}")
    raise ValueError(f"Unknown task: {task}")


class _Backend:
    def predict(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class _TfidfBackend(_Backend):
    def __init__(self, payload: dict[str, Any], task: str) -> None:
        self.vectorizer = payload["vectorizer"]
        self.model = payload["model"]
        self.task = task
        self.variant = payload.get("variant", "full" if task == "sentiment" else "target_context")

    def predict(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        texts = [
            _text_for_variant(
                task=self.task,
                variant=self.variant,
                brand=record.get("brand", ""),
                target=record.get("target", ""),
                context=record.get("context", ""),
                brands=record.get("brands"),
            )
            for record in records
        ]
        matrix = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(matrix)
        preds = probs.argmax(axis=1)
        return preds, probs


class _EncoderBackend(_Backend):
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        metadata_path = model_dir / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.task = self.metadata.get("task")
        if self.task not in {"sentiment", "topic"}:
            raise ValueError(f"Cannot infer task from {model_dir}")
        self.variant = self.metadata.get(
            "variant", "full" if self.task == "sentiment" else "target_context"
        )
        self.max_length = int(self.metadata.get("max_length", 256))
        self.segment = bool(self.metadata.get("segment", False))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        texts = [
            _text_for_variant(
                task=self.task,
                variant=self.variant,
                brand=record.get("brand", ""),
                target=record.get("target", ""),
                context=record.get("context", ""),
                brands=record.get("brands"),
            )
            for record in records
        ]
        if self.segment:
            texts = [
                text.replace("[SEP]", "__SEP__").replace("__SEP__", self.tokenizer.sep_token)
                for text in texts
            ]
            texts = [segment_vietnamese(text) for text in texts]
        else:
            texts = [text.replace("[SEP]", self.tokenizer.sep_token) for text in texts]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        return preds, probs


class _MTLBackend:
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        from src.models.encoders import load_tokenizer, resolve_spec
        from src.models.multitask import SharedEncoderMTL

        model_dir = Path(model_dir)
        metadata_path = model_dir / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        encoder_dir = model_dir / "encoder"
        spec = resolve_spec(str(encoder_dir))
        self.tokenizer = load_tokenizer(spec)
        self.model = SharedEncoderMTL(str(encoder_dir))
        heads = torch.load(model_dir / "heads.pt", map_location="cpu")
        self.model.topic_head.load_state_dict(heads["topic_head"])
        self.model.sentiment_head.load_state_dict(heads["sentiment_head"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.max_length = int(self.metadata.get("max_length", 256))
        self.segment = bool(self.metadata.get("segment", False))

    def _encode(self, texts: list[str]):
        if self.segment:
            texts = [
                text.replace("[SEP]", "__SEP__").replace("__SEP__", self.tokenizer.sep_token)
                for text in texts
            ]
            texts = [segment_vietnamese(text) for text in texts]
        else:
            texts = [text.replace("[SEP]", self.tokenizer.sep_token) for text in texts]
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    @torch.no_grad()
    def predict_sentiment(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        texts = [
            _text_for_variant(
                task="sentiment",
                variant=self.metadata.get("sentiment_variant", "full"),
                brand=record.get("brand", ""),
                target=record.get("target", ""),
                context=record.get("context", ""),
                brands=record.get("brands"),
            )
            for record in records
        ]
        encoded = self._encode(texts)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(task="sentiment", **encoded)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs.argmax(axis=-1), probs

    @torch.no_grad()
    def predict_topic(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        texts = [
            _text_for_variant(
                task="topic",
                variant=self.metadata.get("topic_variant", "target_context"),
                brand=record.get("brand", ""),
                target=record.get("target", ""),
                context=record.get("context", ""),
                brands=record.get("brands"),
            )
            for record in records
        ]
        encoded = self._encode(texts)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(task="topic", **encoded)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs.argmax(axis=-1), probs


class InferenceEngine:
    """Predict topic once per comment and sentiment once per (comment, brand)."""

    def __init__(
        self,
        *,
        sentiment_backend: _Backend | None = None,
        topic_backend: _Backend | None = None,
        mtl_backend: _MTLBackend | None = None,
        sentiment_threshold: float = 0.0,
        topic_threshold: float = 0.0,
    ) -> None:
        self.sentiment_backend = sentiment_backend
        self.topic_backend = topic_backend
        self.mtl_backend = mtl_backend
        self.sentiment_threshold = float(sentiment_threshold)
        self.topic_threshold = float(topic_threshold)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InferenceEngine":
        mtl_config = config.get("mtl")
        sentiment_config = config.get("sentiment")
        topic_config = config.get("topic")
        mtl_backend = _MTLBackend(mtl_config["model_dir"]) if mtl_config else None
        sentiment_backend = None
        topic_backend = None
        if sentiment_config:
            if sentiment_config.get("type") == "tfidf":
                sentiment_backend = _TfidfBackend(joblib.load(sentiment_config["model_path"]), "sentiment")
            else:
                sentiment_backend = _EncoderBackend(sentiment_config["model_dir"])
        if topic_config:
            if topic_config.get("type") == "tfidf":
                topic_backend = _TfidfBackend(joblib.load(topic_config["model_path"]), "topic")
            else:
                topic_backend = _EncoderBackend(topic_config["model_dir"])
        return cls(
            sentiment_backend=sentiment_backend,
            topic_backend=topic_backend,
            mtl_backend=mtl_backend,
            sentiment_threshold=float(config.get("sentiment_threshold", 0.0)),
            topic_threshold=float(config.get("topic_threshold", 0.0)),
        )

    def _validate(self, record: dict[str, Any]) -> None:
        if not str(record.get("comment_id", "")).strip():
            raise ValueError("comment_id is required")
        if not str(record.get("target", "")).strip() and not str(record.get("context", "")).strip():
            raise ValueError("target or context is required")
        if not str(record.get("brand", "")).strip():
            raise ValueError("brand is required")

    def predict_batch(self, records: list[dict[str, Any]], strict: bool = True) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for record in records:
            try:
                self._validate(record)
                valid.append(record)
            except ValueError as error:
                if strict:
                    raise
                errors.append({"comment_id": record.get("comment_id"), "error": str(error)})
        if not valid:
            return errors

        # Topic is a comment-level task: predict one label per unique comment.
        topic_by_comment: dict[str, dict[str, Any]] = {}
        if self.mtl_backend is not None:
            unique_topics: dict[str, dict[str, Any]] = {}
            for record in valid:
                unique_topics.setdefault(str(record["comment_id"]), record)
            keys = list(unique_topics)
            topic_preds, topic_probs = self.mtl_backend.predict_topic([unique_topics[k] for k in keys])
            for key, pred, probs in zip(keys, topic_preds, topic_probs):
                confidence = float(probs[pred])
                topic_by_comment[key] = {
                    "label": int(pred),
                    "topic_id": TOPIC_IDS[int(pred)],
                    "name": TOPIC_ID_TO_NAME[TOPIC_IDS[int(pred)]],
                    "confidence": confidence,
                    "low_confidence": confidence < self.topic_threshold,
                }
        elif self.topic_backend is not None:
            unique_topics = {}
            for record in valid:
                unique_topics.setdefault(str(record["comment_id"]), record)
            keys = list(unique_topics)
            topic_preds, topic_probs = self.topic_backend.predict([unique_topics[k] for k in keys])
            for key, pred, probs in zip(keys, topic_preds, topic_probs):
                confidence = float(probs[pred])
                topic_by_comment[key] = {
                    "label": int(pred),
                    "topic_id": TOPIC_IDS[int(pred)],
                    "name": TOPIC_ID_TO_NAME[TOPIC_IDS[int(pred)]],
                    "confidence": confidence,
                    "low_confidence": confidence < self.topic_threshold,
                }

        if self.mtl_backend is not None:
            sentiment_preds, sentiment_probs = self.mtl_backend.predict_sentiment(valid)
        elif self.sentiment_backend is not None:
            sentiment_preds, sentiment_probs = self.sentiment_backend.predict(valid)
        else:
            sentiment_preds = [None] * len(valid)
            sentiment_probs = [None] * len(valid)

        outputs: list[dict[str, Any]] = []
        for record, pred, probs in zip(valid, sentiment_preds, sentiment_probs):
            comment_id = str(record["comment_id"])
            sentiment = None
            if pred is not None:
                confidence = float(probs[pred])
                sentiment = {
                    "label": int(pred),
                    "name": SENTIMENT_ID_TO_NAME[int(pred)],
                    "confidence": confidence,
                    "low_confidence": confidence < self.sentiment_threshold,
                }
            outputs.append(
                {
                    "comment_id": comment_id,
                    "brand": str(record.get("brand", "")),
                    "sentiment": sentiment,
                    "topic": topic_by_comment.get(comment_id),
                }
            )
        return outputs + errors

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([record])[0]

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        records = frame.to_dict(orient="records")
        outputs = self.predict_batch(records)
        return pd.DataFrame(outputs)
