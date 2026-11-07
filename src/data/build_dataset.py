"""Build canonical tables, leakage-aware splits, and EDA artifacts.

Usage
-----
python -m src.data.build_dataset \
    --raw data/raw/Vietnamese_Social_Listening_Synthetic_1000.xlsx \
    --out data/processed \
    --splits data/splits \
    --reports reports
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from src.data.labels import (
    SENTIMENT_ID_TO_NAME,
    SENTIMENT_ID_TO_VI,
    TOPIC_ID_TO_NAME,
    TOPIC_IDS,
    TOPIC_ID_TO_INDEX,
    normalize_sentiment,
    normalize_topic,
)
from src.data.parse import parse_info
from src.preprocessing.text import build_sentiment_text, build_topic_text, normalize_text
from src.utils.io import ensure_dir, save_json


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _first_unique(series: pd.Series, name: str) -> str:
    values = [str(v).strip() for v in series.dropna().unique() if str(v).strip()]
    if not values:
        return ""
    if len(values) > 1:
        raise ValueError(f"Conflicting {name} values: {values[:5]}")
    return values[0]


def _token_len(text: str) -> int:
    return len(str(text).split())


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Dataset")
    required = {"comment_id", "thread_id", "brand", "info", "sentiment", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    for column in required:
        df[column] = df[column].astype(str).str.strip()
    return df


def parse_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        parsed = parse_info(row.info)
        records.append(
            {
                "comment_id": row.comment_id,
                "thread_id": row.thread_id,
                "brand": row.brand,
                "sentiment_name": row.sentiment,
                "topic_name": row.topic,
                "thread_info": parsed.thread_info,
                "channel": parsed.channel,
                "thread_brand": parsed.thread_brand,
                "context_raw": parsed.context,
                "target_type": parsed.target_type,
                "target_raw": parsed.target,
                "target": parsed.target_body,
            }
        )
    raw = pd.DataFrame(records)
    raw["sentiment_id"] = raw["sentiment_name"].map(normalize_sentiment).astype(int)
    raw["topic_id"] = raw["topic_name"].map(normalize_topic)
    raw["target"] = raw["target"].map(normalize_text)
    raw["context"] = raw["context_raw"].map(normalize_text)
    raw["target_len"] = raw["target"].map(_token_len)
    raw["context_len"] = raw["context"].map(_token_len)
    raw["brand_in_text"] = raw.apply(
        lambda r: r["brand"].lower() in (r["target"] + " " + r["context"]).lower(),
        axis=1,
    )
    return raw


def build_comments(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comment_id, group in raw.groupby("comment_id", sort=True):
        brands = sorted(group["brand"].unique().tolist())
        brand_to_sentiment = {}
        for _, row in group.iterrows():
            brand_to_sentiment.setdefault(row["brand"], row["sentiment_id"])
        if len(brand_to_sentiment) != len(brands):
            raise ValueError(f"Duplicate (comment_id, brand) for {comment_id}")
        target = _first_unique(group["target"], "target")
        context = _first_unique(group["context"], "context")
        topic_id = _first_unique(group["topic_id"], "topic_id")
        rows.append(
            {
                "comment_id": comment_id,
                "thread_id": _first_unique(group["thread_id"], "thread_id"),
                "target": target,
                "context": context,
                "target_type": _first_unique(group["target_type"], "target_type"),
                "channel": _first_unique(group["channel"], "channel"),
                "topic_id": topic_id,
                "topic_name": TOPIC_ID_TO_NAME[topic_id],
                "topic_index": TOPIC_ID_TO_INDEX[topic_id],
                "n_brands": len(brands),
                "brands": "|".join(brands),
                "sentiments": "|".join(str(brand_to_sentiment[b]) for b in brands),
                "brand_in_text": bool(group["brand_in_text"].any()),
                "target_len": len(target.split()),
                "context_len": len(context.split()),
            }
        )
    comments = pd.DataFrame(rows)
    comments["topic_text"] = comments.apply(
        lambda r: build_topic_text(r["target"], r["context"], mode="target_context"),
        axis=1,
    )
    comments["topic_text_target"] = comments["target"]
    comments["topic_text_context"] = comments["context"]
    return comments


def build_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    pairs = raw.copy()
    pairs["sentiment_name_en"] = pairs["sentiment_id"].map(SENTIMENT_ID_TO_NAME)
    pairs["sentiment_name_vi"] = pairs["sentiment_id"].map(SENTIMENT_ID_TO_VI)
    pairs["topic_name"] = pairs["topic_id"].map(TOPIC_ID_TO_NAME)
    pairs["topic_index"] = pairs["topic_id"].map(TOPIC_ID_TO_INDEX)
    pairs["text_sentiment"] = pairs.apply(
        lambda r: build_sentiment_text(
            r["brand"], r["target"], r["context"], include_context=True
        ),
        axis=1,
    )
    pairs = pairs.drop_duplicates(subset=["comment_id", "brand"], keep="first")
    return pairs.reset_index(drop=True)


def add_group_ids(comments: pd.DataFrame, threshold: float = 0.90) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(comments)
    uf = UnionFind(n)

    thread_indices: dict[str, list[int]] = {}
    for idx, thread_id in enumerate(comments["thread_id"].tolist()):
        thread_indices.setdefault(thread_id, []).append(idx)
    for indices in thread_indices.values():
        for other in indices[1:]:
            uf.union(indices[0], other)

    # Exact duplicate targets must never be split across train/test, even when
    # their thread-level context differs.
    target_indices: dict[str, list[int]] = {}
    for idx, target in enumerate(comments["target"].fillna("").tolist()):
        target_indices.setdefault(target, []).append(idx)
    for indices in target_indices.values():
        for other in indices[1:]:
            uf.union(indices[0], other)

    combined_indices: dict[str, list[int]] = {}
    combined = (comments["target"].fillna("") + "\n" + comments["context"].fillna("")).tolist()
    for idx, text in enumerate(combined):
        combined_indices.setdefault(text, []).append(idx)
    for indices in combined_indices.values():
        for other in indices[1:]:
            uf.union(indices[0], other)

    texts = (comments["target"].fillna("") + " " + comments["context"].fillna("")).str.strip()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)

    pairs = [(int(i), int(j), float(similarity[i, j])) for i, j in np.argwhere(similarity >= threshold) if i < j]
    for i, j, _ in pairs:
        uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    root_to_id = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    comments = comments.copy()
    comments["group_id"] = [root_to_id[root] for root in roots]

    duplicate_text_groups = (
        comments.groupby("target")["comment_id"].size().loc[lambda s: s > 1]
    )
    audit = {
        "near_duplicate_threshold": threshold,
        "near_duplicate_pairs": len(pairs),
        "unique_groups": int(comments["group_id"].nunique()),
        "exact_duplicate_target_groups": int(len(duplicate_text_groups)),
        "exact_duplicate_comments": int(duplicate_text_groups.sum()),
        "max_group_size": int(comments.groupby("group_id").size().max()),
    }
    return comments, audit


def make_folds(comments: pd.DataFrame, n_splits: int = 7, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    comments = comments.copy()
    comments["fold"] = -1
    dummy_x = np.zeros((len(comments), 1), dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(
        splitter.split(dummy_x, comments["topic_id"], groups=comments["group_id"])
    ):
        comments.loc[comments.index[valid_idx], "fold"] = fold

    if (comments["fold"] < 0).any():
        raise RuntimeError("Some comments were not assigned to a fold")

    comments["split"] = comments["fold"].map(lambda f: "test" if f == 0 else ("val" if f == 1 else "train"))

    fold_topic = pd.crosstab(comments["fold"], comments["topic_id"])
    fold_brand = pd.crosstab(comments["fold"], comments["brands"].str.split("|").str[0])
    groups_per_fold = comments.groupby("group_id")["fold"].nunique()
    audit = {
        "n_splits": n_splits,
        "seed": seed,
        "fold_sizes": {int(k): int(v) for k, v in comments["fold"].value_counts().sort_index().items()},
        "split_sizes": {str(k): int(v) for k, v in comments["split"].value_counts().items()},
        "groups_spanning_folds": int((groups_per_fold > 1).sum()),
        "topics_missing_in_any_fold": int((fold_topic == 0).sum().sum()),
        "brands_missing_in_any_fold": int((fold_brand == 0).sum().sum()),
        "fold_topic_counts": fold_topic.to_dict(orient="index"),
        "fold_brand_counts": fold_brand.to_dict(orient="index"),
    }
    return comments, audit


def verify_leakage(comments: pd.DataFrame, threshold: float = 0.90) -> dict[str, Any]:
    texts = (comments["target"].fillna("") + " " + comments["context"].fillna("")).str.strip()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    split = comments["split"].to_numpy()
    cross_split_pairs = 0
    for i, j in np.argwhere(similarity >= threshold):
        if i < j and split[i] != split[j]:
            cross_split_pairs += 1
    exact_cross = 0
    duplicate_texts = comments.groupby("target")["split"].nunique()
    exact_cross = int((duplicate_texts > 1).sum())
    return {
        "cross_split_near_duplicates": int(cross_split_pairs),
        "cross_split_exact_duplicates": exact_cross,
    }


def validate_tables(comments: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["comments_unique"] = bool(comments["comment_id"].is_unique)
    checks["pairs_unique"] = bool(not pairs.duplicated(subset=["comment_id", "brand"]).any())
    checks["comments_rows"] = int(len(comments))
    checks["pairs_rows"] = int(len(pairs))
    checks["target_non_empty"] = bool((comments["target"].str.len() > 0).all())
    checks["topic_ids_valid"] = bool(comments["topic_id"].isin(TOPIC_IDS).all())
    checks["sentiment_ids_valid"] = bool(pairs["sentiment_id"].isin([0, 1, 2]).all())
    checks["fold_range_valid"] = bool(comments["fold"].between(0, 6).all())
    checks["topic_consistent_per_comment"] = bool((comments.groupby("comment_id")["topic_id"].nunique() == 1).all())
    checks["all_topic_ids_present"] = bool(set(comments["topic_id"]) == set(TOPIC_IDS))
    if not all(
        checks[key]
        for key in [
            "comments_unique",
            "pairs_unique",
            "target_non_empty",
            "topic_ids_valid",
            "sentiment_ids_valid",
            "fold_range_valid",
            "topic_consistent_per_comment",
            "all_topic_ids_present",
        ]
    ):
        raise ValueError(f"Data validation failed: {checks}")
    return checks


def eda_plots(comments: pd.DataFrame, pairs: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    ensure_dir(out_dir)
    figures: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sentiment_counts = pairs["sentiment_name_en"].value_counts().reindex(["Negative", "Positive", "Neutral"])
    axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=["#d95f5f", "#4c9f70", "#8a8f98"])
    axes[0].set_title("Sentiment distribution (comment-brand pairs)")
    axes[0].set_ylabel("Rows")
    topic_counts = comments["topic_id"].value_counts().reindex(TOPIC_IDS)
    axes[1].bar(topic_counts.index, topic_counts.values, color="#5b7db1")
    axes[1].set_title("Topic distribution (unique comments)")
    axes[1].set_ylabel("Comments")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = out_dir / "label_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    brand_counts = pairs["brand"].value_counts()
    ax.bar(brand_counts.index, brand_counts.values, color="#6b8e9e")
    ax.set_title("Brand distribution (comment-brand pairs)")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    path = out_dir / "brand_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(comments["target_len"], bins=30, color="#5b7db1")
    axes[0].set_title("Target length (whitespace tokens)")
    axes[0].set_xlabel("Tokens")
    non_empty_context = comments.loc[comments["context_len"] > 0, "context_len"]
    axes[1].hist(non_empty_context, bins=20, color="#4c9f70")
    axes[1].set_title("Context length (non-empty)")
    axes[1].set_xlabel("Tokens")
    fig.tight_layout()
    path = out_dir / "length_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(14, 6))
    topic_brand = pd.crosstab(comments["topic_id"], comments["brands"].str.split("|").str[0])
    topic_brand = topic_brand.reindex(TOPIC_IDS)
    sns.heatmap(topic_brand, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Topic x first brand (unique comments)")
    fig.tight_layout()
    path = out_dir / "topic_brand_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    sentiment_brand = pd.crosstab(pairs["brand"], pairs["sentiment_name_en"])
    sentiment_brand = sentiment_brand.reindex(columns=["Negative", "Positive", "Neutral"], fill_value=0)
    sentiment_brand.plot(kind="bar", stacked=True, ax=ax, color=["#d95f5f", "#4c9f70", "#8a8f98"])
    ax.set_title("Sentiment x brand (comment-brand pairs)")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    path = out_dir / "sentiment_brand_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    context_missing = comments.assign(has_context=comments["context_len"] > 0).groupby("topic_id")["has_context"].mean().reindex(TOPIC_IDS)
    ax.bar(context_missing.index, context_missing.values, color="#9b7bb5")
    ax.set_ylim(0, 1)
    ax.set_title("Share of comments with context, by topic")
    ax.set_ylabel("Share")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = out_dir / "context_presence_by_topic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    thread_sizes = comments.groupby("thread_id").size().value_counts().sort_index()
    ax.bar(thread_sizes.index.astype(str), thread_sizes.values, color="#c58b5a")
    ax.set_title("Comments per thread")
    ax.set_xlabel("Comments in thread")
    ax.set_ylabel("Threads")
    fig.tight_layout()
    path = out_dir / "thread_sizes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    group_sizes = comments.groupby("group_id").size().value_counts().sort_index()
    ax.bar(group_sizes.index.astype(str), group_sizes.values, color="#7a9e9f")
    ax.set_title("Comments per leakage group (thread + near-duplicate)")
    ax.set_xlabel("Comments in group")
    ax.set_ylabel("Groups")
    fig.tight_layout()
    path = out_dir / "group_sizes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target_type_counts = comments["target_type"].value_counts()
    axes[0].bar(target_type_counts.index, target_type_counts.values, color="#6b8e9e")
    axes[0].set_title("Target type (unique comments)")
    axes[0].tick_params(axis="x", rotation=20)
    channel_counts = comments["channel"].value_counts()
    axes[1].bar(channel_counts.index, channel_counts.values, color="#b58b6b")
    axes[1].set_title("Channel (unique comments)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = out_dir / "target_type_channel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    return figures


def write_audit_report(
    *,
    raw: pd.DataFrame,
    comments: pd.DataFrame,
    pairs: pd.DataFrame,
    split_audit: dict[str, Any],
    group_audit: dict[str, Any],
    leakage: dict[str, Any],
    validation: dict[str, Any],
    figures: list[str],
    out_path: Path,
    source_path: Path,
) -> None:
    sentiment_counts = pairs["sentiment_name_en"].value_counts().reindex(["Negative", "Positive", "Neutral"])
    topic_counts = comments["topic_id"].value_counts().reindex(TOPIC_IDS)
    brand_counts = pairs["brand"].value_counts()
    missing_target = int((comments["target"].str.len() == 0).sum())
    missing_context = int((comments["context"].str.len() == 0).sum())
    multi_brand = int((comments["n_brands"] > 1).sum())
    brand_not_in_text = int((~pairs["brand_in_text"]).sum())

    lines = [
        "# Data Audit & EDA Report",
        "",
        f"Source file: `{source_path.as_posix()}`",
        "",
        "## 1. Dataset overview",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Raw rows (comment-brand) | {len(raw)} |",
        f"| Unique comments | {len(comments)} |",
        f"| Unique threads | {comments['thread_id'].nunique()} |",
        f"| Unique brands | {pairs['brand'].nunique()} |",
        f"| Multi-brand comments | {multi_brand} |",
        f"| Rows with empty target | {missing_target} |",
        f"| Rows with empty context | {missing_context} |",
        f"| Rows where brand is not in target+context | {brand_not_in_text} |",
        "",
        "## 2. Label distributions",
        "",
        "### Sentiment (comment-brand pairs)",
        "",
        "| Class | Rows | Share |",
        "|---|---:|---:|",
    ]
    for name, count in sentiment_counts.items():
        lines.append(f"| {name} | {int(count)} | {count / len(pairs):.2%} |")
    lines.extend(
        [
            "",
            "### Topic (unique comments)",
            "",
            "| Topic ID | Topic name | Comments | Share |",
            "|---|---|---:|---:|",
        ]
    )
    for topic_id, count in topic_counts.items():
        lines.append(
            f"| {topic_id} | {TOPIC_ID_TO_NAME[topic_id]} | {int(count)} | {count / len(comments):.2%} |"
        )
    lines.extend(
        [
            "",
            "### Brand distribution (comment-brand pairs)",
            "",
            "| Brand | Rows | Share |",
            "|---|---:|---:|",
        ]
    )
    for brand, count in brand_counts.items():
        lines.append(f"| {brand} | {int(count)} | {count / len(pairs):.2%} |")
    lines.extend(
        [
            "",
            "## 3. Data quality and leakage",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Comments unique by `comment_id` | {validation['comments_unique']} |",
            f"| Pairs unique by `(comment_id, brand)` | {validation['pairs_unique']} |",
            f"| Topic consistent per comment | {validation['topic_consistent_per_comment']} |",
            f"| All 13 topic ids present | {validation['all_topic_ids_present']} |",
            f"| Groups spanning folds | {split_audit['groups_spanning_folds']} |",
            f"| Topics missing in any fold | {split_audit['topics_missing_in_any_fold']} |",
            f"| Brands missing in any fold | {split_audit['brands_missing_in_any_fold']} |",
            f"| Cross-split near-duplicates | {leakage['cross_split_near_duplicates']} |",
            f"| Cross-split exact duplicates | {leakage['cross_split_exact_duplicates']} |",
            "",
            f"Near-duplicate detection threshold: cosine >= {group_audit['near_duplicate_threshold']:.2f} on char 3-5 gram TF-IDF.",
            f"Near-duplicate pairs found: {group_audit['near_duplicate_pairs']}.",
            f"Exact duplicate target groups: {group_audit['exact_duplicate_target_groups']}.",
            f"Leakage groups after union of thread and near-duplicate components: {group_audit['unique_groups']}.",
            "",
            "## 4. Split summary",
            "",
            "| Split | Comments | Share |",
            "|---|---:|---:|",
        ]
    )
    for split_name in ["train", "val", "test"]:
        count = int((comments["split"] == split_name).sum())
        lines.append(f"| {split_name} | {count} | {count / len(comments):.2%} |")
    lines.extend(
        [
            "",
            f"Fold assignment: {split_audit['n_splits']}-fold StratifiedGroupKFold, seed {split_audit['seed']}. "
            "Fold 0 = test, fold 1 = validation, folds 2-6 = train.",
            "",
            "## 5. Key observations",
            "",
            f"- The workbook describes itself as **synthetic** (`Summary` sheet). Numbers here are for pipeline validation, not a production estimate.",
            f"- Context is missing for {missing_context}/{len(comments)} unique comments. Context is not always available at inference time.",
            f"- Context mentions the brand in 100% of non-empty cases, while the target mentions the brand in only about half of cases. A topic model that reads context may exploit this shortcut.",
            f"- Topic and brand are strongly associated in this synthetic sample, so topic scores may partly reflect brand memorisation.",
            f"- {multi_brand} comments mention two brands. Sentiment must be predicted per (comment, brand); topic must be predicted once per comment.",
            f"- {brand_not_in_text} comment-brand pairs do not contain the brand name in the target/context text. The model must use the explicit brand field.",
            f"- {group_audit['exact_duplicate_target_groups']} exact duplicate target groups and {group_audit['near_duplicate_pairs']} near-duplicate pairs were found. They are grouped before splitting.",
            "",
            "## 6. Figures",
            "",
        ]
    )
    for figure in figures:
        lines.append(f"- `{Path(figure).name}`")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw/Vietnamese_Social_Listening_Synthetic_1000.xlsx")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--splits", type=str, default="data/splits")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--n-splits", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = ensure_dir(args.out)
    split_dir = ensure_dir(args.splits)
    report_dir = ensure_dir(args.reports)
    figure_dir = ensure_dir(report_dir / "figures")

    raw_df = load_source(raw_path)
    raw = parse_rows(raw_df)
    comments = build_comments(raw)
    pairs = build_pairs(raw)

    comments, group_audit = add_group_ids(comments, threshold=args.duplicate_threshold)
    comments, split_audit = make_folds(comments, n_splits=args.n_splits, seed=args.seed)

    fold_map = comments[["comment_id", "group_id", "fold", "split"]]
    pairs = pairs.merge(fold_map, on="comment_id", how="left", validate="many_to_one")
    if pairs[["fold", "split"]].isna().any().any():
        raise RuntimeError("Some comment-brand pairs did not receive a fold")

    leakage = verify_leakage(comments, threshold=args.duplicate_threshold)
    validation = validate_tables(comments, pairs)
    if leakage["cross_split_near_duplicates"] > 0 or leakage["cross_split_exact_duplicates"] > 0:
        raise RuntimeError(f"Leakage check failed: {leakage}")

    sentiment_counts = pairs["sentiment_name_en"].value_counts().reindex(["Negative", "Positive", "Neutral"])
    topic_counts = comments["topic_id"].value_counts().reindex(TOPIC_IDS)
    brand_counts = pairs["brand"].value_counts()
    missing_target = int((comments["target"].str.len() == 0).sum())
    missing_context = int((comments["context"].str.len() == 0).sum())
    multi_brand = int((comments["n_brands"] > 1).sum())
    brand_not_in_text = int((~pairs["brand_in_text"]).sum())

    comment_columns = [
        "comment_id",
        "thread_id",
        "group_id",
        "fold",
        "split",
        "target",
        "context",
        "target_type",
        "channel",
        "topic_id",
        "topic_name",
        "topic_index",
        "n_brands",
        "brands",
        "sentiments",
        "brand_in_text",
        "target_len",
        "context_len",
        "topic_text",
        "topic_text_target",
        "topic_text_context",
    ]
    pair_columns = [
        "comment_id",
        "thread_id",
        "group_id",
        "fold",
        "split",
        "brand",
        "sentiment_id",
        "sentiment_name_en",
        "sentiment_name_vi",
        "topic_id",
        "topic_name",
        "topic_index",
        "target",
        "context",
        "target_type",
        "channel",
        "brand_in_text",
        "target_len",
        "context_len",
        "text_sentiment",
    ]
    comments[comment_columns].to_csv(out_dir / "comments.csv", index=False, encoding="utf-8")
    pairs[pair_columns].to_csv(out_dir / "comment_brands.csv", index=False, encoding="utf-8")

    folds = (
        comments.groupby(["group_id", "fold", "split"], as_index=False)
        .agg(
            n_comments=("comment_id", "size"),
            topics=("topic_id", lambda s: "|".join(sorted(set(s)))),
            brands=("brands", lambda s: "|".join(sorted({b for value in s for b in value.split("|")}))),
        )
    )
    folds.to_csv(split_dir / "folds.csv", index=False, encoding="utf-8")

    save_json(
        out_dir / "label_maps.json",
        {
            "sentiment": SENTIMENT_ID_TO_NAME,
            "sentiment_vi": SENTIMENT_ID_TO_VI,
            "topic": TOPIC_ID_TO_NAME,
            "topic_ids": TOPIC_IDS,
        },
    )
    save_json(
        report_dir / "data_audit.json",
        {
            "source": str(raw_path),
            "n_raw_rows": int(len(raw)),
            "n_comments": int(len(comments)),
            "n_threads": int(comments["thread_id"].nunique()),
            "n_brands": int(pairs["brand"].nunique()),
            "n_pairs": int(len(pairs)),
            "sentiment_counts": {k: int(v) for k, v in sentiment_counts.items()},
            "topic_counts": {k: int(v) for k, v in topic_counts.items()},
            "brand_counts": {k: int(v) for k, v in brand_counts.items()},
            "missing_target": missing_target,
            "missing_context": missing_context,
            "multi_brand_comments": multi_brand,
            "brand_not_in_text_pairs": brand_not_in_text,
            "group_audit": group_audit,
            "split_audit": split_audit,
            "leakage": leakage,
            "validation": validation,
        },
    )

    figures = eda_plots(comments, pairs, figure_dir)
    write_audit_report(
        raw=raw,
        comments=comments,
        pairs=pairs,
        split_audit=split_audit,
        group_audit=group_audit,
        leakage=leakage,
        validation=validation,
        figures=figures,
        out_path=report_dir / "data_audit.md",
        source_path=raw_path,
    )

    print("Built dataset")
    print(f"  comments: {len(comments)}")
    print(f"  pairs:    {len(pairs)}")
    print(f"  groups:   {comments['group_id'].nunique()}")
    print(f"  splits:   {comments['split'].value_counts().to_dict()}")
    print(f"  report:   {report_dir / 'data_audit.md'}")


if __name__ == "__main__":
    main()
