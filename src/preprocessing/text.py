"""Input construction for encoder, TF-IDF, and decoder models.

Preprocessing is intentionally conservative.  Emoji, teencode, and repeated
characters are kept because Vietnamese social-media studies report that
removing them can hurt sentiment/emotion models.  Normalisation is limited to
Unicode NFC and whitespace cleanup so that the same text can be inspected and
reproduced.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def build_sentiment_text(
    brand: str,
    target: str,
    context: str = "",
    sep: str = "[SEP]",
    include_context: bool = True,
) -> str:
    parts = [f"Brand: {normalize_text(brand)}", f"Target: {normalize_text(target)}"]
    if include_context and normalize_text(context):
        parts.append(f"Context: {normalize_text(context)}")
    return f" {sep} ".join(parts)


def build_topic_text(
    target: str,
    context: str = "",
    sep: str = "[SEP]",
    mode: str = "target_context",
) -> str:
    """Build topic input.

    ``mode`` is one of ``target``, ``context``, or ``target_context``.
    The mixed mode uses target + context when context exists.
    """
    target = normalize_text(target)
    context = normalize_text(context)
    if mode == "target":
        return f"Target: {target}"
    if mode == "context":
        return f"Context: {context}"
    if mode == "target_context":
        if context:
            return f"Target: {target} {sep} Context: {context}"
        return f"Target: {target}"
    raise ValueError(f"Unknown topic input mode: {mode}")


def build_decoder_prompt(
    task: str,
    target: str,
    context: str = "",
    brand: str | None = None,
) -> str:
    lines = [f"### Task: {task}"]
    if task == "sentiment":
        if brand is None:
            raise ValueError("brand is required for sentiment prompts")
        lines.append(f"### Brand: {normalize_text(brand)}")
        lines.append(f"### Target: {normalize_text(target)}")
        if normalize_text(context):
            lines.append(f"### Context: {normalize_text(context)}")
        lines.append("### Allowed labels: 0 = Negative, 1 = Positive, 2 = Neutral")
    elif task == "topic":
        lines.append(f"### Target: {normalize_text(target)}")
        if normalize_text(context):
            lines.append(f"### Context: {normalize_text(context)}")
        lines.append("### Allowed labels: T01, T02, T03, T04, T05, T06, T07, T08, T09, T10, T11, T12, T13")
    else:
        raise ValueError(f"Unknown task: {task}")
    lines.append("### Answer:")
    return "\n".join(lines)
