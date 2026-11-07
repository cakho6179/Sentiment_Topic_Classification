"""Parse the XML-like ``info`` field used by the source workbook.

The field contains three sections::

    <thread_info>
    Kênh: Facebook
    Thương hiệu mục tiêu: Honda
    </thread_info>
    <context>...</context>
    <target type="bình luận">[Bình luận] ...</target>

The parser is deliberately tolerant: missing sections produce empty strings
instead of raising, and the target type prefix (``[Bình luận]``) is removed
from the modelling text while being retained as metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MISSING_MARKERS = {"", "(không có)", "không có", "none", "nan", "null", "n/a"}


@dataclass(frozen=True)
class ParsedInfo:
    thread_info: str
    channel: str
    thread_brand: str
    context: str
    target_type: str
    target: str
    target_body: str


def _first_match(pattern: str, text: str, group: int = 1) -> str:
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return ""
    return match.group(group).strip()


def clean_missing(value: str | None) -> str:
    """Return an empty string for the dataset's missing-value markers."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in _MISSING_MARKERS:
        return ""
    return text


def strip_target_prefix(target: str) -> str:
    """Remove the leading ``[Bài viết]`` / ``[Đánh giá]`` / ``[Bình luận]`` tag."""
    return re.sub(r"^\[[^\]]+\]\s*", "", target).strip()


def parse_info(info: str) -> ParsedInfo:
    text = "" if info is None else str(info)

    thread_info = _first_match(r"<thread_info>(.*?)</thread_info>", text)
    context = clean_missing(_first_match(r"<context>(.*?)</context>", text))

    target_type = _first_match(r'<target type="([^"]*)">', text)
    target = _first_match(r'<target type="[^"]*">(.*?)</target>', text)
    target_body = strip_target_prefix(target)

    channel = _first_match(r"Kênh:\s*([^\n\r]+)", thread_info)
    thread_brand = _first_match(r"Thương hiệu mục tiêu:\s*([^\n\r]+)", thread_info)

    return ParsedInfo(
        thread_info=thread_info,
        channel=channel.strip(),
        thread_brand=thread_brand.strip(),
        context=context,
        target_type=target_type.strip(),
        target=target,
        target_body=target_body,
    )
