"""Canonical label mappings for the project.

The source workbook stores Vietnamese label names.  The modelling code uses
integer ids so that the mapping is explicit and versionable.
"""

from __future__ import annotations

SENTIMENT_TO_ID = {
    "Tiêu cực": 0,
    "Tích cực": 1,
    "Trung lập": 2,
}

SENTIMENT_ID_TO_NAME = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
}

SENTIMENT_ID_TO_VI = {
    0: "Tiêu cực",
    1: "Tích cực",
    2: "Trung lập",
}

TOPIC_TO_ID = {
    "Pin & Sạc": "T01",
    "Chất lượng hạ tầng Sạc/ Đổi Pin": "T02",
    "Chính sách hậu mãi (bảo hành, bảo dưỡng, sửa chữa)": "T03",
    "Quy trình đặt hẹn dịch vụ": "T04",
    "Hệ thống Xưởng Dịch vụ (XDV)": "T05",
    "Chất lượng dịch vụ tại XDV": "T06",
    "Thái độ và chuyên môn của nhân sự XDV": "T07",
    "Chi phí sửa chữa, bảo dưỡng, thay thế phụ tùng": "T08",
    "Chính sách cứu hộ, hỗ trợ khẩn cấp": "T09",
    "Tổng đài CSKH": "T10",
    "Dịch vụ sửa chữa tại nhà": "T11",
    "Đề cập chung": "T12",
    "Chi phí & chính sách sạc": "T13",
}

TOPIC_ID_TO_NAME = {v: k for k, v in TOPIC_TO_ID.items()}

TOPIC_IDS = [f"T{i:02d}" for i in range(1, 14)]

TOPIC_ID_TO_INDEX = {tid: i for i, tid in enumerate(TOPIC_IDS)}
INDEX_TO_TOPIC_ID = {i: tid for tid, i in TOPIC_ID_TO_INDEX.items()}

SENTIMENT_LABELS = [0, 1, 2]
TOPIC_LABELS = list(range(13))


def normalize_sentiment(value: str) -> int:
    value = str(value).strip()
    if value not in SENTIMENT_TO_ID:
        raise ValueError(f"Unknown sentiment label: {value!r}")
    return SENTIMENT_TO_ID[value]


def normalize_topic(value: str) -> str:
    value = " ".join(str(value).split())
    if value not in TOPIC_TO_ID:
        raise ValueError(f"Unknown topic label: {value!r}")
    return TOPIC_TO_ID[value]
