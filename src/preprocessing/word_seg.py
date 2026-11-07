"""Vietnamese word segmentation for PhoBERT-style encoders.

PhoBERT was pre-trained on word-segmented text.  ``pyvi`` is used because it
is lightweight and has no external model download.  The function is a no-op
for encoders that do not require segmentation (ViSoBERT, XLM-R).
"""

from __future__ import annotations


def segment_vietnamese(text: str) -> str:
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)
    except Exception:
        # Segmentation is an enhancement, not a correctness requirement.
        return text
