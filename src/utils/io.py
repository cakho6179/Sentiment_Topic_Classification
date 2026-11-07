"""Small JSON / path helpers used by scripts and kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_input_file(pattern: str, root: str = "/kaggle/input") -> Path:
    """Find the first file matching ``pattern`` under the Kaggle input mount.

    Kaggle mounts datasets at a deep path that varies by dataset owner and
    slug.  Kernels should glob instead of hard-coding that path.
    """
    from glob import glob

    matches = sorted(glob(str(Path(root) / "**" / pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    return Path(matches[0])
