"""YAML configuration loading with simple dotted access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dictionary with attribute access and recursive conversion."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _convert(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _convert(data)


def merge_config(base: Config, override: dict | None) -> Config:
    """Recursively merge ``override`` into ``base`` and return a new Config."""
    import copy

    result = copy.deepcopy(dict(base))
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = dict(merge_config(Config(result[key]), value))
            else:
                result[key] = value
    return _convert(result)
