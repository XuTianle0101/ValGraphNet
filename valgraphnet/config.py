"""Configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply optional recursive overrides."""

    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of two dictionaries."""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def get_cfg(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key from a nested dictionary."""

    current: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

