from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    return obj


def write_json(path: str, payload: Any) -> str:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    return path


def slugify(value: str, *, max_len: int = 80) -> str:
    value = (value or "").strip()
    value = value.replace("/", "__")
    value = value.replace(":", "-")
    value = value.replace("@", "-at-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    value = value.strip("-_.").lower()
    if not value:
        value = "na"
    if len(value) > max_len:
        value = value[:max_len].rstrip("-_.")
    return value


def float_tag(x: float, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


def model_label(model_name: str, label: Optional[str] = None) -> str:
    raw = label if label else model_name
    return slugify(raw, max_len=64)


def timestamp_tag() -> str:
    # Example: 20260417_194522
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_experiment_id(
    prefix: str,
    *,
    cfg: Any,
    num_seeds: int,
    extra_tags: Optional[Iterable[str]] = None,
    include_timestamp: bool = True,
) -> str:
    parts = [
        slugify(prefix, max_len=24),
        f"N{cfg.N}_M{cfg.M}_T{cfg.T}",
        f"p{float_tag(cfg.p_perception)}",
        f"lam{float_tag(cfg.streak.lam)}",
        f"eta{float_tag(cfg.drift.eta)}",
        f"seeds{int(num_seeds)}",
    ]
    if extra_tags:
        parts.extend(slugify(str(x), max_len=24) for x in extra_tags)
    if include_timestamp:
        parts.append(timestamp_tag())
    return "__".join(parts)


def build_match_id(
    row_label: str,
    col_label: str,
    *,
    seed: int,
) -> str:
    return "__".join(
        [
            model_label(row_label),
            "vs",
            model_label(col_label),
            f"seed{int(seed):03d}",
        ]
    )