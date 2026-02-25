"""
io/jsonl.py

Purpose:
- Save episode metadata and per-round logs as JSONL.
- Keeps env free of I/O and enables reproducible analysis.

Return values:
- write_episode(...) writes files and returns their paths.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple

from game_engine.env.types import EpisodeResult, RoundLog


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses and nested structures into JSON-serializable objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    return obj


def write_episode(out_dir: str, run_id: str, episode_id: int, result: EpisodeResult) -> Tuple[str, str]:
    """Write episode metadata and logs.

    Args:
        out_dir: directory to write files into
        run_id: identifier for this run batch
        episode_id: integer episode index within run
        result: EpisodeResult from env

    Returns:
        (meta_path, logs_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, f"{run_id}_ep{episode_id}_meta.json")
    logs_path = os.path.join(out_dir, f"{run_id}_ep{episode_id}_logs.jsonl")

    meta = {
        "run_id": run_id,
        "episode_id": episode_id,
        "schema_v": 1,
        "config": _to_jsonable(result.config),
        "total_rewards": result.total_rewards,
        "final_B": result.final_B,
        "final_streak": result.final_streak,
        "num_rounds": len(result.logs),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(logs_path, "w", encoding="utf-8") as f:
        for rl in result.logs:
            record = _to_jsonable(rl)
            # add identifiers for grouping
            record["run_id"] = run_id
            record["episode_id"] = episode_id
            record["schema_v"] = 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return meta_path, logs_path