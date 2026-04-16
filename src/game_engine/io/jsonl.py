"""
io/jsonl.py

Purpose:
- Save episode metadata and per-round logs as JSONL.
- Keep env free of I/O and enable reproducible analysis.

Return values:
- write_episode(...) writes files and returns their paths.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from game_engine.env.types import EpisodeResult
from game_engine.io.run_paths import ensure_dir, to_jsonable, write_json


def write_episode(
    out_dir: str,
    run_id: str,
    episode_id: int,
    result: EpisodeResult,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
    stem: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Write episode metadata and logs.

    Args:
        out_dir: directory to write files into
        run_id: identifier for this run batch
        episode_id: integer episode index within run
        result: EpisodeResult from env
        extra_meta: optional experiment/match metadata
        stem: optional file stem; defaults to episode_{episode_id:03d}

    Returns:
        (meta_path, logs_path)
    """
    ensure_dir(out_dir)

    stem = stem or f"episode_{episode_id:03d}"
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")
    logs_path = os.path.join(out_dir, f"{stem}_logs.jsonl")

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "episode_id": episode_id,
        "schema_v": 2,
        "config": to_jsonable(result.config),
        "total_rewards": result.total_rewards,
        "final_B": result.final_B,
        "final_streak": result.final_streak,
        "num_rounds": len(result.logs),
    }
    if extra_meta:
        meta["extra_meta"] = to_jsonable(extra_meta)

    write_json(meta_path, meta)

    with open(logs_path, "w", encoding="utf-8") as f:
        for rl in result.logs:
            record = to_jsonable(rl)
            record["run_id"] = run_id
            record["episode_id"] = episode_id
            record["schema_v"] = 2
            if extra_meta:
                record["context"] = to_jsonable(extra_meta)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return meta_path, logs_path