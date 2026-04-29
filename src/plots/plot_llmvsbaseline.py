from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BG = "#faf7f2"
PANEL = "#fffdf9"
GRID = "#d9d3c7"
TEXT = "#1f2430"
MUTED = "#6b7280"

MODEL_COLORS = {
    "DeepSeek-V3.2": "#355070",
    "Qwen3-235B": "#6d597a",
    "GPT-OSS-20B": "#b56576",
    "Gemma-3-27B": "#e56b6f",
    "Llama-3.1-8B": "#2a9d8f",
}

MODEL_MARKERS = {
    "DeepSeek-V3.2": "o",
    "Qwen3-235B": "s",
    "GPT-OSS-20B": "D",
    "Gemma-3-27B": "^",
    "Llama-3.1-8B": "P",
}

MODEL_LINESTYLES = {
    "DeepSeek-V3.2": "-",
    "Qwen3-235B": "-",
    "GPT-OSS-20B": "-.",
    "Gemma-3-27B": "--",
    "Llama-3.1-8B": ":",
}

ENDPOINT_OFFSETS = {
    "DeepSeek-V3.2": 0.030,
    "Qwen3-235B": 0.015,
    "GPT-OSS-20B": 0.000,
    "Gemma-3-27B": -0.015,
    "Llama-3.1-8B": -0.030,
}

MODEL_SHORT = {
    "deepseek_v32": "DeepSeek-V3.2",
    "qwen3_235b_a22b_2507": "Qwen3-235B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "gemma3_27b": "Gemma-3-27B",
    "llama31_8b": "Llama-3.1-8B",
}

PRETTY_BASELINE = {
    "always_cooperate": "Always Cooperate",
    "graded_tft": "Graded TFT",
    "wsls": "WSLS",
    "random_uniform": "Random",
    "always_defect": "Always Defect",
    "grim_trigger": "Grim Trigger",
}

MODEL_ORDER = [
    "DeepSeek-V3.2",
    "Qwen3-235B",
    "GPT-OSS-20B",
    "Gemma-3-27B",
    "Llama-3.1-8B",
]

BASELINE_ORDER = [
    "Always Cooperate",
    "Graded TFT",
    "WSLS",
    "Random",
    "Always Defect",
    "Grim Trigger",
]

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": "#d4ccbf",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "font.size": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--rolling-window", type=int, default=5)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_baseline_label(label: str) -> str:
    s = (label or "").strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "alwayscooperate": "always_cooperate",
        "always_cooperate": "always_cooperate",
        "ac": "always_cooperate",
        "alwaysdefect": "always_defect",
        "always_defect": "always_defect",
        "ad": "always_defect",
        "gradedtft": "graded_tft",
        "graded_tft": "graded_tft",
        "grimtrigger": "grim_trigger",
        "grim_trigger": "grim_trigger",
        "grim": "grim_trigger",
        "winstayloseshift": "wsls",
        "win_stay_lose_shift": "wsls",
        "wsls": "wsls",
        "random": "random_uniform",
        "random_uniform": "random_uniform",
        "randomagent": "random_uniform",
    }

    return aliases.get(s, s)


def infer_seed(match_dir: Path, manifest: dict, meta: dict) -> Optional[int]:
    for obj in [manifest, meta]:
        for key in ["seed", "rng_seed", "episode_seed"]:
            if key in obj:
                try:
                    return int(obj[key])
                except Exception:
                    pass

    m = re.search(r"(?:^|__)s(\d+)(?:__|$)", match_dir.name)
    if m:
        return int(m.group(1))

    m = re.search(r"seed[_-]?(\d+)", match_dir.name, flags=re.I)
    if m:
        return int(m.group(1))

    return None


def infer_focal_and_baseline(manifest: dict) -> Tuple[str, str, int, int]:
    focal = manifest.get("focal_player", {})
    baseline = manifest.get("baseline_player", {}) or manifest.get("opponent_player", {})

    focal_label = focal.get("label") or manifest.get("focal_player_label")
    baseline_label = (
        baseline.get("label")
        or baseline.get("strategy")
        or manifest.get("opponent_player_label")
    )

    if focal_label is None or baseline_label is None:
        players = manifest.get("players", [])
        if len(players) == 2:
            llm_idx = 0
            for i, p in enumerate(players):
                if str(p.get("kind", "")).lower() == "llm":
                    llm_idx = i
                    break

            focal_label = players[llm_idx].get("label")
            baseline_label = (
                players[1 - llm_idx].get("label")
                or players[1 - llm_idx].get("strategy")
            )

            return (
                str(focal_label),
                normalize_baseline_label(str(baseline_label)),
                llm_idx,
                1 - llm_idx,
            )

        raise KeyError("Could not infer focal/baseline labels.")

    if "focal_seat" in manifest:
        focal_seat = int(manifest["focal_seat"])
    else:
        row = manifest.get("row_player", {})
        focal_seat = 0 if row.get("label") == focal_label else 1

    return (
        str(focal_label),
        normalize_baseline_label(str(baseline_label)),
        focal_seat,
        1 - focal_seat,
    )


def coop_from_row(row: dict, seat: int) -> float:
    if "true_coop" in row:
        return float(row["true_coop"][seat])

    if "true_actions" in row:
        action = int(row["true_actions"][seat])

        if "action_semantics" in row:
            mapping = row["action_semantics"].get("index_to_cooperation")
            if mapping is not None:
                return float(mapping[action])

        game_params = row.get("game_parameters", {})
        semantics = game_params.get("action_semantics", {})
        mapping = semantics.get("index_to_cooperation")
        if mapping is not None:
            return float(mapping[action])

        return 1.0 if action == 0 else 0.0

    raise KeyError("No true_coop or true_actions found.")


def reward_from_row(row: dict, seat: int) -> Optional[float]:
    for key in ["rewards", "reward", "payoffs", "true_rewards"]:
        if key not in row:
            continue

        val = row[key]

        if isinstance(val, list):
            return float(val[seat])

        if isinstance(val, dict):
            if str(seat) in val:
                return float(val[str(seat)])
            if seat in val:
                return float(val[seat])

    return None


def load_run_rounds(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for match_dir in sorted(run_dir.iterdir()):
        if not match_dir.is_dir():
            continue

        manifest_path = match_dir / "match_manifest.json"
        meta_path = match_dir / "episode_meta.json"
        logs_path = match_dir / "episode_logs.jsonl"

        if not (manifest_path.exists() and meta_path.exists() and logs_path.exists()):
            continue

        manifest = load_json(manifest_path)
        meta = load_json(meta_path)
        logs = load_jsonl(logs_path)

        model_raw, baseline_raw, focal_seat, _ = infer_focal_and_baseline(manifest)

        model = MODEL_SHORT.get(model_raw, model_raw)
        baseline = PRETTY_BASELINE.get(baseline_raw, baseline_raw)
        seed = infer_seed(match_dir, manifest, meta)

        total_rewards = meta.get("total_rewards", [])
        num_rounds = int(meta.get("num_rounds", len(logs)))

        reward_series = [reward_from_row(r, focal_seat) for r in logs]

        if not all(r is not None for r in reward_series):
            if len(total_rewards) > focal_seat and num_rounds > 0:
                avg_reward = float(total_rewards[focal_seat]) / num_rounds
                reward_series = [avg_reward for _ in logs]
            else:
                reward_series = [np.nan for _ in logs]

        cumulative_reward = 0.0

        for t, log_row in enumerate(logs, start=1):
            reward_t = float(reward_series[t - 1])
            cumulative_reward += 0.0 if np.isnan(reward_t) else reward_t

            rows.append(
                {
                    "match_id": match_dir.name,
                    "model": model,
                    "baseline": baseline,
                    "seed": seed,
                    "round": t,
                    "coop": coop_from_row(log_row, focal_seat),
                    "reward": reward_t,
                    "cumulative_reward": cumulative_reward,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No valid match folders found under {run_dir}")

    return pd.DataFrame(rows)


def ordered_models(df: pd.DataFrame) -> List[str]:
    present = set(df["model"].dropna())
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def ordered_baselines(df: pd.DataFrame) -> List[str]:
    present = set(df["baseline"].dropna())
    return [b for b in BASELINE_ORDER if b in present] + sorted(present - set(BASELINE_ORDER))


def style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=GRID, linewidth=0.9, alpha=0.6)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def save_summary_tables(round_df: pd.DataFrame, out_dir: Path) -> None:
    baselines = ordered_baselines(round_df)
    models = ordered_models(round_df)

    coop_table = (
        round_df
        .groupby(["baseline", "model"])["coop"]
        .mean()
        .reset_index()
        .pivot(index="baseline", columns="model", values="coop")
        .reindex(index=baselines, columns=models)
    )

    reward_table = (
        round_df
        .groupby(["baseline", "model"])["reward"]
        .mean()
        .reset_index()
        .pivot(index="baseline", columns="model", values="reward")
        .reindex(index=baselines, columns=models)
    )

    coop_table.to_csv(out_dir / "table_average_cooperation_rate.csv")
    reward_table.to_csv(out_dir / "table_average_reward_per_round.csv")

    coop_table.round(3).to_latex(out_dir / "table_average_cooperation_rate.tex")
    reward_table.round(3).to_latex(out_dir / "table_average_reward_per_round.tex")

    print("\nAverage cooperation rate:")
    print(coop_table.round(3))

    print("\nAverage reward per round:")
    print(reward_table.round(3))


def detect_overlap_groups(final_values: Dict[str, float], tol: float = 0.03) -> List[List[str]]:
    used = set()
    groups = []

    for model, val in final_values.items():
        if model in used:
            continue

        group = [model]
        used.add(model)

        for other, other_val in final_values.items():
            if other in used:
                continue
            if abs(val - other_val) <= tol:
                group.append(other)
                used.add(other)

        if len(group) >= 2:
            groups.append(group)

    return groups


def plot_one_baseline(
    round_df: pd.DataFrame,
    baseline: str,
    out_path: Path,
    rolling_window: int,
) -> None:
    df = round_df[round_df["baseline"] == baseline].copy()

    df["smooth_coop"] = (
        df.groupby("match_id")["coop"]
        .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    )

    fig, ax = plt.subplots(figsize=(14.8, 7.2))
    style_ax(ax)

    models = ordered_models(df)
    final_values: Dict[str, float] = {}

    for model in models:
        sub = df[df["model"] == model]

        agg = (
            sub.groupby("round", as_index=False)
            .agg(
                mean=("smooth_coop", "mean"),
                std=("smooth_coop", "std"),
            )
            .fillna({"std": 0.0})
        )

        x = agg["round"].to_numpy()
        y = agg["mean"].to_numpy()
        sd = agg["std"].to_numpy()

        color = MODEL_COLORS.get(model, "#444")
        marker = MODEL_MARKERS.get(model, "o")
        linestyle = MODEL_LINESTYLES.get(model, "-")

        final_values[model] = float(y[-1])

        ax.plot(
            x,
            y,
            color=color,
            linewidth=3.0,
            linestyle=linestyle,
            marker=marker,
            markevery=max(1, len(x) // 8),
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.1,
            label=model,
            zorder=5,
        )

        ax.fill_between(
            x,
            np.clip(y - sd, 0, 1),
            np.clip(y + sd, 0, 1),
            color=color,
            alpha=0.10,
            zorder=2,
        )

        final_x = x[-1]
        final_y = y[-1] + ENDPOINT_OFFSETS.get(model, 0.0)

        ax.text(
            final_x + 0.7,
            np.clip(final_y, 0.025, 1.0),
            f"{model}: {y[-1]:.2f}",
            fontsize=9,
            color=color,
            va="center",
            ha="left",
            fontweight="bold",
        )

    overlap_groups = detect_overlap_groups(final_values, tol=0.03)

    if overlap_groups:
        msg_parts = []
        for g in overlap_groups:
            if len(g) <= 3:
                msg_parts.append(" + ".join(g))
            else:
                msg_parts.append(f"{len(g)} models")

        ax.text(
            0.985,
            0.04,
            "Endpoint overlap: " + " | ".join(msg_parts),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color=MUTED,
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor=BG,
                edgecolor="#e5d9c8",
                alpha=0.96,
            ),
        )

    ax.set_xlim(1, int(df["round"].max()) + 8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Round")
    ax.set_ylabel("Rolling cooperation rate")
    ax.set_title(f"LLM behavior against {baseline}", pad=28)

    ax.text(
        0.5,
        1.02,
        f"Each line is one model; shaded band shows variation across seeds. Rolling window = {rolling_window}.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color=MUTED,
    )

    ax.legend(
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=10.5,
    )

    fig.subplots_adjust(left=0.08, right=0.86, top=0.86, bottom=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots_by_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    round_df = load_run_rounds(run_dir)
    round_df.to_csv(out_dir / "round_level_cooperation_reward.csv", index=False)

    save_summary_tables(round_df, out_dir)

    for baseline in ordered_baselines(round_df):
        safe = baseline.lower().replace(" ", "_")
        plot_one_baseline(
            round_df=round_df,
            baseline=baseline,
            out_path=out_dir / f"coop_trajectory_vs_{safe}.png",
            rolling_window=args.rolling_window,
        )

    print(f"\nSaved baseline-wise plots and tables to: {out_dir}")
    print(f"Loaded round rows: {len(round_df)}")
    print(f"Models: {round_df['model'].nunique()}")
    print(f"Baselines: {round_df['baseline'].nunique()}")


if __name__ == "__main__":
    main()