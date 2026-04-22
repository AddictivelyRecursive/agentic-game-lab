from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------
# Styling
# ----------------------------

BG = "#faf7f2"
PANEL = "#fffdf9"
GRID = "#d9d3c7"
TEXT = "#1f2430"
MUTED = "#6b7280"

BACKBONE = "#27364a"
RANDOM_HIGHLIGHT = "#f4d58d"

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

MODEL_SHORT = {
    "deepseek_v32": "DeepSeek-V3.2",
    "qwen3_235b_a22b_2507": "Qwen3-235B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "gemma3_27b": "Gemma-3-27B",
    "llama31_8b": "Llama-3.1-8B",
}

BASELINE_ORDER = [
    "Always Cooperate",
    "Graded TFT",
    "WSLS",
    "★ Random",
    "Always Defect",
    "Grim Trigger",
]

BASELINE_SHORT = {
    "Always Cooperate": "Always\nCooperate",
    "Graded TFT": "Graded\nTFT",
    "WSLS": "WSLS",
    "★ Random": "★ Random",
    "Always Defect": "Always\nDefect",
    "Grim Trigger": "Grim\nTrigger",
}

COMMON_RESPONSE_PATTERN = {
    "always_cooperate": "Cooperate throughout",
    "always_defect": "Cooperate once, then defect",
    "graded_tft": "Cooperate throughout",
    "grim_trigger": "Standard reciprocal / grim-style response",
    "wsls": "Stable cooperation",
    "random_uniform": "Mixed / random retaliation",
}

PRETTY_BASELINE = {
    "always_cooperate": "Always Cooperate",
    "graded_tft": "Graded TFT",
    "wsls": "WSLS",
    "random_uniform": "★ Random",
    "always_defect": "Always Defect",
    "grim_trigger": "Grim Trigger",
}

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
        "axes.titlesize": 17,
        "axes.labelsize": 13,
        "font.size": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse LLM-vs-baseline run outputs and generate a cleaned overlap plot."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


# ----------------------------
# I/O
# ----------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ----------------------------
# Normalization / parsing
# ----------------------------

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
        raise KeyError("Could not infer focal/baseline labels from match manifest.")

    if "focal_seat" in manifest:
        focal_seat = int(manifest["focal_seat"])
    else:
        row = manifest.get("row_player", {})
        focal_seat = 0 if row.get("label") == focal_label else 1

    baseline_seat = 1 - focal_seat
    return str(focal_label), normalize_baseline_label(str(baseline_label)), focal_seat, baseline_seat


def coop_from_log_row(log_row: dict, seat: int) -> float:
    if "true_coop" in log_row:
        return float(log_row["true_coop"][seat])

    if "true_actions" in log_row:
        action = int(log_row["true_actions"][seat])
        return 1.0 if action == 0 else 0.0

    raise KeyError("Neither true_coop nor true_actions found in log row.")


def action_from_log_row(log_row: dict, seat: int) -> int:
    if "true_actions" in log_row:
        return int(log_row["true_actions"][seat])

    if "true_coop" in log_row:
        return 0 if float(log_row["true_coop"][seat]) >= 0.5 else 1

    raise KeyError("Neither true_coop nor true_actions found in log row.")


def first_defection_round(actions: List[int]) -> float:
    for i, a in enumerate(actions, start=1):
        if int(a) == 1:
            return float(i)
    return np.nan


# ----------------------------
# Load raw experiment outputs
# ----------------------------

def load_match_level_dataframe(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue

        manifest_path = child / "match_manifest.json"
        meta_path = child / "episode_meta.json"
        logs_path = child / "episode_logs.jsonl"

        if not (manifest_path.exists() and meta_path.exists() and logs_path.exists()):
            continue

        manifest = load_json(manifest_path)
        meta = load_json(meta_path)
        logs = load_jsonl(logs_path)

        model_label, baseline_label, focal_seat, _ = infer_focal_and_baseline(manifest)

        coop_series = [coop_from_log_row(r, focal_seat) for r in logs]
        action_series = [action_from_log_row(r, focal_seat) for r in logs]

        total_rewards = meta.get("total_rewards", [])
        num_rounds = int(meta.get("num_rounds", len(logs)))
        llm_total_reward = float(total_rewards[focal_seat]) if len(total_rewards) > focal_seat else np.nan
        mean_payoff = llm_total_reward / num_rounds if num_rounds > 0 else np.nan

        rows.append(
            {
                "match_dir": str(child),
                "model_raw": model_label,
                "baseline_raw": baseline_label,
                "model": MODEL_SHORT.get(model_label, model_label),
                "baseline": PRETTY_BASELINE.get(baseline_label, baseline_label),
                "common_response_pattern": COMMON_RESPONSE_PATTERN.get(baseline_label, "—"),
                "mean_coop": float(np.mean(coop_series)) if coop_series else np.nan,
                "mean_payoff": mean_payoff,
                "first_defection_round": first_defection_round(action_series),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No valid match folders found under {run_dir}")

    return pd.DataFrame(rows)


def build_summary_df(match_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        match_df.groupby(
            ["model", "baseline", "common_response_pattern"],
            as_index=False,
        )
        .agg(
            mean_coop=("mean_coop", "mean"),
            mean_payoff=("mean_payoff", "mean"),
            median_first_defection_round=("first_defection_round", "median"),
        )
    )

    order_map = {b: i for i, b in enumerate(BASELINE_ORDER)}
    summary["baseline_order"] = summary["baseline"].map(order_map)
    summary = summary.sort_values(["model", "baseline_order"]).reset_index(drop=True)
    return summary


# ----------------------------
# Plot helpers
# ----------------------------

def style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#d4ccbf")
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8, alpha=0.55)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def summarize_clusters(random_df: pd.DataFrame) -> List[Tuple[float, int, float]]:
    grp = (
        random_df.groupby("mean_coop", as_index=False)
        .agg(
            n_models=("model", "count"),
            payoff=("mean_payoff", "mean"),
        )
        .sort_values("mean_coop", ascending=False)
    )
    return [(float(r["mean_coop"]), int(r["n_models"]), float(r["payoff"])) for _, r in grp.iterrows()]


# ----------------------------
# Main plot
# ----------------------------

def plot_consensus_breakout(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    style_ax(ax)

    present_baselines = [b for b in BASELINE_ORDER if b in set(df["baseline"])]
    x_map = {b: i for i, b in enumerate(present_baselines)}

    if "★ Random" in x_map:
        xr = x_map["★ Random"]
        ax.axvspan(xr - 0.40, xr + 0.40, color=RANDOM_HIGHLIGHT, alpha=0.18, zorder=0)

    consensus = (
        df.groupby("baseline", as_index=False)
        .agg(
            mean_coop=("mean_coop", "mean"),
            min_coop=("mean_coop", "min"),
            max_coop=("mean_coop", "max"),
        )
    )
    consensus["x"] = consensus["baseline"].map(x_map)
    consensus = consensus.sort_values("x")

    ax.fill_between(
        consensus["x"],
        consensus["min_coop"],
        consensus["max_coop"],
        color=BACKBONE,
        alpha=0.08,
        zorder=1,
    )

    ax.plot(
        consensus["x"],
        consensus["mean_coop"],
        color=BACKBONE,
        linewidth=4.8,
        marker="o",
        markersize=8,
        markerfacecolor=BACKBONE,
        markeredgecolor="white",
        markeredgewidth=1.3,
        solid_capstyle="round",
        zorder=3,
    )

    if "★ Random" in x_map:
        random_df = df[df["baseline"] == "★ Random"].copy().reset_index(drop=True)
        jitter = np.linspace(-0.09, 0.09, max(len(random_df), 2))

        for j, (_, row) in enumerate(random_df.iterrows()):
            model = row["model"]
            x = x_map["★ Random"] + jitter[j]
            y = row["mean_coop"]

            ax.scatter(
                x,
                y,
                s=145,
                color=MODEL_COLORS.get(model, "#355070"),
                marker=MODEL_MARKERS.get(model, "o"),
                edgecolor="white",
                linewidth=1.5,
                zorder=6,
            )

    cooperative_block = [b for b in ["Always Cooperate", "Graded TFT", "WSLS"] if b in x_map]
    if cooperative_block:
        xs = [x_map[b] for b in cooperative_block]
        ax.text(
            np.mean(xs),
            0.91,
            "5/5 identical",
            ha="center",
            va="center",
            fontsize=11.5,
            color=MUTED,
            bbox=dict(boxstyle="round,pad=0.24", facecolor=BG, edgecolor="none", alpha=0.96),
            zorder=7,
        )

    defect_block = [b for b in ["Always Defect", "Grim Trigger"] if b in x_map]
    if defect_block:
        xs = [x_map[b] for b in defect_block]
        ax.text(
            np.mean(xs),
            0.095,
            "5/5 identical  •  first defect ≈ 2",
            ha="center",
            va="center",
            fontsize=11.2,
            color=MUTED,
            bbox=dict(boxstyle="round,pad=0.24", facecolor=BG, edgecolor="none", alpha=0.96),
            zorder=7,
        )

    if "★ Random" in x_map:
        random_df = df[df["baseline"] == "★ Random"].copy()
        clusters = summarize_clusters(random_df)
        xr = x_map["★ Random"]

        if len(clusters) >= 1:
            coop, n_models, payoff = clusters[0]
            ax.annotate(
                f"{n_models} models: coop {coop:.2f}, payoff {payoff:.2f}",
                xy=(xr, coop),
                xytext=(xr + 0.52, min(coop + 0.10, 0.60)),
                fontsize=10.6,
                color=TEXT,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=BG, edgecolor="#e5d9c8", alpha=0.98),
                arrowprops=dict(arrowstyle="-", color="#bfae8b", lw=1.2),
                zorder=8,
            )

        if len(clusters) >= 2:
            coop, n_models, payoff = clusters[1]
            ax.annotate(
                f"{n_models} models: coop {coop:.2f}, payoff {payoff:.2f}",
                xy=(xr, coop),
                xytext=(xr + 0.52, max(coop - 0.04, 0.08)),
                fontsize=10.6,
                color=TEXT,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=BG, edgecolor="#e5d9c8", alpha=0.98),
                arrowprops=dict(arrowstyle="-", color="#bfae8b", lw=1.2),
                zorder=8,
            )

        ax.text(
            xr,
            0.02,
            "only baseline showing separation",
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="#9c6b00",
            zorder=8,
        )

    ax.set_xlim(-0.35, len(present_baselines) - 0.20)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("Mean cooperation rate")
    ax.set_xlabel("Baseline strategy")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(range(len(present_baselines)))
    ax.set_xticklabels([BASELINE_SHORT.get(b, b) for b in present_baselines])

    for tick, baseline in zip(ax.get_xticklabels(), present_baselines):
        if baseline == "★ Random":
            tick.set_color("#9c6b00")
            tick.set_fontweight("bold")

    ax.set_title("Consensus and breakout of LLM responses across baselines", pad=36)
    ax.text(
        0.5,
        1.022,
        "Models overlap almost perfectly except against Random.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=MUTED,
    )

    handles = []
    labels = []
    ordered_models = [m for m in MODEL_COLORS if m in set(df["model"])]
    for model in ordered_models:
        h = plt.Line2D(
            [0], [0],
            marker=MODEL_MARKERS.get(model, "o"),
            color="none",
            markerfacecolor=MODEL_COLORS.get(model, "#355070"),
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=10,
            label=model,
        )
        handles.append(h)
        labels.append(model)

    fig.legend(
        handles,
        labels,
        ncol=min(5, len(labels)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10.5,
    )

    fig.text(
        0.985,
        0.065,
        "Backbone = mean across models",
        ha="right",
        va="center",
        fontsize=9.8,
        color=MUTED,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots_llm_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    match_df = load_match_level_dataframe(run_dir)
    summary_df = build_summary_df(match_df)

    summary_csv = out_dir / "llm_vs_baseline_summary.csv"
    plot_png = out_dir / "llm_overlap_consensus_breakout.png"

    summary_df[
        [
            "model",
            "baseline",
            "common_response_pattern",
            "mean_coop",
            "mean_payoff",
            "median_first_defection_round",
        ]
    ].to_csv(summary_csv, index=False)

    plot_consensus_breakout(summary_df, plot_png)

    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved plot to: {plot_png}")
    print(f"Loaded {len(match_df)} matches.")
    print(f"Detected {match_df['model'].nunique()} focal models and {match_df['baseline'].nunique()} baselines.")


if __name__ == "__main__":
    main()