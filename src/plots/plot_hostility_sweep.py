from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap


# ----------------------------
# Configuration / styling
# ----------------------------

BG = "#faf7f2"
PANEL = "#fffdf9"
GRID = "#d9d3c7"
TEXT = "#1f2430"
MUTED = "#6b7280"

# Distinct palette from the paper: earthy + jewel tones
MODEL_COLORS = {
    "deepseek_v32": "#355070",
    "qwen3_235b_a22b_2507": "#6d597a",
    "gpt_oss_20b": "#b56576",
    "gemma3_27b": "#e56b6f",
    "llama31_8b": "#2a9d8f",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_hostility",
    ["#f7f3ec", "#e6d7c3", "#c89f9c", "#7b6d8d", "#355070"],
)

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
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "font.size": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


# ----------------------------
# I/O helpers
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plots for hostility sweep.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()

def fmt_float(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{nd}f}"


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


# ----------------------------
# Metrics
# ----------------------------

def compute_behavior_metrics(logs: List[dict]) -> Dict[str, float]:
    if not logs:
        return {
            "nice": np.nan,
            "retaliatory": np.nan,
            "forgiving": np.nan,
            "switch_rate": np.nan,
        }

    llm_actions = [int(r["true_actions"][0]) for r in logs]
    opp_actions = [int(r["true_actions"][1]) for r in logs]

    nice = 1.0 if llm_actions[0] == 0 else 0.0

    ret_num, ret_den = 0, 0
    forg_num, forg_den = 0, 0
    switches = 0

    for t in range(1, len(logs)):
        if llm_actions[t] != llm_actions[t - 1]:
            switches += 1

        if opp_actions[t - 1] == 1:
            ret_den += 1
            if llm_actions[t] == 1:
                ret_num += 1

        if llm_actions[t - 1] == 1 and opp_actions[t - 1] == 0:
            forg_den += 1
            if llm_actions[t] == 0:
                forg_num += 1

    retaliatory = ret_num / ret_den if ret_den > 0 else np.nan
    forgiving = forg_num / forg_den if forg_den > 0 else np.nan
    switch_rate = switches / (len(logs) - 1) if len(logs) > 1 else 0.0

    return {
        "nice": nice,
        "retaliatory": retaliatory,
        "forgiving": forgiving,
        "switch_rate": switch_rate,
    }


def summarize_match(match_dir: Path) -> Dict[str, float]:
    match_manifest = load_json(match_dir / "match_manifest.json")
    episode_meta = load_json(match_dir / "episode_meta.json")
    logs = load_jsonl(match_dir / "episode_logs.jsonl")

    focal_label = match_manifest["focal_player"]["label"]
    model_id = match_manifest["focal_player"]["model_name"]
    opp_label = match_manifest["opponent_player"]["label"]
    opp_p = float(match_manifest["opponent_player"]["p_cooperate"])
    seed = int(match_manifest["seed"])

    llm_total_reward = float(episode_meta["total_rewards"][0])
    opp_total_reward = float(episode_meta["total_rewards"][1])
    num_rounds = int(episode_meta["num_rounds"])

    llm_coop = float(np.mean([row["true_coop"][0] for row in logs])) if logs else np.nan
    opp_coop = float(np.mean([row["true_coop"][1] for row in logs])) if logs else np.nan

    llm_avg_reward = llm_total_reward / num_rounds if num_rounds else np.nan
    opp_avg_reward = opp_total_reward / num_rounds if num_rounds else np.nan
    payoff_gap = llm_avg_reward - opp_avg_reward

    behavior = compute_behavior_metrics(logs)

    return {
        "match_dir": str(match_dir),
        "model_label": focal_label,
        "model_id": model_id,
        "opponent_label": opp_label,
        "opponent_p": opp_p,
        "seed": seed,
        "num_rounds": num_rounds,
        "llm_total_reward": llm_total_reward,
        "opp_total_reward": opp_total_reward,
        "llm_avg_reward": llm_avg_reward,
        "opp_avg_reward": opp_avg_reward,
        "payoff_gap": payoff_gap,
        "llm_coop_rate": llm_coop,
        "opp_coop_rate": opp_coop,
        "nice": behavior["nice"],
        "retaliatory": behavior["retaliatory"],
        "forgiving": behavior["forgiving"],
        "switch_rate": behavior["switch_rate"],
    }


def load_run_dataframe(run_dir: Path) -> pd.DataFrame:
    rows = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "match_manifest.json").exists() and (child / "episode_logs.jsonl").exists():
            rows.append(summarize_match(child))
    if not rows:
        raise FileNotFoundError(f"No match folders found under {run_dir}")
    df = pd.DataFrame(rows)
    df = df.sort_values(["model_label", "opponent_p", "seed"]).reset_index(drop=True)
    return df


def aggregate_for_lines(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    grp = (
        df.groupby(["model_label", "opponent_p"], as_index=False)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["sem"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
    return grp


# ----------------------------
# Plot helpers
# ----------------------------

def plot_axes(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#d4ccbf")
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8, alpha=0.55)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def short_label(model_label: str) -> str:
    mapping = {
        "deepseek_v32": "DeepSeek-V3.2",
        "qwen3_235b_a22b_2507": "Qwen3-235B",
        "gpt_oss_20b": "GPT-OSS-20B",
        "gemma3_27b": "Gemma-3-27B",
        "llama31_8b": "Llama-3.1-8B",
    }
    return mapping.get(model_label, model_label)


# ----------------------------
# Plots
# ----------------------------

def plot_cooperation_vs_hostility(df: pd.DataFrame, out_dir: Path) -> None:
    stats = aggregate_for_lines(df, "llm_coop_rate")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    plot_axes(ax)

    for model in stats["model_label"].unique():
        sub = stats[stats["model_label"] == model].sort_values("opponent_p")
        color = MODEL_COLORS.get(model, "#355070")

        ax.plot(
            sub["opponent_p"],
            sub["mean"],
            marker="o",
            markersize=7,
            linewidth=3,
            color=color,
            solid_capstyle="round",
            label=short_label(model),
            path_effects=[pe.Stroke(linewidth=4.2, foreground=PANEL), pe.Normal()],
        )

        if sub["count"].max() > 1:
            lo = sub["mean"] - 1.96 * sub["sem"].fillna(0)
            hi = sub["mean"] + 1.96 * sub["sem"].fillna(0)
            ax.fill_between(sub["opponent_p"], lo, hi, color=color, alpha=0.16)

    ax.set_title("Cooperation response to opponent hostility", pad=14)
    ax.set_xlabel("Opponent cooperation probability")
    ax.set_ylabel("Mean LLM cooperation rate")
    ax.set_xticks(sorted(df["opponent_p"].unique()))
    ax.set_ylim(-0.02, 1.02)

    leg = ax.legend(ncol=2, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    ax.text(
        0.99,
        -0.18,
        "Hostility sweep • 100 rounds per match",
        ha="right",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )

    save_fig(fig, out_dir / "cooperation_vs_hostility.png")


def plot_payoff_vs_hostility(df: pd.DataFrame, out_dir: Path) -> None:
    stats = aggregate_for_lines(df, "llm_avg_reward")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    plot_axes(ax)

    for model in stats["model_label"].unique():
        sub = stats[stats["model_label"] == model].sort_values("opponent_p")
        color = MODEL_COLORS.get(model, "#355070")

        ax.plot(
            sub["opponent_p"],
            sub["mean"],
            marker="o",
            markersize=7,
            linewidth=3,
            color=color,
            solid_capstyle="round",
            label=short_label(model),
            path_effects=[pe.Stroke(linewidth=4.2, foreground=PANEL), pe.Normal()],
        )

        if sub["count"].max() > 1:
            lo = sub["mean"] - 1.96 * sub["sem"].fillna(0)
            hi = sub["mean"] + 1.96 * sub["sem"].fillna(0)
            ax.fill_between(sub["opponent_p"], lo, hi, color=color, alpha=0.16)

    ax.set_title("Average payoff across hostility conditions", pad=14)
    ax.set_xlabel("Opponent cooperation probability")
    ax.set_ylabel("Average reward per round")
    ax.set_xticks(sorted(df["opponent_p"].unique()))

    leg = ax.legend(ncol=2, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    ax.text(
        0.99,
        -0.18,
        "Higher is better",
        ha="right",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )

    save_fig(fig, out_dir / "payoff_vs_hostility.png")


def draw_heatmap(matrix_df: pd.DataFrame, title: str, out_path: Path, vmin=None, vmax=None) -> None:
    row_labels = [short_label(x) for x in matrix_df.index]
    col_labels = [f"{x:.1f}" for x in matrix_df.columns]
    values = matrix_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_facecolor(PANEL)

    im = ax.imshow(values, aspect="auto", cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Opponent cooperation probability")
    ax.set_ylabel("Model")

    ax.tick_params(length=0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if val > np.nanmean(values) else TEXT,
                fontweight="bold",
            )

    for i in range(values.shape[0] + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=2)
    for j in range(values.shape[1] + 1):
        ax.axvline(j - 0.5, color=BG, linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=TEXT)

    save_fig(fig, out_path)


def plot_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    payoff_heat = (
        df.pivot_table(index="model_label", columns="opponent_p", values="llm_avg_reward", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    draw_heatmap(
        payoff_heat,
        "Reward landscape",
        out_dir / "heatmap_payoff.png",
    )

    coop_heat = (
        df.pivot_table(index="model_label", columns="opponent_p", values="llm_coop_rate", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    draw_heatmap(
        coop_heat,
        "Cooperation landscape",
        out_dir / "heatmap_cooperation.png",
        vmin=0.0,
        vmax=1.0,
    )

    behavior_cols = ["nice", "retaliatory", "forgiving", "switch_rate"]
    behavior_heat = df.groupby("model_label")[behavior_cols].mean().sort_index()

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    vals = behavior_heat.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=1)

    ax.set_title("Behavioral profile", pad=12)
    ax.set_xticks(np.arange(len(behavior_cols)))
    ax.set_xticklabels(["Nice", "Retaliatory", "Forgiving", "Switching"])
    ax.set_yticks(np.arange(len(behavior_heat.index)))
    ax.set_yticklabels([short_label(x) for x in behavior_heat.index])
    ax.tick_params(length=0)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            val = vals[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if val > 0.58 else TEXT,
                fontweight="bold",
            )

    for i in range(vals.shape[0] + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=2)
    for j in range(vals.shape[1] + 1):
        ax.axvline(j - 0.5, color=BG, linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=TEXT)

    save_fig(fig, out_dir / "heatmap_behavior.png")


def plot_cooperation_payoff_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    plot_axes(ax)

    for model in sorted(df["model_label"].unique()):
        sub = df[df["model_label"] == model].sort_values("opponent_p")
        color = MODEL_COLORS.get(model, "#355070")

        ax.scatter(
            sub["llm_coop_rate"],
            sub["llm_avg_reward"],
            s=120,
            color=color,
            edgecolor=PANEL,
            linewidth=1.5,
            alpha=0.95,
            label=short_label(model),
        )

        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['opponent_p']:.1f}",
                (row["llm_coop_rate"], row["llm_avg_reward"]),
                fontsize=9,
                color=TEXT,
                xytext=(6, 4),
                textcoords="offset points",
            )

    ax.set_title("Cooperation–reward tradeoff", pad=14)
    ax.set_xlabel("Mean cooperation rate")
    ax.set_ylabel("Average reward per round")
    ax.legend(ncol=2, loc="best")

    save_fig(fig, out_dir / "scatter_cooperation_vs_payoff.png")


def plot_small_multiples_trajectories(df: pd.DataFrame, out_dir: Path) -> None:
    models = sorted(df["model_label"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(8.6, 2.4 * len(models)), sharex=True)

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        plot_axes(ax)
        sub = df[df["model_label"] == model].sort_values("opponent_p")

        for _, row in sub.iterrows():
            logs = load_jsonl(Path(row["match_dir"]) / "episode_logs.jsonl")
            coop = pd.Series([r["true_coop"][0] for r in logs], dtype=float)
            rolling = coop.rolling(window=10, min_periods=1).mean()
            p = row["opponent_p"]

            ax.plot(
                np.arange(1, len(rolling) + 1),
                rolling.values,
                linewidth=2.4,
                color=MODEL_COLORS.get(model, "#355070"),
                alpha=0.35 + 0.6 * p,
                label=f"p={p:.1f}",
            )

        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(short_label(model), rotation=0, labelpad=58, va="center")
        ax.legend(ncol=4, loc="upper right", fontsize=9)

    axes[0].set_title("Round-wise cooperation trajectories", pad=12)
    axes[-1].set_xlabel("Round")

    save_fig(fig, out_dir / "trajectories_small_multiples.png")


def write_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "match_level_summary.csv", index=False)

    grouped = (
        df.groupby(["model_label", "opponent_p"], as_index=False)
        .agg(
            llm_coop_rate_mean=("llm_coop_rate", "mean"),
            llm_avg_reward_mean=("llm_avg_reward", "mean"),
            payoff_gap_mean=("payoff_gap", "mean"),
            nice_mean=("nice", "mean"),
            retaliatory_mean=("retaliatory", "mean"),
            forgiving_mean=("forgiving", "mean"),
            switch_rate_mean=("switch_rate", "mean"),
            n=("seed", "count"),
        )
        .sort_values(["model_label", "opponent_p"])
    )
    grouped.to_csv(out_dir / "grouped_summary.csv", index=False)

def build_readme_tables(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns:
    - markdown table aggregated by model x opponent_p
    - markdown table aggregated by model overall
    """
    grouped = (
        df.groupby(["model_label", "opponent_p"], as_index=False)
        .agg(
            coop=("llm_coop_rate", "mean"),
            reward=("llm_avg_reward", "mean"),
            gap=("payoff_gap", "mean"),
            nice=("nice", "mean"),
            retaliatory=("retaliatory", "mean"),
            forgiving=("forgiving", "mean"),
            switching=("switch_rate", "mean"),
            n=("seed", "count"),
        )
        .sort_values(["model_label", "opponent_p"])
    )

    lines = []
    lines.append("| Model | Opponent p | Coop rate | Avg reward | Payoff gap | Nice | Retaliatory | Forgiving | Switch rate | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in grouped.iterrows():
        lines.append(
            f"| {short_label(r['model_label'])} | {r['opponent_p']:.1f} | "
            f"{fmt_float(r['coop'], 3)} | {fmt_float(r['reward'], 3)} | {fmt_float(r['gap'], 3)} | "
            f"{fmt_float(r['nice'], 2)} | {fmt_float(r['retaliatory'], 2)} | "
            f"{fmt_float(r['forgiving'], 2)} | {fmt_float(r['switching'], 2)} | {int(r['n'])} |"
        )
    grouped_md = "\n".join(lines)

    overall = (
        df.groupby("model_label", as_index=False)
        .agg(
            coop=("llm_coop_rate", "mean"),
            reward=("llm_avg_reward", "mean"),
            gap=("payoff_gap", "mean"),
            nice=("nice", "mean"),
            retaliatory=("retaliatory", "mean"),
            forgiving=("forgiving", "mean"),
            switching=("switch_rate", "mean"),
            matches=("match_dir", "count"),
        )
        .sort_values("reward", ascending=False)
    )

    lines2 = []
    lines2.append("| Model | Mean coop | Mean reward | Mean payoff gap | Nice | Retaliatory | Forgiving | Switch rate | Matches |")
    lines2.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in overall.iterrows():
        lines2.append(
            f"| {short_label(r['model_label'])} | {fmt_float(r['coop'], 3)} | "
            f"{fmt_float(r['reward'], 3)} | {fmt_float(r['gap'], 3)} | "
            f"{fmt_float(r['nice'], 2)} | {fmt_float(r['retaliatory'], 2)} | "
            f"{fmt_float(r['forgiving'], 2)} | {fmt_float(r['switching'], 2)} | "
            f"{int(r['matches'])} |"
        )
    overall_md = "\n".join(lines2)

    return grouped_md, overall_md


def build_key_findings(df: pd.DataFrame) -> List[str]:
    findings: List[str] = []

    overall = (
        df.groupby("model_label", as_index=False)
        .agg(
            coop=("llm_coop_rate", "mean"),
            reward=("llm_avg_reward", "mean"),
            nice=("nice", "mean"),
            retaliatory=("retaliatory", "mean"),
            forgiving=("forgiving", "mean"),
            switching=("switch_rate", "mean"),
        )
    )

    if overall.empty:
        return findings

    best_reward = overall.sort_values("reward", ascending=False).iloc[0]
    most_coop = overall.sort_values("coop", ascending=False).iloc[0]
    most_retal = overall.sort_values("retaliatory", ascending=False).iloc[0]
    most_forg = overall.sort_values("forgiving", ascending=False).iloc[0]
    most_switch = overall.sort_values("switching", ascending=False).iloc[0]

    findings.append(
        f"**Best average payoff:** {short_label(best_reward['model_label'])} "
        f"with mean reward **{fmt_float(best_reward['reward'], 3)}** per round."
    )
    findings.append(
        f"**Most cooperative overall:** {short_label(most_coop['model_label'])} "
        f"with mean cooperation **{fmt_float(most_coop['coop'], 3)}**."
    )
    findings.append(
        f"**Most retaliatory:** {short_label(most_retal['model_label'])} "
        f"with retaliatory score **{fmt_float(most_retal['retaliatory'], 2)}**."
    )
    findings.append(
        f"**Most forgiving:** {short_label(most_forg['model_label'])} "
        f"with forgiving score **{fmt_float(most_forg['forgiving'], 2)}**."
    )
    findings.append(
        f"**Most switch-heavy policy:** {short_label(most_switch['model_label'])} "
        f"with switch rate **{fmt_float(most_switch['switching'], 2)}**."
    )

    # Response slope: how strongly cooperation tracks opponent friendliness
    slopes = []
    for model in sorted(df["model_label"].unique()):
        sub = (
            df[df["model_label"] == model]
            .groupby("opponent_p", as_index=False)["llm_coop_rate"]
            .mean()
            .sort_values("opponent_p")
        )
        if len(sub) >= 2:
            slope = np.polyfit(sub["opponent_p"].values, sub["llm_coop_rate"].values, 1)[0]
            slopes.append((model, slope))

    if slopes:
        slopes_df = pd.DataFrame(slopes, columns=["model_label", "slope"])
        most_responsive = slopes_df.sort_values("slope", ascending=False).iloc[0]
        least_responsive = slopes_df.sort_values("slope", ascending=True).iloc[0]
        findings.append(
            f"**Most responsive to opponent friendliness:** {short_label(most_responsive['model_label'])} "
            f"(cooperation slope **{fmt_float(most_responsive['slope'], 3)}** vs opponent p)."
        )
        findings.append(
            f"**Least responsive / flattest reaction curve:** {short_label(least_responsive['model_label'])} "
            f"(slope **{fmt_float(least_responsive['slope'], 3)}**)."
        )

    return findings


def write_readme(run_dir: Path, out_dir: Path, df: pd.DataFrame) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    cfg = manifest.get("config", {})
    num_focals = manifest.get("num_focals", len(df["model_label"].unique()))
    num_urnd = manifest.get("num_urnd_opponents", len(df["opponent_p"].unique()))
    num_seeds = manifest.get("num_seeds", int(df["seed"].nunique()) if "seed" in df.columns else 1)
    run_id = manifest.get("run_id", run_dir.name)

    grouped_md, overall_md = build_readme_tables(df)
    findings = build_key_findings(df)

    finding_block = "\n".join([f"- {x}" for x in findings]) if findings else "- No findings available."

    image_blocks = [
        ("Cooperation vs hostility", "cooperation_vs_hostility_pretty.png"),
        ("Payoff vs hostility", "payoff_vs_hostility_pretty.png"),
        ("Behavioral profile heatmap", "heatmap_behavior_pretty.png"),
        ("Reward landscape heatmap", "heatmap_payoff_pretty.png"),
        ("Cooperation landscape heatmap", "heatmap_cooperation_pretty.png"),
        ("Cooperation–reward tradeoff", "scatter_cooperation_vs_payoff_pretty.png"),
        ("Round-wise cooperation trajectories", "trajectories_small_multiples_pretty.png"),
    ]

    img_md_lines = []
    for title, filename in image_blocks:
        img_md_lines.append(f"### {title}")
        img_md_lines.append(f"![{title}](./{filename})")
        img_md_lines.append("")

    note = (
        "This run contains a single seed, so all figures should be interpreted as "
        "descriptive summaries rather than confidence-interval-based estimates."
        if int(num_seeds) == 1
        else "This run contains multiple seeds, so line plots can be interpreted with uncertainty across repeated episodes."
    )

    md = f"""# Hostility Sweep Results

Auto-generated report for run:

`{run_id}`

## Experiment summary

- **Setup:** 2-player hostility sweep
- **Focal LLMs:** {num_focals}
- **URND opponents:** {num_urnd}
- **Seeds:** {num_seeds}
- **Rounds per match:** {cfg.get("T", "—")}
- **Action levels (M):** {cfg.get("M", "—")}
- **Perception noise:** {cfg.get("p_perception", "—")}
- **Streak lambda:** {cfg.get("streak", {}).get("lam", "—") if isinstance(cfg.get("streak", {}), dict) else "—"}
- **Drift eta:** {cfg.get("drift", {}).get("eta", "—") if isinstance(cfg.get("drift", {}), dict) else "—"}

## Interpretation note

{note}

## Key findings

{finding_block}

## Main figures

{chr(10).join(img_md_lines)}

## Aggregate performance by model

{overall_md}

## Detailed results by model × opponent hostility

{grouped_md}

## Files generated

- `match_level_summary.csv`
- `grouped_summary.csv`
- `cooperation_vs_hostility_pretty.png`
- `payoff_vs_hostility_pretty.png`
- `heatmap_behavior_pretty.png`
- `heatmap_payoff_pretty.png`
- `heatmap_cooperation_pretty.png`
- `scatter_cooperation_vs_payoff_pretty.png`
- `trajectories_small_multiples_pretty.png`

---
Generated automatically by `plot_hostility_sweep_pretty.py`
"""
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(md)
        

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots_hostility"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_run_dataframe(run_dir)

    write_summary_tables(df, out_dir)
    plot_cooperation_vs_hostility(df, out_dir)
    plot_payoff_vs_hostility(df, out_dir)
    plot_heatmaps(df, out_dir)
    plot_cooperation_payoff_scatter(df, out_dir)
    plot_small_multiples_trajectories(df, out_dir)

    write_readme(run_dir, out_dir, df)
    print(f"Saved plots to: {out_dir}")
    print(f"Loaded {len(df)} match summaries.")


if __name__ == "__main__":
    main()