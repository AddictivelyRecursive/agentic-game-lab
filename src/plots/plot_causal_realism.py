#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIR_RE = re.compile(
    r"__n(?P<N>\d+)__m(?P<M>\d+)__p(?P<p>\d+\.\d+)__l(?P<lam>\d+\.\d+)__s(?P<seed>\d+)$"
)


def parse_dir_name(name: str) -> dict:
    m = DIR_RE.search(name)
    if not m:
        raise ValueError(f"Could not parse directory name: {name}")
    d = m.groupdict()
    return {
        "N": int(d["N"]),
        "M": int(d["M"]),
        "p": float(d["p"]),
        "lam": float(d["lam"]),
        "seed": int(d["seed"]),
    }


LEVEL_CMAPS = {
    "coop": "viridis",
    "reward": "plasma",
    "switch": "magma",
    "gtft": "cividis",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



def load_match(match_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    parsed = parse_dir_name(match_dir.name)
    logs_path = match_dir / "episode_logs.jsonl"
    manifest_path = match_dir / "match_manifest.json"
    meta_path = match_dir / "episode_meta.json"

    logs = load_jsonl(logs_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    meta = load_json(meta_path) if meta_path.exists() else {}
    if not logs:
        raise ValueError(f"No logs in {match_dir}")

    first = logs[0]
    ctx = first.get("context", {})

    players = manifest.get("players", [])
    if players:
        seat_to_label = {p["seat"]: p["label"] for p in players}
        seat_to_kind = {p["seat"]: p.get("kind", "unknown") for p in players}
        seat_to_short = {p["seat"]: (p.get("label") or f"p{p['seat']}") for p in players}
    else:
        labels = ctx.get("lineup_labels", [])
        kinds = ctx.get("lineup_kinds", [])
        seat_to_label = {i: labels[i] for i in range(len(labels))}
        seat_to_kind = {i: kinds[i] if i < len(kinds) else "unknown" for i in range(len(labels))}
        seat_to_short = {i: labels[i] for i in range(len(labels))}

    gtft_seat = None
    for seat, label in seat_to_label.items():
        if label == "graded_tft":
            gtft_seat = seat
            break

    round_rows: List[dict] = []
    for row in logs:
        t = row["t"]
        true_actions = row["true_actions"]
        true_coop = row["true_coop"]
        rewards = row["rewards"]
        obs_actions = row.get("obs_actions", [None] * len(true_actions))
        obs_coop = row.get("obs_coop", [None] * len(true_actions))

        for seat in range(len(true_actions)):
            round_rows.append(
                {
                    "match_dir": match_dir.name,
                    "t": t,
                    "seat": seat,
                    "agent": seat_to_label.get(seat, f"seat_{seat}"),
                    "kind": seat_to_kind.get(seat, "unknown"),
                    "action": true_actions[seat],
                    "coop": true_coop[seat],
                    "obs_action": obs_actions[seat],
                    "obs_coop": obs_coop[seat],
                    "reward": rewards[seat],
                    "B_eff": row.get("B_eff"),
                    "B_base": row.get("B_base"),
                    "obs_coop_mean": row.get("obs_coop_mean"),
                    "r_obs_prev": row.get("r_obs_prev"),
                    "r_obs_next": row.get("r_obs_next"),
                    "streak_prev": row.get("streak_prev"),
                    "streak_next": row.get("streak_next"),
                    "gtft_action": true_actions[gtft_seat] if gtft_seat is not None else None,
                    **parsed,
                }
            )

    round_df = pd.DataFrame(round_rows).sort_values(["match_dir", "seat", "t"]).reset_index(drop=True)
    round_df["switched"] = (
        round_df.groupby(["match_dir", "seat"])["action"].diff().fillna(0).ne(0).astype(float)
    )
    if gtft_seat is not None:
        round_df["matches_gtft"] = (round_df["action"] == round_df["gtft_action"]).astype(float)
    else:
        round_df["matches_gtft"] = np.nan

    summary = (
        round_df.groupby(["match_dir", "M", "p", "lam", "seed", "seat", "agent", "kind"], as_index=False)
        .agg(
            mean_coop=("coop", "mean"),
            mean_reward=("reward", "mean"),
            total_reward=("reward", "sum"),
            switch_rate=("switched", "mean"),
            gtft_match_rate=("matches_gtft", "mean"),
            final_action=("action", "last"),
            final_coop=("coop", "last"),
        )
    )

    match_summary = (
        round_df.groupby(["match_dir", "M", "p", "lam", "seed"], as_index=False)
        .agg(
            avg_group_coop=("coop", "mean"),
            avg_group_reward=("reward", "mean"),
            avg_obs_group_coop=("obs_coop", "mean"),
            final_B_eff=("B_eff", "last"),
            final_obs_coop_mean=("obs_coop_mean", "last"),
            final_streak=("streak_next", "last"),
        )
    )

    meta_row = {
        "run_id": meta.get("run_id") or first.get("run_id"),
        "num_rounds": meta.get("num_rounds", len(logs)),
        "lineup_labels": ctx.get("lineup_labels", []),
    }

    return round_df, summary, {"match_summary": match_summary, "meta": meta_row}



def collect_results(root: Path):
    match_dirs = sorted([p for p in root.iterdir() if p.is_dir() and "__m" in p.name])
    if not match_dirs:
        raise FileNotFoundError(f"No match directories found under {root}")

    round_frames = []
    summary_frames = []
    match_frames = []
    meta = None

    for md in match_dirs:
        rdf, sdf, extra = load_match(md)
        round_frames.append(rdf)
        summary_frames.append(sdf)
        match_frames.append(extra["match_summary"])
        if meta is None:
            meta = extra["meta"]

    round_df = pd.concat(round_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    match_df = pd.concat(match_frames, ignore_index=True)
    return round_df, summary_df, match_df, meta



def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def save_csvs(round_df: pd.DataFrame, summary_df: pd.DataFrame, match_df: pd.DataFrame, out_dir: Path):
    round_df.to_csv(out_dir / "round_level_summary.csv", index=False)
    summary_df.to_csv(out_dir / "agent_level_summary.csv", index=False)
    match_df.to_csv(out_dir / "match_level_summary.csv", index=False)



def make_pivot(summary_df: pd.DataFrame, agent: str, M: int, value_col: str, llm_only: bool = False):
    df = summary_df[(summary_df["agent"] == agent) & (summary_df["M"] == M)].copy()
    if llm_only:
        df = df[df["kind"] == "llm"]
    pivot = df.pivot(index="p", columns="lam", values=value_col)
    return pivot.sort_index().sort_index(axis=1)



def draw_heatmap(ax, pivot: pd.DataFrame, title: str, cmap: str, fmt: str = ".2f"):
    if pivot.empty:
        ax.set_axis_off()
        ax.set_title(title)
        return None
    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
    ax.set_xlabel("lambda")
    ax.set_ylabel("noise p")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if pd.notna(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)
    return im



def plot_agent_metric_heatmaps(summary_df: pd.DataFrame, out_dir: Path, value_col: str, fname: str, cmap_key: str, title_prefix: str):
    agents = summary_df[["agent", "seat"]].drop_duplicates().sort_values("seat")["agent"].tolist()
    Ms = sorted(summary_df["M"].unique())

    fig, axes = plt.subplots(len(agents), len(Ms), figsize=(4 * len(Ms), 2.8 * len(agents)), squeeze=False)
    im = None
    for i, agent in enumerate(agents):
        for j, M in enumerate(Ms):
            pivot = make_pivot(summary_df, agent, M, value_col)
            im = draw_heatmap(axes[i, j], pivot, f"{agent} | M={M}", LEVEL_CMAPS[cmap_key])
    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(out_dir / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_llm_gtft_heatmaps(summary_df: pd.DataFrame, out_dir: Path):
    llms = summary_df[summary_df["kind"] == "llm"]["agent"].drop_duplicates().tolist()
    Ms = sorted(summary_df["M"].unique())
    fig, axes = plt.subplots(len(llms), len(Ms), figsize=(4 * len(Ms), 3 * max(1, len(llms))), squeeze=False)
    im = None
    for i, agent in enumerate(llms):
        for j, M in enumerate(Ms):
            pivot = make_pivot(summary_df, agent, M, "gtft_match_rate")
            im = draw_heatmap(axes[i, j], pivot, f"{agent} vs GTFT | M={M}", LEVEL_CMAPS["gtft"])
    fig.suptitle("LLM agreement with graded_tft across noise and streak", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(out_dir / "04_llm_vs_gtft_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_coop_vs_M_small_multiples(summary_df: pd.DataFrame, out_dir: Path):
    ps = sorted(summary_df["p"].unique())
    lams = sorted(summary_df["lam"].unique())
    agents = summary_df[["agent", "seat"]].drop_duplicates().sort_values("seat")["agent"].tolist()

    fig, axes = plt.subplots(len(ps), len(lams), figsize=(4.3 * len(lams), 3.2 * len(ps)), squeeze=False, sharex=True, sharey=True)

    for i, p in enumerate(ps):
        for j, lam in enumerate(lams):
            ax = axes[i, j]
            sub = summary_df[(summary_df["p"] == p) & (summary_df["lam"] == lam)]
            for agent in agents:
                s = sub[sub["agent"] == agent].sort_values("M")
                ax.plot(s["M"], s["mean_coop"], marker="o", label=agent)
            ax.set_title(f"p={p:.2f}, lambda={lam:.2f}")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(sorted(summary_df["M"].unique()))
            ax.set_xlabel("M")
            ax.set_ylabel("mean cooperation")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("How decision granularity changes cooperation under each (p, lambda) cell", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "05_mean_coop_vs_M_small_multiples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_group_dynamics(match_df: pd.DataFrame, round_df: pd.DataFrame, out_dir: Path):
    # Representative cells: two clean baselines and two stressed settings.
    candidates = [
        (2, 0.0, 0.0),
        (5, 0.0, 0.0),
        (5, 0.2, 0.0),
        (5, 0.2, 0.6),
    ]
    existing = []
    all_cells = set(round_df[["M", "p", "lam"]].drop_duplicates().itertuples(index=False, name=None))
    for c in candidates:
        if c in all_cells:
            existing.append(c)
    if not existing:
        return

    fig, axes = plt.subplots(len(existing), 1, figsize=(10, 3.0 * len(existing)), sharex=True)
    if len(existing) == 1:
        axes = [axes]

    for ax, (M, p, lam) in zip(axes, existing):
        sub = round_df[(round_df["M"] == M) & (round_df["p"] == p) & (round_df["lam"] == lam)]
        for agent, grp in sub.groupby("agent"):
            grp = grp.sort_values("t")
            ax.plot(grp["t"], grp["coop"], label=agent, alpha=0.9)
        mean_by_t = sub.groupby("t", as_index=False)["coop"].mean()
        ax.plot(mean_by_t["t"], mean_by_t["coop"], linewidth=3, linestyle="--", label="group mean")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Round-wise true cooperation | M={M}, p={p:.2f}, lambda={lam:.2f}")
        ax.set_ylabel("coop")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("round")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "06_selected_round_dynamics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_group_mean_grid(round_df: pd.DataFrame, out_dir: Path):
    Ms = sorted(round_df["M"].unique())
    for M in Ms:
        sub = round_df[round_df["M"] == M]
        ps = sorted(sub["p"].unique())
        lams = sorted(sub["lam"].unique())
        fig, axes = plt.subplots(len(ps), len(lams), figsize=(4.3 * len(lams), 2.8 * len(ps)), squeeze=False, sharex=True, sharey=True)
        for i, p in enumerate(ps):
            for j, lam in enumerate(lams):
                ax = axes[i, j]
                cell = sub[(sub["p"] == p) & (sub["lam"] == lam)]
                mean_t = cell.groupby("t", as_index=False).agg(
                    mean_true_coop=("coop", "mean"),
                    mean_obs_coop=("obs_coop", "mean"),
                    mean_B_eff=("B_eff", "mean"),
                )
                ax.plot(mean_t["t"], mean_t["mean_true_coop"], label="true coop mean")
                ax.plot(mean_t["t"], mean_t["mean_obs_coop"], label="obs coop mean")
                ax.set_title(f"p={p:.2f}, lambda={lam:.2f}")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.25)
                ax.set_xlabel("round")
                ax.set_ylabel("mean coop")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"Group-level cooperation dynamics for all noise/streak cells | M={M}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"07_group_dynamics_grid_M{M}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)



def plot_llm_profile_bars(round_df: pd.DataFrame, out_dir: Path):
    llms = round_df[round_df["kind"] == "llm"]["agent"].drop_duplicates().tolist()
    Ms = sorted(round_df["M"].unique())
    if not llms:
        return

    fig, axes = plt.subplots(len(llms), len(Ms), figsize=(4 * len(Ms), 2.8 * len(llms)), squeeze=False)
    for i, agent in enumerate(llms):
        for j, M in enumerate(Ms):
            ax = axes[i, j]
            sub = round_df[(round_df["agent"] == agent) & (round_df["M"] == M)]
            counts = sub.groupby("action").size().reindex(range(M), fill_value=0)
            probs = counts / counts.sum() if counts.sum() > 0 else counts
            ax.bar(range(M), probs.values)
            ax.set_title(f"{agent} | M={M}")
            ax.set_xlabel("action index")
            ax.set_ylabel("fraction of rounds")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(M))
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Raw action-index usage across all cells", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "08_llm_action_index_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def write_readme(meta: dict, out_dir: Path):
    text = f"""Causal realism plotting outputs
================================

Run id: {meta.get('run_id')}
Rounds per match: {meta.get('num_rounds')}
Lineup: {meta.get('lineup_labels')}

Files
-----
01_agent_mean_coop_heatmaps.png      : mean true cooperation for every agent across (p, lambda), faceted by M
02_agent_mean_reward_heatmaps.png    : mean reward for every agent across (p, lambda), faceted by M
03_agent_switch_rate_heatmaps.png    : action-switching rate for every agent across (p, lambda), faceted by M
04_llm_vs_gtft_heatmaps.png          : how often each LLM matched graded_tft's action each round
05_mean_coop_vs_M_small_multiples.png: decision-granularity effect under every (p, lambda) cell
06_selected_round_dynamics.png       : representative per-round cooperation trajectories
07_group_dynamics_grid_M*.png        : 3x3 grid of group dynamics for each M
08_llm_action_index_profiles.png     : distribution of raw action indices used by each LLM

CSV outputs
-----------
round_level_summary.csv
agent_level_summary.csv
match_level_summary.csv
"""
    (out_dir / "README.txt").write_text(text, encoding="utf-8")



def main():
    parser = argparse.ArgumentParser(description="Plot causal realism result folders.")
    parser.add_argument("--root", required=True, help="Path to causal realism run directory")
    parser.add_argument("--out", default=None, help="Output directory for plots; default=root/plots")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else (root / "plots")
    ensure_dir(out_dir)

    round_df, summary_df, match_df, meta = collect_results(root)
    save_csvs(round_df, summary_df, match_df, out_dir)

    plot_agent_metric_heatmaps(summary_df, out_dir, "mean_coop", "01_agent_mean_coop_heatmaps.png", "coop", "Mean cooperation across causal conditions")
    plot_agent_metric_heatmaps(summary_df, out_dir, "mean_reward", "02_agent_mean_reward_heatmaps.png", "reward", "Mean reward across causal conditions")
    plot_agent_metric_heatmaps(summary_df, out_dir, "switch_rate", "03_agent_switch_rate_heatmaps.png", "switch", "Action volatility across causal conditions")
    plot_llm_gtft_heatmaps(summary_df, out_dir)
    plot_coop_vs_M_small_multiples(summary_df, out_dir)
    plot_group_dynamics(match_df, round_df, out_dir)
    plot_group_mean_grid(round_df, out_dir)
    plot_llm_profile_bars(round_df, out_dir)
    write_readme(meta, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
