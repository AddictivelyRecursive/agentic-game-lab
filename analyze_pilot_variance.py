"""Read only pilot JSONL; estimate sample variance without testing hypotheses."""
import argparse
from pathlib import Path
from experiments.study_common import (read_study, summarize_cells, ready, cell_table,
                                      new_output, write_json, file_digest, TRAINING_NOTE, MAX_FALLBACK_RATE)


def analyze(input_path, output):
    manifest, rows = read_study(input_path, "pilot")
    cells = summarize_cells(manifest["plan"], rows)
    output = new_output(output)
    rounds = manifest["plan"]["configuration"]["environment"]["T"]
    result = {"rounds_per_episode": rounds,
              "analysis_unit": f"Mean normalized cooperation across all seats and {rounds} actual rounds per episode",
              "source_sha256": file_digest(input_path), "plan_sha256": manifest["plan_sha256"],
              "configuration": manifest["plan"]["configuration"], "models": manifest["plan"]["models"],
              "ready_for_power_calculation": ready(cells), "variance_ddof": 1, "max_fallback_rate_per_cell": MAX_FALLBACK_RATE, "cells": cells}
    write_json(output / "variance_estimates.json", result)
    (output / "variance_summary.md").write_text(
        f"# Pilot variance estimates ({rounds}-round episodes)\n\nNo hypothesis testing was performed.\n\n"
        + f"Each observation is an episode mean across all seats and {rounds} rounds. No conversion to a 50-round variance is applied.\n\n" + cell_table(cells)
        + f"\n\nready for power calculation: {'yes' if ready(cells) else 'no'}\n\n"
        + "Flags are recomputed from the pilot JSONL using the same rules as pilot_summary.md. "
        "Uniform means the observed seed means differ by no more than 1e-12. "
        "Variance uses episode means and ddof=1; Incomplete or uniform cells, and cells with fallback above 12%, block readiness. Smaller fallback rates remain flagged and included; they are not erased.\n\n"
        + TRAINING_NOTE + "\n", encoding="utf-8")
    return result


def plot_variance(result, output):
    """Plot measured episode-mean sample variance at the recorded horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds = result["rounds_per_episode"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, ("structured", "free_form")):
        for model in result["models"]:
            cells = sorted((c for c in result["cells"] if c["model"] == model and c["reasoning_mode"] == mode), key=lambda c: c["N"])
            ax.plot([c["N"] for c in cells], [float("nan") if c["variance"] is None else c["variance"] for c in cells], marker="o", label=model)
        ax.set(title=mode.replace("_", " "), xlabel="Number of players N", xticks=[2, 3, 4, 5])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sample variance of episode mean cooperation (ddof=1)")
    fig.suptitle(f"Pilot variance across seeds - {rounds}-round episodes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    path = Path(output) / "pilot_variance_by_N.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="Write variance plot labelled with the actual episode round count")
    args = parser.parse_args()
    try:
        result = analyze(args.input, args.output)
        if args.plot:
            plot_variance(result, args.output)
        print(f"Wrote variance_summary.md; ready for power calculation: {result['ready_for_power_calculation']}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
