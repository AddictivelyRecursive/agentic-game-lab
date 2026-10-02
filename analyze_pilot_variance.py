"""Read only pilot JSONL; estimate sample variance without testing hypotheses."""
import argparse
from pathlib import Path
from experiments.study_common import (read_study, summarize_cells, ready, cell_table,
                                      new_output, write_json, file_digest, TRAINING_NOTE)


def analyze(input_path, output):
    manifest, rows = read_study(input_path, "pilot")
    cells = summarize_cells(manifest["plan"], rows)
    output = new_output(output)
    result = {"source_sha256": file_digest(input_path), "plan_sha256": manifest["plan_sha256"],
              "configuration": manifest["plan"]["configuration"], "models": manifest["plan"]["models"],
              "ready_for_power_calculation": ready(cells), "variance_ddof": 1, "cells": cells}
    write_json(output / "variance_estimates.json", result)
    (output / "variance_summary.md").write_text(
        "# Pilot variance estimates\n\nNo hypothesis testing was performed.\n\n" + cell_table(cells)
        + f"\n\nready for power calculation: {'yes' if ready(cells) else 'no'}\n\n"
        + "Flags are recomputed from the pilot JSONL using the same rules as pilot_summary.md. "
        "Uniform means the observed seed means differ by no more than 1e-12. "
        "Variance uses episode means and ddof=1; incomplete, uniform, or fallback-contaminated cells block readiness.\n\n"
        + TRAINING_NOTE + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.input, args.output)
        print(f"Wrote variance_summary.md; ready for power calculation: {result['ready_for_power_calculation']}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
