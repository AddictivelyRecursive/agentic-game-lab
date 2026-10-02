"""Score independent human sheets; withhold H5 rates until Cohen's kappa >= 0.7."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from experiments.study_common import read_json, write_json, new_output, digest, file_digest, TRAINING_NOTE
from sample_for_cot_coding import write_csv


def read_codes(path, key):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_id", "reasoning_text", "code"]:
            raise ValueError("Coding sheet must have row_id, reasoning_text, code columns")
        rows = list(reader)
    mapping = {}
    for row in rows:
        row_id = row["row_id"]
        if row_id in mapping or row_id not in key:
            raise ValueError("Duplicate or unknown coding row")
        if digest(row["reasoning_text"]) != key[row_id]["text_sha256"]:
            raise ValueError("Coder changed raw reasoning text")
        code = row["code"].strip()
        allowed = ("NA",) if key[row_id]["missing_reasoning"] else ("0", "1")
        if code not in allowed:
            raise ValueError("Incomplete/invalid code: use 0 or 1; NA only for empty reasoning")
        mapping[row_id] = {**row, "code": None if code == "NA" else int(code)}
    if set(mapping) != set(key):
        raise ValueError("Both coders must complete exactly the entire sample")
    return mapping


def cohen_kappa(a, b):
    if not a or len(a) != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    pa, pb = statistics.mean(a), statistics.mean(b)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if abs(1 - expected) < 1e-15 else (observed - expected) / (1 - expected)


def score(coder1, coder2, key_path, approval_path, history_path, output):
    private = read_json(key_path)
    if private["sample_sha256"] != digest(private["rows"]):
        raise ValueError("Corrupted private sample key")
    key = {r["row_id"]: r for r in private["rows"]}
    if len(key) != len(private["rows"]):
        raise ValueError("Duplicate rows in private sample key")
    approval = read_json(approval_path)
    if (approval.get("confirmed_in_conversation") is not True or not approval.get("approval_reference")
            or approval.get("sample_sha256") != private["sample_sha256"]
            or approval.get("coder1_sha256") != file_digest(coder1)
            or approval.get("coder2_sha256") != file_digest(coder2)):
        raise ValueError("Scoring requires confirmed human coding, bound to this sample and both completed files")
    first, second = read_codes(coder1, key), read_codes(coder2, key)
    history_path = Path(history_path)
    history = read_json(history_path) if history_path.exists() else {"failed_units": [], "attempts": []}
    units = {r["unit_sha256"] for r in private["rows"]}
    if units.intersection(history["failed_units"]):
        raise ValueError("Sample overlaps a failed reliability sample: revise the rubric and use fresh independent data")
    ids = [k for k in key if first[k]["code"] is not None]
    kappa = cohen_kappa([first[k]["code"] for k in ids], [second[k]["code"] for k in ids])
    passed = kappa is not None and kappa >= .7
    output = new_output(output)
    disagreements = [{"row_id": k, "reasoning_text": first[k]["reasoning_text"],
                      "coder1": first[k]["code"], "coder2": second[k]["code"]}
                     for k in ids if first[k]["code"] != second[k]["code"]]
    write_csv(output / "disagreements.csv", ["row_id", "reasoning_text", "coder1", "coder2"], disagreements)
    result = {"sample_sha256": private["sample_sha256"], "kappa": kappa, "reliability_passed": passed,
              "coded_rows": len(ids), "missing_reasoning_rows": len(key) - len(ids), "disagreements": len(disagreements)}
    lines = ["# H5: diluted-responsibility language", f"Cohen's kappa: {kappa if kappa is not None else 'undefined'}.",
             f"Coded rows: {len(ids)}; missing reasoning: {len(key)-len(ids)}; disagreements: {len(disagreements)}."]
    if not passed:
        lines += ["Reliability gate FAILED. No diluted-responsibility rate or H5 conclusion is reported.",
                  "Review disagreements.csv, revise the rubric, and code a fresh independent sample. "
                  "Relabeling or reshuffling these rows does not make a fresh sample. Undefined kappa also fails the gate."]
        history["failed_units"] = sorted(set(history["failed_units"]) | units)
    else:
        groups = defaultdict(list)
        for k in ids:
            groups[key[k]["N"], key[k]["reasoning_mode"]].append(k)
        rates = []
        lines += ["Reliability gate PASSED.", "Rates are reported for each coder separately; disagreements are not silently adjudicated.",
                  "| N | Mode | Texts | Coder 1 rate | Coder 2 rate |", "|---:|---|---:|---:|---:|"]
        for (n, mode), members in sorted(groups.items()):
            r1 = statistics.mean(first[k]["code"] for k in members)
            r2 = statistics.mean(second[k]["code"] for k in members)
            rates.append({"N": n, "reasoning_mode": mode, "texts": len(members), "coder1_rate": r1, "coder2_rate": r2})
            lines.append(f"| {n} | {mode} | {len(members)} | {r1:.6g} | {r2:.6g} |")
        result["rates"] = rates
        lines += ["## H5 comparison: N=5 minus N=2", "Positive differences are in H5's predicted direction."]
        for mode in ("structured", "free_form"):
            lookup = {r["N"]: r for r in rates if r["reasoning_mode"] == mode}
            if set(lookup) == {2, 5}:
                lines.append(f"{mode}: coder 1 difference={lookup[5]['coder1_rate']-lookup[2]['coder1_rate']:.6g}; "
                             f"coder 2 difference={lookup[5]['coder2_rate']-lookup[2]['coder2_rate']:.6g}.")
            else:
                lines.append(f"{mode}: comparison unavailable due to missing reasoning.")
        lines += ["Denominators exclude empty reasoning (NA), never count it as absence of the language. "
                  "Rates describe this bounded coded sample. The supplied protocol specifies no inferential H5 test; "
                  "no independence-based round-level p-value is invented."]
    lines.append(TRAINING_NOTE)
    write_json(output / "h5_results.json", result)
    (output / "h5_summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    history["attempts"].append(result)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(history_path, history)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder1", type=Path, required=True)
    parser.add_argument("--coder2", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--reliability-history", type=Path, required=True,
                        help="Reuse the same study ledger for every attempt; prevents reuse after failed reliability")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args.coder1, args.coder2, args.key, args.approval, args.reliability_history, args.output)
        print(f"Wrote h5_summary.md; reliability passed: {result['reliability_passed']}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
