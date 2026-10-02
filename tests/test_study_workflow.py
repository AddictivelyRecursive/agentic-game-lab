"""Offline study integration tests. Synthetic data is never paper evidence."""
import argparse
import contextlib
import csv
import io
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import study_common as common
from analyze_pilot_variance import analyze as variance_analyze
from sample_for_cot_coding import sample, write_csv
from score_cot_coding import score, cohen_kappa


def deployment():
    models = {name: {"reachable": True, "response": "READY", "hf_identifier": common.resolve_model(name)["repo"],
                     "base_url": f"http://127.0.0.1:{18000+i}/v1", "mean_decision_latency_s": 1,
                     "gpu_count": 1, "evidence_path": "SYNTHETIC TEST FIXTURE"}
              for i, name in enumerate(common.ROSTER)}
    models["gpt-oss-120b"].update(vram_used_gib=70, quantization="synthetic", allocation="separate",
                                 allocation_notes="synthetic test only", isolated_smoke={"N": 2, "T": 5,
                                 "completed": True, "fallback_decisions": 0, "round_latency_s": [4, 4, 4, 4, 4]})
    return {"step0_complete": True, "checked_at": "SYNTHETIC", "models": models}


def fixture_plan(phase="main", seeds=3):
    seeds = 5 if phase == "pilot" else seeds
    ids = list(range(101, 101+seeds))
    return {"schema": common.SCHEMA, "phase": phase, "models": list(common.ROSTER),
            "seeds_per_cell": seeds, "seeds": ids,
            "configuration": common.read_json(ROOT / "experiments/study_config.json"),
            "deployment": deployment(), "cells": [
                {"model": m, "reasoning_mode": mode, "N": n, "episodes": seeds, "seeds": ids}
                for m, mode, n in itertools.product(common.ROSTER, common.MODES, common.N_LEVELS)]}


def synthetic_data(path, phase="main", seeds=3, uniform=False, fallback=False):
    plan = fixture_plan(phase, seeds)
    records = [{"record_type": "manifest", "plan": plan, "plan_sha256": common.digest(plan)}]
    rng = random.Random(501)
    for cell in plan["cells"]:
        m = common.ROSTER.index(cell["model"])
        n = cell["N"]
        for seed in cell["seeds"]:
            target = .72 + (m-2.5)*.025 - (.05 + m*.003)*(n-2)
            if cell["reasoning_mode"] == "free_form":
                target += .035*(n-2) - .02
            target += 0 if uniform else rng.uniform(-.025, .025)
            units = round(target * n * 50 * 4)
            base, remainder = divmod(units, n*50)
            values = [(base + (i < remainder)) / 4 for i in range(n*50)]
            decisions = [{"round": t, "seat": seat, "reasoning_text": "synthetic reasoning",
                          "raw_response": "synthetic", "first_parse_ok": True, "repair_used": False,
                          "fallback_used": fallback, "latency_s": 1}
                         for t, seat in itertools.product(range(1,51), range(n))]
            rounds = [{"t": t, "true_coop": values[(t-1)*n:t*n]} for t in range(1,51)]
            records.append({"record_type": "episode", "episode_id": f"{cell['model']}__{cell['reasoning_mode']}__n{n}__seed{seed}",
                            "model": cell["model"], "reasoning_mode": cell["reasoning_mode"], "N": n,
                            "seed": seed, "status": "completed", "cooperation": sum(values)/len(values),
                            "rounds": rounds, "decisions": decisions})
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


class StudyWorkflowTests(unittest.TestCase):
    def test_dry_run_is_offline_exact_grid_and_hash_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "deployment.json"
            common.write_json(report, deployment())
            argv = ["--models", *common.ROSTER, "--deployment-report", str(report), "--output", str(root/"pilot")]
            with patch.object(common.LocalOpenAIClient, "generate", side_effect=AssertionError("No inference")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(common.run_cli("pilot", argv + ["--dry-run"]), 0)
                with self.assertRaises(SystemExit):
                    common.run_cli("pilot", argv)
            plan = common.read_json(root/"pilot/run_plan.json")
            self.assertEqual(plan["episode_count"], 240)
            self.assertEqual(plan["decision_count"], 42000)
            self.assertEqual(len(plan["cells"]), 48)
            gpt = [c for c in plan["cells"] if c["model"] == "gpt-oss-120b"]
            self.assertTrue(all(c["estimated_seconds"] == 2*c["decisions"] for c in gpt))
            self.assertFalse((root/"pilot/pilot.jsonl").exists())
            approval = root/"approval.json"
            common.write_json(approval, {"plan_sha256": common.digest(plan), "confirmed_in_conversation": True,
                                        "approval_reference": "SYNTHETIC TEST"})
            with patch("experiments.study_common.execute_episode", side_effect=AssertionError("Changed plan must not run")):
                with self.assertRaises(SystemExit):
                    common.run_cli("pilot", argv + ["--approval", str(approval), "--seed-start", "200"])

    def test_missing_deployment_and_missing_main_seed_count_are_rejected(self):
        with self.assertRaises(ValueError):
            common.validate_deployment({"step0_complete": False}, common.ROSTER)
        report = deployment()
        report["models"]["gpt-oss-120b"]["isolated_smoke"]["round_latency_s"] = []
        with self.assertRaises(ValueError):
            common.validate_deployment(report, common.ROSTER)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            common.run_cli("main", ["--models", *common.ROSTER, "--deployment-report", "unused", "--output", "unused"])

    def test_actual_pipeline_records_repairs_and_reasoning_without_a_server(self):
        plan = fixture_plan()
        cell = plan["cells"][0]
        responses = iter(['broken json'] + ['{"a": 2, "reason": "group size matters"}']*100)
        with tempfile.TemporaryDirectory() as directory, patch.object(common.LocalOpenAIClient, "generate", side_effect=lambda **kw: next(responses)):
            row = common.execute_episode(plan, cell, 101, directory)
            self.assertEqual(row["status"], "completed", row.get("error"))
            self.assertEqual(row["cooperation"], .5)
            self.assertEqual(len(row["decisions"]), 100)
            self.assertTrue(row["decisions"][0]["repair_used"])
            self.assertFalse(row["decisions"][0]["first_parse_ok"])
            self.assertEqual(row["decisions"][0]["reasoning_text"], "group size matters")
            self.assertFalse(any(d["fallback_used"] for d in row["decisions"]))
            self.assertNotEqual(row["generation_by_seat"][0]["seed"], row["generation_by_seat"][1]["seed"])

    def test_aborted_episode_keeps_partial_trace_and_has_no_cooperation(self):
        from AI_Agent.agent.api_guard import APIUnavailableError
        plan = fixture_plan()
        with tempfile.TemporaryDirectory() as directory, patch.object(common.LocalOpenAIClient, "generate", side_effect=APIUnavailableError("synthetic outage")):
            row = common.execute_episode(plan, plan["cells"][0], 101, directory)
            self.assertEqual(row["status"], "failed")
            self.assertIsNone(row["cooperation"])
            self.assertTrue(row["decisions"][0]["aborted"])

    def test_pilot_variance_and_quality_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"pilot.jsonl"
            synthetic_data(data, "pilot")
            result = variance_analyze(data, root/"variance")
            self.assertTrue(result["ready_for_power_calculation"])
            self.assertEqual(len(result["cells"]), 48)
            self.assertTrue(all(c["variance"] > 0 for c in result["cells"]))
            synthetic_data(data, "pilot", uniform=True, fallback=True)
            result = variance_analyze(data, root/"flagged")
            self.assertFalse(result["ready_for_power_calculation"])
            self.assertIn("FALLBACK_RATE_GT_0.1", result["cells"][0]["flags"])
            self.assertIn("UNIFORM_ACROSS_SEEDS", result["cells"][0]["flags"])
            records = synthetic_data(data, "main")
            with self.assertRaises(ValueError):
                variance_analyze(data, root/"wrong_phase")
            records.append(records[-1])
            data.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            with self.assertRaises(ValueError):
                common.read_study(data, "main")

    def test_main_completion_and_power_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/"main.jsonl"
            records = synthetic_data(raw)
            manifest, rows = common.read_study(raw, "main")
            with self.assertRaises(ValueError):
                common.require_complete(manifest, rows[:-1])
            common.write_run_summary(root, manifest["plan"], rows[:-1])
            self.assertIn("INCOMPLETE", (root/"main_run_summary.md").read_text())
            dep, power, var = root/"deployment.json", root/"power.json", root/"variance.json"
            common.write_json(dep, deployment())
            common.write_json(var, {"ready_for_power_calculation": True, "models": list(common.ROSTER),
                                   "configuration": common.read_json(ROOT/"experiments/study_config.json")})
            evidence = {"seeds_per_cell": 7, "confirmed_in_conversation": True, "approval_reference": "SYNTHETIC",
                        "rationale": "SYNTHETIC power calculation", "variance_estimates_path": "variance.json", "variance_sha256": common.file_digest(var)}
            common.write_json(power, evidence)
            args = argparse.Namespace(models=list(common.ROSTER), deployment_report=dep, config=ROOT/"experiments/study_config.json",
                                      seeds_per_cell=7, seed_start=101, power_report=power)
            self.assertEqual(common.make_plan(args, "main")["episode_count"], 336)
            args.seeds_per_cell = 5
            with self.assertRaises(ValueError):
                common.make_plan(args, "main")

    def test_all_round_coding_blinding_and_reliability_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/"main.jsonl"
            synthetic_data(raw, seeds=1)
            count = sample(raw, root/"sample")
            self.assertEqual(count, 6*2*(2+5)*50)
            with (root/"sample/coding_sheet.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["row_id", "reasoning_text", "code"])
                rows = list(reader)
            self.assertTrue(all(r["code"] == "" for r in rows))
            a, b = root/"a.csv", root/"b.csv"
            fields = ["row_id", "reasoning_text", "code"]
            for i, row in enumerate(rows):
                row["code"] = str(i%2)
            write_csv(a, fields, rows)
            write_csv(b, fields, rows)
            key_path = root/"sample/private_condition_key.json"
            private = common.read_json(key_path)
            approval = root/"coding_approval.json"
            def approve():
                common.write_json(approval, {"confirmed_in_conversation": True, "approval_reference": "SYNTHETIC",
                    "sample_sha256": private["sample_sha256"], "coder1_sha256": common.file_digest(a), "coder2_sha256": common.file_digest(b)})
            approve()
            result = score(a,b,key_path,approval,root/"history.json",root/"passed")
            self.assertEqual(result["kappa"], 1)
            self.assertEqual(len(result["rates"]), 4)
            for row in rows:
                row["code"] = str(1-int(row["code"]))
            write_csv(b, fields, rows)
            approve()
            result = score(a,b,key_path,approval,root/"history.json",root/"failed")
            self.assertLess(result["kappa"], .7)
            self.assertNotIn("rates", result)
            self.assertIn("No diluted-responsibility rate", (root/"failed/h5_summary.md").read_text())
            with self.assertRaises(ValueError):
                score(a,b,key_path,approval,root/"history.json",root/"reused")

    def test_kappa_undefined_and_known_value(self):
        self.assertIsNone(cohen_kappa([1,1], [1,1]))
        self.assertEqual(cohen_kappa([0,0,1,1], [0,0,1,1]), 1)
        self.assertEqual(cohen_kappa([0,0,1,1], [1,1,0,0]), -1)

    def test_approved_serial_shard_and_resume_do_not_duplicate_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = synthetic_data(root/"synthetic.jsonl", "pilot")
            fixtures = {(*common.cell_key(r), r["seed"]): r for r in records[1:]}
            dep = root/"deployment.json"
            common.write_json(dep, deployment())
            argv = ["--models", *common.ROSTER, "--deployment-report", str(dep), "--output", str(root/"pilot")]
            with contextlib.redirect_stdout(io.StringIO()):
                common.run_cli("pilot", argv+["--dry-run"])
                plan = common.read_json(root/"pilot/run_plan.json")
                approval = root/"approval.json"
                common.write_json(approval, {"plan_sha256": common.digest(plan), "confirmed_in_conversation": True,
                                            "approval_reference": "SYNTHETIC TEST ONLY"})
                shard_args = argv+["--approval", str(approval), "--execute-model", common.ROSTER[0]]
                with patch("experiments.study_common.execute_episode", side_effect=lambda p,c,s,o: fixtures[(*common.cell_key(c),s)]) as execute:
                    common.run_cli("pilot", shard_args)
                    self.assertEqual(execute.call_count, 40)
                    common.run_cli("pilot", shard_args)
                    self.assertEqual(execute.call_count, 40)
            _, rows = common.read_study(root/"pilot/pilot.jsonl", "pilot")
            self.assertEqual(len(rows), 40)
            self.assertIn("ready for power calculation: no", (root/"pilot/pilot_summary.md").read_text())
            self.assertFalse((root/"pilot/.execution.lock").exists())

    def test_mixed_model_and_curve_recover_synthetic_effects(self):
        from fit_glmm_and_curve import analyze
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/"main.jsonl"
            synthetic_data(raw, seeds=5)
            results = analyze(raw, root/"analysis")
            self.assertEqual(results["primary_status"], "converged")
            self.assertLess(results["H1"]["coefficient"], -.03)
            self.assertGreater(results["H4"]["coefficient"], .02)
            self.assertLess(results["H4"]["p_two_sided"], .05)
            self.assertGreater(results["H2"]["delta_aic_flat_minus_dilution"], 0)
            exploratory = (root/"analysis/exploratory_results_summary.md").read_text()
            self.assertIn("non-pre-registered", exploratory)
            self.assertNotIn("## E1", (root/"analysis/confirmatory_results_summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
