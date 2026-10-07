import argparse
import json
from pathlib import Path
import tempfile
import unittest

from experiments import study_common as c


class DeadlineAllocationTests(unittest.TestCase):
    def test_real_evidence_unequal_counts_and_disjoint_shards(self):
        old = c.read_json(c.ROOT/'results/analysis/main_dry_run_review_20261007/run_plan.json')
        variance = c.ROOT/'results/analysis/pilot_variance_20261007/variance_estimates.json'
        counts = dict(zip(c.ROSTER, [14,4,8,3,16,2]))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c.write_json(root/'deployment.json', old['deployment'])
            power = dict(seeds_per_cell=16, model_seed_targets=counts,
                         allocation_method='deadline_budget_20261007', working_endpoint_effect=.15,
                         variance_margin=1.25, accept_lower_power=True,
                         amendment_sha256=c.file_digest(c.ROOT/'docs/H4_deadline_amendment_20261007.txt'),
                         variance_estimates_path=str(variance), variance_sha256=c.file_digest(variance),
                         confirmed_in_conversation=True, approval_reference='TEST', rationale='TEST')
            c.write_json(root/'power.json', power)
            args = argparse.Namespace(models=list(c.ROSTER),config=c.ROOT/'experiments/main_study/config.json',
                deployment_report=root/'deployment.json',generation_overrides=c.ROOT/'experiments/main_study/generation_overrides.json',
                seeds_per_cell=16, power_report=root/'power.json', seed_start=1001)
            plan=c.make_plan(args,'main')
            self.assertEqual(plan['episode_count'],376)
            for cell in plan['cells']:
                self.assertEqual(cell['episodes'],counts[cell['model']])
                self.assertEqual(cell['generation']['temperature'],.7)
                self.assertEqual(cell['generation']['top_p'],1.)
            manifest={'record_type':'manifest','plan':plan,'plan_sha256':c.digest(plan)}
            (root/'main.jsonl').write_text(json.dumps(manifest)+'\n')
            c.read_study(root/'main.jsonl','main')
            self.assertFalse(c.variance_ready_for_reviewed_power(c.read_json(variance),dict(power,accept_lower_power=False)))
            self.assertFalse(c.variance_ready_for_reviewed_power(c.read_json(variance),dict(power,amendment_sha256='bad')))
            seeds=next(cell['seeds'] for cell in plan['cells'] if cell['model']=='deepseek-r1-distill-qwen-14b')
            self.assertFalse(set(seeds[::2]) & set(seeds[1::2]))
            self.assertEqual(len(seeds[::2])+len(seeds[1::2]),16)


if __name__=='__main__':unittest.main()
