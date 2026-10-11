import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from cluster import main_study_jobs as jobs
from test_study_workflow import deployment
from experiments import study_common as common

class MainJobsTests(unittest.TestCase):
    def test_missing_power_count_never_prepares_or_submits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);power=root/'power.json';power.write_text('{"seeds_per_cell":null}')
            with patch.object(jobs.subprocess,'run') as run, self.assertRaises(ValueError):
                jobs.prepare(argparse.Namespace(bundle=root/'bundle',power_report=power))
            run.assert_not_called()
            self.assertFalse((root/'bundle').exists())

    def test_synthetic_prepare_is_offline_and_unapproved_submission_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config=common.read_json(jobs.ROOT/'experiments/main_study/config.json')
            report=deployment()
            variance={'ready_for_power_calculation':True,'configuration':config,'models':list(common.ROSTER)}
            common.write_json(root/'variance.json',variance)
            common.write_json(root/'power.json',{'seeds_per_cell':2,'variance_estimates_path':'variance.json',
                'variance_sha256':common.file_digest(root/'variance.json'),'rationale':'SYNTHETIC TEST ONLY',
                'confirmed_in_conversation':True,'approval_reference':'SYNTHETIC TEST ONLY'})
            common.write_json(root/'deployment.json',report)
            snapshot=root/'model';snapshot.mkdir();(snapshot/'config.json').write_text('{}')
            serving=common.read_json(jobs.ROOT/'experiments/main_study/serving.template.json')
            for i,cfg in enumerate(serving['models'].values()):
                cfg.update(model_path=str(snapshot.resolve()),port=18000+i,walltime='01:00:00',
                           hf_home=str(root.resolve()),extra_serve_args='')
            common.write_json(root/'serving.json',serving)
            original_run=subprocess.run
            calls=[]
            def run(cmd,**kwargs):
                calls.append(cmd)
                if cmd[0]=='bash':return subprocess.CompletedProcess(cmd,0)
                return original_run(cmd,**kwargs)
            args=argparse.Namespace(bundle=root/'bundle',config=jobs.ROOT/'experiments/main_study/config.json',
                generation_overrides=jobs.ROOT/'experiments/main_study/generation_overrides.json',
                deployment_report=root/'deployment.json',power_report=root/'power.json',serving=root/'serving.json')
            with patch.object(jobs.subprocess,'run',side_effect=run), \
                 patch.object(jobs.subprocess,'check_output',return_value='synthetic-test-provenance'):
                jobs.prepare(args)
            self.assertFalse(any(cmd[0]=='sbatch' for cmd in calls))
            for alias in common.ROSTER:
                modelroot=args.bundle/alias
                plan=common.read_json(modelroot/'output/run_plan.json')
                self.assertEqual(plan['configuration']['environment']['T'],30)
                self.assertEqual(plan['seeds_per_cell'],2)
                self.assertEqual(plan['seeds'],[1001,1002])
                self.assertFalse(common.read_json(modelroot/'approval.json')['confirmed_in_conversation'])
            lib=(args.bundle/'main_lib.sh').read_text()
            self.assertIn('run_main.py',lib)
            self.assertIn('--power-report',lib)
            self.assertIn("f.write",lib) if "f.write" in (jobs.ROOT/'cluster/lib.sh').read_text() else None
            # Embedded Python literal escapes must survive shell generation.
            self.assertIn("+'\\n'",lib) if "+'\\n'" in (jobs.ROOT/'cluster/lib.sh').read_text() else None
            with patch.object(jobs.subprocess,'check_output') as submit, self.assertRaises(ValueError):jobs.submit(args)
            submit.assert_not_called()
