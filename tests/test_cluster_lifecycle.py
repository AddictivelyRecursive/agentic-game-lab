"""Exercise Bash lifecycle decisions with fake curl/kill; no Slurm or server."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Requires POSIX Bash")
class ClusterLifecycleTests(unittest.TestCase):
    def shell(self, script, **env):
        return subprocess.run(["bash", "-c", 'set -eo pipefail; source "$LIB"; ' + script],
            env={**os.environ, "LIB": str(ROOT / "cluster/lib.sh"), **env},
            capture_output=True, text=True, timeout=15)

    def test_login_node_is_rejected_before_activation(self):
        result = self.shell('unset SLURM_JOB_ID; run_job smoke')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no login-node inference', result.stderr)

    def test_readiness_checks_exact_served_model(self):
        script = '''
VLLM_PID=123; VLLM_LOG=/dev/null; PORT=18001; HEALTH_WAIT_MIN=1
LLM_BASE_URL=http://127.0.0.1:18001/v1; SERVED_NAME=expected
kill() { return 0; }
curl() { printf '%s' '{"data":[{"id":"expected"}]}'; }
wait_for_vllm
'''
        result = self.shell(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('vLLM ready', result.stdout)

    def test_wrong_model_times_out(self):
        script = '''
VLLM_PID=123; VLLM_LOG=/dev/null; PORT=18001; HEALTH_WAIT_MIN=1
LLM_BASE_URL=http://127.0.0.1:18001/v1; SERVED_NAME=expected
kill() { return 0; }
curl() { printf '%s' '{"data":[{"id":"other"}]}'; }
sleep() { SECONDS=10000; }
wait_for_vllm
'''
        result = self.shell(script)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('timed out', result.stderr)

    def test_dead_server_and_bind_failure_show_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'vllm.log'
            log.write_text('diagnostic: address already in use\n')
            for alive in ('0', '1'):
                result = self.shell('''
VLLM_PID=123; PORT=18001; HEALTH_WAIT_MIN=1
kill() { return "$ALIVE"; }
wait_for_vllm
''', VLLM_LOG=str(log), ALIVE=alive)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('diagnostic: address already in use', result.stderr)
                self.assertIn('occupied' if alive == '0' else 'died during startup', result.stderr)

    def test_cleanup_targets_only_captured_process_groups_and_preserves_status(self):
        result = self.shell('''
kill() { echo "kill $*"; [[ "$1" != -0 ]]; }
wait() { return 0; }
VLLM_PID=123; EXPERIMENT_PID=456
trap cleanup EXIT
exit 7
''')
        self.assertEqual(result.returncode, 7)
        self.assertIn('kill -TERM -- -123', result.stdout)
        self.assertIn('kill -TERM -- -456', result.stdout)
        self.assertNotIn('pkill', result.stdout)

    def test_submission_maps_tp_and_respects_partition_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            sbatch = Path(d) / 'sbatch'
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$MODEL" "$TP_SIZE" "$@"\n')
            sbatch.chmod(0o700)
            result = subprocess.run(['bash', str(ROOT/'cluster/submit.sh'),
                'qwen2.5-72b-instruct', '--time=02:00:00'], env={**os.environ,
                    'PATH': d + os.pathsep + os.environ['PATH'], 'PARTITION': 'gpu_h100_4',
                    'TP_SIZE': '2', 'KIND': 'run'}, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            for value in ('qwen2.5-72b-instruct', '--gres=gpu:2', '--cpus-per-task=8',
                          '--partition=gpu_h100_4', '--time=02:00:00'):
                self.assertIn(value, result.stdout)

    @unittest.skipUnless(shutil.which('setsid'), 'Requires setsid')
    def test_batch_orchestration_with_fake_server_and_experiment(self):
        # Execute the real launcher end to end. Fake binaries cannot load models,
        # allocate GPUs, submit jobs, or make HTTP requests.
        for experiment_status in (0, 7):
            with self.subTest(status=experiment_status), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                bin_dir = root / 'bin'
                bin_dir.mkdir()
                scripts = {
                    'vllm': '#!/bin/bash\nprintf "%s\\n" "$@"\nexec sleep 60\n',
                    'curl': '#!/bin/bash\nprintf \'%s\' \'{"data":[{"id":"qwen3-8b"}]}\'\n',
                    'nvidia-smi': '#!/bin/bash\necho "mock GPU inventory"\n',
                    'git': '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo fakecommit; fi\n',
                    'python': f'''#!{sys.executable}
import os, pathlib, sys
if sys.argv[1:3] == ['-m', 'game_engine.experiments.run_local_episode'] and '--validate-only' not in sys.argv:
    pathlib.Path(os.environ['RUN_DIR'], 'experiment_started').write_text('mock only')
    sys.exit(int(os.environ['FAKE_EXPERIMENT_STATUS']))
os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
''',
                }
                for name, body in scripts.items():
                    path = bin_dir / name
                    path.write_text(body)
                    path.chmod(0o700)
                activation = root / 'activate'
                activation.write_text(':\n')
                (root / 'hf/hub/models--Qwen--Qwen3-8B').mkdir(parents=True)
                result = self.shell('run_job run', PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                    REPO_ROOT=str(ROOT), SLURM_JOB_ID='12345', SLURM_GPUS_ON_NODE='1',
                    SLURM_JOB_NUM_NODES='1', SLURM_JOB_PARTITION='mock', TP_SIZE='1',
                    MODEL='qwen3-8b', MODEL_REPO='Qwen/Qwen3-8B', SERVED_NAME='qwen3-8b',
                    HF_HOME=str(root/'hf'), CONDA_ACTIVATE=str(activation), RUN_DIR=str(root/'run'),
                    EXPERIMENT_CONFIG=str(ROOT/'experiments/local_episode.json'),
                    EXTRA_SERVE_ARGS='--enforce-eager', FAKE_EXPERIMENT_STATUS=str(experiment_status))
                self.assertEqual(result.returncode, 0 if experiment_status == 0 else 1, result.stderr)
                self.assertTrue((root/'run/job.json').exists(), result.stderr)
                self.assertTrue((root/'run/experiment_started').exists(), result.stderr)
                self.assertIn('vLLM ready', result.stdout)
                log = (root/'run/vllm.log').read_text()
                self.assertIn('--generation-config\nvllm', log)
                self.assertIn('--host\n127.0.0.1', log)
                self.assertIn('--enforce-eager', log)


if __name__ == '__main__':
    unittest.main()
