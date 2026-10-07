"""Freeze-reviewed deadline allocation; dry-run every shard before submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/home/jagatsesh/f20220604_ts/agentic-game-lab')
PYTHON = '/home/jagatsesh/miniconda3/envs/slaybench/bin/python'
REFERENCE = 'User 2026-10-07: stop current Qwen and run experiments as planned; shorter 84h budget allocation, fresh v2 data'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    bundle = a.bundle.resolve()
    code = bundle/'code'
    sys.path.insert(0, str(code))
    from experiments.study_common import ROSTER, read_json, file_digest, digest, make_plan
    counts = dict(zip(ROSTER, [14, 4, 8, 3, 16, 2]))
    if (bundle/'submissions.json').exists():
        raise ValueError('Submission ledger exists; refusing duplicate launch')
    oldplan = read_json(code/'evidence/old_plan.json')
    variance = code/'evidence/variance_estimates.json'
    power = dict(seeds_per_cell=16, model_seed_targets=counts, allocation_method='deadline_budget_20261007',
                 working_endpoint_effect=.15, variance_margin=1.25, target_power=.9,
                 accept_lower_power=True, amendment_sha256=file_digest(code/'docs/H4_deadline_amendment_20261007.txt'),
                 variance_estimates_path=str(variance), variance_sha256=file_digest(variance),
                 confirmed_in_conversation=True, approval_reference=REFERENCE,
                 rationale='User-approved budget allocation with reported lower marginal power; zero pilot credit')
    (bundle/'power_report.json').write_text(json.dumps(power, indent=2))
    # Read actual serving commands from original successful deployments; preserve
    # model-specific flags rather than reconstruct defaults.
    manifests = {}
    for root in [BASE/'results/sharanga/resume_20261006/logs', BASE/'results/sharanga']:
        paths = sorted(root.glob('*/job.json'), reverse=True)
        for path in paths:
            record = read_json(path)
            model = record.get('environment', {}).get('MODEL')
            if model in counts and model not in manifests:
                manifests[model] = record
    if set(manifests) != set(counts):
        raise ValueError('Missing previous serving provenance: '+str(set(counts)-set(manifests)))
    deployment = oldplan['deployment']
    for i, model in enumerate(ROSTER):
        deployment['models'][model]['base_url'] = f'http://127.0.0.1:{24000+i}/v1'
    (bundle/'deployment_report.json').write_text(json.dumps(deployment, indent=2))
    lib = (code/'cluster/lib.sh').read_text()
    lib = lib.replace('if command -v git >/dev/null; then',
        'if [[ -n "${AGL_SUBMIT_GIT_COMMIT:-}" ]]; then\n        AGL_GIT_COMMIT="$AGL_SUBMIT_GIT_COMMIT"\n        AGL_GIT_STATUS="$AGL_SUBMIT_GIT_STATUS"\n    elif command -v git >/dev/null; then', 1)
    branch = '''    if [[ "$kind" == main ]]; then
        setsid python "$REPO_ROOT/run_main.py" \\
            --models llama-3.1-8b-instruct qwen3-8b gemma-3-12b-it mistral-small-3.1-24b-instruct deepseek-r1-distill-qwen-14b gpt-oss-120b \\
            --config "$EXPERIMENT_CONFIG" --deployment-report "$PILOT_DEPLOYMENT_REPORT" \\
            --generation-overrides "$PILOT_GENERATION_OVERRIDES" --power-report "$MAIN_POWER_REPORT" \\
            --seeds-per-cell 16 --seed-start 1001 --output "$PILOT_OUTPUT" \\
            --approval "$PILOT_APPROVAL" --execute-model "$MODEL" --execute-seeds $MAIN_EXECUTE_SEEDS \\
            > "$RUN_DIR/experiment.log" 2>&1 &
    elif [[ "$kind" == pilot ]]; then'''
    lib = lib.replace('    if [[ "$kind" == pilot ]]; then', branch, 1)
    (bundle/'main_lib.sh').write_text(lib)
    wrapper = '''#!/bin/bash
set -eo pipefail
source "$HOME/miniconda3/bin/activate" slaybench
python "$MAIN_BUNDLE/code/cluster/launch_deadline_main.py" --bundle "$MAIN_BUNDLE" --run-worker "$MAIN_WORKER"
'''
    # Worker execution is handled by the separate checked-in entry point below.
    wrapper = wrapper.replace('launch_deadline_main.py', 'run_deadline_worker.py')
    (bundle/'worker.sbatch').write_text(wrapper)
    workers = []
    for lane in range(2):
        for stage in ('cheap', 'deep'):
            models = [m for m in ROSTER if m not in ('deepseek-r1-distill-qwen-14b', 'gpt-oss-120b')] if stage == 'cheap' else ['deepseek-r1-distill-qwen-14b']
            workers.append(dict(id=f'{stage}_{lane}', partition='gpu_h100_4', qos='qos_gpu_h100',
                                walltime='20:00:00' if stage == 'cheap' else '72:00:00', lane=lane, models=models))
    workers.append(dict(id='gpt', partition='gpu_h200_8', qos='qos_gpu_h200', walltime='08:00:00', lane=None, models=['gpt-oss-120b']))
    jobs = []
    for worker in workers:
        tasks = []
        for model in worker['models']:
            n = counts[model]
            seeds = list(range(1001, 1001+n))
            if worker['lane'] is not None:
                seeds = seeds[worker['lane']::2]
            root = bundle/'shards'/worker['id']/model
            root.mkdir(parents=True)
            env0 = manifests[model]['environment']
            command = manifests[model]['vllm_command']
            model_path = command[2]
            if not Path(model_path).is_absolute():
                hub = Path(env0['HF_HOME'])/'hub'/('models--'+env0['MODEL_REPO'].replace('/','--'))
                ref = hub/'refs/main'
                if ref.is_file():
                    model_path = str(hub/'snapshots'/ref.read_text().strip())
                else:
                    snapshots = [x for x in (hub/'snapshots').glob('*') if (x/'config.json').is_file()]
                    if len(snapshots) != 1:
                        raise ValueError('Ambiguous cached model revision: '+model)
                    model_path = str(snapshots[0])
            if not (Path(model_path)/'config.json').is_file():
                raise ValueError('Missing cached model '+model_path)
            env = dict(REPO_ROOT=str(code), MODEL=model, MODEL_REPO=env0['MODEL_REPO'], MODEL_PATH=model_path,
                       SERVED_NAME=model, TP_SIZE='1', PORT=str(24000+ROSTER.index(model)),
                       MAX_MODEL_LEN=env0['MAX_MODEL_LEN'], GPU_MEM_UTIL=env0['GPU_MEM_UTIL'],
                       EXTRA_SERVE_ARGS=env0.get('EXTRA_SERVE_ARGS', ''), HEALTH_WAIT_MIN='120',
                       CONDA_ENV='slaybench', HF_HOME=env0['HF_HOME'], LOCAL_TIMEOUT='600',
                       EXPERIMENT_CONFIG=str(code/'experiments/main_study/config.json'),
                       PILOT_DEPLOYMENT_REPORT=str(bundle/'deployment_report.json'),
                       PILOT_GENERATION_OVERRIDES=str(code/'experiments/main_study/generation_overrides.json'),
                       MAIN_POWER_REPORT=str(bundle/'power_report.json'), PILOT_OUTPUT=str(root/'output'),
                       PILOT_APPROVAL=str(root/'approval.json'), MAIN_EXECUTE_SEEDS=' '.join(map(str,seeds)),
                       AGL_SUBMIT_GIT_COMMIT=read_json(code/'evidence/provenance.json')['git_commit'],
                       AGL_SUBMIT_GIT_STATUS='Frozen working-tree sources; see evidence/source_hashes.json')
            cmd = [PYTHON, str(code/'run_main.py'), '--models', *ROSTER, '--config', env['EXPERIMENT_CONFIG'],
                   '--deployment-report', env['PILOT_DEPLOYMENT_REPORT'], '--generation-overrides', env['PILOT_GENERATION_OVERRIDES'],
                   '--power-report', env['MAIN_POWER_REPORT'], '--seeds-per-cell', '16', '--seed-start', '1001',
                   '--output', env['PILOT_OUTPUT'], '--dry-run']
            subprocess.run(cmd, cwd=code, check=True, stdout=subprocess.DEVNULL)
            saved = read_json(root/'output/run_plan.json')
            (root/'approval.json').write_text(json.dumps(dict(plan_sha256=digest(saved), confirmed_in_conversation=True,
                                                           approval_reference=REFERENCE), indent=2))
            tasks.append(dict(model=model, seeds=seeds, environment=env))
        worker['tasks'] = tasks
        jobs.append(worker)
    (bundle/'workers.json').write_text(json.dumps(jobs, indent=2))
    subprocess.run(['bash', '-n', str(bundle/'worker.sbatch'), str(bundle/'main_lib.sh')], check=True)
    assert sum(len(t['seeds'])*8 for w in jobs for t in w['tasks']) == 376
    for model in ROSTER:
        assigned = [seed for w in jobs for t in w['tasks'] if t['model']==model for seed in t['seeds']]
        assert len(assigned) == len(set(assigned)) == counts[model]
    print('Verified dry-runs and disjoint ownership: 376 fresh episodes', flush=True)
    if not a.submit:
        return
    ledger=[]
    for worker in jobs:
        dependency = next((r['job_id'] for r in ledger if r['worker']==f"cheap_{worker['lane']}"), None) if worker['id'].startswith('deep') else None
        env = dict(os.environ, MAIN_BUNDLE=str(bundle), MAIN_WORKER=worker['id'])
        cmd=['sbatch','--parsable','--export=ALL','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=4',
             '--mem='+('128G' if worker['id']=='gpt' else '96G'),'--partition='+worker['partition'],
             '--qos='+worker['qos'],'--time='+worker['walltime'],'--job-name=agl_main_'+worker['id'],
             '--output='+str(bundle/'main_%j.out')]
        if dependency:cmd.append('--dependency=afterok:'+dependency)
        cmd.append(str(bundle/'worker.sbatch'))
        jid=subprocess.check_output(cmd,env=env,cwd=code,text=True).strip().split(';')[0]
        ledger.append(dict(worker=worker['id'],job_id=jid,dependency=dependency))
        (bundle/'submissions.json').write_text(json.dumps(ledger,indent=2))
        print(worker['id'],jid,flush=True)


if __name__=='__main__':main()
