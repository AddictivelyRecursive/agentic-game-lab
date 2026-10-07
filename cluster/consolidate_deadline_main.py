"""Merge only completed, disjoint fresh-main shards; preserve original traces."""
import argparse
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
sys.path.insert(0,str(a.bundle/'code'))
from experiments.study_common import read_study, digest, require_complete, write_run_summary

workers=json.loads((a.bundle/'workers.json').read_text())
manifest=None
rows=[]
sources=[]
seen=set()
for worker in workers:
    for task in worker['tasks']:
        source=Path(task['environment']['PILOT_OUTPUT'])/'main.jsonl'
        current, records=read_study(source,'main')
        if manifest is None:manifest=current
        if current['plan_sha256']!=manifest['plan_sha256']:
            raise ValueError('Shard plan mismatch')
        expected={(task['model'],mode,n,seed) for mode in ('structured','free_form')
                  for n in (2,3,4,5) for seed in task['seeds']}
        actual={(r['model'],r['reasoning_mode'],r['N'],r['seed']) for r in records}
        if actual!=expected or actual & seen or any(r['status']!='completed' for r in records):
            raise ValueError('Incomplete, failed, duplicate or unowned shard: '+str(source))
        seen.update(actual)
        rows.extend(records)
        sources.append({'source':str(source),'episodes':len(records),'worker':worker['id']})
require_complete(manifest,rows)
if len(rows)!=376:raise ValueError('Expected 376 fresh episodes')
a.output.mkdir(parents=True,exist_ok=False)
with (a.output/'main.jsonl').open('x') as stream:
    for row in [manifest,*rows]:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
(a.output/'run_plan.json').write_text(json.dumps(manifest['plan'],indent=2))
(a.output/'consolidation.json').write_text(json.dumps({'pilot_credit':0,'sources':sources,
                                                     'plan_sha256':digest(manifest['plan'])},indent=2))
read_study(a.output/'main.jsonl','main')
write_run_summary(a.output,manifest['plan'],rows)
print('Consolidated 376 fresh episodes; original per-agent traces retained at source paths.')
