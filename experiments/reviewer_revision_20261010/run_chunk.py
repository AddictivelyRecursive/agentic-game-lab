"""Run one fixed whole-episode chunk; no scheduler or enrollment decisions."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--chunk',required=True);p.add_argument('--base-url',required=True);a=p.parse_args()
    chunks=json.loads((HERE/'registration/chunks.json').read_text())['chunks']
    selected=[c for c in chunks if c['chunk_id']==a.chunk]
    if len(selected)!=1: raise ValueError('Unknown chunk')
    c=selected[0]
    if c['cohort_id']!='rr_stage3_fourarm_v2' or c['model'] not in ['mistral-small-3.1-24b-instruct','gemma-3-12b-it']: raise ValueError('Initial authorization is Mistral/Gemma Stage 3 only')
    lock=HERE/'registration/protocol_lock_stage3.json'; settings=json.loads(lock.read_text())['model_settings'][c['model']]
    for episode in c['episode_ids']:
        output=ROOT/'cohort_outputs'/episode
        if (output/'completed.json').exists(): raise ValueError('Completed output exists; refusing rerun')
        subprocess.run([sys.executable,str(HERE/'cohort_runner.py'),'--manifest',str(HERE/'registration/rr_stage3_fourarm_v2_episodes.csv'),'--episode-id',episode,'--registration',str(HERE/'registration/internal_archive_receipt.json'),'--lock',str(lock),'--base-url',a.base_url,'--model-path',settings['snapshot_path'],'--output',str(output),'--max-tokens','2048','--context','16384'],check=True)
if __name__=='__main__':main()
