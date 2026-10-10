"""Freeze a reviewable Stage 3 OSF source packet. Does not register or collect."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REG=HERE/'registration'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    inventory=json.loads((REG/'cache_inventory.json').read_text())
    mapping={'llama-3.1-8b-instruct':'meta-llama/Llama-3.1-8B-Instruct','gemma-3-12b-it':'google/gemma-3-12b-it','mistral-small-3.1-24b-instruct':'mistralai/Mistral-Small-3.1-24B-Instruct-2503'}
    settings={}
    for model,repo in mapping.items():
        found=[r for r in inventory['models'] if r['repo']==repo and r['weights_present']]
        if len(found)!=1: raise ValueError('Expected one verified cached deployment for '+model)
        r=found[0]
        settings[model]=dict(status='LOCKED',repo=repo,snapshot=r['revision'],snapshot_path=r['snapshot_path'],artifact_sha256=r['file_sha256'],weight_sizes=r['weight_sizes'],max_tokens=2048,context=16384,parser='v3_literal_whitespace_only_20261011' if model.startswith('llama-') else 'final_decision_v2',temperature=.7,top_p=1.,quantization=r['quantization'],activation_dtype='bfloat16',tensor_parallel=1,architecture=r['architectures'],extra_serving_flags=['--tokenizer-mode','mistral'] if model.startswith('mistral-') else [])
    files=sorted(set(list((ROOT/'src').rglob('*.py'))+list((ROOT/'src/AI_Agent/prompts').glob('*.txt'))+[HERE/name for name in ['cohort_protocol.py','cohort_runner.py','analyze_cohort.py','parser_v3.py','frozen_parser_v2.py','neutral_policy.txt','test_protocol.py','request_bank.jsonl','cohort.sbatch','run_chunk.py']]+[REG/'INTERNAL_PROSPECTIVE_RECORD.md',REG/'rr_stage3_fourarm_v2_episodes.csv',REG/'chunks.json']))
    lock=dict(protocol_id='rr_stage3_fourarm_v2_20261011',registered=False,files={p.relative_to(ROOT).as_posix():sha(p) for p in files},model_settings=settings,inference_software=inventory['software'],analysis_software={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels']},request_seed_policy='SHA256(episode_id:seat:round:attempt), first four bytes little endian modulo 2147483647',serving_flags=['--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','16384','--max-num-seqs','1','--gpu-memory-utilization','0.90','--generation-config','vllm','--enforce-eager'],note='Frozen prospective source/configuration; internal dated record under the author instruction of 2026-10-11; not externally preregistered. Weight identity uses cached deployment plus configuration/tokenizer hashes and shard sizes; full weight-file SHA256 not yet measured. No historical pooling equivalence claimed.')
    path=REG/'protocol_lock_stage3.json'; path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    archive=REG/'stage3_source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[path,REG/'cache_inventory.json',REG/'cohort_manifest.json']:
            tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    receipt=dict(record_type='internal_prospective_archive',externally_registered=False,archived_at_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(REG/'rr_stage3_fourarm_v2_episodes.csv'),protocol_lock_sha256=sha(path),source_archive_sha256=sha(archive))
    (REG/'internal_archive_receipt.json').write_text(json.dumps(receipt,indent=2))
    upload={p.name:sha(p) for p in [archive,path,REG/'INTERNAL_PROSPECTIVE_RECORD.md',REG/'rr_stage3_fourarm_v2_episodes.csv',REG/'cache_inventory.json']}
    (REG/'stage3_upload_manifest.json').write_text(json.dumps(upload,indent=2))
    print(json.dumps(dict(status='Prepared, not registered',files=len(files),upload_hashes=upload),indent=2))

if __name__=='__main__': main()
