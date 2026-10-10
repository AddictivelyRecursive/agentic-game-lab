"""Read-only deployment evidence; no imports that load model weights."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
HUB=Path('/scratch/jagatsesh/hf/hub')
REPOS=['meta-llama/Llama-3.1-8B-Instruct','mistralai/Mistral-Small-3.1-24B-Instruct-2503','google/gemma-3-12b-it','openai/gpt-oss-120b','meta-llama/Llama-3.3-70B-Instruct']
out=[]
for repo in REPOS:
    folder=HUB/('models--'+repo.replace('/','--'))
    locations=sorted((folder/'snapshots').glob('*'))
    if repo=='mistralai/Mistral-Small-3.1-24B-Instruct-2503': locations.append(Path('/scratch/jagatsesh/agentic-game-lab/model-repair/mistral_snapshot'))
    for snapshot in locations:
        if not snapshot.is_dir(): continue
        cfg=json.loads((snapshot/'config.json').read_text())
        indices=list(snapshot.glob('*.safetensors.index.json'))
        weights=sorted({v for p in indices for v in json.loads(p.read_text())['weight_map'].values()}) if indices else [p.name for p in snapshot.glob('*.safetensors')]
        files=[p for p in snapshot.iterdir() if p.is_file() and (p.name.startswith('tokenizer') or p.name in ['config.json','generation_config.json','special_tokens_map.json','chat_template.jinja','chat_template.json','preprocessor_config.json','processor_config.json','tekken.json','params.json','model.safetensors.index.json'])]
        out.append(dict(repo=repo,revision=snapshot.name,snapshot_path=str(snapshot),architectures=cfg.get('architectures'),context=cfg.get('max_position_embeddings'),quantization=cfg.get('quantization_config'),weights_present=bool(weights) and all((snapshot/w).is_file() and (snapshot/w).stat().st_size>0 for w in weights),weight_bytes=sum((snapshot/w).stat().st_size for w in weights if (snapshot/w).is_file()),weight_sizes={w:(snapshot/w).stat().st_size for w in weights if (snapshot/w).is_file()},file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},note='Presence and configuration only; no runtime compatibility or VRAM claim. mistral_snapshot is the converted local deployment recorded in original main workers, not an upstream revision ID.'))
versions={}
for name in ['vllm','torch','transformers','requests']:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
print(json.dumps(dict(models=out,software=versions),indent=2))
