"""Verify cached weight-file presence and resolve a pinned snapshot, without loading it."""
import argparse
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--hub',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    folder=a.hub/('models--'+a.repo.replace('/','--')); ref=folder/'refs/main'
    pinned={'Qwen/Qwen3-8B':'b968826d9c46dd6066d109eabc6255188de91218','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B':'1df8507178afcc1bef68cd8c393f61a886323761'}
    if a.repo not in pinned: raise ValueError('Snapshot must be explicitly inspected and pinned')
    revision=pinned[a.repo]; snapshot=folder/'snapshots'/revision
    cfg=json.loads((snapshot/'config.json').read_text())
    indices=list(snapshot.glob('*.safetensors.index.json'))
    files=sorted(set(v for p in indices for v in json.loads(p.read_text())['weight_map'].values())) if indices else [p.name for p in snapshot.glob('*.safetensors')]
    if not files or any(not (snapshot/f).is_file() or not (snapshot/f).stat().st_size for f in files): raise ValueError('Cached weight files missing/empty')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(repo=a.repo,revision=revision,snapshot=str(snapshot),architectures=cfg.get('architectures'),max_position_embeddings=cfg.get('max_position_embeddings'),weight_files=files,weight_bytes=sum((snapshot/f).stat().st_size for f in files),note='Presence check; not a historical weight hash or runtime success claim'),indent=2))
    print(snapshot)

if __name__=='__main__': main()
