"""Record prospective diagnostic allocation and software provenance."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import socket

ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
bundle=Path(os.environ['BUNDLE'])
record=dict(kind='prospective saved-state calibration; not episodes',hostname=socket.gethostname(),environment={k:os.getenv(k) for k in ['SLURM_JOB_ID','SLURM_JOB_PARTITION','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES','MODEL','MODEL_REPO','BUNDLE']},packages={p:md.version(p) for p in ['vllm','transformers','torch','requests']},sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file() and p.suffix in ['.py','.sbatch','.jsonl','.md']})
a.output.write_text(json.dumps(record,indent=2))
