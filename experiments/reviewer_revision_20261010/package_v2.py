"""Freeze only the reduced diagnostic code; no remote side effects."""
import hashlib
import json
from pathlib import Path
import tarfile
HERE=Path(__file__).resolve().parent
names=['PLAN.md','README.md','calibrate.py','calibration.sbatch','cache_snapshot.py','runtime_manifest.py','frozen_parser_v2.py','request_bank.jsonl','submit_calibration.py','replace_remote.py']
manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
(HERE/'diagnostic_v2_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(HERE/'diagnostic_v2.tar.gz','w:gz') as tar:
    for name in names+['diagnostic_v2_manifest.json']: tar.add(HERE/name,arcname=name)
print('Frozen v2 diagnostic bundle; DeepSeek=8 requests. Existing Qwen job remains on v1.')
