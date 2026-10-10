"""Offline frozen-code integrity check; no model loading or network calls."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
lock=json.loads((HERE/'registration/protocol_lock_stage3.json').read_text())
for name,expected in lock['files'].items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual!=expected: raise ValueError('Frozen source mismatch: '+name)
print('Verified',len(lock['files']),'frozen source/manifest hashes; no inference')
