"""Integrity-checked access to original or anonymized saved logs; no inference."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SHA256 = '149ac53f8a19286b429d1de83939ae91028845b0b35a18e1dc8dd5cc0468b778'
DEFAULT_INPUT = ROOT / 'results/logs/main.jsonl.gz'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def open_text(path):
    path = Path(path)
    return gzip.open(path, 'rt', encoding='utf8') if path.suffix == '.gz' else path.open(encoding='utf8')


def verify_main(path):
    path = Path(path)
    digest = sha256(path)
    if digest == ORIGINAL_SHA256:
        return digest
    manifest = json.loads((ROOT / 'results/MANIFEST.json').read_text())
    expected = manifest['logs']['main']
    if digest != expected['sha256']:
        raise ValueError('Unapproved saved-log digest')
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected['uncompressed_sha256']:
        raise ValueError('Uncompressed saved-log digest mismatch')
    return digest


def verify_agent_inputs(episode_ids):
    manifest = json.loads((ROOT / 'results/MANIFEST.json').read_text())
    expected = manifest['logs']['agent_inputs']
    path = ROOT / expected['file']
    if sha256(path) != expected['sha256']:
        raise ValueError('Agent-input digest mismatch')
    seen = set()
    with open_text(path) as f:
        for line in f:
            row = json.loads(line)
            key = (row['episode_id'], row['seat'], row['round'])
            assert key not in seen and key[0] in episode_ids
            seen.add(key)
            assert row['raw_input']['temporal_context'] == dict(
                total_rounds=30, rounds_remaining_including_current=31-row['round'])
    assert len(seen) == 39480 and {key[0] for key in seen} == episode_ids
    return len(seen), [dict(sha256=expected['sha256'], records=len(seen))]
