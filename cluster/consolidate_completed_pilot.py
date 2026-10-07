"""Consolidate the selected temperature-0.7 pilot shards without changing rows."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.study_common import read_study, file_digest, write_json, write_run_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--gpt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original, old_rows = read_study(args.original, 'pilot')
    gpt, gpt_rows = read_study(args.gpt, 'pilot')
    a, b = original['plan'], gpt['plan']
    for key in ('configuration', 'models', 'seeds', 'seeds_per_cell', 'generation_seed_policy'):
        if a[key] != b[key]:
            raise ValueError(f'Shard mismatch: {key}')
    for ac, bc in zip(a['cells'], b['cells']):
        for key in ('model', 'reasoning_mode', 'N', 'seeds', 'generation'):
            if ac[key] != bc[key]:
                raise ValueError(f'Cell mismatch: {key}')
    if any(r['model'] == 'gpt-oss-120b' for r in old_rows):
        raise ValueError('Original shard unexpectedly already contains GPT')
    if any(r['model'] != 'gpt-oss-120b' for r in gpt_rows):
        raise ValueError('GPT shard contains another model')
    rows = old_rows + gpt_rows
    if len(rows) != 240 or any(r['status'] != 'completed' for r in rows):
        raise ValueError('Expected exactly 240 completed episodes')
    for r in rows:
        if len(r['generation_by_seat']) != r['N'] or any(
            g['temperature'] != .7 or g['top_p'] != 1.0 for g in r['generation_by_seat']
        ):
            raise ValueError('Episode sampling settings mismatch')
    provenance = {
        'selection': 'User selected completed original temperature-0.7 pilot; fresh Llama/Qwen reruns excluded.',
        'sources': [
            {'path': str(args.original.resolve()), 'sha256': file_digest(args.original),
             'plan_sha256': original['plan_sha256'], 'episodes': len(old_rows), 'parser': 'original extraction (v1)'},
            {'path': str(args.gpt.resolve()), 'sha256': file_digest(args.gpt),
             'plan_sha256': gpt['plan_sha256'], 'episodes': len(gpt_rows), 'parser': 'final_decision_v2'},
        ],
        'limitation': 'Parser version differs by model (GPT v2, others original). This is pilot variance evidence, not confirmatory evidence.',
        'sampling_verified': {'temperature': .7, 'top_p': 1.0, 'rounds': 30, 'seeds': a['seeds']},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / 'pilot.jsonl'
    manifest = dict(original, consolidation_provenance=provenance)
    with target.open('w', encoding='utf-8') as stream:
        for record in [manifest, *rows]:
            stream.write(json.dumps(record) + '\n')
    read_study(target, 'pilot')
    write_json(args.output / 'consolidation_provenance.json', provenance)
    write_run_summary(args.output, a, rows)
    print('Validated 240 completed episodes, temperature=0.7, top_p=1.0')


if __name__ == '__main__':
    main()
