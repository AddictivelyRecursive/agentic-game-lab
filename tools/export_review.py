"""Build a history-free review ZIP from the explicit release allowlist."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def release_files():
    roots = ['src/AI_Agent', 'src/game_engine', 'tests', 'experiments/main_study']
    files = []
    for root in roots:
        files.extend(p for p in (ROOT / root).rglob('*') if p.is_file()
                     and p.suffix in {'.py', '.txt', '.json'})
    files.extend(ROOT.glob('*.py'))
    files = [p for p in files if p.name != 'token_audit.py']
    for name in ['README.md', '.gitignore', '.gitattributes', 'requirements-local.txt',
                 'requirements-study.txt', 'experiments/__init__.py',
                 'experiments/study_common.py', 'experiments/study_config.json',
                 'experiments/local_episode.json', 'experiments/local_smoke.json']:
        files.append(ROOT / name)
    files.extend((ROOT / 'experiments').glob('*.example.json'))
    files.extend((ROOT / 'analysis').glob('*.py'))
    files = [p for p in files if p.name not in {'submit_cpu.py', 'verify_server_copy.py'}]
    for name in ['README.md', 'SCHEMA.md', 'requirements.txt', 'detector_rules.py']:
        files.append(ROOT / 'analysis' / name)
    files.extend((ROOT / 'analysis/validation').glob('*.py'))
    for name in ['study_specification.md', 'experiment_workflow.md',
                 'H4_amendment_20261007.txt', 'H4_deadline_amendment_20261007.txt',
                 'H4_seed_optimization_amendment_20261007.txt', 'REVIEW_RELEASE.md']:
        files.append(ROOT / 'docs' / name)
    for name in ['README.md', 'main_study_jobs.py', 'lib.sh', 'submit.sh',
                 'sharanga_run.sbatch', 'sharanga_pilot.sbatch', 'sharanga_smoke.sbatch']:
        files.append(ROOT / 'cluster' / name)
    for pattern in ['*.tex', '*.sty', '*.bst', '*.bib', 'README.md', 'tables/*.tex', 'figures/*.pdf']:
        files.extend((ROOT / 'Thesis').glob(pattern))
    files.append(Path(__file__).resolve())
    return sorted({p for p in files if p.is_file() and '__pycache__' not in p.parts})

def main():
    files = release_files()
    forbidden = re.compile(r'/home/(?!reviewer(?:/|\b))[^/\s]+|[A-Z]:\\Users\\[^\\\s]+', re.I)
    for path in files:
        if path.suffix != '.pdf' and path != Path(__file__).resolve():
            if forbidden.search(path.read_text(encoding='utf-8-sig')):
                raise SystemExit(f'Identifying path or account in {path.relative_to(ROOT)}')
    output = ROOT / 'dist/acl-review.zip'
    output.parent.mkdir(exist_ok=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    print(f'{output}: {len(files)} files; no Git history or raw experiment logs')

if __name__ == '__main__':
    main()
