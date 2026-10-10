"""Package preparation evidence separately from the frozen registered source."""
from pathlib import Path
import tarfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.jsonl','.csv','.tex','.txt','.sbatch'] and '__pycache__' not in p.parts]
with tarfile.open(HERE/'preparation_evidence.tar.gz','w:gz') as tar:
    for p in files: tar.add(p,arcname=p.relative_to(ROOT).as_posix())
print('Prepared',len(files),'evidence files; this archive does not launch anything')
