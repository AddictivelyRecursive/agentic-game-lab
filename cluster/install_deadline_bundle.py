"""Install our archive into a new immutable main-run directory and submit."""
from pathlib import Path
import subprocess
import tarfile

bundle=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/main_deadline_20261007_v2')
bundle.mkdir()
with tarfile.open('/tmp/main_deadline_20261007.tar.gz') as archive:
    for member in archive.getmembers():
        target=(bundle/member.name).resolve()
        if not target.is_relative_to(bundle) or member.issym() or member.islnk():
            raise ValueError('Unsafe archive member')
    archive.extractall(bundle)
subprocess.run(['/home/jagatsesh/miniconda3/envs/slaybench/bin/python',
                str(bundle/'code/cluster/launch_deadline_main.py'),'--bundle',str(bundle),'--submit'],check=True)
