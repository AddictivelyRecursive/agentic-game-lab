"""Restore submission automation with short, persistent cron invocations."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

root=Path('/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/main_short_chunks_20261007')
prior=subprocess.check_output(['crontab','-l'],text=True)
(root/'crontab_before_feeder.txt').write_text(prior)
shutil.copy2('/tmp/feed_short_main_once.py',root/'feed_short_main.py')
wrapper=root/'feed_once.sh'
wrapper.write_text('#!/bin/bash\nexport PATH='+shlex.quote(os.environ['PATH'])+'\nexec /home/jagatsesh/miniconda3/envs/slaybench/bin/python -u '+str(root/'feed_short_main.py')+' --once\n')
line='*/5 * * * * /bin/bash '+str(wrapper)+' >> '+str(root/'feeder.log')+' 2>&1 # AGL_SHORT_MAIN_20261007'
clean=[v for v in prior.splitlines() if '# AGL_SHORT_MAIN_20261007' not in v]
subprocess.run(['crontab','-'],input='\n'.join([*clean,line])+'\n',text=True,check=True)
subprocess.run(['/bin/bash',str(wrapper)],check=True)
print(subprocess.check_output(['crontab','-l'],text=True))
counts={}
for source in root.glob('shards/*/*/output/main.jsonl'):
    for text in source.read_text().splitlines()[1:]:
        row=json.loads(text)
        key=(row['model'],row['status'])
        counts[key]=counts.get(key,0)+1
print('Episode counts:',counts)
