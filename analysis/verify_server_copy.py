"""POST-HOC / EXPLORATORY: compare isolated server result copy to local tables."""
import argparse
import json
from pathlib import Path
import pandas as pd
from common import HERE, SOURCE_HASH

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('server_analysis',type=Path); a=ap.parse_args(); checks=[]
    for file in sorted((HERE/'results').glob('*.csv')):
        remote=a.server_analysis/'results'/file.name
        if not remote.exists(): raise FileNotFoundError(remote)
        pd.testing.assert_frame_equal(pd.read_csv(file),pd.read_csv(remote),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
        checks.append(file.name)
    gate=json.loads((a.server_analysis/'results/00_gate.json').read_text()); assert gate['passed'] and gate['source_sha256']==SOURCE_HASH
    result=dict(label='POST-HOC / EXPLORATORY',source_sha256=SOURCE_HASH,all_csvs_match=True,tolerance=dict(rtol=1e-10,atol=1e-10),tables=checks,remote_packages=json.loads((a.server_analysis/'package_versions.json').read_text()),jobs=json.loads((a.server_analysis/'logs/20261010T172837Z/submissions.json').read_text()))
    (HERE/'server_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print('Verified',len(checks),'CSV tables against Sharanga')

if __name__=='__main__': main()
