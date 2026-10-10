"""POST-HOC / EXPLORATORY: sequential saved-data analysis with hard gate."""
import argparse
import subprocess
import sys
from common import DEFAULT_INPUT, HERE

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default=DEFAULT_INPUT); ap.add_argument('--export-dir',default=HERE.parent/'results/analysis/main_h5_coding_20261010'); args=ap.parse_args()
    for name in ['00_reproduction.py','01_robustness.py','02_heterogeneity.py','03_persistence.py','04_mismatch.py']:
        cmd=[sys.executable,str(HERE/name),'--input',str(args.input)]
        if name=='04_mismatch.py': cmd+=['--export-dir',str(args.export_dir)]
        subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,str(HERE/'finalize.py')],check=True)

if __name__=='__main__': main()
