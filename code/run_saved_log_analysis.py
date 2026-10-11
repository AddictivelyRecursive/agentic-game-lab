"""Run the saved-log reproduction gate followed by exploratory analyses."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('Provide the original main JSONL; raw logs are not included in this release.')
    here = Path(__file__).resolve().parent
    for name in ['00_reproduction.py', '01_robustness.py', '02_heterogeneity.py', '03_persistence.py']:
        subprocess.run([sys.executable, str(here/'analysis'/name), '--input', str(args.input.resolve())], check=True)
    subprocess.run([sys.executable, str(here/'analysis/narrative_support_20261011.py'),
                    '--input', str(args.input.resolve())], check=True)


if __name__ == '__main__':
    main()
