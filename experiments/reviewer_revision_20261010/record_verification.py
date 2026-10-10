"""Record observed offline verification; no new tests, model calls or registration."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
lock=HERE/'registration/protocol_lock_stage3.json'
record=dict(client_date='2026-10-11',client_timezone='Asia/Calcutta',server_preparation_directory='/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/reviewer_preparation_final_20261011',local_offline_tests_passed=4,sharanga_offline_tests_passed=4,sharanga_verified_source_hashes=71,example_registration_receipt_correctly_rejected=True,incomplete_cohort_analysis_correctly_withheld_inference=True,model_inference_in_verification=False,source_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),source_archive_sha256=hashlib.sha256((HERE/'registration/stage3_source.tar.gz').read_bytes()).hexdigest(),note='Observed tool results from this session; preparation only, not a registered or collected cohort')
(HERE/'registration/server_preparation_verification.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
