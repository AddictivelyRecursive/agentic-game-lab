# Prospective reviewer revision

Read PLAN.md for the staged design and gates. The first submitted batch contains only 48 saved-state diagnostic requests, not new game episodes. Two single-GPU jobs, two hours each, cap this batch at four allocated GPU-hours. Larger grids are not auto-submitted.

Local preparation:

```
python experiments/reviewer_revision_20261010/build_bank.py
python experiments/reviewer_revision_20261010/calibrate.py --model qwen3-8b --bank experiments/reviewer_revision_20261010/request_bank.jsonl --validate-only
python experiments/reviewer_revision_20261010/calibrate.py --model deepseek-r1-distill-qwen-14b --bank experiments/reviewer_revision_20261010/request_bank.jsonl --validate-only
```

Frozen Sharanga directory: `/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/reviewer_revision_20261010`.

Submitted jobs: Qwen 388834; DeepSeek 388835. Inspect `submissions.json`, `runs/cal_JOBID.out`, and per-job runtime/response records. The submission script refuses duplicate submission after a receipt exists. Cached snapshots are explicitly pinned in cache_snapshot.py; no model downloads.

`python monitor.py` records one availability/status snapshot; `python monitor.py --loop` checks every five minutes until stopped. No persistent monitor is currently installed. Slurm already starts pending resource requests when eligible GPUs become available; repeatedly submitting after polling would risk duplicate jobs. The monitor therefore never submits or bypasses scheduler allocations. Reported global GPU inventory does not imply permission to claim a GPU outside Slurm.

The paper source/PDF were backed up locally before removing main-paper explanation-audit claims and the unimplemented human-coding protocol. Historical post-hoc outputs remain unchanged. `revise_paper.py` is a one-time edit script and refuses reapplication after its backup exists.
