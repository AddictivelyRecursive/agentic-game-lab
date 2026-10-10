# Internal prospective record — 2026-10-11

The author superseded the external OSF requirement on 2026-10-11. Collection is authorized using the dated plan, fixed seed manifests, analysis specification and hashed source archive in this repository. This is an internal prospective record, NOT external preregistration. The appendix write-up may follow later. Original completed-cohort analyses remain post-hoc / exploratory.


---

# Prospective reviewer revision

Read PLAN.md for the staged design and gates. The current exact-message batch contains 32 saved-state requests (Qwen 24, DeepSeek 8), not new game episodes. Larger grids are not auto-submitted. Stage 0b is complete; see STAGE_0B_MEMO.md and PARSER_V3_SENSITIVITY.md.

Local preparation:

```
python experiments/reviewer_revision_20261010/build_bank.py
python experiments/reviewer_revision_20261010/calibrate.py --model qwen3-8b --bank experiments/reviewer_revision_20261010/request_bank.jsonl --validate-only
python experiments/reviewer_revision_20261010/calibrate.py --model deepseek-r1-distill-qwen-14b --bank experiments/reviewer_revision_20261010/request_bank.jsonl --validate-only
```

Current frozen Sharanga diagnostic directory: `/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/reviewer_revision_v2_exact_20261011`.

Current jobs: Qwen 388864 (120 minutes), DeepSeek 388865 (99 minutes). Previous jobs 388834/388835 were cancelled: the initial prompt bank used compact JSON instead of the original indent=2 formatting. Its records remain archived and are non-qualifying. The old Qwen allocation consumed 1,133 seconds; old DeepSeek consumed zero. Including both replacement job limits, total calibration allocation is capped at 3.965 GPU-hours. See replacement_receipt.json and submissions_v2_exact.json. The submission script refuses duplicate receipts. Cached snapshots are explicitly pinned; no model downloads.

Preparation and offline checks:

```
python experiments/reviewer_revision_20261010/diagnose_saved.py
python experiments/reviewer_revision_20261010/parser_sensitivity.py
python experiments/reviewer_revision_20261010/prepare_cohorts.py
python experiments/reviewer_revision_20261010/test_protocol.py
python experiments/reviewer_revision_20261010/freeze_registration.py
```

Submit `registration/STAGE3_OSF_REGISTRATION.md` and its attachments using your OSF account before enrollment. Stage 1/2/4 drafts and unresolved dependencies are in registration/OTHER_COHORT_PACKETS.md. The source archive and protocol lock must not be changed after registration; changes require a new version. The runner refuses unresolved model settings or absent/mismatched external-registration receipts. No human-rating task is part of this extension.

`python monitor.py` records one availability/status snapshot; `python monitor.py --loop` checks every five minutes until stopped. No persistent monitor is currently installed. Slurm already starts pending resource requests when eligible GPUs become available; repeatedly submitting after polling would risk duplicate jobs. The monitor therefore never submits or bypasses scheduler allocations. Reported global GPU inventory does not imply permission to claim a GPU outside Slurm.

The paper source/PDF were backed up locally before removing main-paper explanation-audit claims and the unimplemented human-coding protocol. Historical post-hoc outputs remain unchanged. `revise_paper.py` is a one-time edit script and refuses reapplication after its backup exists.

## Superseding execution update — 2026-10-11

Author instruction removed the OSF prerequisite. The frozen design, seeds and analysis specification are internal prospective records, not external preregistration. The appendix write-up is deferred. See registration/INTERNAL_PROSPECTIVE_RECORD.md and internal_archive_receipt.json; source archive SHA256 bf1520b7c7832136823277385b9e4ad301af03dd7f74ac56bd15c2c324953f33.

Initial Stage 3 collection accepted by Sharanga: Gemma job 388987 (16 episodes), Mistral 388988 and 388989 (8 episodes each). Both first jobs have afterany dependencies on calibration 388864 and 388865; Mistral part 2 depends on 388988. At most two study GPUs can run concurrently. This is one balanced block per model, not full-grid enrollment. Check measured runtime and reliability before submitting further blocks. GPT-OSS remains gated on instrumentation; Qwen/DeepSeek remain gated on calibration. Scheduler acceptance does not mean completed episodes.

Mistral preserves --tokenizer-mode mistral from its original deployment. The collector uses serving-side /tokenize for context checks; tokenizer/context failures leave an incomplete episode rather than generating artificial fallback outcomes. Four offline tests passed locally and on Sharanga; shell syntax checked, and the submission driver verified all 74 frozen source hashes before submission. Full transport responses and finish reasons are retained before parsing.
