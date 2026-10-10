# Internal prospective record — 2026-10-11

The author superseded the external OSF requirement on 2026-10-11. Collection is authorized using the dated plan, fixed seed manifests, analysis specification and hashed source archive in this repository. This is an internal prospective record, NOT external preregistration. The appendix write-up may follow later. Original completed-cohort analyses remain post-hoc / exploratory.


---

# Execution status â€” 2026-10-11 (Asia/Calcutta)

Original 376-episode primary results and raw logs remain unchanged. Retrospective diagnoses below are **POST-HOC / EXPLORATORY**; future cohorts are prospective follow-ups informed by inspected original data. No new cohort episodes have been collected.

| Work | Status / evidence |
| --- | --- |
| Stage 0 exact-message calibration | Qwen job 388864 (24 requests, 120-minute limit), DeepSeek 388865 (8 requests, 99-minute limit); pending scheduler priority at latest check |
| Original diagnostic correction | Jobs 388834/388835 cancelled; original bank used compact JSON instead of indent=2. Old Qwen records are non-qualifying and retained; 1,133 allocated seconds consumed; DeepSeek consumed zero |
| Calibration GPU envelope | Old allocation plus both replacement job limits = 3.96472 GPU-hours; at most two GPUs concurrently |
| Stage 0b | Complete, no new inference; failure taxonomy, episode-cluster intervals, conservative v3 re-extraction, conditional outcome sensitivity, CSV and LaTeX |
| Stage 1 | Seed manifest prepared; token allowances and OSF packet await exact-message calibration |
| Stage 2 | Independent 192-episode manifest and draft analysis prepared; GPT-OSS instrumentation diagnostic still required before enrollment |
| Stage 3 | 576-episode manifest, four-arm prompts, runner, analysis and OSF packet prepared; awaits author's immutable OSF registration |
| Stage 4 | Separate 16-episode smoke and 192-episode primary frozen manifests prepared; F-only floor rule fixed before outcomes; dynamic controls shared with Stage 3 only under exact protocol match |
| Stage 5 | Existing cache/configuration/tokenizer evidence collected for 70B; runtime/VRAM/five-round smoke untested; no model enrolled or downloaded |
| Stage 6 | Deferred under the plan's go/no-go rules; submission cycle remains unselected |

## Stage 0b findings

Llama has 871 initial validity failures. Parser v3 recovers 800 without changing any v2-valid action, raising pooled first-attempt validity from 92.59% to 99.40%. It permits only literal LF/CR/TAB in complete JSON strings; it does not fill missing braces, infer actions or choose among multiple decisions. New Llama cohorts use a distinct protocol ID and remain separate from the original dataset.

GPT-OSS has 195 failures: 99 malformed/otherwise unextractable outputs and 96 calls with a logged no-final-content error but no saved output body. v3 recovers only 6, raising validity from 88.39% to 88.75%. Saved finish reasons and usage are unavailable, so truncation versus reasoning/template behavior cannot be resolved from these logs. Full GPT-OSS enrollment stays gated on an instrumented diagnostic.

Conditional F early/late cooperation is 23.40%/21.44% under v2-eligible first attempts and 23.58%/22.35% under v3-eligible first attempts for Llama. GPT-OSS is 13.51%/0.00% under both eligible subsets. These condition on successful extraction and retain observed histories; they are not counterfactual rerun trajectories. All executed primary outcomes remain unchanged.

## Precision and costs

At 12 seeds per cell, old-variance D CI half-widths are Mistral 4.21 pp and GPT-OSS 8.11 pp. At twice old variance they are 5.95 pp and 11.47 pp. The bounded-outcome t benchmark is 60.28 pp at n=12; plug-in targets are not guarantees. See results/planning_precision.csv and its LaTeX table for n=12/20/40 and all sensitivity margins.

Historical 96-episode execution estimates, excluding loading, are Mistral 10.36 GPU-hours and GPT-OSS 23.31. Corrected-budget runtime can differ substantially. Qwen's real cost sheet awaits qualifying calibration responses; no cost is fabricated from incomplete or nonmatching prompt runs.

The 768 Stage 2/3 episodes are provisionally divided into 96 whole-episode, model-homogeneous chunks. Their 60â€“120 minute limits use a 15-minute loading assumption and 2x historical episode runtime, and must be revised from measured timings where protocols change. None is submitted. The future execution rule uses two global Slurm dependency lanes; a TP=2 window drains both. Slurm allocates queued jobs; the five-minute monitor never duplicates requests or claims GPUs outside an allocation. No persistent monitor is installed.

## Registration and implementation

The author selected OSF and will submit the packet. Ready packet: registration/STAGE3_OSF_REGISTRATION.md, with fixed manifest, protocol lock, source archive, upload hashes and analysis. Other cohorts' draft specifications identify unresolved decisions rather than inventing protocol values. The runner refuses collection without a matching immutable-registration receipt, locked model settings, source/configuration integrity and a Slurm allocation. It records full responses without substituting reasoning text, retains fallback episodes, and preserves interrupted attempts before a same-ID restart from the initial boundary. Completed episodes cannot be rerun. The analysis refuses significance tests for an incomplete cohort.

Offline checks cover exact original-message reconstruction, EV information/order isolation, parser rejection of truncated/ambiguous/invalid actions, full repair-to-a4 fallback, frozen B_eff and D=3 times the slope difference. Four tests pass locally and on Sharanga. All 71 frozen source/manifest hashes pass on Sharanga. The invalid example OSF receipt is correctly refused before any model call; the incomplete-cohort analysis correctly withholds significance tests.

## Deployment caveats and paper housekeeping

DeepSeek temperature .7 is in its recommended range, but the original system message conflicts with the card's recommendation to place instructions in the user message. Calibration retains original messages to identify output-budget differences; a card-consistent prompt is a separate intervention, not a silent fix. [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B#usage-recommendations), checked 2026-10-11.

Mistral's standard hub snapshot contains metadata but lacks weights. Its complete converted local deployment, `/scratch/jagatsesh/agentic-game-lab/model-repair/mistral_snapshot`, matches the path in the original main worker manifests; configuration/tokenizer hashes and shard sizes are recorded. No download or checkpoint substitution occurred, and no old/new pooling equivalence is claimed.

Equation (4) already has the correct factor 3 and required no change. The execution appendix's self-reference now points to its actual serving/reliability tables. The Conclusion now says execution reliability checks; no explanation audit or human-rating promise is implied. The manuscript compiles successfully.

Current diagnostic bundle: `/home/jagatsesh/f20220604_ts/agentic-game-lab/results/sharanga/reviewer_revision_v2_exact_20261011`. Prepared cohort source/evidence: `results/sharanga/reviewer_preparation_final_20261011` under the same server project. Original diagnostic bundle is retained separately.

## Superseding execution update — 2026-10-11

Author instruction removed the OSF prerequisite. The frozen design, seeds and analysis specification are internal prospective records, not external preregistration. The appendix write-up is deferred. See registration/INTERNAL_PROSPECTIVE_RECORD.md and internal_archive_receipt.json; source archive SHA256 bf1520b7c7832136823277385b9e4ad301af03dd7f74ac56bd15c2c324953f33.

Initial Stage 3 collection accepted by Sharanga: Gemma job 388987 (16 episodes), Mistral 388988 and 388989 (8 episodes each). Both first jobs have afterany dependencies on calibration 388864 and 388865; Mistral part 2 depends on 388988. At most two study GPUs can run concurrently. This is one balanced block per model, not full-grid enrollment. Check measured runtime and reliability before submitting further blocks. GPT-OSS remains gated on instrumentation; Qwen/DeepSeek remain gated on calibration. Scheduler acceptance does not mean completed episodes.

Mistral preserves --tokenizer-mode mistral from its original deployment. The collector uses serving-side /tokenize for context checks; tokenizer/context failures leave an incomplete episode rather than generating artificial fallback outcomes. Four offline tests passed locally and on Sharanga; shell syntax checked, and the submission driver verified all 74 frozen source hashes before submission. Full transport responses and finish reasons are retained before parsing.
