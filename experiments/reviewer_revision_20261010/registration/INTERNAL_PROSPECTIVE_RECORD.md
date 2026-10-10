# Internal prospective record — 2026-10-11

The author superseded the external OSF requirement on 2026-10-11. Collection is authorized using the dated plan, fixed seed manifests, analysis specification and hashed source archive in this repository. This is an internal prospective record, NOT external preregistration. The appendix write-up may follow later. Original completed-cohort analyses remain post-hoc / exploratory.

The previously prepared Stage 3 OSF draft below supplies the unchanged design and analysis specification. Its external submission instructions are superseded by this paragraph. No human-rating promises or validated explanation-audit claims are authorized. Maximum study concurrency remains two GPUs; all other calibration and model eligibility gates remain.

# OSF registration packet: concurrent scaffold ablation

Status: prospective follow-up motivated by already inspected original results. No Stage 3 episodes have been collected. This document is prepared for the author to submit as an immutable OSF registration. An OSF project draft or this dated file alone is not a preregistration.

## Question and hypotheses

Do supplied immediate-payoff scores, their presentation order, and the original structured wording have distinguishable effects on cooperation? Does Llama's previously observed positive scaffold-by-group-size interaction replicate in a fresh cohort? Original results are known: Llama D=+14.02 pp was the only individual interaction surviving six-model Holm correction. The new cohort is independent, with no retrospective pooling in the primary analysis.

## Design, sample and allocation

Three fixed models: Llama-3.1-8B-Instruct, Gemma-3-12B-IT, Mistral-Small-3.1-24B-Instruct-2503. Four group sizes N=2,3,4,5; 30 rounds; homogeneous self-play; four arms; 12 independent episodes per model-by-N-by-arm cell: 576 episodes total. The exact disjoint seeds, episode IDs, balanced blocks and randomized within-block order are in `rr_stage3_fourarm_v2_episodes.csv`. Twelve is fixed, not an interim significance stopping rule. No original or diagnostic episode belongs to this cohort.

Arms:

1. F: original free-form instruction and state, without immediate-payoff scores.
2. S: original structured instruction and descending immediate-payoff ranking.
3. EV-canonical: neutral instruction, same scores as EV-sorted, ordered by action 0 through 4.
4. EV-sorted: exactly the same neutral instruction and score values as EV-canonical, descending by score.

The score is the action-dependent term -8(1-a/4); the opponent term and K cancel across own actions. `neutral_policy.txt` fixes the common EV instruction. Message construction, schema, histories and action map are identical between EV arms except score-list order; offline tests verify this. The canonical-versus-F contrast measures the supplied-score presentation package, including the accompanying neutral instruction; it cannot isolate arithmetic from every wording difference. Sorted-versus-canonical isolates order under that common package. S-versus-sorted measures the original versus neutral framing.

Environment: M=5; a=0 full cooperation and a=4 full defection; c=1-a/4. B0=12, C=8, K=0, Bmin=9, Bmax=15; rolling drift window 8, eta=.35, rstar=.55; streak theta=.5, lambda=.25, tau=4; observation history and summary windows 10; perception noise 0. No communication or model memory.

## Protocol and execution

Temperature .7, top-p 1, 2,048 output tokens, 16,384 context, sequential requests, native chat templates, no reasoning parser or constrained decoding. A deterministic request seed derives from episode ID, seat, round and attempt using SHA256; it is not shown to agents. Llama uses `v3_literal_whitespace_only_20261011`; Gemma and Mistral retain v2. v3 only accepts complete decision JSON additionally containing literal LF/CR/TAB in strings. It never completes truncated JSON, infers prose actions, changes v2-valid actions or chooses among multiple decisions. The saved-log motivation and sensitivity are disclosed separately.

The original repair instruction is retained, with at most three repair calls after the initial call. After four unsuccessful calls the action is 4. Full response JSON, final and reasoning fields, finish reason, usage, HTTP errors and timings are saved. Reasoning content is never used in place of missing final content. Prompt-plus-output context is checked without input truncation. HTTP timeout is 600 seconds; no hidden transport retries. Failed calls are not replaced by new episode seeds. The exact tokenizer/configuration hashes, local converted Mistral deployment, software versions and source hashes are in `protocol_lock_stage3.json` and `cache_inventory.json`. No model download or environment upgrade is part of this registration.

Whole episodes form short, model-homogeneous chunks. At most two study GPUs are allocated concurrently. Changing a prompt, parser, checkpoint, serving stack or output allowance after collection starts creates a new protocol and cohort; it does not overwrite this cohort.

## Outcomes and estimands

The independent unit is the episode. Primary outcome is episode-mean cooperation across all seats and 30 rounds, expressed in percent. Four N cells receive equal weight. Seat-rounds are never treated as independent observations.

Primary family: the equal-N cooperation-level contrast for EV-canonical minus F, EV-sorted minus EV-canonical, and S minus EV-sorted, separately for each of the three models. These nine two-sided tests form one Holm family at alpha=.05. Direction is not selected from the results.

Secondary interaction family: the same nine arm comparisons for D(A,B)=3(beta_A-beta_B)=0.6 sum_N (N-3.5)(C_A,N-C_B,N), in percentage points. These nine two-sided tests form a separate Holm family and are explicitly secondary; no overall alpha=.05 guarantee is claimed across the two families.

Llama replication: D(F,S)>0, a single prespecified one-sided Welch t test at .05, reported separately as a directional replication. The original estimate and its interval are displayed alongside the independent estimate. A two-sided 95% CI is reported; a successful sign test alone does not demonstrate equality to +14.02 pp. This test is not added to either nine-test family.

Levels use cell weights +/-0.25. D uses weights +/-0.6(N-3.5). For either contrast, variance is sum of squared weights times the sample episode variance divided by its cell count. Welch-Satterthwaite df equals V squared divided by sum of cell variance contributions squared/(n_cell-1). Use t-based 95% intervals. A cell with fewer than two episodes or zero total estimated contrast variance makes t inference not estimable. Such slots remain in the fixed Holm family with p=1 for correction; no estimate/p-value is fabricated for the unestimable slot.

Temporal summaries (F/S and EV arms): late-minus-early cooperation (rounds 26--30 minus 1--5) and fraction of exactly-zero late-cooperation episodes; 5,000 episode-bootstrap draws within N cells, equal-N aggregation, seed 20261010. These are descriptive. Floor shares, output lengths, first-attempt validity, repairs, fallbacks and truncation are also descriptive. A floor result is reported without a mechanistic interpretation.

## Missingness, exclusion and stopping

No episode is excluded for repairs, fallback, cooperation direction or a failed reliability target. Primary analysis retains completed fallback-affected trajectories. Partial episodes retain their telemetry and are prominently reported; they are not completed-outcome observations, replaced seeds or silently counted as successful episodes. Resume at validated episode boundaries, using the interrupted-attempt rules below. If an episode cannot be completed without a protocol change, report the shortfall and do not present the fixed-allocation confirmatory analysis as complete. The locked analysis refuses significance tests until all 576 planned episodes have completed under their prescribed protocol.

No effect-driven interim testing or stopping. If reliability or deployment requires a protocol change, pause collection, retain all existing records, register the replacement cohort separately, and label any incomplete original cohort analysis descriptive. Hardware failure retries must be recorded, use the same prescribed episode identity and deterministic request seeds, and never select among completed outcomes. An interrupted partial episode can restart only from its initial boundary after the prior Slurm allocation is verified terminal, with exactly the same protocol lock. The entire previous attempt is archived under its original episode ID. It does not contribute a second episode or disappear from the attrition/cost report. The first completed attempt is primary; completed episodes are never rerun. Repeatedly failing identities are reported as shortfalls, not replaced by other seeds.

## Data visibility and provenance

Original six-model outcomes and saved-log diagnostics were inspected before this plan. No new Stage 3 cooperation outcomes have been seen or collected. The registration is a prospective replication and ablation informed by prior findings, not a claim that hypotheses arose without previous data. Code, prompts, manifest, protocol lock and analysis script accompany the registration. All results and deviations will be reported, including nulls and floors. No explanation-audit prevalence or human-rating claim is part of this study.

## Attachments

- `rr_stage3_fourarm_v2_episodes.csv` and `cohort_manifest.json`
- `protocol_lock_stage3.json` and `cache_inventory.json`
- `stage3_source.tar.gz` (frozen source, prompts, runner, analysis and tests)
- `../results/planning_precision.csv` and its LaTeX table
- `../STAGE_0B_MEMO.md` and `../PARSER_V3_SENSITIVITY.md`

Author supplies the OSF registration URL and UTC timestamp after submission. The runner requires a matching `registration_receipt.json`; it cannot collect with the illustrative example receipt.
