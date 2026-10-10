# Internal prospective record — 2026-10-11

The author superseded the external OSF requirement on 2026-10-11. Collection is authorized using the dated plan, fixed seed manifests, analysis specification and hashed source archive in this repository. This is an internal prospective record, NOT external preregistration. The appendix write-up may follow later. Original completed-cohort analyses remain post-hoc / exploratory.


---

# Reviewer-Response Research Plan: Prospective Extension (v2)

Revision of the original plan, incorporating review feedback. Original logs, paper versions and post-hoc results remain archived. All new experiments are separate prospective cohorts and are never retroactively part of the original 376 episodes. No promise of acceptance or of significant effects.

## Standing constraints (unchanged)

Strengthen the existing paper first. No human raters, and no audit or human-rating claims in the main paper. At most two GPUs concurrently for this study. No communication-channel retrofit for the theme track. No model downloads or environment upgrades are authorized by this plan.

## What changed in v2

1. Stage 3 now has four concurrent arms (free-form, original structured, EV canonical order, EV sorted order). This gives a same-cohort comparison and a fresh replication test of Llama's +14 pp interaction.
2. Stage 4 (frozen benefit) starts with a smoke test and a prespecified floor-effect rule, and shares its dynamic arms with Stage 3 so they are paid for once.
3. A new Stage 0b (zero-GPU) resolves the Llama and GPT-OSS first-attempt parse-rate question before any top-up data is collected.
4. Stage 2 top-ups are independent 12-seed cohorts by default. Pooling with old data becomes a secondary sensitivity analysis under a concrete equivalence rule.
5. External registration (OSF or AsPredicted) before each new cohort, with a confirmatory versus exploratory table.
6. Stage 6 has explicit go/no-go criteria, a soft decision date, and a main-text versus appendix allocation.
7. Smaller changes: Stage 0 is rescoped around Qwen, the truncation gate is relaxed to a bounded rate, Qwen3-32B is deferred, and independent stages run in parallel on the second GPU.

## Priority and what each stage resolves

| Stage | Concern | Design | Gate / deliverable |
| --- | --- | --- | --- |
| 0 | Qwen truncation | Qwen3-8B: 12 saved first-attempt states, budget 2,048 vs 16,384. DeepSeek: 4 states, 8,192 vs 16,384 | 32 requests total, with raw responses, finish reasons, usage, latency |
| 0b | Llama / GPT-OSS parse failures | Failure taxonomy from saved logs. No new inference | Decision memo: parser, prompt/schema, or budget fix, or none |
| 1 | Reliability in real trajectories | Fresh 30-round Qwen and DeepSeek episodes: N=2/5, S/F, 2 seeds per cell per model (16 episodes) | Freeze protocol; report failures, truncations, repairs, fallbacks |
| 2 | Tiny replication | Mistral and GPT-OSS: new independent cohorts, 12 seeds per cell | CI-width targets; fixed allocation; no effect-driven stopping |
| 3 | Weak scaffold manipulation | Llama-8B, Gemma-12B, Mistral-24B, N=2..5, 12 seeds per cell, four concurrent arms | Separates ordering/salience from extra arithmetic; replicates Llama |
| 4 | Dynamic-environment specificity | Frozen B\_eff=12 (eta=0 and lambda=0); smoke test first | Floor rule prespecified; matched dynamic arms shared with Stage 3 |
| 5 | Small-model scope | Llama-3.3-70B (TP=2). Optionally one recent mid-size model | Five-round smoke per model; exclusive two-GPU windows |
| 6 | External generalization | Threshold public goods game | Only if go/no-go criteria are met |

Stages 1 to 6 are staged designs, not automatically submitted jobs.

## Stage 0: frozen diagnostic (Qwen first)

This is a diagnostic, not hypothesis testing. Qwen3-8B is the priority because its first-attempt parse rates were 31.5% structured and 7.9% free-form, with 3,417 of 3,664 extraction errors containing an unterminated thinking block. DeepSeek already parsed at 98 to 99%, so it gets a reduced check only.

State selection (by metadata, before any new output is seen):

- Qwen: seat 0, seed 1001, N=2/5, S/F, rounds 1/15/30 (12 states). Budgets 2,048 and 16,384 (24 requests).
- DeepSeek: seat 0, seed 1001, N=2/5, S/F, round 15 (4 states). Budgets 8,192 and 16,384 (8 requests).

Protocol:

- Reconstruct the original system and user messages from saved raw\_input with the frozen prompt builder. Do not replay repair prompts.
- Keep temperature .7, top\_p 1, the original action map and the v2 validator. Use a fixed per-state generation seed shared across the two budgets, and randomize request order. These are new diagnostic seeds, not a claim of bitwise reproduction of the original unseeded calls.
- Keep native thinking enabled in both arms. A thinking-disabled arm is a separate policy intervention and is never called a pipeline fix.
- No constrained decoding, repairs, fallbacks or new episodes in this stage.
- Save the full response JSON immediately: content, reasoning fields if present, finish\_reason and usage. Validate only final content with the v2 extractor. Reasoning text is never turned into an action.
- Serve with context 32,768. Count the templated prompt locally and reject, rather than truncate, any input where input plus configured output exceeds the context. Max output must never equal the full context.
- If 16,384 still hits the length limit, a predeclared follow-up may test 24,576 where input length permits. It is not run automatically.
- Changing context extension (YaRN), chat template or reasoning parser creates a new protocol version.

Resource limits: one GPU per model job, four CPUs, at most two hours per job, at most four allocated GPU-hours in total. HTTP timeout at most 600 seconds and no transport retries. Incomplete jobs keep all records and cannot be relabeled as successful calibration.

Reading the result: with a shared seed, the 2,048-token output should usually be close to a prefix of the 16,384-token output, so completion-token counts in the 16,384 arm largely answer the truncation question directly. Batching nondeterminism may break exact prefix matching, so keep both arms as a check but treat usage counts as the primary evidence. Before enrolling Qwen at 16,384, compute the cost sheet: mean latency per decision, expected episode time and the GPU-hours needed for 12 seeds per cell.

## Stage 0b: Llama and GPT-OSS parse diagnosis (no GPU)

First-attempt parse was 89.7% for Llama free-form and 83.3% for GPT-OSS free-form, below the plan's own 95% readiness target. Fallback rates were only 1.3% and 0.7%, so repair prompts rescued most failures. This matters because new cohorts should not inherit an unexamined weakness.

Procedure, using only saved logs and no new inference:

1. Classify every first-attempt failure for these two models into: malformed JSON or markdown wrapper, missing field, invalid action value, truncation or empty output, other.
2. Report counts by model, mode and N, with episode-cluster uncertainty.
3. Apply a decision rule recorded before looking at downstream effects:
   - If most failures are extraction-format issues, a parser v3 is warranted. Re-extract the old logs under v3 as a sensitivity analysis and use v3 in the new cohorts.
   - If most failures are generation-side (invalid actions, truncation), change the budget, schema or prompt in a new protocol version and re-run the Stage 0-style diagnostic first.
   - If failures are rare or already handled by the repair path, keep the protocol and report the rates plainly.
4. In every case the new cohort is analyzed separately from the old one under a new protocol ID. It is never silently pooled.

Stages 2 and 3 for Llama and GPT-OSS wait for this memo. Stages 2 and 3 for Mistral and Gemma (parse about 99 to 100%) do not.

## Stage 1: reliability in real trajectories

After Stage 0, run fresh 30-round episodes for Qwen (corrected budget) and DeepSeek: N=2 and 5, S and F, two seeds per cell per model (16 episodes). Retain every episode. Report failures, truncations and the first-valid, repair and fallback shares with timing.

Readiness gate (replaces the previous zero-truncation rule, which is too strict for reasoning models with 16 episodes): first-attempt validity of at least 95%, fallback at most 1%, and observed length truncations at most 1%, each reported with episode-cluster uncertainty and an upper bound. Sixteen episodes cannot certify these rates, so they are screening thresholds, not guarantees.

If a target fails, diagnose and version the change. Never replace failed seeds or drop a troublesome model without reporting it. If reliable Qwen remains infeasible, remove it from headline claims and report the deployment limitation.

## Stage 2: replication of low-seed models

Mistral (3 seeds per cell in the original) and GPT-OSS (2) get new independent cohorts of 12 seeds per cell: 96 episodes each, 192 in total. Seeds are disjoint from 1001 to 1016.

This replaces the previous plan of topping up to 12 and pooling. Independent cohorts remove the dependence on a pooling test that was not concretely defined, at a cost of about 40 extra episodes.

Pooling with the original data is a secondary sensitivity analysis only, permitted if all of the following hold:

- Prompt, parser version, checkpoint, quantization, sampling settings and inference stack match by recorded hash and version.
- In a model with cohort, mode and N terms, the 90% confidence interval for the cohort-by-mode effect on level lies within a prespecified margin of plus or minus 5 pp, and the cohort-by-mode-by-N term is not materially different from zero by the same criterion.
- If either condition fails, report cohorts separately and treat the new cohort as primary.

Precision planning: tabulate expected D confidence-interval half-widths at n=12, 20 and 40 using completed-main variances, with 1.5x and 2x variance sensitivity. Target a practically useful interval (provisional half-width 10 pp), not a significant p-value. Freeze allocation after the reliability pilot, run a first balanced block and then the fixed allocation, and never stop on the basis of effects. Include a worst-case bounded-outcome analysis next to plug-in estimates.

## Stage 3: scaffold ablation with concurrent controls

Models: Llama-8B, Gemma-12B, Mistral-24B. N=2..5, 12 seeds per cell. Four arms, collected concurrently under one cohort ID with disjoint seeds and randomized job order:

| Arm | Description |
| --- | --- |
| F | Free-form, no ranking (fresh control) |
| S | Original structured prompt (immediate-payoff ranking, descending order) |
| EV-canonical | Identical neutral instruction and identical score values, presented in canonical action order |
| EV-sorted | Same instruction and scores, presented in descending immediate-payoff order |

Size: 3 models x 4 N x 12 seeds x 4 arms = 576 episodes.

What each contrast identifies:

- EV-canonical vs F: effect of supplying score values alone.
- EV-sorted vs EV-canonical: effect of ordering and salience, holding information fixed.
- S vs EV-sorted: effect of the original framing wording relative to a neutral instruction.
- S vs F within Llama: a fresh, prespecified replication of the +14 pp interaction, the only original result that survived Holm correction. Report it as one directional replication test labeled as such. It is not pooled into the Holm family, and the original estimate is shown alongside it.

Other rules: the neutral instruction text must be identical in both EV arms, and the schema and state must be identical. Record output-length differences. A separate action-label permutation test is optional, must change mapping and scoring consistently, and is not mixed with ordering. Gemma structured and possibly Mistral structured may sit at the floor, so report floor shares and do not interpret D there as a clean estimate. Llama arms wait for the Stage 0b memo.

## Stage 4: frozen-benefit control

Freeze B\_eff=12 with B0=Bmin=Bmax=12, eta=0, lambda=0 (both are required, since setting drift to zero alone leaves the streak multiplier active), C=8, K=0, T=30, N=2..5, S and F.

Smoke test first: Llama and Mistral, two seeds per cell at N=2 and 5, S and F (16 episodes). Assert B\_eff=12 in every round of every log. User clarification, 2026-10-11: before looking at any slopes, apply the floor rule to free-form only. If frozen free-form cooperation is below 5% separately at both N=2 and N=5 for a model, that model's frozen slope comparison is declared uninformative and reported as a floor result, not as evidence about mechanisms. Structured results remain descriptive. Smoke seeds are separate from the primary frozen cohort.

Models: Llama and Mistral plus one validated larger model (Stage 5), chosen before new behavior is inspected. Size at 12 seeds per cell: 96 episodes per model per dynamics condition for S/F.

Sharing: for Llama and Mistral, the dynamic S and F arms in the Stage 3 cohort serve as the matched dynamic controls, provided the protocol, prompt versions and software match exactly. Only the frozen arms are new (2 models x 4 N x 2 modes x 12 seeds = 192 episodes). The larger model needs its own matched dynamic arms. This is a payoff-dynamics ablation, not a second game.

## Stage 5: larger-model scope

Primary candidate: Llama-3.3-70B, which tests whether Llama's positive interaction survives scaling. Before enrollment verify the exact snapshot and tokenizer hashes, weight-file completeness, model context, vLLM architecture compatibility, VRAM and quantization. One isolated five-round smoke per added model. No unverified model is silently substituted.

TP=2 uses both study GPUs, so schedule 70B jobs in exclusive windows when no other study job is running. Keep model loading and GPU timing in the cost ledger.

Deferred: Qwen3-32B. It would reintroduce the thinking-budget problem, so enroll it only if Stage 0 and Stage 1 resolve Qwen.

Optional: one additional recent mid-size open model, if one passes the same deployment and smoke checks. Reviewers may find the current roster dated, and a recent model adds more than a second large Llama-family checkpoint. State the model choice and rationale before collection.

## Stage 6: second game (go/no-go)

Only proceed if all of the following hold at a decision review:

- Stages 0 to 3 are complete and their results are written up.
- Qwen is either validated or formally removed from headline claims.
- The Stage 3 and Stage 4 results are interpretable (not all at floors).
- There is enough time before the target submission to also complete a separate preregistered analysis specification and implementation validation.

Otherwise, move the second game to a limitations and future-work statement. Proposed game: threshold public goods with fixed payoff rules. No communication channel is added.

Soft deadline: pick the ARR cycle to target and write its date here: \_\_\_\_. Hold the go/no-go review at least six weeks before that date. Check the current ARR calendar for the exact deadline.

## Statistical lock and registration

Primary outcome remains episode mean cooperation. Four N cells get equal weight. D = 3(beta\_F - beta\_S), with eight-cell Welch intervals and Holm across a predeclared roster. Levels, D and temporal persistence remain distinct measurements.

Before each new cohort is collected, submit a timestamped registration (OSF or AsPredicted) that fixes the models, arms, seeds, protocol IDs, primary contrasts, Holm families and exclusion rules. Internal dated notes are not described as preregistration.

| Analysis | Status |
| --- | --- |
| Stage 2 D, level and persistence per model (new cohorts) | Confirmatory, Holm across the registered roster |
| Stage 3 EV contrasts (EV-canonical vs F, EV-sorted vs EV-canonical, S vs EV-sorted) | Confirmatory, separate Holm family per model set |
| Llama S vs F replication of +14 pp | Single directional replication test, reported separately |
| Stage 4 frozen-benefit contrasts | Confirmatory only if the floor rule is passed, else descriptive |
| Pooled old-plus-new estimates | Sensitivity analysis |
| Reliability, trajectories, round-level patterns | Descriptive |
| Stage 0 and Stage 1 diagnostics | Descriptive, never part of D or temporal tests |

Twelve seeds per cell is a replication floor, not a claim of 90% power. Corrected-budget Qwen and DeepSeek cohorts are always analyzed separately from the old-budget cohort. Diagnostic saved-state calls are not episode replications.

A full prospective eight-model balanced replication at 20 seeds per cell would be 1,280 episodes. Treat it as a long-term option that needs a measured cost sheet before launch.

## Scheduling and parallelism

Use short model-homogeneous Slurm jobs, targeting 1 to 4 hours per chunk after measured load and episode times. Keep whole episodes within a chunk, preserve partial records and resume only from validated episode boundaries. Reload overhead is measured, not assumed free. Study GPU ceiling is two. Never cancel unrelated jobs, and never use GPUs outside an allocation.

Suggested ordering:

- GPU A: Stage 0 (Qwen, then DeepSeek), then Stage 1.
- GPU B, in parallel: Stage 0b (CPU only), then Stage 2 and Stage 3 for Mistral and Gemma. Llama and GPT-OSS jobs begin after the 0b memo.
- Stage 4 smoke test once Stage 3 protocols for Llama and Mistral are frozen.
- Stage 5 (70B, TP=2) in an exclusive window after other jobs are drained.

Every run records cohort ID, configuration and code hashes, model and tokenizer snapshot IDs, software versions, full request and response telemetry, allocation and latency, failures and finish reasons.

## Paper work

In the abstract and introduction, lead with the concrete level, slope and persistence dissociations, and move the measurement grid into the introduction. Remove the explanation audit and H5 from the main paper. Keep any detector evidence only as a clearly exploratory supplement, and do not treat the 11.46% detector rate as a validated prevalence. Remove Qwen from headline strategic conclusions until its corrected pipeline is evaluated. State the ranking manipulation plainly. Keep null findings, Llama sensitivity and legitimate limitations visible.

Housekeeping to fix before the next submission:

- Check Eq. (4) against the source: given the weights in Eq. (5), it should read 3(beta\_free,m - beta\_struct,m).
- Appendix B refers to itself ("Appendix B reports actual serving and reliability"). Fix the cross-reference.
- Confirm the Conclusion and the recommendations box no longer imply the removed audit.

Main text versus appendix: the main text carries the measurement framework, the original six-model results, the Stage 3 ablation and one control (frozen benefit or a larger model). The appendix carries the reliability tables, Stage 2 pooling sensitivity, the Qwen pipeline account and any extra model results. A second game and an eight-model 20-seed replication will probably not both fit in an 8-page paper, so decide at the Stage 6 review which, if either, goes in.

The theme track is not a reason to add communication. That changes the research question and is outside this plan.

## Provenance

Original confirmatory results remain unchanged. New analyses are explicitly prospective follow-ups motivated by already-inspected data, and no retroactive preregistration claim is made.

## Sources checked for calibration

- Qwen3-8B official model card (huggingface.co/Qwen/Qwen3-8B): native 32,768 context, thinking enabled by default, explicit thinking switch. A larger max\_tokens cannot exceed input plus output context.
- DeepSeek-R1-Distill-Qwen-14B official model card (huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B): evaluation generation allowance of 32,768 and sampling guidance. Verify that the plan's system-prompt and temperature setup does not conflict with the card's recommendations.

## Open decisions

Execution record (2026-10-11): see [STATUS.md](STATUS.md). Stage 0b is complete and the author selected an OSF packet for manual submission. Exact-message replacement calibration jobs are 388864/388865; prior jobs are archived and cancelled after the JSON-formatting discrepancy. Stage 3 source, seeds, runner and analysis are frozen for registration; new cohorts are not yet registered or submitted.

- Target ARR cycle and date for the soft deadline.
- Total GPU-hour envelope after Stage 0 timing and the Qwen cost sheet.
- CI-width target and fixed sample allocation.
- Which additional recent model, if any, to enroll in Stage 5.
- Parser v3 versus protocol change after the Stage 0b memo.

Latest execution status: the OSF prerequisite above was superseded by the author on 2026-10-11; see STATUS.md for the accepted internal-archive collection jobs.
