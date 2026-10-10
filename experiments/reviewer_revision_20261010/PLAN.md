# Reviewer-response research plan — prospective extension

User direction: strengthen the existing paper first; no NAACL deadline; no human raters; remove audit claims and human-rating promises from the main paper; at most two GPUs concurrently for this study. Original logs, paper versions and post-hoc results remain archived. New experiments are separate prospective cohorts, never retroactively part of the original 376 episodes. No promise of acceptance or significant effects.

## Priority and what each step resolves

| Stage | Concern addressed | Concrete design | Gate / deliverable |
|---|---|---|---|
| 0 | Reliability and suspected truncation | Qwen3-8B and DeepSeek-R1-Distill-Qwen-14B, 12 saved first-attempt prompt states each; original budget versus 16,384 output tokens; identical prompts and native thinking behavior | 48 total requests, raw responses, finish reasons, token usage, parsed/valid status, latency; <=4 allocated GPU-hours |
| 1 | Reliability in actual trajectories | After calibration, fresh 30-round episodes for Qwen and DeepSeek: N=2/5, S/F, two seeds per cell per model (16 episodes total) | All episodes retained; report failures, truncations, first-valid/repair/fallback and timing; freeze protocol before full replication |
| 2 | Tiny replication / uncertain intervals | Add Mistral 9 seeds per cell (72 episodes), GPT-OSS 10 (80), reaching 12/cell if stack equivalence passes; otherwise collect independent 12/cell cohorts | Precision planning from main variance with sensitivity margins; first balanced block then fixed allocation, never stop for significance |
| 3 | Weak scaffold manipulation | Llama-8B, Gemma-12B, Mistral-24B: two new EV arms, canonical action order and descending immediate-payoff order, with identical neutral instruction and identical score values; N=2..5, 12 seeds/cell (288 episodes) | Distinguishes ordering/salience from supplying extra arithmetic; original S/F remains a separate comparison |
| 4 | Dynamic-environment specificity | Frozen B_eff=12, B0=Bmin=Bmax=12, eta=0, lambda=0, C=8, K=0; retain T=30, N=2..5 and S/F | Validate constant B_eff on every round; compare same-budget dynamic controls. This is a payoff-dynamics ablation, not a second game |
| 5 | Small-model scope | Primary larger-model candidate Llama-3.3-70B; second candidate Qwen3-32B. Cached names are present, but weights, runtime support, memory and throughput require deployment checks | One isolated five-round smoke per added model; 70B TP=2 consumes both study GPUs. No unverified model substituted silently |
| 6 | External generalization | Only after stages above: a genuinely different game, proposed threshold public goods with fixed payoff rules | Separate preregistered-in-advance local analysis specification and implementation validation; no communication/theme-track retrofit |

Stages 1–6 are staged designs, not automatically submitted jobs. Only Stage 0 is currently prepared for bounded GPU submission. Its outcome determines valid budgets and realistic cost estimates; it cannot determine which cooperation direction to retain.

## Stage 0: frozen diagnostic, not hypothesis testing

Select seat 0, seed 1001, N=2/5, S/F, rounds 1/15/30: twelve states/model, chosen by metadata before new outputs. Reconstruct exactly the original initial system/user messages from saved raw_input and the frozen prompt builder. Do not replay repair prompts. Use Qwen budgets 2,048 and 16,384; DeepSeek 8,192 and 16,384. Preserve temperature .7, top_p 1, the action map and v2 validator. Use a fixed per-state generation seed shared across the two budgets and randomize request order. Calibration generation seeds are new diagnostic settings, not a claim of bitwise reproduction of the unseeded original calls.

Native Qwen thinking remains enabled/default in both budget arms. A later thinking-disabled arm is a separate policy intervention, never silently called a pipeline fix. Do not add constrained decoding while assessing budget effects. Preserve complete response JSON including content, reasoning fields if present, finish_reason and usage. Validate only final content using the exact v2 extractor; never turn reasoning text into a submitted action. No repairs, fallback substitutions or new episodes in this stage.

Serve with context 32,768, leaving room for input and completion. Locally count the fully templated prompt; reject rather than truncate input when input+configured output exceeds the context. Record actual server usage too. Output max_tokens must not equal the entire context. If 16,384 still reaches length limits, a predeclared next calibration step may test 24,576 where input length permits; it is not automatically run in this first batch. Changing context extension/YaRN, templates or reasoning parsers creates a separate protocol version.

Each model job requests one GPU, four CPUs, at most two hours; maximum first-batch allocation four GPU-hours. Max 24 requests/model and 90 minutes request runtime after readiness; HTTP timeout at most 600 seconds; no transport retries. Save each response immediately. Incomplete jobs retain all records and cannot be relabeled as successful calibration. Scheduler waiting is not an inference result.

Pilot readiness targets are first-attempt action validity >=95%, observed length truncations zero and fallback <=1%, assessed in the trajectory pilot, with episode-cluster uncertainty and all denominators. Twelve calibration states cannot certify these rates. If a target fails, diagnose and version changes; never repeatedly replace failed seeds or drop troublesome models without reporting the failure. If reliable Qwen remains infeasible, remove it from headline claims and report its deployment limitation.

## Replication and statistical lock

Primary future outcome remains episode mean cooperation. Four N cells receive equal weight; D=3(beta_F-beta_S), eight-cell Welch intervals, Holm across the predeclared roster. Levels, D and temporal persistence remain distinct. Specify a single primary contrast family before the large cohort; added models expand that family, not selected individual discoveries. Controls have prespecified separate families and all results are reported.

Twelve seeds/cell is a minimum replication floor, not a 90% power claim. Prepare anticipated CI widths at n=12/20/40 from completed-main variances, with 1.5x and 2x variance sensitivity and an explicit statement that variance may change under corrected budgets. Target a practically useful D interval (provisional half-width 10 pp), not a significant p-value. Freeze allocation after the reliability pilot; use no effect-driven optional stopping. A worst-case bounded-outcome analysis accompanies plug-in estimates. Full prospective balanced replication of eight models at 20 seeds/cell would be 1,280 episodes, a long-term option requiring a measured cost sheet rather than immediate launch.

Mistral/GPT-OSS top-ups use new seeds disjoint from 1001–1016. Pool with old data only after prompt, parser, checkpoint, quantization, sampling and inference-stack equivalence checks; otherwise report old and new cohorts separately. Corrected-budget Qwen/DeepSeek are always separate from their old-budget cohort. Diagnostic saved-state calls are not episode replications and never enter D or temporal tests.

## Controls before breadth

The EV ablation should be cheap relative to reasoning-model reruns. Its neutral instruction must be identical in the sorted and canonical-order arms; score values, schema and state must be identical, and output length differences recorded. It identifies a presentation effect, not structured reasoning generally. A separate action-label permutation test is optional and must change mapping and scoring consistently; do not mix it with ordering in the first ablation.

Frozen benefit requires BOTH eta=0 and lambda=0: setting drift to zero alone leaves the streak multiplier active. Assert B_eff=12 in all logs. Use at least Llama and Mistral plus one validated larger model, chosen before new behavior is inspected. At 12 seeds/cell this is 288 frozen episodes for three models; matched new dynamic runs must use the same validated protocol. Do not call this cross-game evidence. A true second game comes later if these controls justify the added scope.

Llama-70B and Qwen-32B are larger cached candidates, not claims to be the newest or strongest available. Verify exact snapshot and tokenizer hashes, weight-file completeness, model context, vLLM architecture compatibility, VRAM and quantization before enrollment. No model downloads or environment upgrades are authorized by this bundle. Keep model loading/GPU timing in the cost ledger.

## Paper work now

Lead the abstract/introduction with concrete level/slope/persistence dissociations. Move the existing measurement grid to the introduction. Remove the anecdotal explanation audit as a main contribution and remove H5/human-rating promises; retain historical protocols only in the archive and any unvalidated detector evidence only as a clearly exploratory supplement. Avoid treating the 11.46% detector rate as a validated misunderstanding prevalence. Remove Qwen from headline strategic conclusions until its corrected pipeline is evaluated. State the tested ranking manipulation plainly; keep null findings and Llama sensitivity visible. Do not delete legitimate limitations to make the paper sound stronger.

The theme track is not a reason to bolt on communication. A communication channel changes the research question, requires message-budget and observability controls, and is outside this strengthening-first plan. No human baseline or mechanism-validation claim is added.

## Scheduler and provenance

Use short model-homogeneous jobs; target 1–4 hour future chunks after measured load/episode times. Keep whole episodes within a chunk, preserve partial records and resume only from validated episode boundaries. Reload overhead is measured, not assumed free. Use Slurm for slot dispatch rather than probing other users' processes or taking GPUs outside an allocation. The study GPU ceiling is two; TP=2 uses both. Unrelated jobs are never cancelled. Initial calibration jobs are independent, one GPU each, on supported H100/H200 partitions, with frozen local bundle paths and job-scoped process-group cleanup.

Every new run records cohort ID, configuration and code hashes, model/tokenizer snapshot IDs, software versions, full request/response telemetry, allocation/latency, failures and finish reasons. Original confirmatory results remain unchanged. New analyses are explicitly prospective follow-ups motivated by already inspected data; no retroactive preregistration claim.

## Sources checked for calibration

- [Qwen3-8B official model card](https://huggingface.co/Qwen/Qwen3-8B): native 32,768 context, thinking enabled by default, explicit thinking switch, long-generation examples. Larger max_tokens alone cannot exceed input+output context.
- [DeepSeek-R1-Distill-Qwen-14B official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B): evaluation generation allowance 32,768 and sampling guidance. These recommendations motivate diagnostic budgets, not an assertion that every task needs 32k.

Open later decisions: final total GPU-hour envelope after Stage 0 timing; CI-width target and fixed sample allocation; whether the second larger model adds enough value beyond Llama-70B. None blocks the small, already bounded calibration batch.
