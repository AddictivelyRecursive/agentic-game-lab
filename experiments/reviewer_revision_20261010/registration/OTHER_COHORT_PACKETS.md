# Remaining OSF packets: dependency log

These are concrete draft specifications, not completed registrations. Do not submit unresolved protocol fields as though already fixed.

## Stage 1: 16 diagnostic trajectory episodes

Use `rr_stage1_reliability_v2_episodes.csv`: Qwen and DeepSeek, N=2/5, S/F, two disjoint seeds per cell, 30 rounds. No hypothesis tests on cooperation. Retain all outcomes. Before registration, attach exact-message calibration results and choose each model's fixed output allowance and context. Readiness screens: first-attempt validity >=95%, fallback <=1%, length finishes <=1%, with episode-cluster intervals and upper bounds. Bootstrap episodes within model/mode/N; when zero events yield a degenerate bootstrap, additionally report an exact one-sided upper bound on the probability an episode has any event, not an independent-decision binomial bound. These small samples cannot certify decision-level reliability. Protocol changes produce new cohorts, not replacement failed seeds.

DeepSeek caveat: temperature .7 is within its official recommended range, but the original experiment's system prompt conflicts with the model card's recommendation to place all instructions in the user message. The current calibration deliberately retains original messages to isolate output-budget differences. It must not be described as a fully model-card-compliant deployment. Moving instructions is a separate protocol intervention to specify before a future cohort, not a silent budget correction. Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B#usage-recommendations (checked 2026-10-11).

Status: exact-message calibration pending; model settings not locked. Do not collect yet.

## Stage 2: 192 independent replication episodes

Use `rr_stage2_independent_v2_episodes.csv`: Mistral and GPT-OSS, N=2..5, S/F, 12 new seeds per cell, 96 episodes per model. Use the same environment as Stage 3. Mistral retains v2, .7/top-p 1, 2,048 tokens/16,384 context. GPT-OSS settings await a fully instrumented diagnostic; the saved logs cannot distinguish missing final content, length limits and template behavior. No guessed reasoning-content substitution.

Proposed fixed confirmatory family: six two-sided tests—level gap F-S, D(F,S), and F late-minus-early, for each of the two models—Holm across all six at .05. Four N cells weighted equally. Levels and D use the eight-cell Welch estimator described in the Stage 3 packet. Persistence uses the episode-level late-minus-early difference, four weights .25 and four-cell Welch uncertainty. Report two-sided 95% CIs. Other temporal/reliability summaries are descriptive. Exact analysis is `analyze_cohort.py`.

Allocation remains 12/cell; the 10-pp half-width target is provisional and not guaranteed. The saved-variance planning sheet covers n=12/20/40, 1/1.5/2x variance and a bounded-outcome benchmark. If a larger allocation is selected before enrollment, regenerate and freeze the manifest and packet before registering, not after inspecting effects.

Pooling is secondary and only if protocol/hash/stack equivalence passes, then both 90% intervals lie inside +/-5 pp: cohort difference in the F-S level gap, and cohort difference in D(F,S). This makes the group-size term's margin apply to the full N=2..5 fitted contrast, not an unscaled per-N coefficient. Otherwise no pooling; new independent cohorts remain primary. Parser, sampling-seed policy or software differences fail the strict protocol-equivalence check. Exclusion/missingness/stopping rules match Stage 3.

Status: Mistral deployment is identified; GPT-OSS generation diagnostic is unresolved. Do not describe the two-model packet as ready to register until its complete protocol is fixed. Mistral may be enrolled after a separately fixed registration that preserves the planned family and before any new outcomes are inspected.

## Stage 4 smoke: 16 frozen-benefit episodes

Use `rr_stage4_floor_smoke_v2_episodes.csv`: Llama/Mistral, N=2/5, S/F, two seeds/cell, T=30. B0=Bmin=Bmax=12, eta=0 and lambda=0, C=8,K=0. Assert B_eff=12 on every logged round. Parser/prompt/runtime must match the corresponding Stage 3 controls. Freeze the diagnostic protocol before registration.

Floor rule, clarified by the author on 2026-10-11: a model is declared uninformative for the frozen slope comparison if its F cooperation is <5% at N=2 AND <5% at N=5. Means use the two smoke episodes in each F cell. This test does not use S, and no slope is inspected before the decision. Report S descriptively. Smoke episodes are separate and excluded from the primary frozen cohort by design, not based on their outcomes.

The primary two-model frozen manifest contains 192 episodes at 12/cell. Dynamic S/F controls are exactly the corresponding Stage 3 rows and are collected once. A larger model is not substituted or added after viewing its new behavior; it needs its own prior deployment decision, matched dynamic arms and registration. Main-cohort contrasts and the full fixed Holm roster must be locked before that cohort's enrollment. Do not register the larger-model comparison with an unnamed model.

## Stages 5 and 6

Llama-3.3-70B cache presence/configuration/tokenizer have been inspected; runtime/VRAM and five-round smoke remain untested. TP=2 requires an exclusive two-GPU window. No additional model has been selected, downloaded or enrolled. Stage 6 remains gated on the earlier stages and a future target-cycle decision. No ARR date is invented and no second-game episodes are authorized by this dependency log alone.
