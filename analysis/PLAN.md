# POST-HOC / EXPLORATORY analysis plan

Prepared 2026-10-10. Status: planned, not executed. Deployment means copying this plan to Sharanga; it does not launch analysis, episodes, inference, vLLM, or GPU jobs. No optional judge is authorized. Implement and execute tasks strictly 0 → 1 → 2 → 3 → 4, subject to the Task 0 gate.

## Scope and immutable inputs

Analyze only the completed fresh-main study: six models, structured/free_form, N=2,3,4,5, T=30; seeds per cell 14/4/8/3/16/2 for Llama/Qwen/Gemma/Mistral/DeepSeek/GPT-OSS. Expect 376 episodes, 48 cells and 39,480 decisions. Never merge pilots, reruns or partial attempts. Preserve raw logs and existing paper assets.

Local canonical input: `results/server_copies/final_main_20261010/main_complete_20261010/main.jsonl`, SHA256 `149ac53f8a19286b429d1de83939ae91028845b0b35a18e1dc8dd5cc0468b778`. Existing explanation export: `results/analysis/main_h5_coding_20261010/coding_sheet.csv`, with a private condition key beside it. Expect 19,740 N=2/5 decisions and 274 empty explanations. On Sharanga, discover and hash the completed export before choosing an input; do not assume the local snapshot path exists there. If needed, transfer the verified saved input, not experimental code or active worker directories.

## Initial schema inspection (before calculations)

Write `SCHEMA.md` with paths, hashes, record types, field types, keys, join rules, completeness checks and unresolved ambiguities. Preliminary inspection of one saved episode found:

| Concept | Saved field / derivation to verify across all records |
|---|---|
| Model, mode, N, seed, episode | `model`, `reasoning_mode`, `N`, `seed`, `episode_id` |
| Round and seat | `decisions[].round`, `decisions[].seat`; round records use `rounds[].t` |
| Submitted action | Join decision to `rounds[t].true_actions[seat]`; no direct action field in inspected decision |
| Cooperation | `1-action/4`; cross-check `rounds[].true_coop` and episode `cooperation` |
| Repairs | Original `repair_used`, `repair_calls` |
| Fallback | Original `fallback_used`; fallback implies submitted action 4 |
| Explanation | `reasoning_text`; retain raw response separately |
| Parse / first-attempt validity | `first_parse_ok` is historical; recompute with existing v2 extractor from first `model_calls[].output` |
| Parser provenance | `model_calls[].json_extraction_version`, call `stage`; validate order and parser consistency |
| Completion | Episode `status`, decision `aborted` |

Use original repair/fallback flags, as `Thesis/build_complete_results.py` does for Table 5. Distinguish JSON extraction success from an action that is actually valid. Apply the existing action validator's exact rules; document coercion rather than inventing a new parser. Verify every seat-round occurs exactly once, rounds are 1–30, actions are integers 0–4, all expected cells/seeds exist, and no final action differs between evidence sources. Missing required fields or irreconcilable joins stop the work and trigger a report; never infer fallback from action 4 or repairs solely from call counts.

## Shared implementation and statistical conventions

- Python: pandas, numpy, scipy, statsmodels, matplotlib. `common.py` owns validated loaders, episode aggregation, Welch contrasts, Holm correction, bootstrap and CSV/booktabs exporters. Separate `00_reproduction.py` through `04_mismatch.py`, plus `detector_rules.py` and `run_all.py`. Record actual package versions and exact CLI commands in README.md.
- Seed all random procedures at 20261010, with deterministic named random streams and fixed model/cell ordering. Save parameters and source hashes in a run manifest. No API keys, external calls or judge code on the default path.
- Episode is the inference unit. Aggregate seats within rounds and rounds within episodes. Average the four N-cell means equally for model-mode summaries, preserving this weighting after filtering. Resample whole episodes within model × mode × N cells, 5,000 draws unless specified. Do not assume cross-mode seed pairing.
- Save each output table as CSV and booktabs LaTeX, figures as PDF. Include POST-HOC / EXPLORATORY metadata in script/table/figure headers or captions; retain machine-readable numeric values. Display cooperation percentages and pp with two decimals; distinguish percent from fractions and pp.
- Conditional decision-filtered estimates first aggregate retained decisions within episode, then cell, then equally across N; report denominators and episodes/cells with no eligible decisions. Also report explicitly requested descriptive seat-decision rates. No seat-level inferential tests.
- Record constant predictors, missing denominators, small cells and zero estimated variances explicitly. Never silently drop a cell, model or bootstrap draw. Log invalid bootstrap counts; do not manufacture finite inverse-variance weights. Resolve material nonestimability in the report before interpreting a bootstrap CI.
- Update RESULTS.md and commit only this task's files after each completed task, leaving unrelated dirty files untouched. Never state that an unexecuted task passed.

## Task 0 — reproduction gate (approximately 1 hour)

Create a literal published-anchor registry covering every user-specified level, temporal mean, Gemma N mean, decision count, repair/fallback rate, D estimate/CI/p/Holm, Q/p/I². Reproduce from saved decisions, not copied summary tables. Compare existing confirmatory outputs only as a diagnostic.

For D in pp use weights +0.6(N−3.5) for F and their negatives for S, unbiased episode sample variances (ddof=1), variance terms w²v/n, Welch df `(sum terms)^2 / sum(terms^2/(n−1))`, two-sided t CI/p and six-model Holm. Check the equivalent separate-mode OLS slope contrast. Compute inverse-variance Q and I² without claiming a population-wide effect.

Tolerances: cooperation ±0.01 pp; D ±0.05 pp; exact counts. For other rounded anchors compare at published precision, documenting the rounding interval (Table 5 rates 0.1 pp, D CI 0.01 pp, p/Holm 0.001, Q 0.01, heterogeneity p 0.0001, I² 0.1 pp). Use a numerical epsilon only at rounding boundaries; no tuning tolerance after seeing results.

Produce `results/00_reproduction.csv` with `quantity,published,recomputed,abs_diff,pass`, its LaTeX equivalent, reusable episode/cell estimates and RESULTS.md's gate section. Any failed anchor stops run_all before Tasks 1–4. Diagnose source scope, flag definitions, exclusions, N weighting, parser and variance implementation, then report immediately. Continue only after all anchors pass or the user explicitly accepts identified discrepancies.

## Task 1 — repair/fallback robustness (approximately 3 hours)

1. For all 12 model-mode combinations, produce round repair/fallback seat-decision rates and equally N-weighted episode cooperation trajectories. Plot rate and cooperation panels; highlight Qwen F/S. Report Qwen F late fallback share and nonfallback late cooperation, with explicit denominators.
2. Compare unfiltered episodes against no-fallback, repair-share ≤25%, and ≤10% sets. Save retained counts for every cell, episode means and early/late outcomes. Estimate D with retained cell sample sizes and Welch uncertainty only if all eight model cells retain ≥2 episodes; otherwise `not estimable`.
3. Estimate early/late cooperation conditional on first-attempt-valid, unrepaired, nonfallback decisions. Report selection bias and eligibility loss.
4. Use separate within-model-mode Spearman correlations of episode late cooperation with episode repair and fallback shares. Bootstrap whole episodes within N cells, 5,000 draws. Mark constant-variable correlations nonestimable. These associations are descriptive and may reflect N or endogenous pipeline behavior.
5. Report unfiltered and filtered early-to-late differences and retention together. Use the four allowed verdict labels SURVIVES / WEAKENED / DOES NOT SURVIVE / NOT ESTIMABLE, always with numbers. Treat persistence of negative late-minus-early across estimable filters as survival; attenuation or inconsistent signs as weakening; clear reversal/removal across estimable filters as non-survival. State when the baseline has no decay and when selective attrition prevents a verdict; do not force a decay narrative for Mistral or Gemma.

Outputs: `01_fallback_by_round.csv`, `01_filtered_outcomes.csv`, `01_dose_response.csv`, corresponding LaTeX tables and `figures/01_*.pdf`; RESULTS.md section and ≤150-word real-number paper insert; task commit.

## Task 2 — heterogeneity (approximately 1.5 hours)

- Leave each model out, recompute weighted mean, Q(4), chi-square p and I²; identify exclusions changing p<.05 and explicitly answer whether significance survives removing Llama.
- Parametric null: 20,000 Normal draws around the inverse-variance common mean with fixed observed SEs; repeat using t draws with df=max(2, Welch df). Use scale=observed SE for the t version and disclose its heavier variance; compare simulated Q to exact recomputed observed Q, also reporting rounded 18.23. Use `(exceedances+1)/(draws+1)` and Monte Carlo uncertainty.
- Nonparametric: 5,000 within-cell episode bootstrap draws, recomputing D, variance, Welch df, Q, I², six-model Holm and Llama significance each time. Report percentile CIs, significant fraction and all nonestimable-draw counts. Sparse GPT-OSS cells may yield zero variance: no silent discarding or arbitrary variance floor.
- Endpoint weights: F5 +1, F2 −1, S5 −1, S2 +1. Use independent-cell Welch CIs alongside the four-point D. Forest plot shows six D estimates and descriptive inverse-variance pooled estimate/CI with its assumptions clearly stated.

Outputs: `02_leave_one_out.csv`, `02_Q_resampling.csv`, `02_endpoint_vs_fourpoint.csv`, LaTeX equivalents, `02_forest.pdf`, RESULTS.md with candid Llama sensitivity and paper insert; task commit.

## Task 3 — level, slope and persistence (approximately 2 hours)

For each model-mode, and supplementary each N cell, bootstrap late-minus-early, mean per-episode OLS slope over rounds 1–30, and fraction of episodes with exactly zero late cooperation. For the cell-mean half-opening crossing, compute the first round strictly below 0.5 × opening; if opening=0, explicitly note threshold zero and report never where no crossing exists. Bootstrap crossing times with never retained as a distinct outcome; report never proportion and finite-crossing quantiles rather than silently treating never as missing.

Build one six-row summary grid with F−S level gap, Task 0 D/CI, F late-minus-early/CI and fraction zero-late. Two Spearman correlations across six models are descriptive only: level gap versus |D| and versus late-minus-early. Plot level gap vs D; mark decay if F late-minus-early <0 and persistence otherwise, disclosing the sign-based descriptive classification and uncertainty.

Outputs: `03_persistence.csv`, supplementary by-N table, `03_summary_grid.csv`, all LaTeX equivalents, `figures/03_*.pdf`, RESULTS.md and summary-caption insert; task commit.

## Task 4 — explanation audit (approximately 3–4 hours; lowest priority)

Join the N=2/5 export to canonical decisions using the private condition key; verify counts/actions/text and report all 274 empty explanations separately. If export linkage or required fields are unavailable, stop this task and report rather than guess.

Freeze small documented R1–R4 rules before aggregate prevalence inspection. Match explicit claims about the selected action, handle local negation, distinguish hypothetical/other-player action mentions from chosen-action assertions, and compare explicit numeric cooperation claims to 1−a/4 with documented numeric-format tolerance. Preserve per-rule evidence spans and per-decision flags. Verify the specified Gemma S N2 seed1001 round15 seat0 and Llama round1 seat0 examples; identify the latter precisely from saved records. Only repair failures against these known examples without inspecting other outcomes for rule tuning.

Report rates by model-mode and buckets 1–5/6–25/26–30, plus valid unrepaired decisions. Supply requested Wilson intervals as descriptive binomial intervals with an explicit warning that they do not account for within-episode dependence; add within-cell episode-bootstrap CIs for inferential interpretation. Do not describe Wilson intervals as episode-cluster inference.

Export 75 flagged and 75 unflagged cases stratified across models, using fixed allocations with transparent shortage handling. Random IDs hide model/mode; export only requested fields and blank human_label. Keep selection probabilities, episode IDs, stratum weights and detector flags in a separate private key, never the labeling sheet. `validation/score_validation.py` validates real labels, reports UNCLEAR/unlabeled counts, Wilson precision and unflagged miss-rate intervals; estimates recall using population flag prevalence and stratified sampling weights, with clear sampling/cluster limitations. A balanced case-control sample's naive recall is invalid. No fabricated labels, no judge without supplied API access plus explicit go, and at most 400 judge calls if later authorized.

Outputs: `04_mismatch_by_model_mode.csv`, `04_mismatch_by_round.csv`, empty-explanation and rule-detail tables, LaTeX equivalents, `validation/validation_sample.csv`, private key, `validation/score_validation.py`, `figures/04_*.pdf`, RESULTS.md and paper insert. Describe only detectable inconsistencies under this rule-based detector, exploratory and limited to N=2/5; no claim of misunderstanding or completed human validation. Task commit.

## Completion and operating boundaries

RESULTS.md begins with bold POST-HOC / EXPLORATORY and contains each task's methods, real key numbers, one-paragraph verdict, caveats and discrepancies, plus four ≤150-word PAPER INSERTS. README.md records exact gated reproduction commands and package versions. No manuscript replacement or publication is part of this plan.

Use CPU analysis only, within cluster policy; choose a CPU allocation after inspecting available resources, never reuse inference launch scripts. Preserve checkpoints for long bootstrap tasks and communicate meaningful progress. If time constrained, omit the already-unauthorized optional judge first, then Task 3 rank correlations, recording omissions. Never bypass the reproduction gate or replace missing results with proposed text.
