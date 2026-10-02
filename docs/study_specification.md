# AgenticGameLab — Scaling-Induced Defection & Decision Scaffolding Study
# Full implementation and experiment plan (supersedes prior partial prompts)

This is the complete, current specification. Treat it as the source of
truth for this project going forward — do not merge it with assumptions
from earlier, partial instructions in this conversation.

## 1. Model roster

Six models, one per major lab, deployed locally via the existing
LocalOpenAIClient. No pre-assigned behavioral archetypes — model identity
is an empirical factor in the analysis, not a predetermined category.

| Alias | HF identifier | Role in design |
|---|---|---|
| llama-3.1-8b-instruct | meta-llama/Llama-3.1-8B-Instruct | Standard instruction-tuned baseline; most-cited open baseline in this literature |
| qwen3-8b | Qwen/Qwen3-8B | Standard instruction-tuned, same scale as Llama pick — enables a same-scale, different-lab comparison |
| gemma-3-12b-it | google/gemma-3-12b-it | Standard instruction-tuned, mid scale step-up |
| mistral-small-3.1-24b-instruct | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | Standard instruction-tuned, larger scale step-up |
| deepseek-r1-distill-qwen-14b | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | Reasoning-distilled model, for exploratory comparison to standard models |
| gpt-oss-120b | openai/gpt-oss-120b | Reasoning-tuned model, largest in roster, for exploratory comparison to standard models |

Note in any writeup: deepseek-r1-distill-qwen-14b is a DeepSeek-trained
distillation on a Qwen base — address this directly as still representing
an independent training process/organization, not as a hidden Qwen
duplicate.

### Step 0 — Deployment confirmation (do this first, every time this
roster changes)

1. Confirm all 6 models are deployed and reachable via LocalOpenAIClient.
   Report each individually (alias, confirmed reachable y/n, response to a
   trivial test prompt through the existing pipeline) — do not report
   "all working" as a single line.
2. For gpt-oss-120b specifically (largest model, different scale class
   from the rest): report VRAM/quantization used, confirm it coexists with
   the serving requirements of the other 5 models or requires a separate
   allocation, and run one isolated single-episode smoke test (N=2, 5
   rounds) BEFORE including it in any grid below. Report per-round
   latency. If this model's cost makes the full 6-model grid infeasible in
   the remaining timeline, report the specific cost figures and propose
   which model to drop — do not drop one unilaterally.
3. Do not assume any model's behavior based on prior work at a different
   parameter scale (this applies to qwen3-8b vs. Qwen3-235B, and to
   gpt-oss-120b vs. gpt-oss-20b, both referenced in earlier project
   history) — treat all 6 as empirically open.

## 2. Hypotheses

Environment fixed at M=5, p=0.0, theta=0.50, T=50 unless stated otherwise.
Grid: N ∈ {2,3,4,5} × reasoning_mode ∈ {structured, free_form} × 6 models
= 48 cells.

**H1 (confirmatory, primary).** Mean cooperation rate C̄ decreases over
N ∈ {2,3,4,5} in the structured condition.

**H2 (confirmatory, primary).** The decline in H1 is fit better by a
1/(N−1) dilution curve than by a flat (no-effect) null model (AIC/BIC
comparison), on pooled structured-condition data across all 6 models.

**H4 (confirmatory, primary novelty claim).** The slope of cooperation
over N differs between structured and free_form reasoning modes (tested
as an N × reasoning_mode interaction term; two-sided, no assumed
direction). This is the paper's central contribution.

**H5 (secondary, confirmatory, bounded scope).** Chain-of-thought
reasoning text shows more "diluted responsibility" language (explicit
reasoning tying the decision to group size, diffusion of impact, or
reduced individual accountability) at N=5 than N=2, coded on a bounded
sample (N=2 and N=5 cells only, both modes, all 6 models).

**E1 (exploratory, NOT pre-registered, NOT confirmatory).** Reasoning-
tuned models (deepseek-r1-distill-qwen-14b, gpt-oss-120b) show a steeper
scaling-induced decline than standard instruction-tuned models (the other
4), connecting to prior work on reasoning models and free-riding. Report
this only as a descriptive/exploratory comparison, explicitly labeled as
such, never folded into H1/H2/H4's confirmatory results.

Descoped, explicit future work, not run in this cycle: opponent-
composition-matched sweep; B_eff-frozen ablation; N beyond 5.

Pre-registered fallback: a null H4 result is a valid, reportable outcome
(paper reframes around H1/H2/H5 as a rigor contribution). Do not treat a
null H4 as requiring new post-hoc hypotheses.

## 3. Experiment files — one file, one fixed purpose, each with its own
   summary output

Every script below writes a summary file in addition to raw output — a
human should be able to read the summary alone and know whether to
proceed to the next step, without re-parsing raw JSONL.

### `run_pilot.py` → tests: none (variance estimation only, feeds H1/H2/H4 design)
- Fixed grid: 4 N-levels × 2 reasoning_modes × 6 models × 5 seeds = 240
  episodes. Models passed via CLI, not hardcoded.
- `--dry-run` prints full run plan, episode count, and compute/time
  estimate (accounting for gpt-oss-120b's latency from Step 0). No real
  execution without explicit approval of the dry-run output.
- Output: `pilot_summary.md` — per-cell episode count completed, per-cell
  C̄ variance, flags for any cell with fallback_rate > 0.1 or suspiciously
  uniform C̄ across seeds, and a plain "ready for power calculation: yes/no"
  line.

### `analyze_pilot_variance.py` → feeds the (external, human-run) power
calculation
- Reads pilot JSONL only. No hypothesis testing of any kind here.
- Output: `variance_estimates.json` (per-cell variance, ready for a power
  calc) and `variance_summary.md` (plain-language version of the same,
  plus the fallback/uniformity flags carried over from run_pilot.py for
  visibility in one place).

### `run_main.py` → generates data for H1, H2, H4, E1
- Same 48-cell grid. `seeds_per_cell` is a REQUIRED argument with no
  default — must come from the power calculation, never guessed or
  inherited from the pilot's 5.
- `--dry-run` required before real execution, same pattern as run_pilot.py.
- Output: `main_run_summary.md` — total episodes completed per cell,
  reliability metrics (parse/repair/fallback rate) per model × mode,
  explicit flag of any cell that failed to complete its target seed count.

### `fit_glmm_and_curve.py` → tests H1, H2, H4, E1
- PRIMARY model (H1, H4): cooperation rate ~ N * reasoning_mode, model
  identity as random intercept (random slope for N by model if it
  converges). Report N main effect (H1) and N×mode interaction (H4) with
  coefficients, CIs, significance.
- SECONDARY, descriptive only: refit with model identity as fixed effect,
  for a per-model breakdown table/figure. Label explicitly as descriptive,
  not a second confirmatory test.
- H2: flat vs. 1/(N−1) curve fit via AIC/BIC on pooled structured-
  condition data across all 6 models. Report separately from H1/H4.
- E1: compare fitted N-slopes between the 2 reasoning-tuned models and the
  4 standard models. Label clearly as exploratory/non-pre-registered in
  the output itself, not just in a comment.
- Output: `confirmatory_results_summary.md` (H1, H2, H4 — the pre-
  registered tests, clearly separated from each other) and
  `exploratory_results_summary.md` (E1 and the descriptive per-model
  breakdown, clearly separated from the confirmatory file so the two are
  never accidentally cited interchangeably).

### `sample_for_cot_coding.py` → prepares data for H5
- Pulls every round from N=2 and N=5 cells only, both reasoning modes, all
  6 models, from run_main.py's output.
- Exports a coding sheet (CSV) with raw reasoning text, a blank code
  column, and condition labels hashed/stripped where possible so coders
  can code blind to condition.
- Output: `cot_sample_summary.md` — total rows sampled, breakdown by
  model/N/mode, so it's clear the sample is balanced before coding starts.

### `score_cot_coding.py` → tests H5
- Reads both coders' completed sheets. Computes Cohen's kappa. If kappa <
  0.7, outputs disagreement cases for rubric revision and does NOT report
  a diluted-responsibility rate — stops short of a conclusion until a
  fresh sample clears 0.7.
- Output: `h5_summary.md` — kappa, and (only if kappa ≥ 0.7) the coded
  rate of diluted-responsibility language by N and reasoning_mode.

## 4. Sequencing and gates (do not skip or reorder)

Step 0 (deployment) → run_pilot.py --dry-run → [human review] →
run_pilot.py (real) → analyze_pilot_variance.py → [human power
calculation, external to these scripts] → run_main.py --dry-run → [human
review] → run_main.py (real) → fit_glmm_and_curve.py → in parallel once
run_main.py completes: sample_for_cot_coding.py → [human coding, external]
→ score_cot_coding.py.

No script proceeds past a `--dry-run` or past a human-gated step (power
calculation, coding) without explicit confirmation in this conversation.
Report back after Step 0, after each dry-run, and after each summary file
is produced.