# Local study scripts

The [study specification](study_specification.md) is authoritative. The six
entrypoints live at the repository root. They never SSH, provision inference,
submit Slurm jobs, or start servers. Only explicitly approved pilot/main
execution calls already configured loopback endpoints through LocalOpenAIClient.
Imports, `--help`, dry-runs, analysis, and coding preparation make no model calls.

## Files and outputs

| Script | Purpose | Summary |
|---|---|---|
| `run_pilot.py` | 48 cells × 5 seeds = 240 episodes; variance only | `pilot_summary.md` |
| `analyze_pilot_variance.py` | Sample variance from pilot JSONL only | `variance_summary.md`, plus `variance_estimates.json` |
| `run_main.py` | Same grid, required external power-based seed count | `main_run_summary.md` |
| `fit_glmm_and_curve.py` | H1/H4 mixed model, H2 curve comparison, descriptive E1 | `confirmatory_results_summary.md`, `exploratory_results_summary.md` |
| `sample_for_cot_coding.py` | All seat decisions in all rounds at N=2 and N=5 | `cot_sample_summary.md` |
| `score_cot_coding.py` | Two independent coders and kappa gate | `h5_summary.md` |

Shared validation/execution lives in `experiments/study_common.py`. No model is
selected by default: `--models` requires all six distinct aliases. The roster
constant validates the supplied CLI list; it does not choose a lineup for you.

## Setup and frozen choices

```powershell
python -m pip install -r requirements-study.txt
python run_pilot.py --help
python run_main.py --help
```

`experiments/study_config.json` fixes M=5, p=0, theta=0.50, T=50. The unspecified
payoff/drift/history/generation settings are inherited from the existing local
smoke configuration and are printed in the reviewable plan. Review these before
approving collection. N and reasoning mode come from the grid; the runner's
`--seed-start` and seed count replace the example config's seed list.

Each cell uses the selected model in every seat with independent agent state.
This follows the existing same-model local runner. It is explicit in the plan;
the scripts do not introduce an opponent-composition sweep. Memory is
`history_only`, with opponent prediction disabled. Structured mode uses the
existing expected-utility ranking; free_form uses the existing free-form prompt.

To make repetitions meaningful with p=0, the runner derives generation seeds
from phase/model/mode/N/episode-seed/seat, instead of reusing an identical fixed
sampling seed across repetitions. The derived seed is stable within a seat's
episode. It is recorded in each episode's `generation_by_seat` metadata and determined
by the plan's seed policy. Provider/GPU numerical nondeterminism can remain.

## Step 0 and dry-run

Deployment evidence must be measured externally and supplied in the format of
`experiments/deployment_report.example.json`. Its placeholders intentionally
fail validation. Include literal trivial responses, canonical model IDs,
evidence references, endpoints, and measured decision latency for each model.
GPT-OSS also requires measured VRAM, quantization, allocation/coexistence notes,
and five isolated smoke round latencies. Alias presence in the registry is not
deployment evidence. Do not fill unknown measurements with guesses.

These reports are human-supplied attestations, not cryptographic proof that a
model is loaded. Endpoints must still serve the specified aliases at execution.
Reconfirm Step 0 whenever deployment configuration or the roster changes.

```powershell
$models = @('llama-3.1-8b-instruct', 'qwen3-8b', 'gemma-3-12b-it', 'mistral-small-3.1-24b-instruct', 'deepseek-r1-distill-qwen-14b', 'gpt-oss-120b')
python run_pilot.py --models $models --deployment-report deployment_report.json --output output/pilot --dry-run
```

This writes `run_plan.json`, `dry_run_summary.md`, and the initial
`pilot_summary.md`, then exits. It prints all 48 cells, all seed IDs, episode and
decision counts, serial time, and active GPU-hours. GPT-OSS timing comes directly
from its five-round N=2 smoke (mean round time / 2 seats). Estimates exclude
queueing, loading, and idle reservations; histories, modes, and repairs can
change latency. No dollar cost or full-grid feasibility claim is fabricated.

## Approval and collection

After the user explicitly approves that dry-run in conversation, make a receipt
following `experiments/run_approval.example.json`: copy the plan SHA256, set
`confirmed_in_conversation` true, and record the actual approval reference.
The scripts cannot inspect conversation history. A receipt records approval;
it must not be created as a substitute for obtaining it.

```powershell
python run_pilot.py --models $models --deployment-report deployment_report.json --output output/pilot --approval pilot_approval.json
python analyze_pilot_variance.py --input output/pilot/pilot.jsonl --output output/pilot_variance
```

Real execution compares the complete plan, code fingerprint, configuration,
deployment evidence, and seed count with the saved dry-run. A changed plan needs
a new dry-run/output directory and renewed approval. Inference uses only local
loopback endpoints; there is no provider fallback or provisioning logic.

For separate serving allocations, `--execute-model ALIAS` executes just that
model's cells within the already reviewed full plan. Execute shards serially
against the same output directory. The full grid and missing cells remain in
the summaries; a shard alone cannot pass the analysis gate. The output directory
must be accessible to the process executing each shard, and its loopback
endpoint must exist there. The scripts do not transfer or merge remote files.

Completed episodes are skipped on another invocation of the same approved plan.
Failed episodes remain failed and are never silently replaced. Resolve failures
and review a new run rather than select favorable repetitions. A lock rejects
concurrent writers. Following an ungraceful process kill, inspect partial data
and verify that no process is running before manually removing a stale lock.

The consolidated `pilot.jsonl` or `main.jsonl` starts with a plan manifest and
then contains one episode record per attempt: true cooperation by round,
model-provided explanation, raw responses and repair attempts, reliability
flags, seed, condition, and timings. Original simulator files and full agent
traces are retained under `episodes/`. Interrupted/failed episodes cannot be
mistaken for completed 50-round episodes. A summary is updated after each
attempt and at runner exit.

## External power calculation and main

The variance analyzer reads only pilot JSONL. It performs no tests. Uniformity
means all completed seed means differ by at most 1e-12. Incomplete cells,
uniform cells, and any fallback contamination block readiness; fallback rates
over 0.1 are separately flagged. Variance is sample variance (ddof=1).

The human conducts the power calculation and confirms its result in conversation.
Create a report using `experiments/power_report.example.json`, tying the chosen
seed count to the SHA256 of the ready variance estimates and recording the
effect sizes/power assumptions in `rationale`. Paths in this report are relative
to the report file. `--seeds-per-cell` is required and must match the report.

```powershell
python run_main.py --models $models --deployment-report deployment_report.json --seeds-per-cell 20 --power-report power_report.json --output output/main --dry-run
# 20 is illustrative: substitute the actual externally calculated value.
# Only after separate, explicit approval of the main dry-run:
python run_main.py --models $models --deployment-report deployment_report.json --seeds-per-cell 20 --power-report power_report.json --output output/main --approval main_approval.json
python fit_glmm_and_curve.py --input output/main/main.jsonl --output output/analysis
python sample_for_cot_coding.py --input output/main/main.jsonl --output output/coding
```

The main summary reports completed/target counts for every cell and decision-
weighted first-call JSON parse, repair, and fallback rates by model × mode.
JSON parse success does not mean the action passed validation. A repair rate
counts decisions requiring at least one repair, not the number of repair calls.

Analysis/coding preparation require all 48 cells at their target seed counts
and zero fallback. This conservative rule prevents deterministic fallback
behavior from becoming model behavior. Changing that rule requires a separate
reviewed analysis decision; the scripts do not silently discard contaminated
episodes or cells.

## Statistical implementation

The analysis unit is the episode mean of normalized true cooperation across
50 rounds and N seats. Individual rounds are not independent replicates.
Cooperation is graded at M=5, so the implementation uses a **Gaussian mixed
model with identity link**, not a binary/binomial model. The specification
does not name a likelihood/link; this choice is explicit for review before data
collection. The requested filename `fit_glmm_and_curve.py` is retained.

- **H1/H4:** ML fit of `cooperation ~ N * free_mode`; structured is reference.
  Model identity is a random intercept, with random N slope attempted first.
  Unusable/nonconverged fits retry another optimizer, then the random-intercept
  model. If none is usable, the summary says the effects are not estimable;
  this is not reported as a null result. Coefficients, 95% Wald CIs, and
  two-sided asymptotic p-values are separate for H1 and H4. H1 support additionally
  requires a negative structured slope. Per-test alpha is 0.05, with no
  multiplicity correction specified. Six random-effect groups limit precision.
- **H2:** pooled structured episode means from all six models, flat `C=a`
  versus `C=a+b/(N-1)`, Gaussian ML AIC/BIC including residual variance in the
  parameter count. Positive flat-minus-dilution IC differences favor dilution;
  positive b means decline with N. H2 is reported separately.
- **Descriptive fixed effects:** `cooperation ~ N * free_mode * C(model)`
  allows genuinely different per-model/mode slopes. CSV and Markdown tables
  satisfy the requested per-model breakdown. No second confirmatory test.
- **E1:** equally weighted group mean slopes for two reasoning-tuned versus
  four standard models, by mode. Explicitly exploratory/non-pre-registered;
  architecture, scale, and training differences remain confounded.

Fit warnings/optimizer attempts, marginal predictions/residuals, analysis data,
and structured numerical results accompany the summaries. Boundary fits are
visible in diagnostics; no automatic post-hoc hypothesis is introduced for null
H4. See the [statsmodels MixedLM API](https://www.statsmodels.org/stable/generated/statsmodels.formula.api.mixedlm.html)
for the random intercept/slope interface used here.

DeepSeek-R1-Distill-Qwen-14B is a DeepSeek-trained distillation on a Qwen base,
representing an independent training organization/process. It is not silently
classified as a duplicate Qwen model. No behavioral archetypes are assumed.

## Human coding and H5

Give each coder only `coding_sheet.csv`. Do not share the private condition key
or sample summary. IDs are random and order is shuffled. Text is unchanged;
it can mention group size or reveal mode, so blinding is necessarily partial.
These are returned model explanations, not hidden chain-of-thought extraction.

Every seat/round in N=2 and N=5 cells is included. Consequently N=5 has 2.5 times
as many rows as N=2; episodes and rounds are balanced, not raw seat counts.
Codes are `1` for explicit diluted-responsibility reasoning, `0` otherwise,
and `NA` only for empty explanations. Both sheets must contain exactly the
same IDs and unchanged text. Missing text is reported and excluded, not coded 0.

After independent coding and explicit conversational confirmation, create the
receipt in `experiments/coding_approval.example.json` using the private key's
sample SHA256 and each completed CSV's file SHA256:

```powershell
python score_cot_coding.py --coder1 coder1.csv --coder2 coder2.csv --key output/coding/private_condition_key.json --approval coding_approval.json --reliability-history output/h5_reliability_history.json --output output/h5
```

Kappa <0.7 (or undefined kappa) produces disagreements and suppresses **all H5
rates and conclusions**. Reuse the same reliability-history ledger on every
attempt: it rejects overlap with failed samples, including reshuffled/renamed
rows. Since sampling includes all requested rows, a fresh independent sample
requires newly collected, separately approved data with distinct episode seeds;
the scripts never collect it automatically. Do not reset the ledger to bypass
the gate. Kappa ≥0.7 reports each coder's rates separately by N/mode, plus their
N=5 minus N=2 differences, without inventing adjudicated labels or a new H5
inferential test that the supplied protocol did not specify.

## Offline verification

```powershell
python -m unittest discover -s tests -v
```

Tests mock LocalOpenAIClient and generate synthetic fixtures in temporary
directories. They check execution gates, exact grids, variance flags, real
pipeline repair/abort recording, complete H5 coverage, kappa/freshness gates,
and recovery of known synthetic mixed-model effects. No real deployment,
pilot, main experiment, power calculation, or human coding is implied.
