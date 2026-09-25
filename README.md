# README

# Agentic Game Lab

A modular framework for simulating and analyzing LLM agents in game-theoretic environments, with a focus on Iterated Prisoner’s Dilemma (IPD) and multi-agent strategic behavior.

## Features

- Multi-player IPD with configurable parameters:
  - Number of players (N)
  - Action granularity (M)
  - Noise
  - Payoff dynamics (drift, streak incentives)
- Plug-and-play LLM agents
- Structured logging (JSONL)
- Experiment runners for systematic sweeps

## Setup

```bash
git clone https://github.com/AddictivelyRecursive/agentic-game-lab
cd agentic-game-lab

pip install -r requirements.txt
```

Add API keys (OpenRouter) in a `.env` file.

## Running

```bash
python -m game_engine.experiments.run_causal_M
```

Other scripts:

- `run_llmvsbaseline.py`
- `run_hostility_sweep.py`
- `run_baseline_test.py`
- `run_causal_noise.py`
- `run_causal_theta.py`
- `run_causal_N_hetrogeneous.py`
- `run_causal_NxMxP.py`

python -m src.plots.plot_llmvsbaseline --run-dir "src\results\llm_vs_baseline\lvb__N2_M5_T50__p0.05__lam0.25__eta0.35__seeds5__f5__b6__swap0__t50__20260427_123810"

## Controlled sensitivity experiments

Run commands from `src` so package imports and prompt paths resolve:

```powershell
cd src
python -m game_engine.experiments.run_sensitivity --dry-run --output results/sensitivity_preview
python -m game_engine.experiments.run_sensitivity --model YOUR_OPENROUTER_MODEL --repeats 3 --output results/sensitivity_run
```

The paired snapshot suite varies N, M, perception noise p, and drift strength eta.
It records actions, normalized cooperation, explanations, optional expectations,
and validity. Failed model decisions must not be treated as model behavior.
N duplicates an opponent history, holding observed mean cooperation constant;
M preserves cooperation levels. The eta comparison disables the streak bonus.
These are hypothetical fixed observations, not sampled match trajectories.
Repeated calls are unseeded model samples. Each case resets the agent and uses
history-only memory so unrelated cases do not carry notes forward.

## Memory and prediction experiments

For full-match memory comparisons, run the baseline experiment separately:

```powershell
$env:AGENT_MEMORY_MODE = "history_only"
python -m game_engine.experiments.run_llmvsbaseline
$env:AGENT_MEMORY_MODE = "model_memory"
python -m game_engine.experiments.run_llmvsbaseline
```

History-only is the default. Model-memory carries four model-written text fields:
`hypothesis`, `evidence`, `uncertainty`, and `reconsider_if`, each capped at 400
characters. Notes replace previous notes after a successful model decision and
are cleared on reset. Fallback decisions never update memory. These notes are
fallible hypotheses and do not constitute a validated opponent model or proof of
exploration. Other runners can choose `LLMWrapperAgent(memory_mode=...)`.

Opponent predictions are now an explicit ablation, off by default. Set
`AGENT_PREDICT_OPPONENTS=1` for the baseline comparison runner to request them.
Older runs that always requested predictions use a different prompt condition.
Set `AGENT_MAX_RETRIES` to fix the repair budget (default 3; zero disables repairs).
Both settings are also constructor arguments `predict_opponents` and `max_retries`
on `LLMAgent` and `LLMWrapperAgent`.

Episode metadata records model validity, memory and prediction conditions, and
repair budgets. Run manifests record the baseline runner settings. Episodes with
fallback decisions are marked invalid for model comparison; exclude or separately
report them before aggregating model rewards.

## Agent decision pipeline

`src/AI_Agent/agent/pipeline.py` implements:

```text
Prepare context -> LLM decision -> Validate -> Save memory, return, log
                                      |
                                   Invalid
                                      |
                              Bounded repair -> Validate
                                      |
                              Budget exhausted
                                      |
                            Flag failure + fallback
```

Context preparation includes the observation-noise rules, including p=0.
A normal turn makes one LLM call. Repairs use the same evidence and latest failed
output. The fallback chooses the immediate-payoff action and is explicitly marked
as a model failure. The old graph runner, schema, node classes, and separate return
node have been removed. Traces now use descriptive `stage` entries and retain each
model attempt in `model_calls`. Legacy raw-output keys `N6` and `N8` remain for
existing analysis scripts.

## Experiment settings and provider failures

Shared full-match defaults in `src/game_engine/experiments/settings.py` use a
20-round horizon, 10-round history/statistics, and an 8-round drift window.
The snapshot suite uses the same horizon; its supplied four-round history and
drift window remain matched.

Runners that enable `stop_on_api_failure` stop on authentication/access/credit
errors (HTTP 401/402/403), or three consecutive provider request failures. A
successful API response resets the counter; JSON parsing errors do not count as
provider failures. Aborted calls are logged and never update memory.
