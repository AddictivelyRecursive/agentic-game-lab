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