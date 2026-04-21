# Agentic Game Lab

## Environment Setup

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

If you prefer to export it directly instead of using `.env`:

### Linux / macOS

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### Windows (PowerShell)

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

---

## Run Single Match

```bash
cd src
python -m game_engine.experiments.run_baseline_test
```

---

## Run Tournament

```bash
cd src
python -m game_engine.experiments.run_crossplay_tournament
```

---

## Make Plots

```bash
cd src
python -m plotting_scripts.plot_hostility_sweep --run-dir "results/hostility_sweep/hs__N2_M2_T100__p0.00__lam0.00__eta0.00__seeds1__f5__u4__t100__20260421_150314"       
```