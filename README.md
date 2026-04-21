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
python "plots/plotting_scripts.py"  --root "results/causal_realism/cr__N5__M2-3-5__p3__l3__s1__t50__20260418_220804"        
```