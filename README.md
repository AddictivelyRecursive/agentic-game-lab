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
python -m game_engine.experiments.run_tournament
```

> Replace `run_tournament` with your actual tournament module name if different.
