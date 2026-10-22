# Which experiment next?

Every lab experiment costs time and money, so the real question when optimising a reaction is which experiment to run next, given the results so far. This project compares decision-makers that each get the same small budget of experiments on fully measured reaction screens: random search, Gaussian-process Bayesian optimisation, a TabPFN surrogate, an LLM agent that reasons from the results so far, and a hybrid where the LLM shortlists and TabPFN ranks. Because every condition has already been run, an experiment is a table lookup. The deliverable is an experiment-planning copilot for chemists, validated on real data, as a marimo app.

## Install

From this directory:

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS and Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## API key

Set `OPENAI_API_KEY` in the environment before running an LLM strategy. The key stays in the environment and out of the repository. `.env` is gitignored.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-key-here"
```

bash:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Configuration

`config.yaml` holds the batch size (`q: 5`), the initial design size (`initial_design: 5`), budgets (`BH` 20, `SM` 20, `DA` 30, `MOCK` 20), seed counts (`statistical: 10`, `llm: 3`), the hit rule, and the model name `gpt-6-luna`. The campaign runner reads this file.

## Run the MOCK task

MOCK is a 120-condition screen (4 ligands × 3 bases × 10 additives) with fixed yields, used until the real tasks are available. From this directory, with the virtual environment active:

```bash
python -m src.campaign --strategy random --module src.strategies.random_s --tasks MOCK --task-ids MOCK --seeds 2
python -m src.campaign --strategy oracle --module src.strategies.oracle --tasks MOCK --task-ids MOCK --seeds 2
python -m src.collect
```

`--seeds 2` runs seeds 0 and 1. Each run writes `results/runs/<strategy>__<task_id>__<seed>.csv`. `python -m src.collect` merges those files into `results/runs.csv`.

Columns: strategy, dataset, task_id, seed, n, round, pick_idx, yield, best_so_far, norm_best, is_hit, any_hit, filled_random.

Datasets and the other strategies are owned by teammates and will appear as their modules land.
