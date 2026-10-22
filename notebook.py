# Marimo app: experiment-planning copilot, validation, and method.
# Opening this notebook does not call the network and does not need an API key.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Which experiment next?")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return mo, root


@app.cell
def _(root):
    import json

    try:
        import pandas as pd
    except Exception:
        pd = None

    def seed_label(value):
        try:
            number = float(value)
            if number.is_integer():
                return str(int(number))
        except Exception:
            pass
        return str(value)

    real_runs = root / "results" / "runs.csv"
    fixture_runs = root / "writeup" / "fixture_runs.csv"
    summary_path = root / "results" / "summary.json"
    table_path = root / "results" / "summary_table.csv"
    fig_dir = root / "results" / "figures"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    showing_fixture_runs = not (real_runs.exists() and real_runs.stat().st_size > 0)
    figures_are_fixture = bool(summary.get("fixture", showing_fixture_runs))
    fixture = showing_fixture_runs or figures_are_fixture
    runs_path = fixture_runs if showing_fixture_runs else real_runs
    runs = None
    table = None
    if pd is not None and runs_path.exists():
        try:
            runs = pd.read_csv(runs_path)
        except Exception:
            runs = None
    if pd is not None and table_path.exists():
        try:
            table = pd.read_csv(table_path)
        except Exception:
            table = None
    return fig_dir, fixture, runs, seed_label, summary, table


@app.cell
def _():
    def load_task_ids():
        try:
            from src.tasks import list_tasks
        except Exception:
            return []
        attempts = (
            lambda: list_tasks(),
            lambda: list_tasks(["BH", "DA", "SM", "MOCK"]),
            lambda: list_tasks(["MOCK"]),
        )
        for attempt in attempts:
            try:
                found = attempt()
            except Exception:
                continue
            if found:
                return [str(item) for item in list(found)]
        return []

    task_ids = load_task_ids()
    return (task_ids,)


@app.cell
def _(mo, task_ids):
    if task_ids:
        task_picker = mo.ui.dropdown(options=task_ids, value=task_ids[0], label="Task")
        task_note = "Tasks listed by src.tasks.list_tasks."
    else:
        task_picker = mo.ui.text(value="MOCK", label="Task id")
        task_note = "src.tasks.list_tasks is not available. Type a task id. Default is MOCK."
    return task_note, task_picker


@app.cell
def _(mo):
    placeholder_view = mo.md(
        """
The five rows below are placeholders for the shared initial design. They are not measured yields.

| n | round | pick | yield | note |
|---|---|---|---|---|
| 1 | 0 | placeholder | — | Initial design placeholder. Not a campaign result. |
| 2 | 0 | placeholder | — | Initial design placeholder. Not a campaign result. |
| 3 | 0 | placeholder | — | Initial design placeholder. Not a campaign result. |
| 4 | 0 | placeholder | — | Initial design placeholder. Not a campaign result. |
| 5 | 0 | placeholder | — | Initial design placeholder. Not a campaign result. |
"""
    )
    suggest = mo.ui.run_button(label="Suggest next 5")
    return placeholder_view, suggest


@app.cell
def _(mo, suggest, task_picker):
    clicked = bool(getattr(suggest, "value", False))
    if not clicked:
        suggest_view = mo.md(
            "Suggestions appear after **Suggest next 5**. "
            "Opening this notebook does not call the network and does not need an API key. "
            "The request uses an empty history, because the five rows above are placeholders."
        )
    else:
        task_id = str(task_picker.value or "MOCK")
        try:
            from src.recommend import recommend
        except Exception:
            recommend = None
        recs = []
        if recommend is not None:
            try:
                recs = recommend(task_id, [], [], q=5, strategy="hybrid")
            except Exception:
                recs = []
        if isinstance(recs, dict):
            if "reason" in recs or "names" in recs:
                recs = [recs]
            elif isinstance(recs.get("recommendations"), list):
                recs = recs["recommendations"]
            else:
                recs = []
        if not recs:
            suggest_view = mo.md(
                "set OPENAI_API_KEY or wait for cached results."
            )
        else:
            lines = ["| names | reason |", "|---|---|"]
            for rec in recs:
                if not isinstance(rec, dict):
                    lines.append(f"| {str(rec).replace('|', '/')} | |")
                    continue
                names = rec.get("names") or rec.get("labels") or {}
                if isinstance(names, dict):
                    name_text = "; ".join(f"{key}: {val}" for key, val in names.items())
                else:
                    name_text = str(names)
                reason = str(rec.get("reason") or "")
                name_text = " ".join(name_text.replace("|", "/").split())
                reason = " ".join(reason.replace("|", "/").split())
                lines.append(f"| {name_text} | {reason} |")
            suggest_view = mo.md("\n".join(lines))
    return (suggest_view,)


@app.cell
def _(fig_dir, fixture, mo, summary, table):
    blocks = []
    if fixture:
        blocks.append(mo.md("**FIXTURE DATA, not the hackathon result.**"))
    disclaimer = summary.get("disclaimer") if isinstance(summary, dict) else None
    if disclaimer:
        blocks.append(mo.md(str(disclaimer)))
    figures = (
        ("figure1_norm_best.png", "Figure 1. Mean normalised best-so-far versus n."),
        ("figure2_hit_rate.png", "Figure 2. Found-hit rate at the budget, by dataset."),
        ("figure3_ablation.png", "Figure 3. Name ablation at n = 10 and at the budget."),
    )
    for filename, caption in figures:
        path = fig_dir / filename
        blocks.append(mo.md(f"**{caption}**"))
        if path.exists():
            try:
                blocks.append(mo.image(src=str(path.resolve())))
            except Exception:
                blocks.append(mo.md(f"`{filename}` is on disk but could not be displayed."))
        else:
            blocks.append(
                mo.md(f"`{filename}` is not available. See `results/figures/MISSING.txt` when a figure was skipped.")
            )
    missing_path = fig_dir / "MISSING.txt"
    if missing_path.exists():
        missing_text = missing_path.read_text(encoding="utf-8").strip()
        if missing_text:
            blocks.append(mo.md("**Figures that were not drawn**\n\n" + missing_text))
    blocks.append(mo.md("**Summary table.** When the fixture banner is showing, this table is not the hackathon result."))
    if table is not None and not getattr(table, "empty", True):
        try:
            blocks.append(mo.ui.table(table))
        except Exception:
            blocks.append(mo.md("summary_table.csv could not be rendered."))
    else:
        blocks.append(mo.md("summary_table.csv is not available yet."))
    validation_body = mo.vstack(blocks)
    return (validation_body,)


@app.cell
def _(mo, runs):
    if runs is None or getattr(runs, "empty", True) or "strategy" not in getattr(runs, "columns", []):
        strategy_pick = None
        replay_note = "No run rows are available to replay."
    else:
        _options = sorted(runs["strategy"].astype(str).unique())
        strategy_pick = mo.ui.dropdown(options=_options, value=_options[0], label="Strategy")
        replay_note = "Rows are listed in increasing n, the order the experiments were taken."
    return replay_note, strategy_pick


@app.cell
def _(mo, runs, strategy_pick):
    if strategy_pick is None or runs is None:
        task_pick = None
    else:
        _rows = runs[runs["strategy"].astype(str) == str(strategy_pick.value)]
        _options = sorted(_rows["task_id"].astype(str).unique())
        task_pick = (
            mo.ui.dropdown(options=_options, value=_options[0], label="Task") if _options else None
        )
    return (task_pick,)


@app.cell
def _(mo, runs, seed_label, strategy_pick, task_pick):
    if task_pick is None or strategy_pick is None or runs is None:
        seed_pick = None
    else:
        _rows = runs[
            (runs["strategy"].astype(str) == str(strategy_pick.value))
            & (runs["task_id"].astype(str) == str(task_pick.value))
        ]
        _options = sorted({seed_label(value) for value in _rows["seed"].tolist()})
        seed_pick = (
            mo.ui.dropdown(options=_options, value=_options[0], label="Seed") if _options else None
        )
    return (seed_pick,)


@app.cell
def _(mo, replay_note, runs, seed_label, seed_pick, strategy_pick, task_pick):
    if seed_pick is None or strategy_pick is None or task_pick is None or runs is None:
        replay_view = mo.md(replay_note)
    else:
        _rows = runs[
            (runs["strategy"].astype(str) == str(strategy_pick.value))
            & (runs["task_id"].astype(str) == str(task_pick.value))
        ].copy()
        _rows = _rows[_rows["seed"].map(seed_label) == str(seed_pick.value)]
        if "n" in _rows.columns:
            _rows = _rows.sort_values("n")
        preferred = [
            "n",
            "round",
            "pick_idx",
            "yield",
            "best_so_far",
            "norm_best",
            "is_hit",
            "any_hit",
            "filled_random",
            "strategy",
            "dataset",
            "task_id",
            "seed",
        ]
        columns = [column for column in preferred if column in _rows.columns]
        shown = _rows[columns].reset_index(drop=True) if columns else _rows
        try:
            table_view = mo.ui.table(shown)
        except Exception:
            table_view = mo.md("The selected run could not be rendered.")
        replay_view = mo.vstack(
            [
                mo.md(replay_note),
                mo.hstack([strategy_pick, task_pick, seed_pick]),
                table_view,
            ]
        )
    return (replay_view,)


@app.cell
def _(mo, root):
    def read_text(relative, fallback):
        path = root / relative
        if not path.exists():
            return fallback
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            return fallback
        return text or fallback

    datasets = read_text("data/processed/DATASETS.md", "Datasets not landed yet.")
    method_p3 = read_text("results/method_p3.md", "P3 method notes have not landed yet.")
    method_p4 = read_text("results/method_p4.md", "P4 method notes have not landed yet.")
    problem = mo.md(
        """
## Problem

Every lab experiment costs time and money, so the real question when optimising a reaction is which experiment to run next, given the results so far.

We compare decision-makers that each get the same small budget on fully measured reaction datasets. Because every condition has already been run, running an experiment is a table lookup. The deliverable is an experiment-planning copilot: a chemist picks a reaction and sees the next suggested batch with written reasons, plus the validation results.
"""
    )
    protocol = mo.md(
        """
## Protocol

5 random initial, batch q=5, BH budget 20, DA budget 30, seeds 10 statistical / 3 LLM, hit = top 1% or top 3 whichever set is larger.

The first five conditions are identical across strategies for a given task and seed. A strategy never sees the yield of a condition it has not picked. Statistical arms are random, GP-EI, and TabPFN or the random-forest fallback. LLM arms are the chemist, the hybrid, and the anonymised ablation.

```
history of (condition, yield) -> strategy proposes 5 conditions -> table lookup returns yields -> history
```
"""
    )
    strategies = mo.md(
        """
## Strategies

- `random`: uniform without replacement.
- `gp_ei`: Gaussian process, Matern kernel, expected improvement, top q.
- `tabpfn_ucb` or `rf_ucb`: TabPFN, or a random-forest fallback, top q by the 0.9 quantile.
- `llm`: the language model picks q from the history and the chemical names.
- `hybrid`: the language model shortlists 20 and TabPFN ranks them.
- `llm_anon`: the same agent with component names hidden.
- `llm_zeroshot` and `oracle` and `tabpfn_transfer`: included when those runs exist.
"""
    )
    related = mo.md(
        """
## Related work

- Shields et al., Nature 2021: Bayesian optimisation beat expert chemists on reaction optimisation.
- Ramos et al., 2023, BO-ICL: Bayesian optimisation with LLM in-context learning.
- Liu et al., ICLR 2024, LLAMBO.
- Kristiadi et al., ICML 2024: a critical look at LLMs for Bayesian optimisation over molecules.
- Hollmann et al., Nature 2025, TabPFN.

These are cited, not claimed as new. The pieces this project adds are TabPFN as the surrogate on these reaction screens, the LLM-shortlist plus TabPFN-rank hybrid, a head-to-head across tasks, and the anonymised-names ablation.
"""
    )
    limitations = mo.md(
        """
## Limitations

- The datasets are retrospective lookup tables; a live lab adds noise and failed runs.
- These datasets are well known, so the LLM may have seen them in training. The anonymised ablation addresses this partly but cannot rule it out.
- The batch rule (top q by acquisition) is simple.
- There are few tasks and seeds, so treat effect sizes with caution.
"""
    )
    method_view = mo.vstack(
        [
            problem,
            protocol,
            mo.md("## Datasets\n\n" + datasets),
            strategies,
            mo.md("## Notes from the classical arms\n\n" + method_p3),
            mo.md("## Notes from the LLM arms\n\n" + method_p4),
            related,
            limitations,
        ]
    )
    return (method_view,)


@app.cell
def _(
    method_view,
    mo,
    placeholder_view,
    replay_view,
    suggest,
    suggest_view,
    task_note,
    task_picker,
    validation_body,
):
    title = mo.md(
        """
# Which experiment next?

An experiment-planning copilot for reaction screens. This page does not spend API credits when it opens.
"""
    )
    copilot = mo.vstack(
        [
            mo.md(task_note),
            task_picker,
            placeholder_view,
            suggest,
            suggest_view,
        ]
    )
    validation = mo.vstack(
        [
            validation_body,
            mo.md("### Replay one run"),
            replay_view,
        ]
    )
    try:
        view = mo.ui.tabs(
            {"Copilot": copilot, "Validation": validation, "Method": method_view}
        )
    except Exception:
        view = mo.vstack(
            [
                mo.md("## Copilot"),
                copilot,
                mo.md("## Validation"),
                validation,
                mo.md("## Method"),
                method_view,
            ]
        )
    mo.vstack([title, view])
    return


if __name__ == "__main__":
    app.run()
