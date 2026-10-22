"""Draw campaign figures from summary.json and the runs file.

PNG files are 200 dpi. A figure with no data is not drawn; the absence is
recorded in results/figures/MISSING.txt. Fixture charts are titled as fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analysis import ROOT, build_report, display_path, jsonable, load_runs, resolve_runs_path, sort_strategies
except ImportError:  # pragma: no cover - direct execution fallback
    from src.analysis import ROOT, build_report, display_path, jsonable, load_runs, resolve_runs_path, sort_strategies

FIG_DIR = ROOT / "results" / "figures"
SUMMARY_PATH = ROOT / "results" / "summary.json"
ABLATION = ["llm", "llm_anon", "llm_zeroshot"]

LABELS = {
    "random": "Random",
    "oracle": "Oracle",
    "gp_ei": "GP-EI",
    "rf_ucb": "RF q90",
    "tabpfn_ucb": "TabPFN q90",
    "tabpfn_transfer": "TabPFN transfer",
    "llm": "LLM",
    "llm_anon": "LLM anonymised",
    "llm_zeroshot": "LLM zero-shot",
    "hybrid": "Hybrid",
}

STYLES = {
    "random": {"color": "grey", "linestyle": "-", "linewidth": 2.0},
    "gp_ei": {"color": "blue", "linestyle": "-", "linewidth": 2.0},
    "tabpfn_ucb": {"color": "green", "linestyle": "-", "linewidth": 2.0},
    "rf_ucb": {"color": "green", "linestyle": "-", "linewidth": 2.0},
    "tabpfn_transfer": {"color": "green", "linestyle": "-.", "linewidth": 2.0},
    "llm": {"color": "orange", "linestyle": "-", "linewidth": 2.0},
    "hybrid": {"color": "red", "linestyle": "-", "linewidth": 3.5},
    "llm_anon": {"color": "orange", "linestyle": "--", "linewidth": 2.0},
    "oracle": {"color": "lightgrey", "linestyle": "-", "linewidth": 1.0},
    "llm_zeroshot": {"color": "orange", "linestyle": ":", "linewidth": 2.0},
}


def _style(name: str) -> dict:
    return STYLES.get(name, {"color": "black", "linestyle": "-", "linewidth": 2.0})


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _source_clause(fixture: bool) -> str:
    if fixture:
        return "FIXTURE data, not the hackathon result"
    return "real campaign runs"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _retire(path: Path) -> None:
    if path.exists():
        path.unlink()


def _as_float_array(values) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)


def _legend(ax, count: int) -> None:
    if count > 4:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        ax.legend(frameon=False)


def _ylim_from_values(values: np.ndarray, default_top: float = 1.05) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, default_top
    top = max(default_top, float(np.nanmax(finite)) * 1.08)
    return 0.0, top


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        print("summary.json not found. Figures are computed from the runs file.", file=sys.stderr)
        return {}
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"summary.json could not be parsed ({exc}). Figures use the runs file.", file=sys.stderr)
        return {}


def figure1(report: dict, dest: Path) -> bool:
    curves = report.get("curves") or {}
    names = [name for name in sort_strategies(curves) if curves[name].get("n")]
    if not names:
        _retire(dest)
        return False
    fixture = bool(report.get("fixture"))
    fig, ax = plt.subplots(figsize=(11.2, 6.6), constrained_layout=True)
    plotted = []
    band_values = []
    for name in names:
        curve = curves[name]
        steps = curve["n"]
        means = _as_float_array(curve["mean"])
        low = _as_float_array(curve["ci_low"])
        high = _as_float_array(curve["ci_high"])
        if not np.isfinite(means).any():
            continue
        style = _style(name)
        ax.plot(
            steps,
            means,
            label=_label(name),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=4 if name == "hybrid" else 2,
        )
        if np.isfinite(low).any() and np.isfinite(high).any():
            ax.fill_between(
                steps,
                low,
                high,
                color=style["color"],
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            band_values.extend([low, high])
        plotted.append(means)
        band_values.append(means)
    if not plotted:
        plt.close(fig)
        _retire(dest)
        return False
    ax.axvline(10, color="black", linestyle="--", linewidth=1.2, label="n = 10", zorder=3)
    counts_change = any(
        len(set(curves[name].get("n_pairs") or [])) > 1 for name in names
    )
    if counts_change:
        ax.set_xlabel("Experiments (n)\nEach point averages only the runs that reached that n")
    else:
        ax.set_xlabel("Experiments (n)")
    ax.set_ylabel("Mean normalised best-so-far")
    ax.set_title(f"Mean normalised best-so-far versus n\n{_source_clause(fixture)}")
    stacked = np.concatenate(band_values) if band_values else np.array([])
    ax.set_ylim(*_ylim_from_values(stacked))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _legend(ax, len(names) + 1)
    _save(fig, dest)
    return True


def figure2(report: dict, dest: Path) -> bool:
    by_dataset = (report.get("metrics") or {}).get("by_dataset") or {}
    datasets = list(by_dataset)
    strategies = sort_strategies(
        {name for block in by_dataset.values() for name in block}
    )
    if not datasets or not strategies:
        _retire(dest)
        return False
    matrix = []
    any_value = False
    for name in strategies:
        row = []
        for dataset in datasets:
            block = by_dataset[dataset].get(name) or {}
            rate = block.get("found_hit_rate")
            if rate is None:
                row.append(np.nan)
            else:
                row.append(float(rate))
                any_value = True
        matrix.append(row)
    if not any_value:
        _retire(dest)
        return False

    fixture = bool(report.get("fixture"))
    fig, ax = plt.subplots(figsize=(11.2, 6.6), constrained_layout=True)
    x = np.arange(len(datasets))
    width = min(0.72 / len(strategies), 0.32)
    drawn = []
    for index, name in enumerate(strategies):
        offset = (index - (len(strategies) - 1) / 2) * width
        style = _style(name)
        heights = np.asarray(matrix[index], dtype=float)
        ax.bar(
            x + offset,
            heights,
            width=width,
            label=_label(name),
            color=style["color"],
            edgecolor="black",
            linewidth=1.6 if name == "hybrid" else 0.4,
        )
        drawn.append(heights)
    ax.set_xticks(x, datasets, rotation=0 if len(datasets) < 5 else 20)
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_ylabel("Found-hit rate at budget")
    ax.set_xlabel("Dataset")
    ax.set_title(f"Found-hit rate at the budget by dataset\n{_source_clause(fixture)}")
    ax.set_ylim(*_ylim_from_values(np.concatenate(drawn), default_top=1.05))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _legend(ax, len(strategies))
    _save(fig, dest)
    return True


def _error_bar(block: dict, key: str) -> tuple[float, float, float]:
    stats = block.get(key) or {}
    mean = stats.get("mean")
    low = stats.get("ci_low")
    high = stats.get("ci_high")
    if mean is None:
        return np.nan, np.nan, np.nan
    mean_f = float(mean)
    err_low = np.nan if low is None else mean_f - float(low)
    err_high = np.nan if high is None else float(high) - mean_f
    return mean_f, err_low, err_high


def figure3(report: dict, dest: Path) -> str | None:
    overall = (report.get("metrics") or {}).get("overall") or {}
    present = [name for name in ABLATION if name in overall]
    missing = [name for name in ABLATION if name not in overall]
    if not present:
        _retire(dest)
        return (
            "Figure 3 was not written. Inputs absent: strategies "
            + ", ".join(ABLATION)
            + ". No bar was invented."
        )
    means = []
    yerr = []
    usable = False
    for name in present:
        pair = []
        errors = [[], []]
        for key in ("norm_best_n10", "norm_best_budget"):
            mean, err_low, err_high = _error_bar(overall[name], key)
            pair.append(mean)
            errors[0].append(err_low)
            errors[1].append(err_high)
            if np.isfinite(mean):
                usable = True
        means.append(pair)
        yerr.append(errors)
    if not usable:
        _retire(dest)
        return (
            "Figure 3 was not written. "
            + ", ".join(present)
            + " have no norm_best values at n=10 or at the budget."
        )

    fixture = bool(report.get("fixture"))
    title = f"LLM ablation at n=10 and at the budget\n{_source_clause(fixture)}"
    if missing:
        title += "\nAbsent, and not plotted: " + ", ".join(missing)
    fig, ax = plt.subplots(figsize=(11.2, 6.8), constrained_layout=True)
    x = np.arange(2)
    width = min(0.8 / len(present), 0.36)
    hatches = {"llm": "", "llm_anon": "///", "llm_zeroshot": "..."}
    collected = []
    for index, name in enumerate(present):
        offset = (index - (len(present) - 1) / 2) * width
        style = _style(name)
        heights = np.asarray(means[index], dtype=float)
        ax.bar(
            x + offset,
            heights,
            width=width,
            label=_label(name),
            color=style["color"],
            edgecolor="black",
            linewidth=0.6,
            hatch=hatches.get(name, ""),
            yerr=np.asarray(yerr[index], dtype=float),
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )
        collected.append(heights)
    ax.set_xticks(x, ["n = 10", "budget"])
    ax.set_ylabel("Mean normalised best-so-far")
    ax.set_title(title)
    ax.set_ylim(*_ylim_from_values(np.concatenate(collected)))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _legend(ax, len(present))
    _save(fig, dest)
    return None


def _seed_key(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def figure4(df, report: dict, dest: Path) -> str | None:
    """Annotate a real trajectory only when logged reasons match run rows."""
    if report.get("fixture"):
        _retire(dest)
        return (
            "Figure 4 was not written. The runs are fixture data, so no reason "
            "annotations were drawn."
        )
    reasons_path = ROOT / "results" / "llm_reasons.jsonl"
    if not reasons_path.exists():
        _retire(dest)
        return (
            "Figure 4 was not written. Input absent: results/llm_reasons.jsonl. "
            "No annotations were invented."
        )
    records = []
    for line in reasons_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and str(item.get("reason") or "").strip():
            records.append(item)
    if not records:
        _retire(dest)
        return (
            "Figure 4 was not written. results/llm_reasons.jsonl has no usable "
            "reason records. No annotations were invented."
        )

    work = df.copy()
    work["seed_key"] = work["seed"].map(_seed_key)
    grouped: dict[tuple[str, str, str], list] = {}
    for item in records:
        key = (
            str(item.get("strategy") or ""),
            str(item.get("task_id") or ""),
            _seed_key(item.get("seed")),
        )
        grouped.setdefault(key, []).append(item)

    chosen = None
    notes = []
    for (strategy, task_id, seed_key), items in grouped.items():
        sub = work[
            (work["strategy"] == strategy)
            & (work["task_id"] == task_id)
            & (work["seed_key"] == seed_key)
        ]
        if sub.empty:
            continue
        matched = []
        for item in items:
            try:
                row_idx = float(item.get("row_idx"))
            except (TypeError, ValueError):
                continue
            picks = pd_to_float(sub["pick_idx"])
            hit = sub[np.isfinite(picks) & np.isclose(picks, row_idx)]
            if hit.empty:
                continue
            step = int(hit.sort_values("n").iloc[0]["n"])
            matched.append((step, str(item.get("reason")).strip()))
        if matched:
            chosen = (task_id, seed_key)
            notes = matched[:3]
            break
    if chosen is None:
        _retire(dest)
        return (
            "Figure 4 was not written. Reason records did not match any run row. "
            "No annotation was invented."
        )

    task_id, seed_key = chosen
    task_runs = work[(work["task_id"] == task_id) & (work["seed_key"] == seed_key)]
    fig, ax = plt.subplots(figsize=(11.2, 6.8), constrained_layout=True)
    for name, sub in task_runs.groupby("strategy"):
        ordered = sub.sort_values("n")
        style = _style(str(name))
        ax.plot(
            ordered["n"],
            ordered["norm_best"],
            label=_label(str(name)),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    for step, reason in notes:
        rows = task_runs[task_runs["n"] == step]
        if rows.empty:
            continue
        y_value = float(rows["norm_best"].iloc[0])
        short = " ".join(reason.split())
        if len(short) > 90:
            short = short[:87] + "..."
        ax.annotate(
            short,
            xy=(step, y_value),
            textcoords="offset points",
            xytext=(8, 14),
            fontsize=11,
            arrowprops={"arrowstyle": "->", "color": "black"},
        )
    ax.axvline(10, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Experiments (n)")
    ax.set_ylabel("Normalised best-so-far")
    ax.set_title(
        f"Task {task_id}, seed {seed_key}: logged reasons on the trajectory\nreal campaign runs"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    _legend(ax, task_runs["strategy"].nunique())
    _save(fig, dest)
    return None


def pd_to_float(series):
    return np.asarray(series, dtype=float)


def _write_notes(notes: list[str]) -> None:
    path = FIG_DIR / "MISSING.txt"
    if notes:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"Wrote {display_path(path)}")
    elif path.exists():
        path.unlink()


def main() -> int:
    _apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    try:
        runs_path, fixture = resolve_runs_path(None)
    except SystemExit as exc:
        _write_notes([f"No figures were written. {exc}"])
        return 1
    if summary and bool(summary.get("fixture")) != fixture:
        print(
            "summary.json fixture flag does not match the runs file. "
            "Figures follow the runs file.",
            file=sys.stderr,
        )
    frame = load_runs(runs_path)
    report = jsonable(build_report(frame, fixture=fixture, source=display_path(runs_path)))
    # summary.json is read above so a stale flag is visible. Plot data comes from
    # the same runs file analysis.py would summarise, including its fixture flag.
    notes = []
    fig1 = FIG_DIR / "figure1_norm_best.png"
    fig2 = FIG_DIR / "figure2_hit_rate.png"
    fig3 = FIG_DIR / "figure3_ablation.png"
    fig4 = FIG_DIR / "figure4_reasons.png"

    if figure1(report, fig1):
        print(f"Wrote {display_path(fig1)}")
    else:
        notes.append(
            "Figure 1 was not written. The runs file has no norm_best curve to plot."
        )
    if figure2(report, fig2):
        print(f"Wrote {display_path(fig2)}")
    else:
        notes.append(
            "Figure 2 was not written. Found-hit rates by dataset were absent."
        )
    note3 = figure3(report, fig3)
    if note3:
        notes.append(note3)
    else:
        print(f"Wrote {display_path(fig3)}")
    note4 = figure4(frame, report, fig4)
    if note4:
        notes.append(note4)
    else:
        print(f"Wrote {display_path(fig4)}")
    _write_notes(notes)
    if report.get("fixture"):
        print("DRAFT / fixture: figure titles mark this source. Not a hackathon result.")
    return 0 if fig1.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
