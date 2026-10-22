"""Summarise campaign runs.

Reads results/runs.csv when it exists. Otherwise reads writeup/fixture_runs.csv
and marks the summary as fixture data. Never writes results/runs.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
BOOT_SEED = 0

REQUIRED_COLUMNS = [
    "strategy",
    "dataset",
    "task_id",
    "seed",
    "n",
    "round",
    "pick_idx",
    "yield",
    "best_so_far",
    "norm_best",
    "is_hit",
    "any_hit",
    "filled_random",
]

STRATEGY_ORDER = [
    "random",
    "oracle",
    "gp_ei",
    "rf_ucb",
    "tabpfn_ucb",
    "tabpfn_transfer",
    "llm_zeroshot",
    "llm_anon",
    "llm",
    "hybrid",
]

DEFINITIONS = {
    "norm_best_n10": "Mean of norm_best at n=10 over (task, seed) pairs.",
    "norm_best_budget": "Mean of norm_best at the task budget, the maximum n observed for that task.",
    "found_hit_rate": "Fraction of (task, seed) runs with any_hit true at that budget row.",
    "median_experiments_to_first_hit": (
        "Median of the first n where any_hit is true. A run that never hits contributes budget+1. "
        "The median is numpy's median, so an even count averages the two central values."
    ),
    "bootstrap": (
        "2000 resamples of the (task, seed) values, numpy Generator seed 0, "
        "percentiles 2.5 and 97.5. The interval is null when fewer than 2 pairs are available."
    ),
}


def sort_strategies(names) -> list[str]:
    labels = {str(name) for name in names}
    rank = {name: index for index, name in enumerate(STRATEGY_ORDER)}
    return sorted(labels, key=lambda name: (rank.get(name, len(STRATEGY_ORDER)), name))


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_runs_path(explicit: str | None = None) -> tuple[Path, bool]:
    """Return (path, fixture_flag). A real results/runs.csv wins over the fixture."""
    fixture_path = (ROOT / "writeup" / "fixture_runs.csv").resolve()
    if explicit:
        raw = Path(explicit)
        path = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
        if not path.exists():
            alt = (ROOT / raw).resolve()
            path = alt if alt.exists() else path
        if not path.exists():
            raise SystemExit(f"Runs file not found: {explicit}")
        return path, path.resolve() == fixture_path

    real = ROOT / "results" / "runs.csv"
    if real.exists() and real.stat().st_size > 0:
        return real.resolve(), False
    if real.exists() and real.stat().st_size == 0:
        print("results/runs.csv is empty. Using writeup/fixture_runs.csv.", file=sys.stderr)
    if not fixture_path.exists():
        raise SystemExit(
            "No runs file found. Expected results/runs.csv or writeup/fixture_runs.csv."
        )
    return fixture_path, True


def _normalize_seed(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return int(number) if number.is_integer() else number
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    def one(value) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return False
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    return series.map(one).astype(bool)


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SystemExit(f"{display_path(path)} is missing columns: {', '.join(missing)}")
    frame = df.copy()
    for column in ("strategy", "dataset", "task_id"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["n"] = pd.to_numeric(frame["n"], errors="coerce")
    frame["norm_best"] = pd.to_numeric(frame["norm_best"], errors="coerce")
    frame["pick_idx"] = pd.to_numeric(frame["pick_idx"], errors="coerce")
    frame["seed"] = frame["seed"].map(_normalize_seed)
    frame["any_hit"] = _as_bool(frame["any_hit"])
    frame = frame.dropna(subset=["n", "norm_best"])
    frame = frame[frame["seed"].notna()]
    frame = frame[frame["strategy"] != ""]
    frame = frame[frame["strategy"].str.lower() != "nan"]
    frame = frame[frame["task_id"] != ""]
    frame["n"] = frame["n"].astype(int)
    frame = frame.drop_duplicates(
        ["strategy", "dataset", "task_id", "seed", "n"], keep="last"
    )
    return frame


def mean_ci(values) -> dict:
    """Mean and a percentile bootstrap interval. Seed 0 on every call."""
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_pairs": 0}
    mean = float(array.mean())
    if count < 2:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n_pairs": count}
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.integers(0, count, size=(N_BOOT, count))
    means = array[draw].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": count,
    }


def per_run_frame(df: pd.DataFrame) -> pd.DataFrame:
    budgets = df.groupby("task_id")["n"].max().astype(int).to_dict()
    records = []
    grouped = df.groupby(["strategy", "dataset", "task_id", "seed"], sort=True)
    for (strategy, dataset, task_id, seed), group in grouped:
        budget = int(budgets[task_id])
        ordered = group.sort_values("n")
        at_10 = ordered.loc[ordered["n"] == 10, "norm_best"]
        at_budget = ordered.loc[ordered["n"] == budget, "norm_best"]
        hit_at_budget = ordered.loc[ordered["n"] == budget, "any_hit"]
        hit_steps = ordered.loc[ordered["any_hit"], "n"]
        first = int(hit_steps.min()) if len(hit_steps) else budget + 1
        records.append(
            {
                "strategy": str(strategy),
                "dataset": str(dataset),
                "task_id": str(task_id),
                "seed": seed,
                "budget": budget,
                "norm_n10": float(at_10.iloc[-1]) if len(at_10) else np.nan,
                "norm_budget": float(at_budget.iloc[-1]) if len(at_budget) else np.nan,
                "found_hit": bool(hit_at_budget.iloc[-1]) if len(hit_at_budget) else None,
                "first_hit_n": first,
            }
        )
    return pd.DataFrame.from_records(records)


def _summarise_block(runs: pd.DataFrame) -> dict:
    hit_values = [bool(value) for value in runs["found_hit"].tolist() if value is not None]
    first = runs["first_hit_n"].to_numpy(dtype=float)
    return {
        "n_runs": int(len(runs)),
        "norm_best_n10": mean_ci(runs["norm_n10"].to_numpy(dtype=float)),
        "norm_best_budget": mean_ci(runs["norm_budget"].to_numpy(dtype=float)),
        "found_hit_rate": float(np.mean(hit_values)) if hit_values else None,
        "found_hit_n": int(len(hit_values)),
        "median_experiments_to_first_hit": float(np.median(first)) if len(first) else None,
    }


def build_curves(df: pd.DataFrame) -> dict:
    curves = {}
    for strategy, sub in df.groupby("strategy", sort=True):
        steps = sorted(int(value) for value in sub["n"].unique())
        means, lows, highs, counts = [], [], [], []
        for step in steps:
            stats = mean_ci(sub.loc[sub["n"] == step, "norm_best"].to_numpy(dtype=float))
            means.append(stats["mean"])
            lows.append(stats["ci_low"])
            highs.append(stats["ci_high"])
            counts.append(stats["n_pairs"])
        curves[str(strategy)] = {
            "n": steps,
            "mean": means,
            "ci_low": lows,
            "ci_high": highs,
            "n_pairs": counts,
        }
    return curves


def wilcoxon_block(runs: pd.DataFrame) -> dict:
    """Paired test vs gp_ei. Omit the test, with a reason, when it should not be run."""
    base = {
        "reference": "gp_ei",
        "endpoint": "norm_best at the budget",
        "performed": False,
        "reason": None,
        "tests": [],
    }
    if runs.empty or "gp_ei" not in set(runs["strategy"]):
        base["reason"] = (
            "gp_ei is not among the strategies, so the paired Wilcoxon test was omitted."
        )
        return base
    try:
        from scipy.stats import wilcoxon
    except Exception:
        base["reason"] = "scipy is not available, so the paired Wilcoxon test was omitted."
        return base

    usable = runs.dropna(subset=["norm_budget"]).copy()
    reference = usable.loc[
        usable["strategy"] == "gp_ei", ["task_id", "seed", "norm_budget"]
    ]
    others = [name for name in sort_strategies(usable["strategy"].unique()) if name != "gp_ei"]
    if not others:
        base["reason"] = "gp_ei is the only strategy, so there is no paired comparison."
        return base

    tests = []
    for name in others:
        other = usable.loc[usable["strategy"] == name, ["task_id", "seed", "norm_budget"]]
        merged = reference.merge(other, on=["task_id", "seed"], suffixes=("_gp_ei", "_other"))
        n_pairs = int(len(merged))
        if n_pairs < 5:
            tests.append(
                {
                    "strategy": name,
                    "performed": False,
                    "n_pairs": n_pairs,
                    "reason": (
                        "Need at least 5 paired (task, seed) runs against gp_ei; "
                        f"found {n_pairs}. The test was omitted."
                    ),
                }
            )
            continue
        left = merged["norm_budget_other"].to_numpy(dtype=float)
        right = merged["norm_budget_gp_ei"].to_numpy(dtype=float)
        if np.allclose(left, right, equal_nan=True):
            tests.append(
                {
                    "strategy": name,
                    "performed": False,
                    "n_pairs": n_pairs,
                    "reason": (
                        "All paired differences are zero, so the Wilcoxon test is "
                        "undefined and was omitted."
                    ),
                }
            )
            continue
        try:
            result = wilcoxon(left, right, alternative="two-sided", zero_method="wilcox")
        except Exception as exc:
            tests.append(
                {
                    "strategy": name,
                    "performed": False,
                    "n_pairs": n_pairs,
                    "reason": f"The Wilcoxon test could not be computed ({exc}). It was omitted.",
                }
            )
            continue
        if hasattr(result, "pvalue"):
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        else:
            statistic = float(result[0])
            p_value = float(result[1])
        significant = bool(p_value < 0.05)
        tests.append(
            {
                "strategy": name,
                "performed": True,
                "n_pairs": n_pairs,
                "statistic": statistic,
                "p_value": p_value,
                "significant_at_0.05": significant,
                "note": (
                    "The two-sided test met p < 0.05."
                    if significant
                    else "The test did not show a difference at the 0.05 level. Do not claim significance."
                ),
            }
        )

    performed = any(item["performed"] for item in tests)
    reason = None
    if not performed:
        reason = "No paired Wilcoxon test was run. " + " ".join(
            item["reason"] for item in tests if item.get("reason")
        )
    base["performed"] = performed
    base["reason"] = reason
    base["tests"] = tests
    return base


def build_report(df: pd.DataFrame, fixture: bool, source: str) -> dict:
    runs = per_run_frame(df)
    strategies = sort_strategies(df["strategy"].unique()) if not df.empty else []
    overall = {}
    for name in strategies:
        overall[name] = _summarise_block(runs[runs["strategy"] == name])
    by_dataset: dict[str, dict] = {}
    if not runs.empty:
        for dataset in sorted(runs["dataset"].unique()):
            subset = runs[runs["dataset"] == dataset]
            by_dataset[str(dataset)] = {
                name: _summarise_block(subset[subset["strategy"] == name])
                for name in sort_strategies(subset["strategy"].unique())
            }
    if fixture:
        disclaimer = (
            "DRAFT / fixture. These numbers are a schema check, not the hackathon result."
        )
    else:
        disclaimer = (
            "Computed from campaign runs. Claim a difference only when "
            "wilcoxon_vs_gp_ei performed the test and significant_at_0.05 is true."
        )
    return {
        "fixture": bool(fixture),
        "source": source,
        "strategies": strategies,
        "n_bootstrap": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "n_rows": int(len(df)),
        "n_runs": int(len(runs)),
        "definitions": DEFINITIONS,
        "metrics": {"overall": overall, "by_dataset": by_dataset},
        "curves": build_curves(df) if not df.empty else {},
        "wilcoxon_vs_gp_ei": wilcoxon_block(runs),
        "disclaimer": disclaimer,
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def write_table(report: dict, path: Path) -> None:
    rows = []

    def add(scope: str, dataset: str, strategy: str, block: dict) -> None:
        at_10 = block["norm_best_n10"]
        at_budget = block["norm_best_budget"]
        rows.append(
            {
                "scope": scope,
                "dataset": dataset,
                "strategy": strategy,
                "n_runs": block["n_runs"],
                "n_pairs_n10": at_10["n_pairs"],
                "mean_norm_best_n10": at_10["mean"],
                "ci95_low_n10": at_10["ci_low"],
                "ci95_high_n10": at_10["ci_high"],
                "n_pairs_budget": at_budget["n_pairs"],
                "mean_norm_best_budget": at_budget["mean"],
                "ci95_low_budget": at_budget["ci_low"],
                "ci95_high_budget": at_budget["ci_high"],
                "found_hit_rate_budget": block["found_hit_rate"],
                "median_experiments_to_first_hit": block["median_experiments_to_first_hit"],
            }
        )

    for strategy, block in report["metrics"]["overall"].items():
        add("overall", "ALL", strategy, block)
    for dataset, strategies in report["metrics"]["by_dataset"].items():
        for strategy, block in strategies.items():
            add("dataset", dataset, strategy, block)
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise reaction-planner campaign runs.")
    parser.add_argument(
        "--runs",
        default=None,
        help="Path to runs.csv. Default: results/runs.csv if it exists, else the fixture.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Directory for summary.json and summary_table.csv. Default: results.",
    )
    args = parser.parse_args(argv)

    runs_path, fixture = resolve_runs_path(args.runs)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_runs(runs_path)
    report = jsonable(build_report(frame, fixture=fixture, source=display_path(runs_path)))
    json_path = out_dir / "summary.json"
    table_path = out_dir / "summary_table.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_table(report, table_path)

    print(f"Wrote {display_path(json_path)} and {display_path(table_path)}")
    print(
        f"fixture={str(report['fixture']).lower()} "
        f"source={report['source']} "
        f"strategies={', '.join(report['strategies']) or '(none)'}"
    )
    if report["fixture"]:
        print("DRAFT / fixture: do not treat these numbers as the hackathon result.")
    wilcoxon = report["wilcoxon_vs_gp_ei"]
    if not wilcoxon.get("performed"):
        print(f"Wilcoxon omitted: {wilcoxon.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
