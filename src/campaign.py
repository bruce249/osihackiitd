"""Campaign runner. Strategies are loaded by import path; this file does not name them.

Example:
    python -m src.campaign --strategy random --module src.strategies.random_s --tasks MOCK --seeds 2
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks import get_task, list_tasks

COLUMNS = [
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


def load_config() -> dict:
    path = ROOT / "config.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = ["q", "initial_design", "budgets", "seeds", "hit_rule", "model"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"config.yaml is missing keys: {missing}")
    for dataset in ("BH", "SM", "DA", "MOCK"):
        if dataset not in config["budgets"]:
            raise SystemExit(f"config.yaml budgets is missing {dataset}.")
    for arm in ("statistical", "llm"):
        if arm not in config["seeds"]:
            raise SystemExit(f"config.yaml seeds is missing {arm}.")
    return config


def draw_initial_design(task_id: str, seed: int, n_rows: int, n_init: int) -> list[int]:
    """Shared starting rows for every strategy on this (task_id, seed).

    Drawn before any strategy code runs. The generator is local and is not
    passed into the strategy.
    """
    rng = np.random.default_rng((zlib.crc32(task_id.encode()) + int(seed)) % (2**32))
    picked = rng.choice(int(n_rows), size=int(n_init), replace=False)
    return [int(i) for i in np.asarray(picked).ravel()]


def _strategy_rng(task_id: str, strategy_name: str, seed: int) -> np.random.Generator:
    text = task_id + ":" + strategy_name
    return np.random.default_rng((zlib.crc32(text.encode()) + int(seed)) % (2**32))


def _budget(task, config: dict) -> int:
    if task.budget is not None:
        return int(task.budget)
    return int(config["budgets"][task.dataset])


def _as_list(proposed) -> list:
    if proposed is None:
        return []
    if isinstance(proposed, (int, np.integer)):
        return [int(proposed)]
    if isinstance(proposed, np.ndarray):
        return [int(i) for i in proposed.ravel().tolist()]
    return list(proposed)


def _select_picks(proposed, pool_idx, q_call: int, rng) -> list[tuple[int, bool]]:
    """Keep valid unique pool indices, then fill any shortfall at random."""
    pool = [int(i) for i in np.asarray(pool_idx, dtype=int).ravel()]
    pool_set = set(pool)
    chosen: list[int] = []
    seen: set[int] = set()
    for raw in _as_list(proposed):
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx in pool_set and idx not in seen:
            chosen.append(idx)
            seen.add(idx)
        if len(chosen) >= q_call:
            break
    flags = [False] * len(chosen)
    need = q_call - len(chosen)
    if need > 0:
        remain = np.asarray([i for i in pool if i not in seen], dtype=int)
        take = min(need, int(remain.size))
        if take > 0:
            draw = rng.choice(remain, size=take, replace=False)
            for idx in np.asarray(draw, dtype=int).ravel():
                chosen.append(int(idx))
                flags.append(True)
    return list(zip(chosen, flags))


def _resolve_task_ids(tasks: str | None, task_ids: str | None) -> list[str]:
    if task_ids:
        ids = [part.strip() for part in task_ids.split(",") if part.strip()]
        if not ids:
            raise SystemExit("--task-ids was empty.")
        return ids
    if not tasks:
        raise SystemExit("Pass --tasks, --task-ids, or both.")
    datasets = [part.strip() for part in tasks.split(",") if part.strip()]
    ids = list_tasks(datasets)
    # MOCK is runnable even when data/processed/tasks.json does not list it.
    if "MOCK" in datasets and "MOCK" not in ids:
        ids.append("MOCK")
    if not ids:
        raise SystemExit(
            "No tasks matched. Pass --tasks MOCK or --task-ids MOCK to run the mock screen."
        )
    return ids


def _run_job(job: tuple[str, int, str]) -> dict:
    task_id, seed, module_path = job
    seed = int(seed)
    config = load_config()
    private = get_task(task_id)
    if private.y is None:
        raise RuntimeError(f"Task {task_id} has no yields, so the runner cannot score it.")

    n_rows = int(len(private.conditions))
    budget = _budget(private, config)
    q = int(config["q"])
    n_init = int(config["initial_design"])
    if n_init < 1:
        raise RuntimeError("initial_design in config.yaml must be at least 1.")
    if budget < n_init:
        raise RuntimeError(f"Task {task_id} budget {budget} is smaller than the initial design {n_init}.")
    if n_rows < budget:
        raise RuntimeError(f"Task {task_id} has {n_rows} rows, fewer than budget {budget}.")

    # Draw the shared design first. Do this before importing the strategy.
    initial = draw_initial_design(task_id, seed, n_rows, n_init)

    module = importlib.import_module(module_path)
    if not hasattr(module, "make_strategy"):
        raise RuntimeError(f"{module_path} has no make_strategy().")
    strategy = module.make_strategy()
    name = str(strategy.name)
    one_shot = bool(getattr(strategy, "one_shot", False))
    if not hasattr(strategy, "propose"):
        raise RuntimeError(f"{module_path}.make_strategy() returned no propose().")

    public = private.public()
    assert public.y is None
    strategy_rng = _strategy_rng(task_id, name, seed)

    y = np.asarray(private.y, dtype=float).ravel()
    threshold = float(private.hit_threshold)
    max_yield = float(private.max_yield)
    rows: list[dict] = []
    evaluated: list[int] = []
    eval_y: list[float] = []
    best: float | None = None
    any_hit = False

    def append_pick(idx: int, round_no: int, filled: bool) -> None:
        nonlocal best, any_hit
        yld = float(y[int(idx)])
        best = yld if best is None else max(best, yld)
        is_hit = bool(yld >= threshold)
        any_hit = any_hit or is_hit
        norm = (best / max_yield) if max_yield != 0 else 0.0
        rows.append(
            {
                "strategy": name,
                "dataset": private.dataset,
                "task_id": private.id,
                "seed": seed,
                "n": len(rows) + 1,
                "round": int(round_no),
                "pick_idx": int(idx),
                "yield": yld,
                "best_so_far": float(best),
                "norm_best": float(norm),
                "is_hit": is_hit,
                "any_hit": bool(any_hit),
                "filled_random": bool(filled),
            }
        )

    for idx in initial:
        append_pick(idx, 0, False)
        evaluated.append(int(idx))
        eval_y.append(float(y[int(idx)]))

    round_no = 1
    while len(evaluated) < budget:
        evaluated_set = set(evaluated)
        pool_idx = np.asarray([i for i in range(n_rows) if i not in evaluated_set], dtype=int)
        if pool_idx.size == 0:
            break
        remaining = budget - len(evaluated)
        q_call = remaining if one_shot else min(q, remaining)
        q_call = min(int(q_call), int(pool_idx.size))
        if q_call < 1:
            break

        view = private.public()
        assert view.y is None
        hist_idx = list(evaluated)
        hist_y = list(eval_y)
        assert len(hist_idx) == len(hist_y)
        proposed = strategy.propose(
            view,
            pool_idx,
            hist_idx,
            hist_y,
            q_call,
            strategy_rng,
            seed,
            round_no,
        )
        picks = _select_picks(proposed, pool_idx, q_call, strategy_rng)
        before = len(evaluated)
        for idx, filled in picks:
            if len(evaluated) >= budget:
                break
            append_pick(idx, round_no, filled)
            evaluated.append(int(idx))
            eval_y.append(float(y[int(idx)]))
        if len(evaluated) == before:
            raise RuntimeError(
                f"{name} on {task_id} seed {seed} made no progress in round {round_no}."
            )
        if one_shot:
            break
        round_no += 1

    if len(rows) != budget:
        raise RuntimeError(
            f"{name} on {task_id} seed {seed} recorded {len(rows)} rows; budget is {budget}."
        )

    out_dir = ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = name.replace("\\", "_").replace("/", "_")
    path = out_dir / f"{safe}__{private.id}__{seed}.csv"
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(path, index=False)
    last = rows[-1]
    return {
        "strategy": name,
        "dataset": private.dataset,
        "task_id": private.id,
        "seed": seed,
        "n": len(rows),
        "norm_best": float(last["norm_best"]),
        "any_hit": bool(last["any_hit"]),
        "path": str(path),
    }


def _dispatch(jobs: list[tuple[str, int, str]], workers: int, use_threads: bool) -> list[dict]:
    if not jobs:
        return []
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as executor:
        return list(executor.map(_run_job, jobs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one strategy across tasks and seeds.")
    parser.add_argument("--strategy", required=True, help="Name recorded if it matches the module.")
    parser.add_argument("--module", required=True, help="Import path, e.g. src.strategies.random_s")
    parser.add_argument("--tasks", default=None, help="Comma-separated dataset names, e.g. BH or MOCK")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids. Used instead of --tasks.")
    parser.add_argument("--seeds", type=int, required=True, help="Number of seeds. Runs use 0 .. N-1.")
    parser.add_argument("--workers", type=int, default=1, help="Pool size. Default 1.")
    parser.add_argument(
        "--threads",
        action="store_true",
        help="Use a thread pool. Otherwise use a process pool.",
    )
    args = parser.parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    config = load_config()
    print(
        f"Config q={config['q']} initial_design={config['initial_design']} "
        f"model={config['model']}"
    )
    preview = importlib.import_module(args.module).make_strategy()
    if str(preview.name) != args.strategy:
        print(
            f"NOTE: --strategy {args.strategy!r} differs from module name {preview.name!r}. "
            f"Runs are recorded as {preview.name!r}."
        )

    task_ids = _resolve_task_ids(args.tasks, args.task_ids)
    jobs = [(task_id, seed, args.module) for task_id in task_ids for seed in range(args.seeds)]
    (ROOT / "results" / "runs").mkdir(parents=True, exist_ok=True)
    summaries = _dispatch(jobs, args.workers, args.threads)
    for item in summaries:
        print(
            f"{item['strategy']} {item['task_id']} seed={item['seed']}: "
            f"n={item['n']} norm_best={item['norm_best']:.4f} any_hit={item['any_hit']} "
            f"{item['path']}"
        )
    print(f"Finished {len(summaries)} run(s).")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
