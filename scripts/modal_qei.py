"""Run gp_qei on Modal and write campaign CSVs under results/runs/.

One container per (task, seed). The remote loop matches src/campaign.py:
shared 5-row initial design, batches of q=5, separate strategy RNG, and the
same columns. Fantasy yields never leave propose().
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "gp-qei"
REMOTE_ROOT = "/app"
TASK_IDS = ["BH_t1", "BH_t2", "BH_t3", "BH_t4", "BH_t5", "DA"]
SEEDS = list(range(10))
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
TOKEN_NAMES = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
ENV_PATHS = (Path(r"d:\AIS and Web3\OsiHack\.env"), ROOT / ".env")

def _ignore_upload(path: Path) -> bool:
    name = path.name
    return name in {".env", ".venv", "venv", "__pycache__", "data"} or name.endswith(
        (".pyc", ".pyo")
    )


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "numpy==2.5.4",
        "pandas==3.1.0",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "pyyaml==6.0.3",
    )
    .add_local_dir(ROOT / "src", remote_path=f"{REMOTE_ROOT}/src", ignore=_ignore_upload)
    .add_local_file(ROOT / "config.yaml", remote_path=f"{REMOTE_ROOT}/config.yaml")
    .add_local_dir(
        ROOT / "data" / "processed" / "tasks",
        remote_path=f"{REMOTE_ROOT}/data/processed/tasks",
        ignore=_ignore_upload,
    )
)

app = modal.App(APP_NAME)


def _load_token_env() -> None:
    """Copy MODAL_TOKEN_ID / MODAL_TOKEN_SECRET into the process if a file has them."""
    for path in ENV_PATHS:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip().removeprefix("export").strip()
            value = value.strip().strip('"').strip("'")
            if key in TOKEN_NAMES and value and not os.environ.get(key):
                os.environ[key] = value


def _assert_modal_auth() -> str:
    proc = subprocess.run(
        ["modal", "profile", "current"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = []
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if text and "token" not in text.lower() and "secret" not in text.lower():
            lines.append(text)
    if proc.returncode != 0 or not lines:
        sys.stderr.write(
            "MODAL_AUTH_MISSING\n"
            "modal token set --token-id ... --token-secret ...\n"
            "Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.\n"
        )
        raise SystemExit(1)
    return lines[-1]


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "token" in lowered or "secret" in lowered or "ak-" in lowered:
        return f"{type(exc).__name__}: details redacted"
    return text[:500]


def _write_csv(rows: list[dict]) -> Path:
    task_id = str(rows[0]["task_id"])
    seed = int(rows[0]["seed"])
    path = ROOT / "results" / "runs" / f"gp_qei__{task_id}__{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in COLUMNS})
    return path


@app.function(image=image, cpu=2, memory=2048, timeout=1800, max_containers=60)
def run_one(task_id: str, seed: int) -> dict:
    """One campaign job. Returns plain rows; the local process writes the CSV."""
    import sys as _sys

    _sys.path.insert(0, REMOTE_ROOT)

    import numpy as np

    from src.campaign import (
        _budget,
        _select_picks,
        _strategy_rng,
        draw_initial_design,
        load_config,
    )
    from src.strategies.gp_qei import make_strategy
    from src.tasks import get_task

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

    initial = draw_initial_design(task_id, seed, n_rows, n_init)
    strategy = make_strategy()
    name = str(strategy.name)
    one_shot = bool(getattr(strategy, "one_shot", False))
    if name != "gp_qei":
        raise RuntimeError(f"strategy name is {name!r}, expected 'gp_qei'.")
    if not hasattr(strategy, "propose"):
        raise RuntimeError("make_strategy() returned no propose().")

    view = private.public()
    if view.y is not None:
        raise RuntimeError("public() leaked yields.")
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
        public = private.public()
        if public.y is not None:
            raise RuntimeError("public() leaked yields.")
        hist_idx = list(evaluated)
        hist_y = list(eval_y)
        proposed = strategy.propose(
            public,
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
            raise RuntimeError(f"{name} on {task_id} seed {seed} made no progress in round {round_no}.")
        if one_shot:
            break
        round_no += 1

    if len(rows) != budget:
        raise RuntimeError(f"{name} on {task_id} seed {seed} recorded {len(rows)} rows; budget is {budget}.")
    print(f"{name} {private.id} seed={seed}: n={len(rows)} any_hit={any_hit}", flush=True)
    return {"task_id": private.id, "seed": seed, "n": len(rows), "rows": rows}


def main() -> None:
    _load_token_env()
    profile = _assert_modal_auth()
    print(f"Modal profile: {profile}", flush=True)
    print(f"App: {APP_NAME}", flush=True)
    pairs = [(task_id, seed) for task_id in TASK_IDS for seed in SEEDS]
    failures: list[str] = []
    written = 0
    with app.run():
        results = run_one.starmap(pairs, return_exceptions=True, order_outputs=True)
        for (task_id, seed), result in zip(pairs, results):
            if isinstance(result, BaseException):
                failures.append(f"{task_id} seed={seed}: {_safe_error(result)}")
                print(f"FAIL {task_id} seed={seed}: {_safe_error(result)}", flush=True)
                continue
            path = _write_csv(result["rows"])
            written += 1
            print(f"wrote n={result['n']} {path.name}", flush=True)
    print(f"Wrote {written} CSV(s); failures {len(failures)}.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
