"""Merge results/runs/*.csv into results/runs.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"
OUT_PATH = ROOT / "results" / "runs.csv"
KEY = ["strategy", "task_id", "seed", "n"]
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


def main() -> None:
    files = []
    if RUNS_DIR.is_dir():
        files = sorted(
            (path for path in RUNS_DIR.glob("*.csv") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
    if not files:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=COLUMNS).to_csv(OUT_PATH, index=False)
        print(f"No run files in {RUNS_DIR}. Wrote an empty {OUT_PATH}.")
        return

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [column for column in KEY if column not in frame.columns]
        if missing:
            raise SystemExit(f"{path.name} is missing columns {missing}.")
        frame["_source"] = path.name
        frames.append(frame)

    # Files are oldest-first, so keep="last" retains the newest file's rows.
    merged = pd.concat(frames, ignore_index=True)
    duplicate = merged.duplicated(KEY, keep=False)
    if duplicate.any():
        print(
            "WARNING: duplicate (strategy, task_id, seed, n); keeping the last file's rows.",
            file=sys.stderr,
        )
        grouped = merged.loc[duplicate].groupby(KEY, sort=False)["_source"]
        for key, sources in grouped:
            names = list(dict.fromkeys(sources.tolist()))
            print(f"  {key}: {names}", file=sys.stderr)

    merged = merged.drop_duplicates(KEY, keep="last")
    ordered = [column for column in COLUMNS if column in merged.columns]
    extras = [column for column in merged.columns if column not in ordered and column != "_source"]
    merged = merged[ordered + extras]
    merged = merged.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(merged)} rows from {len(files)} files).")


if __name__ == "__main__":
    main()
