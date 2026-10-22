"""Task objects shared by every strategy and by the campaign runner."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TASK_DIR = PROCESSED / "tasks"


@dataclass
class Task:
    id: str                    # "BH_t1" ... "BH_t5", "DA", "SM_t1" ..., "MOCK"
    dataset: str               # "BH" | "DA" | "SM" | "MOCK"
    reaction_text: str         # one line describing the reaction and fixed substrate, for the LLM
    var_cols: list[str]        # e.g. ["ligand", "base", "additive"]
    conditions: pd.DataFrame   # one row per measured condition; columns = var_cols; values = labels like "L2"
    X: np.ndarray              # one-hot features, row-aligned with conditions
    options: dict[str, list[dict]]  # var -> [{"label": "L1", "name": "XPhos", "smiles": "..."}]
    hit_threshold: float       # yield at or above this is a hit
    max_yield: float
    budget: int
    y: np.ndarray | None       # yields, row-aligned; ONLY the campaign runner reads this

    def public(self) -> "Task":
        return replace(self, y=None)


def get_task(task_id: str) -> Task:
    """Return the MOCK screen, or a Task pickled at data/processed/tasks/<id>.pkl.

    The id "MOCK" is case-sensitive and does not need an entry in tasks.json.
    """
    if task_id == "MOCK":
        from src.mock import build_mock_task

        return build_mock_task()

    path = TASK_DIR / f"{task_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved task {task_id!r} at {path}. get_task('MOCK') does not need a pickle."
        )
    with path.open("rb") as handle:
        task = pickle.load(handle)
    if not isinstance(task, Task):
        raise TypeError(f"{path} did not contain a Task (got {type(task).__name__}).")
    return task


def list_tasks(datasets: list[str]) -> list[str]:
    """Task ids for these datasets, from data/processed/tasks.json.

    Returns [] when the index is missing. MOCK stays available through get_task
    even when the index does not mention it.
    """
    path = PROCESSED / "tasks.json"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        data = data["tasks"]

    wanted = set(datasets)
    found: list[str] = []

    def add(task_id: object) -> None:
        text = str(task_id)
        if text and text not in found:
            found.append(text)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                dataset = item.get("dataset")
                task_id = item.get("id", item.get("task_id"))
                if dataset in wanted and task_id:
                    add(task_id)
            elif isinstance(item, str) and _id_matches_dataset(item, wanted):
                add(item)
    elif isinstance(data, dict):
        list_values = bool(data) and all(isinstance(value, list) for value in data.values())
        if list_values:
            for dataset, task_ids in data.items():
                if dataset in wanted:
                    for task_id in task_ids:
                        add(task_id)
        else:
            for key, meta in data.items():
                if isinstance(meta, dict):
                    dataset = meta.get("dataset")
                    task_id = meta.get("id", meta.get("task_id", key))
                    if dataset in wanted:
                        add(task_id)
                elif meta in wanted:
                    add(key)
    return found


def _id_matches_dataset(task_id: str, datasets: set[str]) -> bool:
    if task_id in datasets:
        return True
    return any(task_id.startswith(dataset + "_") for dataset in datasets)
