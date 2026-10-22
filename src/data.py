"""Historical substrate data for the transfer surrogate.

``get_transfer_data`` stacks ``X`` and ``y`` from the other tasks of the same
dataset and appends a substrate one-hot block. ``pad_with_substrate_block``
appends the same block to a matrix from the current task so the columns match.

The substrate block is not part of ``task.X``. Campaign strategies read
``task.X`` directly; only ``tabpfn_transfer`` uses these two functions.

Substrate order is the sorted list of task ids in that dataset (stable).
With a single task (DA), the transfer matrices have zero rows and width
``X.shape[1] + 1``, and padding still adds that one column of ones.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_TASKS_JSON = _ROOT / "data" / "processed" / "tasks.json"
_TASK_DIR = _ROOT / "data" / "processed" / "tasks"


def _load_catalogue() -> list[dict]:
    if not _TASKS_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_TASKS_JSON}. Build data/processed/tasks.json and the "
            "task pickles before calling get_transfer_data or "
            "pad_with_substrate_block."
        )
    with _TASKS_JSON.open(encoding="utf-8") as handle:
        catalogue = json.load(handle)
    if not isinstance(catalogue, list):
        raise ValueError(
            f"{_TASKS_JSON} must be a list of task records, got {type(catalogue).__name__}."
        )
    return catalogue


def _dataset_task_ids(task_id: str) -> list[str]:
    catalogue = _load_catalogue()
    dataset = None
    for record in catalogue:
        if record.get("id") == task_id:
            dataset = record.get("dataset")
            break
    if dataset is None:
        known = [record.get("id") for record in catalogue]
        raise KeyError(
            f"Task {task_id!r} is not listed in {_TASKS_JSON}. Known ids: {known}."
        )
    ids = sorted(
        record["id"] for record in catalogue if record.get("dataset") == dataset
    )
    if task_id not in ids:
        raise KeyError(f"Task {task_id!r} missing from dataset {dataset!r} ids {ids}.")
    return ids


def _load_task(task_id: str):
    path = _TASK_DIR / f"{task_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing task pickle {path}. It is listed in tasks.json but was not built."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _substrate_block(n_rows: int, n_substrates: int, index: int) -> np.ndarray:
    block = np.zeros((n_rows, n_substrates), dtype=np.float64)
    if n_rows:
        block[:, index] = 1.0
    return block


def get_transfer_data(task_id: str) -> tuple[np.ndarray, np.ndarray]:
    """``X``, ``y`` from the other tasks of this dataset, plus a substrate block.

    Rows are concatenated in sorted task-id order. Each row's substrate
    column is 1 at that other task's index. If this dataset has only one
    task, both arrays have shape ``(0, n_features + 1)`` and ``(0,)``.
    """
    ids = _dataset_task_ids(task_id)
    n_substrates = len(ids)
    others = [tid for tid in ids if tid != task_id]
    if not others:
        current = _load_task(task_id)
        n_features = int(np.asarray(current.X).shape[1])
        return (
            np.zeros((0, n_features + n_substrates), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    width = None
    for tid in others:
        task = _load_task(tid)
        x = np.asarray(task.X, dtype=np.float64)
        y = np.asarray(task.y, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"{tid} X must be 2-D, got shape {x.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"{tid} has {x.shape[0]} feature rows and {y.shape[0]} yields."
            )
        if width is None:
            width = x.shape[1]
        elif x.shape[1] != width:
            raise ValueError(
                f"{tid} X width {x.shape[1]} does not match {width}. "
                "Tasks in one dataset must share the global one-hot order."
            )
        block = _substrate_block(x.shape[0], n_substrates, ids.index(tid))
        x_parts.append(np.concatenate([x, block], axis=1))
        y_parts.append(y)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def pad_with_substrate_block(task_id: str, X) -> np.ndarray:
    """Append this task's substrate one-hot columns so ``X`` matches transfer ``X``."""
    ids = _dataset_task_ids(task_id)
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"pad_with_substrate_block expected a 2-D array, got shape {matrix.shape}."
        )
    block = _substrate_block(matrix.shape[0], len(ids), ids.index(task_id))
    return np.concatenate([matrix, block], axis=1)
