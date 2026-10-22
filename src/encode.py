"""One-hot encoding for reaction-condition tables.

Column order is stable: for each variable in ``var_cols``, columns follow
``options[var]`` in the order given. That order is the global option list
stored in ``data/processed/options_<DATASET>.json``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_hot_column_names(var_cols, options) -> list[str]:
    """Names aligned with the columns of :func:`one_hot`.

    Each name is ``"<variable>=<label>"``, for example ``"ligand=L2"``.
    """
    names: list[str] = []
    for var in var_cols:
        if var not in options:
            raise KeyError(f"options has no entry for variable {var!r}")
        for opt in options[var]:
            label = opt["label"]
            names.append(f"{var}={label}")
    return names


def one_hot(df: pd.DataFrame, var_cols, options) -> np.ndarray:
    """Return a float64 one-hot matrix, one row per row of ``df``.

    Unknown labels raise ``ValueError``. Options that do not appear in
    ``df`` stay as all-zero columns so tasks in one dataset share a width.
    """
    n = len(df)
    blocks: list[np.ndarray] = []
    for var in var_cols:
        if var not in options:
            raise KeyError(f"options has no entry for variable {var!r}")
        if var not in df.columns:
            raise KeyError(f"dataframe has no column {var!r}")
        labels = [opt["label"] for opt in options[var]]
        index = {lab: i for i, lab in enumerate(labels)}
        if len(index) != len(labels):
            raise ValueError(f"Duplicate labels for variable {var!r}: {labels}")
        codes = np.empty(n, dtype=np.int64)
        for row_i, value in enumerate(df[var].tolist()):
            try:
                codes[row_i] = index[value]
            except KeyError:
                raise ValueError(
                    f"Unknown label {value!r} for variable {var!r}. "
                    f"Known labels: {labels}"
                ) from None
        block = np.zeros((n, len(labels)), dtype=np.float64)
        if n:
            block[np.arange(n), codes] = 1.0
        blocks.append(block)
    if not blocks:
        return np.zeros((n, 0), dtype=np.float64)
    return np.concatenate(blocks, axis=1)
