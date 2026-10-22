"""Fixed 120-row screen used before the real reaction tasks exist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.tasks import Task

# 4 ligands x 3 bases x 10 additives.
LIGANDS = ["L1", "L2", "L3", "L4"]
BASES = ["B1", "B2", "B3"]
ADDITIVES = [f"A{i}" for i in range(1, 11)]
VAR_COLS = ["ligand", "base", "additive"]
REACTION_TEXT = (
    "Mock Pd-catalysed Buchwald-Hartwig amination of bromobenzene with aniline; "
    "choose the ligand, base and additive."
)


def build_mock_task() -> Task:
    """Build the MOCK task. Yields come from a fixed RNG and do not change between calls."""
    rows: list[dict[str, str]] = []
    yields: list[float] = []
    rng = np.random.default_rng(0)

    # Yield model. Effects are deterministic functions of the option index.
    # Noise is one Normal(0, 2) draw per row from np.random.default_rng(0),
    # taken in this loop order (ligand, then base, then additive):
    #
    #   ligand_effect(i)   = (i - 1.5) / 3.0        # i = 0..3
    #   base_effect(j)     = (j - 1.0) / 2.0        # j = 0..2
    #   additive_effect(k) = (k - 4.5) / 9.0        # k = 0..9
    #   interaction(i, j)  = ligand_effect(i) * base_effect(j)
    #   yield = clip(
    #       50
    #       + 20 * ligand_effect
    #       + 15 * base_effect
    #       + 10 * additive_effect
    #       + 10 * interaction
    #       + noise,
    #       0,
    #       100,
    #   )
    for i, ligand in enumerate(LIGANDS):
        for j, base in enumerate(BASES):
            for k, additive in enumerate(ADDITIVES):
                rows.append({"ligand": ligand, "base": base, "additive": additive})
                ligand_effect = (i - 1.5) / 3.0
                base_effect = (j - 1.0) / 2.0
                additive_effect = (k - 4.5) / 9.0
                interaction = ligand_effect * base_effect
                noise = float(rng.normal(0.0, 2.0))
                raw = (
                    50.0
                    + 20.0 * ligand_effect
                    + 15.0 * base_effect
                    + 10.0 * additive_effect
                    + 10.0 * interaction
                    + noise
                )
                yields.append(float(np.clip(raw, 0.0, 100.0)))

    conditions = pd.DataFrame(rows, columns=VAR_COLS)
    y = np.asarray(yields, dtype=np.float64)
    if len(conditions) != 120 or y.shape != (120,):
        raise RuntimeError(f"MOCK screen should have 120 rows, got {len(conditions)}.")

    x_ligand = _one_hot(conditions["ligand"], LIGANDS)
    x_base = _one_hot(conditions["base"], BASES)
    x_additive = _one_hot(conditions["additive"], ADDITIVES)
    features = np.concatenate([x_ligand, x_base, x_additive], axis=1)
    if features.shape != (120, 17) or not np.allclose(features.sum(axis=1), 3.0):
        raise RuntimeError("MOCK one-hot features are not row-aligned with conditions.")

    options = {
        "ligand": [_option(label, f"Ligand {_letter(i)}") for i, label in enumerate(LIGANDS)],
        "base": [_option(label, f"Base {_letter(i)}") for i, label in enumerate(BASES)],
        "additive": [_option(label, f"Additive {_letter(i)}") for i, label in enumerate(ADDITIVES)],
    }
    return Task(
        id="MOCK",
        dataset="MOCK",
        reaction_text=REACTION_TEXT,
        var_cols=list(VAR_COLS),
        conditions=conditions,
        X=features,
        options=options,
        hit_threshold=float(np.sort(y)[-3]),
        max_yield=float(np.max(y)),
        budget=20,
        y=y,
    )


def _option(label: str, name: str) -> dict[str, str]:
    return {"label": label, "name": name, "smiles": "C"}


def _letter(index: int) -> str:
    return chr(ord("A") + index)


def _one_hot(values: pd.Series, levels: list[str]) -> np.ndarray:
    positions = {label: i for i, label in enumerate(levels)}
    index = values.map(positions)
    if index.isna().any():
        missing = sorted(set(values) - set(levels))
        raise RuntimeError(f"MOCK condition labels missing from levels: {missing}")
    return np.eye(len(levels), dtype=np.float64)[index.to_numpy(dtype=int)]
