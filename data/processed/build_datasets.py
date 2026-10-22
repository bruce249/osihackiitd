"""Build processed reaction tables and Task pickles.

Cleaning section
----------------
Buchwald-Hartwig source:
https://github.com/rxn4chemistry/rxn_yields
raw file data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx
downloaded from
https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx

Only sheet FullCV_01 is used. rxn_yields loads that sheet
(``pd.read_excel(..., sheet_name='FullCV_01')``). The other sheets are
reshuffles of the same screen, not extra reactions.

Yield scale: the Output column is already a percentage on 0-100
(observed minimum 0, maximum just under 100). It is NOT multiplied by 100.
Fractions on 0-1 would have been multiplied by 100; this file is not on
that scale.

The amine is 4-methylaniline. Ahneman et al., Science 2018, name it
4-methylaniline, and rxn_yields hard-codes ``methylaniline = 'Cc1ccc(N)cc1'``
(p-toluidine) in ``generate_buchwald_hartwig_rxns``.

Direct arylation source:
https://github.com/b-shields/edbo
experiments/data/direct_arylation/experiment_index.csv
plus ligand-list.csv, base-list.csv and solvent-list.csv.
Yields in that file are already on a 0-100 scale.

Suzuki-Miyaura is not built here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.encode import one_hot, one_hot_column_names  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TASK_DIR = PROCESSED / "tasks"

BH_XLSX = RAW / "Dreher_and_Doyle_input_data.xlsx"
BH_URL = (
    "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
    "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
)
DA_INDEX = RAW / "direct_arylation_experiment_index.csv"
DA_LIGANDS = RAW / "direct_arylation_ligand-list.csv"
DA_BASES = RAW / "direct_arylation_base-list.csv"
DA_SOLVENTS = RAW / "direct_arylation_solvent-list.csv"
DA_URL = (
    "https://raw.githubusercontent.com/b-shields/edbo/master/"
    "experiments/data/direct_arylation/experiment_index.csv"
)

# Names checked against the SMILES in FullCV_01. Labels are assigned later
# by sorting these raw strings, not by this dictionary's order.
BH_LIGAND_NAMES = {
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "tBuBrettPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "tBuXPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC": "AdBrettPhos",
}
BH_BASE_NAMES = {
    "CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG",
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et",
    "CN1CCCN2C1=NCCC2": "MTBD",
}
BH_ADDITIVE_NAMES = {
    "C1(C2=CC=CC=C2)=CC=NO1": "5-phenylisoxazole",
    "C1(C2=CC=CC=C2)=CON=C1": "4-phenylisoxazole",
    "C1(C2=CC=CC=C2)=NOC=C1": "3-phenylisoxazole",
    "C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=CC=NO1": "5-(dibenzylamino)isoxazole",
    "C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=NOC=C1": "3-(dibenzylamino)isoxazole",
    "C12=C(C=CC=C2)ON=C1": "1,2-benzisoxazole",
    "C12=CON=C1C=CC=C2": "2,1-benzisoxazole",
    "CC1=C(C(OCC)=O)C=NO1": "ethyl 5-methylisoxazole-4-carboxylate",
    "CC1=CC(C(OCC)=O)=NO1": "ethyl 5-methylisoxazole-3-carboxylate",
    "CC1=CC(C)=NO1": "3,5-dimethylisoxazole",
    "CC1=CC(N2C=CC=C2)=NO1": "5-methyl-3-(pyrrol-1-yl)isoxazole",
    "CC1=CC=NO1": "5-methylisoxazole",
    "CC1=NOC(C(OCC)=O)=C1": "ethyl 3-methylisoxazole-5-carboxylate",
    "CC1=NOC(C2=CC=CC=C2)=C1": "3-methyl-5-phenylisoxazole",
    "CC1=NOC=C1": "3-methylisoxazole",
    "CCOC(C1=CON=C1)=O": "ethyl isoxazole-4-carboxylate",
    "CCOC(C1=NOC=C1)=O": "ethyl isoxazole-3-carboxylate",
    "COC1=NOC(C(OCC)=O)=C1": "ethyl 3-methoxyisoxazole-5-carboxylate",
    "FC(C=CC=C1F)=C1C2=CC=NO2": "5-(2,6-difluorophenyl)isoxazole",
    "O=C(OC)C1=CC=NO1": "methyl isoxazole-5-carboxylate",
    "O=C(OC)C1=NOC(C2=CC=CO2)=C1": "methyl 5-(furan-2-yl)isoxazole-3-carboxylate",
    "O=C(OC)C1=NOC(C2=CC=CS2)=C1": "methyl 5-(thiophen-2-yl)isoxazole-3-carboxylate",
}
BH_ARYL_NAMES = {
    "BrC1=CC=C(C(F)(F)F)C=C1": ("1-bromo-4-(trifluoromethyl)benzene", "Br"),
    "BrC1=CC=C(CC)C=C1": ("1-bromo-4-ethylbenzene", "Br"),
    "BrC1=CC=C(OC)C=C1": ("4-bromoanisole", "Br"),
    "BrC1=CN=CC=C1": ("3-bromopyridine", "Br"),
    "BrC1=NC=CC=C1": ("2-bromopyridine", "Br"),
    "ClC1=CC=C(C(F)(F)F)C=C1": ("1-chloro-4-(trifluoromethyl)benzene", "Cl"),
    "ClC1=CC=C(CC)C=C1": ("1-chloro-4-ethylbenzene", "Cl"),
    "ClC1=CC=C(OC)C=C1": ("4-chloroanisole", "Cl"),
    "ClC1=CN=CC=C1": ("3-chloropyridine", "Cl"),
    "ClC1=NC=CC=C1": ("2-chloropyridine", "Cl"),
    "IC1=CC=C(C(F)(F)F)C=C1": ("1-iodo-4-(trifluoromethyl)benzene", "I"),
    "IC1=CC=C(CC)C=C1": ("1-ethyl-4-iodobenzene", "I"),
    "IC1=CC=C(OC)C=C1": ("4-iodoanisole", "I"),
    "IC1=CN=CC=C1": ("3-iodopyridine", "I"),
    "IC1=NC=CC=C1": ("2-iodopyridine", "I"),
}

# Chosen to span maximum yield and to include Cl, Br and I.
# Ordered by increasing maximum yield. 1-chloro-4-ethylbenzene is excluded
# by the max-yield < 30% rule and is not in this list.
BH_TASK_SMILES = [
    "ClC1=CC=C(OC)C=C1",
    "BrC1=CC=C(C(F)(F)F)C=C1",
    "ClC1=CN=CC=C1",
    "IC1=CC=C(CC)C=C1",
    "IC1=NC=CC=C1",
]

BH_BASE_READABLE = {
    "BTMG": "BTMG",
    "P2Et": "P2Et",
    "MTBD": "MTBD",
}
DA_BASE_READABLE = {
    "KOAc": "potassium acetate",
    "KOPiv": "potassium pivalate",
    "CsOAc": "cesium acetate",
    "CsOPiv": "cesium pivalate",
}
DA_SOLVENT_READABLE = {
    "BuOAc": "n-butyl acetate",
    "p-Xylene": "p-xylene",
    "BuCN": "butyronitrile",
    "DMAc": "N,N-dimethylacetamide",
}


def hit_threshold(y: np.ndarray) -> tuple[float, int]:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(values.size)
    k = max(3, int(math.ceil(0.01 * n)))
    ordered = np.sort(values)[::-1]
    return float(ordered[k - 1]), k


def _require_names(kind: str, raw_values, names: dict) -> None:
    missing = [value for value in raw_values if value not in names]
    extra = [value for value in names if value not in set(raw_values)]
    if missing or extra:
        raise SystemExit(
            f"{kind} name table does not match the file.\n"
            f"missing ({len(missing)}): {missing}\n"
            f"extra ({len(extra)}): {extra}"
        )


def assign_options(raw_values, prefix: str, name_of, smiles_of) -> list[dict]:
    ordered = sorted(raw_values, key=lambda value: (str(type(value)), str(value)))
    # Numeric strings and numbers: caller passes already-sorted values when
    # lexicographic order would be wrong. This helper sorts strings.
    options = []
    for i, raw in enumerate(ordered, start=1):
        options.append(
            {
                "label": f"{prefix}{i}",
                "name": name_of(raw),
                "smiles": smiles_of(raw),
            }
        )
    return options


def assign_options_sorted(pairs, prefix: str, name_of, smiles_of) -> list[dict]:
    """``pairs`` is an iterable of raw values already in the desired order."""
    options = []
    for i, raw in enumerate(pairs, start=1):
        options.append(
            {
                "label": f"{prefix}{i}",
                "name": name_of(raw),
                "smiles": smiles_of(raw),
            }
        )
    return options


def load_bh() -> tuple[pd.DataFrame, dict]:
    # See module docstring: FullCV_01, yields already 0-100.
    df = pd.read_excel(BH_XLSX, sheet_name="FullCV_01")
    expected = ["Ligand", "Additive", "Base", "Aryl halide", "Output"]
    if list(df.columns) != expected:
        raise SystemExit(f"Unexpected BH columns: {list(df.columns)}")
    df = df.rename(
        columns={
            "Ligand": "ligand",
            "Additive": "additive",
            "Base": "base",
            "Aryl halide": "aryl_halide",
            "Output": "yield",
        }
    )
    df["yield"] = pd.to_numeric(df["yield"], errors="raise")
    ymax = float(df["yield"].max())
    ymin = float(df["yield"].min())
    scaled = False
    if ymax <= 1.5 and ymin >= -1e-9:
        df["yield"] = df["yield"] * 100.0
        scaled = True
    grouped = (
        df.groupby(["aryl_halide", "ligand", "base", "additive"], as_index=False)["yield"]
        .mean()
    )
    n_dup_groups = int(len(df) - len(grouped))
    _require_names("ligand", grouped["ligand"].unique(), BH_LIGAND_NAMES)
    _require_names("base", grouped["base"].unique(), BH_BASE_NAMES)
    _require_names("additive", grouped["additive"].unique(), BH_ADDITIVE_NAMES)
    _require_names("aryl_halide", grouped["aryl_halide"].unique(), BH_ARYL_NAMES)
    meta = {
        "source_url": BH_URL,
        "sheet": "FullCV_01",
        "rows_before_average": int(len(df)),
        "rows_after_average": int(len(grouped)),
        "duplicate_rows_collapsed": n_dup_groups,
        "yield_min": float(grouped["yield"].min()),
        "yield_max": float(grouped["yield"].max()),
        "yield_multiplied_by_100": scaled,
        "n_aryl": int(grouped["aryl_halide"].nunique()),
        "n_ligand": int(grouped["ligand"].nunique()),
        "n_base": int(grouped["base"].nunique()),
        "n_additive": int(grouped["additive"].nunique()),
    }
    return grouped, meta


def load_da() -> tuple[pd.DataFrame, dict]:
    index = pd.read_csv(DA_INDEX)
    ligands = pd.read_csv(DA_LIGANDS)
    bases = pd.read_csv(DA_BASES)
    solvents = pd.read_csv(DA_SOLVENTS)
    lig_map = dict(zip(ligands["Ligand_SMILES"], ligands["Ligand"]))
    base_map = dict(zip(bases["Base_SMILES"], bases["Base"]))
    solv_map = dict(zip(solvents["Solvent_SMILES"], solvents["Solvent"]))
    missing_l = sorted(set(index["Ligand_SMILES"]) - set(lig_map))
    missing_b = sorted(set(index["Base_SMILES"]) - set(base_map))
    missing_s = sorted(set(index["Solvent_SMILES"]) - set(solv_map))
    if missing_l or missing_b or missing_s:
        raise SystemExit(
            f"DA name lists do not cover the index. "
            f"ligands {missing_l}, bases {missing_b}, solvents {missing_s}"
        )
    df = pd.DataFrame(
        {
            "base": index["Base_SMILES"].astype(str),
            "ligand": index["Ligand_SMILES"].astype(str),
            "solvent": index["Solvent_SMILES"].astype(str),
            "concentration": index["Concentration"].map(lambda v: _num_token(v)),
            "temperature": index["Temp_C"].map(lambda v: _num_token(v)),
            "yield": pd.to_numeric(index["yield"], errors="raise"),
        }
    )
    ymax = float(df["yield"].max())
    ymin = float(df["yield"].min())
    scaled = False
    if ymax <= 1.5 and ymin >= -1e-9:
        df["yield"] = df["yield"] * 100.0
        scaled = True
    before = len(df)
    df = (
        df.groupby(
            ["base", "ligand", "solvent", "concentration", "temperature"],
            as_index=False,
        )["yield"].mean()
    )
    meta = {
        "source_url": DA_URL,
        "rows_before_average": int(before),
        "rows_after_average": int(len(df)),
        "duplicate_rows_collapsed": int(before - len(df)),
        "yield_min": float(df["yield"].min()),
        "yield_max": float(df["yield"].max()),
        "yield_multiplied_by_100": scaled,
        "ligand_names": lig_map,
        "base_codes": base_map,
        "solvent_codes": solv_map,
    }
    return df, meta


def _num_token(value) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    text = f"{number:.6f}".rstrip("0").rstrip(".")
    return text


def options_from_raw(raw_values, prefix, names, as_smiles=True) -> tuple[list[dict], dict]:
    ordered = sorted(set(raw_values), key=lambda value: str(value))
    options = []
    lookup = {}
    for i, raw in enumerate(ordered, start=1):
        label = f"{prefix}{i}"
        record = {
            "label": label,
            "name": names[raw],
            "smiles": raw if as_smiles else "",
        }
        options.append(record)
        lookup[raw] = label
    return options, lookup


def numeric_options(tokens, prefix, name_fmt) -> tuple[list[dict], dict]:
    ordered = sorted(set(tokens), key=lambda value: float(value))
    options = []
    lookup = {}
    for i, raw in enumerate(ordered, start=1):
        label = f"{prefix}{i}"
        record = {"label": label, "name": name_fmt(raw), "smiles": ""}
        options.append(record)
        lookup[raw] = label
    return options, lookup


def map_labels(df: pd.DataFrame, columns: dict[str, dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column, lookup in columns.items():
        out[column] = df[column].map(lookup)
        if out[column].isna().any():
            bad = df.loc[out[column].isna(), column].unique().tolist()
            raise SystemExit(f"Unmapped values in {column}: {bad}")
    return out


def summarise_aryl(bh: pd.DataFrame) -> list[dict]:
    rows = []
    for smiles, sub in bh.groupby("aryl_halide"):
        name, halide = BH_ARYL_NAMES[smiles]
        rows.append(
            {
                "smiles": smiles,
                "name": name,
                "halide": halide,
                "n": int(len(sub)),
                "max_yield": float(sub["yield"].max()),
                "mean_yield": float(sub["yield"].mean()),
            }
        )
    rows.sort(key=lambda row: (row["max_yield"], row["name"]))
    return rows


def build_condition_table(labeled: pd.DataFrame, yields: pd.Series, var_cols: list[str]):
    table = labeled.loc[:, var_cols].copy()
    table["yield"] = yields.to_numpy()
    table = table.sort_values(var_cols, kind="mergesort").reset_index(drop=True)
    if table.duplicated(var_cols).any():
        raise SystemExit(f"Duplicate condition rows remain for {var_cols}")
    conditions = table.loc[:, var_cols].copy()
    y = table["yield"].to_numpy(dtype=np.float64)
    return conditions, y


def validate_matrix(conditions, y, options, var_cols, dataset_name: str) -> np.ndarray:
    if not np.isfinite(y).all():
        raise SystemExit(f"{dataset_name} has non-finite yields")
    if y.min() < -1e-6 or y.max() > 100 + 1e-6:
        raise SystemExit(
            f"{dataset_name} yields outside 0-100: {y.min()} .. {y.max()}"
        )
    x = one_hot(conditions, var_cols, options)
    names = one_hot_column_names(var_cols, options)
    if x.shape != (len(conditions), len(names)):
        raise SystemExit(f"{dataset_name} one-hot shape {x.shape} != {(len(conditions), len(names))}")
    if not np.allclose(x.sum(axis=1), len(var_cols)):
        raise SystemExit(f"{dataset_name} one-hot rows do not sum to {len(var_cols)}")
    return x


def name_rows(dataset: str, variable: str, options: list[dict]) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "variable": variable,
            "label": opt["label"],
            "name": opt["name"],
            "smiles": opt["smiles"],
        }
        for opt in options
    ]


def try_import_task():
    try:
        from src.tasks import Task
    except Exception as exc:
        print(f"Task import unavailable ({type(exc).__name__}: {exc}). Pickles skipped.")
        return None
    return Task


def main() -> None:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    bh, bh_meta = load_bh()
    da, da_meta = load_da()

    bh_lig_opts, bh_lig_map = options_from_raw(
        bh["ligand"].unique(), "L", BH_LIGAND_NAMES, as_smiles=True
    )
    bh_base_opts, bh_base_map = options_from_raw(
        bh["base"].unique(), "B", BH_BASE_NAMES, as_smiles=True
    )
    bh_add_opts, bh_add_map = options_from_raw(
        bh["additive"].unique(), "A", BH_ADDITIVE_NAMES, as_smiles=True
    )
    bh_options = {"ligand": bh_lig_opts, "base": bh_base_opts, "additive": bh_add_opts}
    bh_var_cols = ["ligand", "base", "additive"]

    da_base_names = {
        smiles: DA_BASE_READABLE[code] for smiles, code in da_meta["base_codes"].items()
    }
    da_lig_names = da_meta["ligand_names"]
    da_solv_names = {
        smiles: DA_SOLVENT_READABLE[code]
        for smiles, code in da_meta["solvent_codes"].items()
    }
    da_base_opts, da_base_map = options_from_raw(da["base"].unique(), "B", da_base_names)
    da_lig_opts, da_lig_map = options_from_raw(da["ligand"].unique(), "L", da_lig_names)
    da_solv_opts, da_solv_map = options_from_raw(da["solvent"].unique(), "S", da_solv_names)
    da_conc_opts, da_conc_map = numeric_options(
        da["concentration"].unique(), "C", lambda token: f"{token} M"
    )
    da_temp_opts, da_temp_map = numeric_options(
        da["temperature"].unique(), "T", lambda token: f"{token} °C"
    )
    da_options = {
        "base": da_base_opts,
        "ligand": da_lig_opts,
        "solvent": da_solv_opts,
        "concentration": da_conc_opts,
        "temperature": da_temp_opts,
    }
    da_var_cols = ["base", "ligand", "solvent", "concentration", "temperature"]

    bh.to_csv(PROCESSED / "BH.csv", index=False)
    da.to_csv(PROCESSED / "DA.csv", index=False)
    (PROCESSED / "options_BH.json").write_text(
        json.dumps(bh_options, indent=2), encoding="utf-8"
    )
    (PROCESSED / "options_DA.json").write_text(
        json.dumps(da_options, indent=2), encoding="utf-8"
    )

    aryl_summary = summarise_aryl(bh)
    kept = []
    for smiles in BH_TASK_SMILES:
        match = next(row for row in aryl_summary if row["smiles"] == smiles)
        if match["max_yield"] < 30:
            raise SystemExit(f"Selected substrate {match['name']} has max yield {match['max_yield']} < 30")
        kept.append(match)
    halides = {row["halide"] for row in kept}
    if not {"Cl", "Br", "I"} <= halides:
        raise SystemExit(f"BH tasks do not cover Cl, Br and I: {halides}")
    dropped = [row for row in aryl_summary if row["max_yield"] < 30]

    catalogue = []
    prepared = []
    names = []
    names.extend(name_rows("BH", "ligand", bh_lig_opts))
    names.extend(name_rows("BH", "base", bh_base_opts))
    names.extend(name_rows("BH", "additive", bh_add_opts))

    bh_labeled_all = map_labels(
        bh,
        {"ligand": bh_lig_map, "base": bh_base_map, "additive": bh_add_map},
    )
    for i, row in enumerate(kept, start=1):
        task_id = f"BH_t{i}"
        mask = bh["aryl_halide"] == row["smiles"]
        conditions, y = build_condition_table(
            bh_labeled_all.loc[mask], bh.loc[mask, "yield"], bh_var_cols
        )
        x = validate_matrix(conditions, y, bh_options, bh_var_cols, task_id)
        threshold, k = hit_threshold(y)
        reaction = (
            "Pd-catalysed Buchwald-Hartwig amination of "
            f"{row['name']} with 4-methylaniline; choose the ligand, base and isoxazole additive."
        )
        record = {
            "id": task_id,
            "dataset": "BH",
            "substrate": row["name"],
            "variables": bh_var_cols,
            "row_count": int(len(conditions)),
            "max_yield": float(np.max(y)),
            "hit_threshold": threshold,
            "budget": 20,
            "halide": row["halide"],
            "hit_k": k,
            "smiles": row["smiles"],
        }
        catalogue.append({key: record[key] for key in (
            "id", "dataset", "substrate", "variables", "row_count",
            "max_yield", "hit_threshold", "budget",
        )})
        prepared.append(
            {
                "id": task_id,
                "dataset": "BH",
                "reaction_text": reaction,
                "var_cols": bh_var_cols,
                "conditions": conditions,
                "X": x,
                "options": bh_options,
                "hit_threshold": threshold,
                "max_yield": float(np.max(y)),
                "budget": 20,
                "y": y,
            }
        )
        names.append(
            {
                "dataset": "BH",
                "variable": "aryl_halide",
                "label": task_id,
                "name": row["name"],
                "smiles": row["smiles"],
            }
        )

    da_labeled = map_labels(
        da,
        {
            "base": da_base_map,
            "ligand": da_lig_map,
            "solvent": da_solv_map,
            "concentration": da_conc_map,
            "temperature": da_temp_map,
        },
    )
    da_conditions, da_y = build_condition_table(da_labeled, da["yield"], da_var_cols)
    da_x = validate_matrix(da_conditions, da_y, da_options, da_var_cols, "DA")
    da_threshold, da_k = hit_threshold(da_y)
    da_reaction = (
        "Pd-catalysed direct C-H arylation of an imidazole to form "
        "5-(2-fluorophenyl)-1-methyl-1H-imidazole-4-carbonitrile "
        "(Shields et al., Nature 2021, reaction 3); choose the base, ligand, "
        "solvent, concentration and temperature."
    )
    catalogue.append(
        {
            "id": "DA",
            "dataset": "DA",
            "substrate": "5-(2-fluorophenyl)-1-methyl-1H-imidazole-4-carbonitrile",
            "variables": da_var_cols,
            "row_count": int(len(da_conditions)),
            "max_yield": float(np.max(da_y)),
            "hit_threshold": da_threshold,
            "budget": 30,
        }
    )
    prepared.append(
        {
            "id": "DA",
            "dataset": "DA",
            "reaction_text": da_reaction,
            "var_cols": da_var_cols,
            "conditions": da_conditions,
            "X": da_x,
            "options": da_options,
            "hit_threshold": da_threshold,
            "max_yield": float(np.max(da_y)),
            "budget": 30,
            "y": da_y,
        }
    )
    for variable, options in da_options.items():
        names.extend(name_rows("DA", variable, options))
    names.append(
        {
            "dataset": "DA",
            "variable": "substrate",
            "label": "DA",
            "name": "5-(2-fluorophenyl)-1-methyl-1H-imidazole-4-carbonitrile",
            "smiles": "",
        }
    )

    pd.DataFrame(names).to_csv(PROCESSED / "names.csv", index=False)

    # Unknown-label guard.
    bad = da_conditions.copy()
    bad.iloc[0, 0] = "NOT_A_LABEL"
    try:
        one_hot(bad, da_var_cols, da_options)
    except ValueError:
        pass
    else:
        raise SystemExit("one_hot did not raise on an unknown label")

    Task = try_import_task()
    pickled = []
    if Task is not None:
        import pickle

        for spec in prepared:
            task = Task(
                id=spec["id"],
                dataset=spec["dataset"],
                reaction_text=spec["reaction_text"],
                var_cols=spec["var_cols"],
                conditions=spec["conditions"],
                X=spec["X"],
                options=spec["options"],
                hit_threshold=spec["hit_threshold"],
                max_yield=spec["max_yield"],
                budget=spec["budget"],
                y=spec["y"],
            )
            path = TASK_DIR / f"{spec['id']}.pkl"
            with path.open("wb") as handle:
                pickle.dump(task, handle, protocol=4)
            pickled.append(spec["id"])
        (PROCESSED / "tasks.json").write_text(
            json.dumps(catalogue, indent=2), encoding="utf-8"
        )
    else:
        print("tasks.json was not written because Task could not be imported.")

    summary = {
        "bh": bh_meta,
        "da": {
            "source_url": da_meta["source_url"],
            "rows_before_average": da_meta["rows_before_average"],
            "rows_after_average": da_meta["rows_after_average"],
            "duplicate_rows_collapsed": da_meta["duplicate_rows_collapsed"],
            "yield_min": da_meta["yield_min"],
            "yield_max": da_meta["yield_max"],
            "yield_multiplied_by_100": da_meta["yield_multiplied_by_100"],
            "hit_k": da_k,
            "hit_threshold": da_threshold,
            "n_features": int(da_x.shape[1]),
        },
        "aryl": aryl_summary,
        "bh_tasks": [
            {
                "id": f"BH_t{i}",
                "name": row["name"],
                "halide": row["halide"],
                "n": next(item["row_count"] for item in catalogue if item["id"] == f"BH_t{i}"),
                "max_yield": next(item["max_yield"] for item in catalogue if item["id"] == f"BH_t{i}"),
                "hit_threshold": next(item["hit_threshold"] for item in catalogue if item["id"] == f"BH_t{i}"),
                "smiles": row["smiles"],
            }
            for i, row in enumerate(kept, start=1)
        ],
        "dropped_max_under_30": dropped,
        "pickled": pickled,
        "bh_feature_width": int(prepared[0]["X"].shape[1]),
        "da_conc": [opt["name"] for opt in da_conc_opts],
        "da_temp": [opt["name"] for opt in da_temp_opts],
        "bh_ligands": [(opt["label"], opt["name"]) for opt in bh_lig_opts],
        "bh_bases": [(opt["label"], opt["name"]) for opt in bh_base_opts],
        "bh_additives": [(opt["label"], opt["name"]) for opt in bh_add_opts],
        "da_ligands": [(opt["label"], opt["name"]) for opt in da_lig_opts],
        "da_bases": [(opt["label"], opt["name"]) for opt in da_base_opts],
        "da_solvents": [(opt["label"], opt["name"]) for opt in da_solv_opts],
    }
    (PROCESSED / "build_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "bh_rows": bh_meta["rows_after_average"],
        "bh_scaled": bh_meta["yield_multiplied_by_100"],
        "da_rows": da_meta["rows_after_average"],
        "da_scaled": da_meta["yield_multiplied_by_100"],
        "da_hit_k": da_k,
        "da_hit_threshold": da_threshold,
        "pickled": pickled,
        "dropped": [row["name"] for row in dropped],
        "bh_tasks": summary["bh_tasks"],
    }, indent=2))


if __name__ == "__main__":
    main()
