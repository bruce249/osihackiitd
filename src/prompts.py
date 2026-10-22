"""Prompts for the experiment-planning LLM.

The static prefix (reaction text and option lists) comes first so it can be
cached. The experiment history and the batch instruction come last.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert synthetic chemist planning experiments to maximise reaction yield.\n"
    "Each experiment chooses one option for every variable listed. You will see the results\n"
    "of experiments already run. Choose the next batch to find the highest-yield conditions\n"
    "in as few experiments as possible. Use chemical knowledge (ligand and base effects,\n"
    "substrate reactivity, solvent effects) and the observed results. Balance exploiting\n"
    "good regions with testing informative new combinations. Never repeat an experiment.\n"
    "Return JSON only."
)


def build_messages(task, hist_idx, hist_y, k, mode: str) -> tuple[str, str]:
    """Return ``(system, user)`` for named or anonymised mode."""
    if mode not in ("named", "anon"):
        raise ValueError("mode must be 'named' or 'anon'")
    static = _static_block(task, mode)
    dynamic = _dynamic_block(task, hist_idx, hist_y, k)
    return SYSTEM_PROMPT, static + "\n\n" + dynamic


def parse_picks(payload: dict, task) -> list[tuple[int, str]]:
    """Map a picks payload to ``(row_idx, reason)``.

    Combinations that are not in ``task.conditions`` are dropped. Duplicate
    rows are dropped. The order of the first valid occurrence is kept.
    """
    if not isinstance(payload, dict):
        return []
    if "picks" not in payload and isinstance(payload.get("response"), dict):
        payload = payload["response"]
    picks = payload.get("picks", [])
    if not isinstance(picks, list):
        return []

    var_cols = [str(col) for col in task.var_cols]
    lookup = {col.lower(): col for col in var_cols}
    index = condition_index(task)
    seen: set[int] = set()
    out: list[tuple[int, str]] = []
    for pick in picks:
        if not isinstance(pick, dict):
            continue
        normalised: dict[str, str] = {}
        for key, value in pick.items():
            if str(key).lower() == "reason":
                continue
            canon = lookup.get(str(key).lower())
            if canon is None or value is None:
                continue
            normalised[canon] = str(value).strip()
        if any(not normalised.get(col) for col in var_cols):
            continue
        key = tuple(normalised[col] for col in var_cols)
        row = index.get(key)
        if row is None or row in seen:
            continue
        seen.add(row)
        reason = pick.get("reason", "")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            reason = str(reason)
        out.append((row, reason.strip()))
    return out


def condition_index(task) -> dict[tuple[str, ...], int]:
    """Map a label tuple, in ``var_cols`` order, to a positional row index."""
    cols = [str(col) for col in task.var_cols]
    index: dict[tuple[str, ...], int] = {}
    for i in range(_n_rows(task)):
        labels = row_labels(task, i)
        key = tuple(labels[col] for col in cols)
        index.setdefault(key, i)
    return index


def row_labels(task, idx: int) -> dict[str, str]:
    """Positional row ``idx`` as ``{variable: label}``."""
    cols = [str(col) for col in task.var_cols]
    cond = task.conditions
    row = cond.iloc[int(idx)] if hasattr(cond, "iloc") else cond[int(idx)]
    return {col: _cell(row[col]) for col in cols}


def _static_block(task, mode: str) -> str:
    lines: list[str] = []
    reaction = _reaction_text(task, mode)
    if reaction:
        lines.append(reaction)
        lines.append("")
    lines.append("Variables and options:")
    options = getattr(task, "options", {}) or {}
    var_cols = [str(col) for col in task.var_cols]
    for var in var_cols:
        lines.append(f"{var}:")
        opts = list(options.get(var, []) or [])
        if not opts:
            opts = _labels_as_options(task, var)
        pretty = var.replace("_", " ").title()
        for i, opt in enumerate(opts, start=1):
            label = _cell(opt.get("label", ""))
            if mode == "anon":
                lines.append(f"{label}: {pretty} option {i}")
                continue
            name = _cell(opt.get("name", ""))
            smiles = _cell(opt.get("smiles", ""))
            if smiles.lower() == "nan":
                smiles = ""
            if smiles:
                lines.append(f"{label}: {name} [SMILES {smiles}]")
            elif name:
                lines.append(f"{label}: {name}")
            else:
                lines.append(label)
        lines.append("")
    return "\n".join(lines).rstrip()


def _dynamic_block(task, hist_idx, hist_y, k) -> str:
    lines = ["Experiments so far (best first):"]
    rows: list[tuple[float, int, str]] = []
    n = _n_rows(task)
    cols = [str(col) for col in task.var_cols]
    for pos, (idx, y) in enumerate(zip(hist_idx, hist_y)):
        try:
            ii = int(idx)
        except (TypeError, ValueError):
            continue
        if ii < 0 or ii >= n:
            continue
        labels = row_labels(task, ii)
        text = " | ".join(labels[col] for col in cols)
        try:
            yv = float(y)
        except (TypeError, ValueError):
            yv = float("nan")
        shown = "nan%" if yv != yv else f"{yv:.1f}%"
        rows.append((yv, pos, f"{text} -> {shown}"))
    rows.sort(key=lambda item: _history_sort_key(item[0], item[1]))
    for _yv, _pos, line in rows:
        lines.append(line)
    lines.append("")
    lines.append(_instruction(cols, k))
    return "\n".join(lines)


def _instruction(var_cols, k: int) -> str:
    keys = ", ".join(f'"{col}": "<label>"' for col in var_cols)
    example = '{"picks": [{' + keys + ', "reason": "<one sentence>"}]}'
    return (
        f"Propose exactly {int(k)} NEW experiments not listed above. Return JSON:\n"
        f"{example}"
    )


def _history_sort_key(yv: float, pos: int) -> tuple:
    if yv != yv:
        return (1, 0.0, pos)
    return (0, -yv, pos)


def _reaction_text(task, mode: str) -> str:
    text = str(getattr(task, "reaction_text", "") or "").strip()
    if mode != "anon" or not text:
        return text
    secrets: list[str] = []
    options = getattr(task, "options", {}) or {}
    for opts in options.values():
        for opt in opts or []:
            label = _cell(opt.get("label", ""))
            for key in ("name", "smiles"):
                value = _cell(opt.get(key, ""))
                if value and value != label and len(value) >= 3 and value.lower() != "nan":
                    secrets.append(value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(secret, "[hidden]")
    return text


def _labels_as_options(task, var: str) -> list[dict]:
    seen: list[str] = []
    for i in range(_n_rows(task)):
        label = row_labels(task, i)[var]
        if label not in seen:
            seen.append(label)
    return [{"label": label, "name": "", "smiles": ""} for label in seen]


def _n_rows(task) -> int:
    return len(task.conditions)


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
