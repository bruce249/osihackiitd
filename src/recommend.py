"""Live next-experiment recommendations for the app.

Uses the on-disk LLM cache. Calls the API only when the cache file is missing
and OPENAI_API_KEY is set. Returns an empty list instead of raising.
"""

from __future__ import annotations

import logging

import numpy as np

from src.llm_client import redact
from src.prompts import row_labels

log = logging.getLogger(__name__)


def recommend(task_id, hist_idx, hist_y, q=5, strategy="hybrid") -> list[dict]:
    """Return up to ``q`` suggestions for the next experiments.

    Each item is ``{"row_idx", "labels", "names", "reason"}``. ``names`` come
    from ``task.options``. The strategy sees ``task.public()``. The pool is
    every row not in ``hist_idx``. The live RNG seed is 0. The cache round
    is ``len(hist_idx)``, so a longer experiment log does not reuse an earlier
    response.
    """
    try:
        return _recommend(task_id, hist_idx, hist_y, q=q, strategy=strategy)
    except Exception as exc:
        log.warning("recommend failed: %s", redact(str(exc))[:400])
        return []


def _recommend(task_id, hist_idx, hist_y, q, strategy) -> list[dict]:
    from src.tasks import get_task

    task = get_task(task_id)
    public = task.public() if hasattr(task, "public") else task
    hist_idx = [int(i) for i in hist_idx]
    hist_y = [float(y) for y in hist_y]
    n = len(public.conditions)
    taken = set(hist_idx)
    pool = np.asarray([i for i in range(n) if i not in taken], dtype=int)
    rng = np.random.default_rng(0)
    strat = _make(strategy)
    if strat is None:
        log.warning("recommend: unknown strategy %s", strategy)
        return []
    chosen = strat.propose(
        public,
        pool,
        hist_idx,
        hist_y,
        int(q),
        rng,
        0,
        len(hist_idx),
    )
    reasons = getattr(strat, "last_reasons", {})
    out = []
    for idx in chosen:
        labels = row_labels(public, int(idx))
        out.append(
            {
                "row_idx": int(idx),
                "labels": labels,
                "names": _names(public, labels),
                "reason": str(reasons.get(int(idx), "")),
            }
        )
    return out


def _make(strategy: str):
    key = str(strategy).strip().lower()
    if key == "hybrid":
        from src.strategies.hybrid import make_strategy

        return make_strategy()
    if key == "llm":
        from src.strategies.llm_agent import make_strategy

        return make_strategy(mode="named", one_shot=False)
    if key == "llm_anon":
        from src.strategies.llm_agent import make_strategy

        return make_strategy(mode="anon", one_shot=False)
    if key == "llm_zeroshot":
        from src.strategies.llm_agent import make_strategy

        return make_strategy(mode="named", one_shot=True)
    return None


def _names(task, labels: dict) -> dict:
    options = getattr(task, "options", {}) or {}
    names = {}
    for var, label in labels.items():
        name = str(label)
        for opt in options.get(var, []) or []:
            if str(opt.get("label", "")).strip() == str(label).strip():
                raw = opt.get("name")
                if raw:
                    name = str(raw)
                break
        names[var] = name
    return names
