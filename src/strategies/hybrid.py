"""Hybrid strategy: a named-LLM shortlist of 20, ranked by surrogate q90."""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from src.strategies.llm_agent import fetch_picks

log = logging.getLogger(__name__)

SHORTLIST = 20


def _mean_quantile_stub(X_hist, y_hist, X_cand, seed):
    """Score every candidate with the mean of y_hist. q90 is that constant."""
    y = np.asarray([] if y_hist is None else list(y_hist), dtype=float).ravel()
    n = int(len(X_cand))
    fill = float(np.mean(y)) if y.size else 0.0
    band = np.full(n, fill, dtype=float)
    return band, band.copy(), band.copy()


_USING_STUB = False
try:
    from src.surrogates import fit_predict_quantiles
except Exception:
    try:
        from ..surrogates import fit_predict_quantiles
    except Exception:
        _USING_STUB = True

        def fit_predict_quantiles(X_hist, y_hist, X_cand, seed):
            # stub until P3 delivers: score candidates by mean of y_hist, q90 = that constant
            return _mean_quantile_stub(X_hist, y_hist, X_cand, seed)


def make_strategy(**kwargs):
    """Return the hybrid arm.

    Explicit ``strategy`` or ``name`` kwargs win. Otherwise ``--strategy``
    is read when ``--module`` points at this file, then STRATEGY_NAME.
    This module only builds the hybrid arm, so the recorded name stays
    ``hybrid`` for that selection and for a bare ``make_strategy()`` call.
    """
    requested = kwargs.get("strategy")
    if requested is None:
        requested = kwargs.get("name")
    if requested is None:
        requested = _cli_strategy_name()
    if requested is not None and str(requested).lower() not in ("hybrid", ""):
        requested = "hybrid"
    if requested and not os.environ.get("STRATEGY_NAME", "").strip():
        os.environ["STRATEGY_NAME"] = "hybrid"
    return Strategy()


def _cli_strategy_name():
    argv = sys.argv
    module = _argv_value(argv, "--module")
    strategy = _argv_value(argv, "--strategy")
    if module and strategy:
        norm = module.replace("\\", ".").replace("/", ".")
        if norm.endswith("hybrid") or norm.endswith("hybrid.py"):
            if str(strategy).lower() == "hybrid":
                os.environ["STRATEGY_NAME"] = "hybrid"
            return strategy
    env_name = os.environ.get("STRATEGY_NAME", "").strip()
    if env_name:
        return env_name
    return None


def _argv_value(argv, flag: str):
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    prefix = flag + "="
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


class Strategy:
    name = "hybrid"
    one_shot = False

    def __init__(self):
        self.last_reasons: dict[int, str] = {}

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        q = int(q)
        pool = _unique_ints(pool_idx)
        self.last_reasons = {}
        if q <= 0 or not pool:
            return []
        pool_set = set(pool)
        shortlist = fetch_picks(
            strategy_name=self.name,
            task=task,
            pool_idx=pool,
            hist_idx=hist_idx,
            hist_y=hist_y,
            k=SHORTLIST,
            seed=seed,
            round_idx=round_idx,
            mode="named",
        )
        reasons = {idx: reason for idx, reason in shortlist}
        short_idx = [idx for idx, _reason in shortlist if idx in pool_set]
        no_hist = _hist_len(hist_y) == 0

        if not short_idx and _USING_STUB and no_hist:
            chosen = _random_picks(pool, q, rng)
            self.last_reasons = {
                idx: "Random fill; no LLM shortlist and no history for a surrogate."
                for idx in chosen
            }
            return chosen

        if short_idx:
            ranked = _rank(task, hist_idx, hist_y, short_idx, seed, rng, no_hist)
        else:
            ranked = _rank(task, hist_idx, hist_y, pool, seed, rng, no_hist)

        chosen: list[int] = []
        seen: set[int] = set()
        for idx in ranked:
            if idx in pool_set and idx not in seen:
                chosen.append(idx)
                seen.add(idx)
            if len(chosen) >= q:
                break
        if len(chosen) < q:
            rest = [idx for idx in pool if idx not in seen]
            extra = _rank(task, hist_idx, hist_y, rest, seed, rng, no_hist)
            for idx in extra:
                if idx in pool_set and idx not in seen:
                    chosen.append(idx)
                    seen.add(idx)
                if len(chosen) >= q:
                    break
        gap = "Ranked by surrogate q90."
        self.last_reasons = {idx: reasons.get(idx, gap) for idx in chosen}
        return chosen


def _rank(task, hist_idx, hist_y, cand_idx, seed, rng, no_hist: bool) -> list[int]:
    cands = _unique_ints(cand_idx)
    if not cands:
        return []
    try:
        scores = _q90(task, hist_idx, hist_y, cands, seed, fit_predict_quantiles)
    except Exception as exc:
        log.warning("hybrid surrogate ranking failed: %s", str(exc)[:300])
        if _USING_STUB and no_hist:
            return _random_picks(cands, len(cands), rng)
        try:
            scores = _q90(task, hist_idx, hist_y, cands, seed, _mean_quantile_stub)
        except Exception:
            return cands
    return [cands[i] for i in _order_desc_stable(scores)]


def _q90(task, hist_idx, hist_y, cands, seed, scorer) -> np.ndarray:
    features = np.asarray(task.X, dtype=float)
    hist = np.asarray(list(hist_idx), dtype=int).ravel()
    if hist.size == 0:
        x_hist = np.zeros((0, features.shape[1]), dtype=float)
    else:
        x_hist = features[hist]
    x_cand = features[np.asarray(cands, dtype=int)]
    _q10, _q50, q90 = scorer(x_hist, list(hist_y), x_cand, seed)
    scores = np.asarray(q90, dtype=float).ravel()
    if scores.shape != (len(cands),):
        raise ValueError(f"q90 length {scores.shape[0]} != {len(cands)} candidates")
    return scores


def _order_desc_stable(scores) -> list[int]:
    arr = np.asarray(scores, dtype=float).ravel()
    safe = np.where(np.isfinite(arr), arr, -np.inf)
    return [int(i) for i in np.argsort(-safe, kind="stable")]


def _random_picks(pool, q, rng) -> list[int]:
    pool = _unique_ints(pool)
    if not pool or int(q) <= 0:
        return []
    k = min(int(q), len(pool))
    chosen = rng.choice(np.asarray(pool, dtype=int), size=k, replace=False)
    return [int(i) for i in np.asarray(chosen).ravel()]


def _hist_len(hist_y) -> int:
    try:
        return len(list(hist_y))
    except TypeError:
        return 0


def _unique_ints(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out
