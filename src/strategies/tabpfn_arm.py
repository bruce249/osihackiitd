"""Pick the next conditions by the surrogate's 0.9 predictive quantile."""

from __future__ import annotations

import numpy as np

from ..surrogates import SURROGATE_NAME, fit_predict_quantiles


def make_strategy(**kwargs):
    return Strategy()


def top_by_score(pool_idx, scores, q, rng) -> list[int]:
    """Highest scores first. Equal scores are shuffled with rng."""
    pool = np.asarray(pool_idx, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    n = int(pool.size)
    if n == 0 or int(q) <= 0:
        return []
    if scores.shape != (n,):
        raise ValueError(f"scores shape {scores.shape} != {(n,)}")
    k = min(int(q), n)
    perm = rng.permutation(n)
    tie = np.empty(n, dtype=np.int64)
    tie[perm] = np.arange(n, dtype=np.int64)
    rank = np.where(np.isfinite(scores), -scores, np.inf)
    order = np.lexsort((tie, rank))
    return [int(i) for i in pool[order[:k]]]


class Strategy:
    name = "tabpfn_ucb" if SURROGATE_NAME == "tabpfn" else "rf_ucb"
    one_shot = False

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        pool = np.asarray(pool_idx, dtype=int).ravel()
        if pool.size == 0 or int(q) <= 0:
            return []
        hist = np.asarray(list(hist_idx), dtype=int).ravel()
        X_all = np.asarray(task.X, dtype=float)
        if hist.size == 0:
            X_hist = np.zeros((0, X_all.shape[1]), dtype=float)
        else:
            X_hist = X_all[hist]
        X_cand = X_all[pool]
        _q10, _q50, q90 = fit_predict_quantiles(X_hist, hist_y, X_cand, seed)
        return top_by_score(pool, q90, q, rng)
