"""q90 selection with other substrates added to the surrogate context.

If transfer data or src.data is missing, this falls back to the single-task
arm and keeps the name tabpfn_transfer.
"""

from __future__ import annotations

import numpy as np

from ..surrogates import fit_predict_quantiles
from .tabpfn_arm import Strategy as _ArmStrategy
from .tabpfn_arm import top_by_score


def make_strategy(**kwargs):
    return Strategy()


def _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
    return _ArmStrategy().propose(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)


class Strategy:
    name = "tabpfn_transfer"
    one_shot = False

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        try:
            return self._propose_transfer(
                task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx
            )
        except Exception:
            return _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)

    def _propose_transfer(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx):
        try:
            from ..data import get_transfer_data, pad_with_substrate_block
        except Exception:
            return _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)

        pool = np.asarray(pool_idx, dtype=int).ravel()
        if pool.size == 0 or int(q) <= 0:
            return []

        loaded = get_transfer_data(task.id)
        if loaded is None:
            return _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)
        X_tr, y_tr = loaded
        X_tr = np.asarray(X_tr, dtype=float)
        y_tr = np.asarray(y_tr, dtype=float).ravel()
        if X_tr.ndim != 2 or X_tr.shape[0] == 0 or y_tr.shape[0] != X_tr.shape[0]:
            return _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)

        if X_tr.shape[0] > 1000:
            chosen = rng.choice(X_tr.shape[0], size=1000, replace=False)
            X_tr = X_tr[chosen]
            y_tr = y_tr[chosen]

        X_all = np.asarray(task.X, dtype=float)
        hist = np.asarray(list(hist_idx), dtype=int).ravel()
        X_pool = np.asarray(pad_with_substrate_block(task.id, X_all[pool]), dtype=float)
        if hist.size == 0:
            X_ctx = X_tr
            y_ctx = y_tr
        else:
            X_hist = np.asarray(
                pad_with_substrate_block(task.id, X_all[hist]), dtype=float
            )
            y_hist = np.asarray(list(hist_y), dtype=float).ravel()
            if (
                X_hist.ndim != 2
                or X_pool.ndim != 2
                or X_hist.shape[1] != X_tr.shape[1]
                or X_pool.shape[1] != X_tr.shape[1]
                or X_hist.shape[0] != y_hist.shape[0]
            ):
                return _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)
            X_ctx = np.vstack([X_hist, X_tr])
            y_ctx = np.concatenate([y_hist, y_tr])

        if X_pool.shape[1] != np.asarray(X_ctx).shape[1]:
            return _fallback(task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx)

        _q10, _q50, q90 = fit_predict_quantiles(X_ctx, y_ctx, X_pool, seed)
        return top_by_score(pool, q90, q, rng)
