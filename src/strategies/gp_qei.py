"""Batch expected improvement by the Kriging-believer heuristic."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# Same optimiser budget as gp_ei. A DA proposal is well under a second.
_N_RESTARTS = 2


def make_strategy(**kwargs):
    return Strategy()


def _random_picks(pool_idx, q, rng) -> list[int]:
    pool = np.asarray(pool_idx, dtype=int).ravel()
    if pool.size == 0 or int(q) <= 0:
        return []
    k = min(int(q), int(pool.size))
    chosen = rng.choice(pool, size=k, replace=False)
    return [int(i) for i in np.asarray(chosen).ravel()]


def _top_by_score(pool_idx, scores, q, rng) -> list[int]:
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


def _expected_improvement(mu, sd, y_real) -> np.ndarray:
    """EI for maximisation. `y_real` is observed yield only; fantasy values are excluded."""
    mu = np.asarray(mu, dtype=float).ravel()
    sd = np.asarray(sd, dtype=float).ravel()
    y = np.asarray(y_real, dtype=float).ravel()
    best = float(np.max(y))
    xi = 0.01 * (float(np.max(y) - np.min(y)) + 1e-9)
    ei = np.zeros(mu.size, dtype=float)
    valid = np.isfinite(mu) & np.isfinite(sd) & (sd >= 1e-9)
    if np.any(valid):
        improv = mu[valid] - best - xi
        z = improv / sd[valid]
        ei[valid] = improv * norm.cdf(z) + sd[valid] * norm.pdf(z)
    return ei


def _fit_predict(x_hist, y_hist, x_pool, seed):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=_N_RESTARTS,
        random_state=0 if seed is None else int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", module=r"sklearn\.gaussian_process")
        gp.fit(x_hist, y_hist)
        mu, sd = gp.predict(x_pool, return_std=True)
    return np.asarray(mu, dtype=float).ravel(), np.asarray(sd, dtype=float).ravel()


class Strategy:
    name = "gp_qei"
    one_shot = False

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        pool = np.asarray(pool_idx, dtype=int).ravel()
        hist = np.asarray(list(hist_idx), dtype=int).ravel()
        y_real = np.asarray(list(hist_y), dtype=float).ravel()
        if y_real.size < 2 or y_real.size != hist.size or pool.size == 0 or int(q) <= 0:
            return _random_picks(pool, q, rng)

        x_all = np.asarray(task.X, dtype=float)
        remain = pool.copy()
        fit_x = np.asarray(x_all[hist], dtype=float)
        fit_y = np.asarray(y_real, dtype=float).copy()
        chosen: list[int] = []

        for _step in range(int(q)):
            if remain.size == 0:
                break
            try:
                mu, sd = _fit_predict(fit_x, fit_y, x_all[remain], seed)
                ei = _expected_improvement(mu, sd, y_real)
            except Exception:
                chosen.extend(_random_picks(remain, int(q) - len(chosen), rng))
                break
            pick = _top_by_score(remain, ei, 1, rng)
            if not pick:
                chosen.extend(_random_picks(remain, int(q) - len(chosen), rng))
                break
            idx = int(pick[0])
            pos = int(np.flatnonzero(remain == idx)[0])
            fantasy = float(mu[pos])
            chosen.append(idx)
            remain = remain[remain != idx]
            if not np.isfinite(fantasy):
                chosen.extend(_random_picks(remain, int(q) - len(chosen), rng))
                break
            fit_x = np.vstack([fit_x, x_all[idx][None, :]])
            fit_y = np.append(fit_y, fantasy)
        return chosen
