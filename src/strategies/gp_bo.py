"""Gaussian-process Bayesian optimisation with expected improvement."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


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


class Strategy:
    name = "gp_ei"
    one_shot = False

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        pool = np.asarray(pool_idx, dtype=int).ravel()
        hist = np.asarray(list(hist_idx), dtype=int).ravel()
        y = np.asarray(list(hist_y), dtype=float).ravel()
        if y.size < 2 or pool.size == 0:
            return _random_picks(pool, q, rng)

        X_all = np.asarray(task.X, dtype=float)
        X_hist = X_all[hist]
        X_pool = X_all[pool]

        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-2)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=0 if seed is None else int(seed),
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.filterwarnings("ignore", module=r"sklearn\.gaussian_process")
                gp.fit(X_hist, y)
                mu, sd = gp.predict(X_pool, return_std=True)
        except Exception:
            return _random_picks(pool, q, rng)

        mu = np.asarray(mu, dtype=float).ravel()
        sd = np.asarray(sd, dtype=float).ravel()
        best = float(np.max(y))
        xi = 0.01 * (float(np.max(y) - np.min(y)) + 1e-9)
        ei = np.zeros(pool.size, dtype=float)
        valid = np.isfinite(mu) & np.isfinite(sd) & (sd >= 1e-9)
        if np.any(valid):
            improv = mu[valid] - best - xi
            z = improv / sd[valid]
            ei[valid] = improv * norm.cdf(z) + sd[valid] * norm.pdf(z)
        return _top_by_score(pool, ei, q, rng)
