"""Uniform random search without replacement."""

from __future__ import annotations

import numpy as np


def make_strategy(**kwargs):
    return Strategy()


class Strategy:
    name = "random"
    one_shot = False

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        pool = np.asarray(pool_idx, dtype=int).ravel()
        if pool.size == 0 or int(q) <= 0:
            return []
        k = min(int(q), int(pool.size))
        chosen = rng.choice(pool, size=k, replace=False)
        return [int(i) for i in np.asarray(chosen).ravel()]
