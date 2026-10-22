"""Picks the highest true yields. This exists only to test the campaign metric."""

from __future__ import annotations

import numpy as np

from src.tasks import get_task


def make_strategy(**kwargs):
    return Strategy()


class Strategy:
    name = "oracle"
    one_shot = False

    def propose(self, task, pool_idx, hist_idx, hist_y, q, rng, seed, round_idx) -> list[int]:
        # The task handed to a strategy has y=None. Read the private yields.
        private = get_task(task.id)
        if private.y is None:
            raise RuntimeError(f"Oracle has no yields for task {task.id}.")
        pool = np.asarray(pool_idx, dtype=int).ravel()
        if pool.size == 0 or int(q) <= 0:
            return []
        yields = np.asarray(private.y, dtype=float).ravel()[pool]
        k = min(int(q), int(pool.size))
        # Highest true yield first. A permutation from rng breaks exact ties.
        perm = rng.permutation(pool.size)
        tie = np.empty(pool.size, dtype=np.int64)
        tie[perm] = np.arange(pool.size, dtype=np.int64)
        order = np.lexsort((tie, -yields))
        return [int(i) for i in pool[order[:k]]]
