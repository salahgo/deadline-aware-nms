"""Deadline metrics: the tail, the miss ratio, and an interval that respects clustering.

A mean latency cannot express a deadline. These are the three quantities the paper
reports instead, and the resampling unit is the part that is easy to get wrong.
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantile", "deadline_miss_ratio", "session_bootstrap_ci"]


def quantile(samples, q: float = 0.95) -> float:
    """Linear-interpolated quantile, the convention every latency figure here uses.

    A p95 over fewer than about twenty samples is the maximum in disguise; report the
    sample count beside any percentile taken from a short run.
    """
    a = np.asarray(samples, dtype=float)
    if a.size == 0:
        raise ValueError("empty sample")
    return float(np.quantile(a, q))


def deadline_miss_ratio(samples, deadline_ms: float) -> float:
    """Fraction of responses later than ``deadline_ms``."""
    a = np.asarray(samples, dtype=float)
    if a.size == 0:
        raise ValueError("empty sample")
    return float((a > deadline_ms).mean())


def session_bootstrap_ci(sessions, statistic, n_boot: int = 10000,
                         alpha: float = 0.05, seed: int = 0):
    """Two-sided interval for ``statistic`` under a hierarchical resample.

    ``sessions`` is a sequence of sequences: one inner sequence per independent timing
    session, each holding that session's per-image observations.

    RESAMPLE SESSIONS, THEN IMAGES WITHIN THEM. Replayed requests over one corpus are
    not independent trials -- the same image seen twice contributes the same difficulty
    twice -- so pooling every observation and resampling flat treats correlated
    measurements as independent and returns an interval far too narrow. An exact
    binomial limit on a miss ratio has the same defect and is never used here.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(s, dtype=float) for s in sessions]
    if not groups or any(g.size == 0 for g in groups):
        raise ValueError("every session needs at least one observation")

    n = len(groups)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        drawn = [g[rng.integers(0, g.size, size=g.size)] for g in (groups[i] for i in pick)]
        boots[b] = statistic(np.concatenate(drawn))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
