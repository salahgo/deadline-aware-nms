"""Choosing the cap from a deadline instead of from a round number."""
from __future__ import annotations

import math

__all__ = ["upper_confidence_limit", "select_k"]


def upper_confidence_limit(values, conf: float = 0.95) -> float:
    """One-sided Student-t upper limit on the mean of a few session statistics.

    Written for the case it is actually used in: a handful of sessions. At n=3 the
    multiplier is 4.303 rather than 1.96, and a rule that hides that behind a normal
    approximation will select a cap it cannot defend.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("no sessions")
    mean = sum(xs) / n
    if n == 1:
        return mean
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    # one-sided t quantiles at 95%, indexed by degrees of freedom
    t = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
         8: 1.860, 9: 1.833, 10: 1.812}.get(n - 1, 1.645)
    if conf != 0.95:
        raise ValueError("only the 95% limit is tabulated here")
    return mean + t * math.sqrt(var / n)


def select_k(grid, suppression_sessions, accuracy_cost, budget_ms: float,
             max_accuracy_cost: float):
    """Largest cap whose bounded tail fits the budget and whose accuracy cost is allowed.

    ``grid``                  candidate values of K, any order
    ``suppression_sessions``  K -> sequence of per-session suppression p95 values (ms)
    ``accuracy_cost``         K -> clean accuracy given up at that cap, a PAIRED
                              difference taken within one detection set
    ``budget_ms``             the suppression screen the tail must fit inside
    ``max_accuracy_cost``     the deployment's policy limit on that loss

    Returns the selected K, or ``None`` if no grid point satisfies both conditions.

    The tail term is an UPPER CONFIDENCE LIMIT on the p95, not the p95 itself: a cap
    chosen on a point estimate is chosen on one draw of the noise. Selecting the
    boundary value is also not the same as deploying it -- the margin between the
    selected K and the deployed one is a declared choice, not a consequence of the
    sweep.
    """
    ok = []
    for k in sorted(grid):
        sessions = suppression_sessions.get(k)
        if not sessions:
            continue
        if upper_confidence_limit(sessions) > budget_ms:
            continue
        if accuracy_cost.get(k, float("inf")) > max_accuracy_cost:
            continue
        ok.append(k)
    return max(ok) if ok else None
