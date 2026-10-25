"""Choosing the cap from a deadline instead of from a round number."""
from __future__ import annotations

import math

__all__ = ["upper_confidence_limit", "select_k", "select_k_certified"]


def upper_confidence_limit(values, conf: float = 0.95) -> float:
    """DEPRECATED since 2.0.0: a one-sided Student-t limit on the MEAN of session statistics.

    It is not a confidence bound on the p95 of future requests (a quantile of a mixture is
    not the mean of the conditional quantiles), and it carries no between-input uncertainty
    when the sessions replay the same inputs. Use ``select_k_certified`` with
    ``bounds.exceedance_bound``. Kept only so code written against 1.x still imports.

    The multiplier is the ONE-SIDED 95% Student-t quantile (2.920 at n = 3, i.e. 2 degrees of
    freedom); 4.303 is the two-sided 95% value, which the submitted manuscript used by mistake.
    Beyond 11 sessions it falls back to the normal quantile 1.645, an approximation.
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
    """DEPRECATED since 2.0.0. Largest cap whose mean-of-session-p95 limit fits the budget.

    This is NOT a confidence limit on the p95 of future requests (see
    ``upper_confidence_limit``). Use ``select_k_certified``.

    ``grid``                  candidate values of K, any order
    ``suppression_sessions``  K -> sequence of per-session suppression p95 values (ms)
    ``accuracy_cost``         K -> clean accuracy given up at that cap, a PAIRED
                              difference taken within one detection set
    ``budget_ms``             the suppression screen the tail must fit inside
    ``max_accuracy_cost``     the deployment's policy limit on that loss

    Returns the selected K, or ``None`` if no grid point satisfies both conditions.

    The tail term is a Student-t limit on the MEAN of session p95 values, kept only so code
    written against 1.x still runs. Selecting the boundary value is also not the same as
    deploying it -- the margin between the selected K and the deployed one is a declared
    choice, not a consequence of the sweep.
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


def select_k_certified(grid, latencies, accuracy_cost, budget_ms: float,
                       max_accuracy_cost: float, rho: float = 0.05, alpha: float = 0.05):
    """Largest cap whose exceedance probability is CERTIFIED within rho (v2.0.0).

    ``latencies``  K -> {input id: [suppression latencies of all its requests, ms]}, with
                   inputs sampled independently (e.g. one frame per video sequence)
    ``accuracy_cost`` K -> paired clean accuracy given up at that cap

    A K qualifies only if ``bounds.exceedance_bound`` returns PASS at ``budget_ms``: an
    INCONCLUSIVE grid point is not selected. Returns (K or None, {K: bound result}); every
    grid point with data is evaluated and reported.

    The grid is tested in FIXED SEQUENCE: in increasing K, stopping at the first K whose bound
    does not PASS, and the selected K is the largest one before that stop that also meets the
    accuracy condition (a deterministic policy filter, not a test). Fixed-sequence testing
    keeps the 1 - alpha level for the selected K without a multiplicity correction and without
    assuming that the verdict is monotone in K; a PASS at a larger K after the stop is reported
    but never selected. A grid point with no latency data is reported as MISSING and stops the
    sequence like any other non-PASS. The bound is conditional on the runtime states the latencies were
    measured in (see ``bounds``).
    """
    from .bounds import exceedance_bound

    results, ok, stopped = {}, [], False
    for k in sorted(grid):
        by_input = latencies.get(k)
        if not by_input:
            # A declared grid point without data is not a PASS: it ends the sequence, as an
            # untested hypothesis must, rather than being skipped over.
            results[k] = {"verdict": "MISSING", "C": 0}
            stopped = True
            continue
        r = exceedance_bound(by_input, budget_ms, rho=rho, alpha=alpha)
        results[k] = r
        if r["verdict"] != "PASS":
            stopped = True
        if not stopped and accuracy_cost.get(k, float("inf")) <= max_accuracy_cost:
            ok.append(k)
    return (max(ok) if ok else None), results
