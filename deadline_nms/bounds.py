"""Exact, cluster-level confidence bounds on a deadline-exceedance probability.

WHY THIS REPLACES THE t-LIMIT AND THE BOOTSTRAP (v2.0.0)
---------------------------------------------------------------------------
A deadline claim is a statement about future requests. For any latency distribution,

    Q_0.95(T) <= D   if and only if   P(T > D) <= 0.05      (up to ties at D),

so a confidence statement about the 95th percentile of future requests is a confidence
statement about the exceedance probability theta = P(T > D). Version 1.x bounded the MEAN
of a few session p95 values with a Student-t limit (which is not a bound on the p95 of
future requests), and bounded miss ratios with a bootstrap that collapses to [0, 0] or
[1, 1] when every observation is at a boundary.

THE BOUND
---------------------------------------------------------------------------
Draw C inputs independently and observe each through one or more requests, dependent in
any way. Let A be the number of inputs with at least one request over D, and B the number
with every request over D. One request drawn at random from each input gives
X ~ Binomial(C, theta) with B <= X <= A. The exact one-sided Clopper-Pearson limits are
monotone in the count, so

    U = cp_upper(A, C)   is a valid one-sided upper bound on theta,
    L = cp_lower(B, C)   is a valid one-sided lower bound,

for any within-input dependence. What is needed is independence BETWEEN the inputs' latency
vectors. Sampling the inputs independently (for video, one frame per sequence) does not by
itself provide it: inputs measured in a shared session, process or machine state share that
state, and a state that slows every input at once couples them. The bound is therefore a
statement CONDITIONAL on the runtime state(s) under which the data were taken. It says nothing
about states that were not sampled, and repeating requests inside the same states adds no
information about them. (In the JRTIP revision, worker processes fell into a fast or a slow
state at start-up, about 3.5 ms apart; a verdict taken in one state did not transfer to the
other.) To cover runtime states, sample them: treat independently started runtime episodes
as the unit, or condition the claim explicitly. With A = 0, U = 1 - alpha**(1/C): certifying
theta <= 0.05 at 95% needs C >= 59 independent units.

Verdict against a policy limit rho: PASS if U <= rho, FAIL if L > rho, otherwise
INCONCLUSIVE, which must never promote.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

__all__ = ["cp_upper", "cp_lower", "exceedance_bound", "verdict", "min_inputs_for_pass"]


def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _binom_cdf(x: int, n: int, p: float) -> float:
    if x < 0:
        return 0.0
    if x >= n:
        return 1.0
    return min(1.0, sum(math.exp(_log_pmf(k, n, p)) for k in range(x + 1)))


def _bisect(pred, lo: float, hi: float, iters: int = 200) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper limit: the largest p with P(X <= x; n, p) >= alpha."""
    if n <= 0 or x >= n:
        return 1.0
    return _bisect(lambda p: _binom_cdf(x, n, p) >= alpha, x / n, 1.0)


def cp_lower(x: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided lower limit: the smallest p with P(X >= x; n, p) >= alpha."""
    if n <= 0 or x <= 0:
        return 0.0
    return _bisect(lambda p: 1.0 - _binom_cdf(x - 1, n, p) < alpha, 0.0, x / n)


def verdict(upper: float, lower: float, rho: float = 0.05) -> str:
    """PASS if the upper bound is within rho, FAIL if the lower bound exceeds it."""
    if upper <= rho:
        return "PASS"
    if lower > rho:
        return "FAIL"
    return "INCONCLUSIVE"


def exceedance_bound(latencies_by_input: Mapping[object, Sequence[float]] | Iterable[Sequence[float]],
                     deadline_ms: float, rho: float = 0.05, alpha: float = 0.05) -> dict:
    """Cluster-level bound on P(T > deadline) from per-input latency samples.

    ``latencies_by_input`` maps each independently sampled input (or is an iterable of
    per-input sequences) to the latencies of all its requests, in any number and with any
    dependence. Returns the counts, both limits and the verdict.

    Scope. The limits are exact Clopper-Pearson limits when the inputs' miss probabilities
    are equal. When they differ (one frame per video sequence, say), the count is
    Poisson-binomial and only the PASS/FAIL VERDICTS are shown to keep their error rates, and
    only where the critical counts satisfy Hoeffding's (1956) conditions: k_pass <= C*rho - 1
    and k_fail >= C*rho + 1. That was checked for rho = alpha = 0.05 at C = 100, 136 and 137
    (the paper's sample sizes); it is not a general property of every C, rho and alpha, and
    one-sided CP limits are not completely robust to unequal trials in general (Mattner and
    Tasto, arXiv:1403.0229). Inputs must be independent of each other: a shared runtime state
    that makes them dependent is outside what this bound covers.
    """
    _check_unit_interval("alpha", alpha)
    _check_unit_interval("rho", rho)
    if not (isinstance(deadline_ms, (int, float)) and math.isfinite(deadline_ms) and deadline_ms > 0):
        raise ValueError(f"deadline_ms must be a finite positive number, got {deadline_ms!r}")
    groups = list(latencies_by_input.values()) if isinstance(latencies_by_input, Mapping) \
        else [list(g) for g in latencies_by_input]
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError("every input needs at least one request")
    for g in groups:
        for x in g:
            # A NaN compares False with everything and would silently count as "no miss";
            # a negative or infinite latency is a measurement fault, not an observation.
            if not (isinstance(x, (int, float)) and math.isfinite(x) and x >= 0):
                raise ValueError(f"latencies must be finite and non-negative, got {x!r}")
    c = len(groups)
    a = sum(1 for g in groups if any(x > deadline_ms for x in g))
    b = sum(1 for g in groups if all(x > deadline_ms for x in g))
    u, l = cp_upper(a, c, alpha), cp_lower(b, c, alpha)
    return {"C": c, "A_any_over": a, "B_all_over": b, "upper": u, "lower": l,
            "verdict": verdict(u, l, rho)}


def min_inputs_for_pass(rho: float = 0.05, alpha: float = 0.05) -> int:
    """Smallest number of independent units for which zero misses can certify theta <= rho.

    Closed form: U(0, C) = 1 - alpha**(1/C) <= rho  <=>  C >= ln(alpha) / ln(1 - rho).
    """
    _check_unit_interval("alpha", alpha)
    _check_unit_interval("rho", rho)
    c = max(1, math.ceil(math.log(alpha) / math.log1p(-rho) - 1e-12))
    while c > 1 and cp_upper(0, c - 1, alpha) <= rho:     # guard against rounding at the edge
        c -= 1
    while cp_upper(0, c, alpha) > rho:
        c += 1
    return c


def _check_unit_interval(name: str, v: float) -> None:
    if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {v!r}")
