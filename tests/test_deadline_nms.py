"""The properties the defense is claimed on, not a coverage exercise."""
from __future__ import annotations

import numpy as np
import pytest

from deadline_nms import (admit_topk, deadline_miss_ratio, greedy_nms,
                          pairwise_bound, quantile, select_k,
                          upper_confidence_limit)


def test_admission_is_deterministic_when_the_cutoff_is_tied():
    """The property argpartition alone does NOT give: stable MEMBERSHIP at the cutoff.

    Ten candidates share the cutoff score and only three places remain. An unstable
    partition may return any three of them, so the result has to be pinned by the
    lexicographic rule, not by whatever the library happened to select.
    """
    scores = np.array([0.9, 0.8] + [0.5] * 10)
    first = admit_topk(scores, 5)
    for _ in range(50):
        np.testing.assert_array_equal(admit_topk(scores, 5), first)
    # two strictly above the cutoff, then the lowest-indexed ties
    np.testing.assert_array_equal(first, [0, 1, 2, 3, 4])


def test_admission_never_exceeds_the_cap_and_is_a_noop_when_disabled():
    rng = np.random.default_rng(0)
    scores = rng.random(5000)
    assert admit_topk(scores, 1024).size == 1024
    assert admit_topk(scores, 0).size == 5000       # 0 disables the cap
    assert admit_topk(scores, 9999).size == 5000    # cap above the input


def test_admission_keeps_the_highest_scores():
    rng = np.random.default_rng(1)
    scores = rng.random(2000)
    kept = admit_topk(scores, 100)
    assert scores[kept].min() >= np.sort(scores)[-100]


@pytest.mark.parametrize("k, expected", [(0, 0), (1, 0), (2, 1), (1024, 523776)])
def test_pairwise_bound(k, expected):
    assert pairwise_bound(k) == expected


def test_nms_output_is_bounded_by_the_cap():
    """The bound is on WORK, so the check is that admitted input, not output, is capped."""
    rng = np.random.default_rng(2)
    n, k = 20_000, 512
    boxes = np.stack([rng.uniform(0, 1000, n), rng.uniform(0, 1000, n)], axis=1)
    boxes = np.concatenate([boxes, boxes + 20], axis=1)
    scores = rng.random(n)
    keep = admit_topk(scores, k)
    assert keep.size == k
    assert len(greedy_nms(boxes[keep], scores[keep], 0.65)) <= k


def test_metrics():
    xs = list(range(1, 101))
    assert quantile(xs, 0.95) == pytest.approx(95.05, abs=0.5)
    assert deadline_miss_ratio(xs, 90) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        quantile([])


def test_upper_limit_uses_the_small_sample_multiplier():
    """At three sessions the multiplier is 2.920, not 1.645; the limit must reflect it."""
    v = [30.0, 31.0, 32.0]
    limit = upper_confidence_limit(v)
    assert limit > sum(v) / len(v)
    assert limit == pytest.approx(31.0 + 2.920 * (1.0 / 3) ** 0.5, rel=1e-3)


def test_select_k_respects_both_conditions():
    grid = [256, 512, 1024, 2048]
    tails = {256: [1.0, 1.1, 1.0], 512: [4.0, 4.2, 4.1],
             1024: [10.0, 10.2, 10.1], 2048: [40.0, 41.0, 40.5]}
    # 2048 is excluded by the deadline, 1024 by the accuracy limit
    cost = {256: 0.0, 512: 0.0, 1024: 0.02, 2048: 0.0}
    assert select_k(grid, tails, cost, 33.3, max_accuracy_cost=1e-3) == 512
    assert select_k(grid, tails, {k: 0.0 for k in grid}, 33.3, 1e-3) == 1024
    assert select_k(grid, tails, cost, 0.1, 1e-3) is None


# --- v2.0.0: the cluster-level exceedance bound ---------------------------------
from deadline_nms import cp_lower, cp_upper, exceedance_bound, min_inputs_for_pass, select_k_certified


def test_cp_limits_match_known_values():
    # zero misses on 48 inputs: 1 - 0.05**(1/48); R binom.test(5, 100) one-sided limits
    assert cp_upper(0, 48) == pytest.approx(1 - 0.05 ** (1 / 48), rel=1e-9)
    assert cp_upper(5, 100) == pytest.approx(0.10225, abs=1e-4)
    assert cp_lower(5, 100) == pytest.approx(0.019906, abs=1e-4)
    assert cp_lower(48, 48) == pytest.approx(0.05 ** (1 / 48), rel=1e-9)


def test_bound_does_not_degenerate_at_the_boundaries():
    """The bootstrap returns [0,0] for all-pass data; the bound must not."""
    r = exceedance_bound({i: [10.0, 11.0, 12.0] for i in range(48)}, 33.3)
    assert r["A_any_over"] == 0 and r["upper"] > 0.05 and r["verdict"] == "INCONCLUSIVE"
    r = exceedance_bound({i: [40.0] * 3 for i in range(137)}, 33.3)
    assert r["verdict"] == "FAIL" and r["lower"] > 0.95


def test_fifty_nine_inputs_are_needed_for_a_zero_miss_pass():
    assert min_inputs_for_pass(0.05) == 59
    assert exceedance_bound({i: [1.0] for i in range(59)}, 33.3)["verdict"] == "PASS"


def test_within_input_dependence_cannot_buy_a_pass():
    """Repeating requests on the same 48 inputs adds no independent information."""
    many = {i: [10.0] * 1000 for i in range(48)}
    assert exceedance_bound(many, 33.3)["verdict"] == "INCONCLUSIVE"


def test_select_k_certified_refuses_inconclusive_grid_points():
    lat = {k: {i: [k / 1000.0] * 3 for i in range(137)} for k in (512, 1024, 4096)}
    lat[8192] = {i: ([40.0] if i < 3 else [20.0]) * 3 for i in range(137)}   # 3 inputs over
    k, res = select_k_certified([512, 1024, 4096, 8192], lat, {k: 0.0 for k in lat}, 33.3, 1e-3)
    assert res[8192]["verdict"] == "INCONCLUSIVE" and k == 4096


def test_select_k_certified_stops_at_the_first_grid_point_that_does_not_pass():
    """Fixed sequence: a PASS after the first non-PASS is reported but never selected."""
    lat = {k: {i: [5.0] * 3 for i in range(137)} for k in (512, 1024, 4096)}
    lat[2048] = {i: ([40.0] if i < 3 else [20.0]) * 3 for i in range(137)}   # INCONCLUSIVE
    k, res = select_k_certified([512, 1024, 2048, 4096], lat, {k: 0.0 for k in lat}, 33.3, 1e-3)
    assert res[2048]["verdict"] == "INCONCLUSIVE" and res[4096]["verdict"] == "PASS"
    assert k == 1024


def test_select_k_certified_accuracy_filter_does_not_stop_the_sequence():
    """Small caps that cost too much accuracy are skipped, not treated as a failed test."""
    lat = {k: {i: [5.0] * 3 for i in range(137)} for k in (512, 1024, 2048)}
    cost = {512: 0.02, 1024: 0.0005, 2048: 0.0001}
    k, _ = select_k_certified([512, 1024, 2048], lat, cost, 33.3, 1e-3)
    assert k == 2048


def test_select_k_certified_stops_at_a_grid_point_without_data():
    """A declared K with no measurements is not a PASS; it ends the fixed sequence."""
    lat = {k: {i: [5.0] * 3 for i in range(137)} for k in (512, 1024, 4096)}   # 2048 missing
    k, res = select_k_certified([512, 1024, 2048, 4096], lat, {k: 0.0 for k in lat}, 33.3, 1e-3)
    assert res[2048]["verdict"] == "MISSING" and res[4096]["verdict"] == "PASS"
    assert k == 1024


# ---- v2.0.0 input validation (external review, F.4) -----------------------------------
import math as _math
import pytest as _pytest
from deadline_nms.bounds import exceedance_bound as _eb, min_inputs_for_pass as _mip, cp_upper as _cpu


def test_nan_latency_is_rejected_not_counted_as_a_pass():
    groups = {i: [1.0] for i in range(137)}
    groups[0] = [float("nan")]
    with _pytest.raises(ValueError):
        _eb(groups, 33.3)


def test_negative_and_infinite_latencies_are_rejected():
    for bad in (-1.0, float("inf"), float("-inf")):
        with _pytest.raises(ValueError):
            _eb({0: [bad], 1: [1.0]}, 33.3)


def test_invalid_policy_parameters_are_rejected():
    g = {i: [1.0] for i in range(137)}
    for kw in ({"alpha": 0.0}, {"alpha": 1.0}, {"rho": 0.0}, {"rho": 1.5}):
        with _pytest.raises(ValueError):
            _eb(g, 33.3, **kw)
    with _pytest.raises(ValueError):
        _eb(g, float("nan"))
    with _pytest.raises(ValueError):
        _mip(rho=0.0)


def test_min_inputs_closed_form():
    assert _mip(0.05, 0.05) == 59
    assert _cpu(0, 59) <= 0.05 < _cpu(0, 58)
    # a Bonferroni split over 11 grid points still passes with zero misses at C = 137
    assert _cpu(0, 137, 0.05 / 11) < 0.05
