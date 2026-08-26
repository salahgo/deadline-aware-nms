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
