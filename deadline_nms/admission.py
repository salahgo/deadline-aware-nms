"""Bounded candidate admission: the cap that makes suppression work independent of input.

An NMS-based detector does an amount of work set by the number of candidates that
survive its confidence threshold, and that number is an input-controlled quantity. A
candidate-inflation attack drives it up; the deadline is what pays. Admitting at most
``k`` candidates bounds the pairwise comparisons the suppression stage can be made to
perform, without retraining or touching the weights.
"""
from __future__ import annotations

import numpy as np

__all__ = ["admit_topk", "pairwise_bound"]


def admit_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` admitted candidates, smallest under the key ``(-s_j, j)``.

    Returns every index when ``k <= 0`` or ``k >= len(scores)``, so the cap can be
    switched off by configuration without a second code path.

    WHY THE TIE RULE IS PART OF THE SPECIFICATION, NOT AN IMPLEMENTATION DETAIL.
    ``np.argpartition`` uses introselect, which is not stable. When more candidates tie
    at the cutoff score than there are places left, it returns an ARBITRARY subset of
    them -- and sorting whatever it returned afterwards settles the order of the
    survivors while doing nothing about which ones survived. Two runs, two library
    versions or two machines can then admit different boxes and emit different
    detections while both "apply Top-K". Exact ties are reachable: fp16 and quantized
    score tensors have few enough distinct values that the cutoff is often shared.

    So membership is defined lexicographically rather than left to the partition:
    everything strictly above the cutoff score is admitted, and the remaining places
    are filled from the candidates equal to the cutoff in ascending index order. The
    partition is still what finds the cutoff, so the cut stays O(n) -- it compares
    scores, but it adds no pairwise IoU comparison, which is the term the bound below
    is about.
    """
    n = int(scores.shape[0])
    if k <= 0 or n <= k:
        return np.arange(n)

    part = np.argpartition(-scores, k - 1)[:k]
    cutoff = scores[part].min()

    above = np.flatnonzero(scores > cutoff)          # unambiguously admitted
    if above.size >= k:                              # cutoff not actually reached
        return np.sort(above[:k])
    ties = np.flatnonzero(scores == cutoff)          # ascending index by construction
    return np.sort(np.concatenate([above, ties[: k - above.size]]))


def pairwise_bound(k: int) -> int:
    """Most IoU comparisons a serial greedy NMS can perform on ``k`` admitted boxes.

    ``k(k-1)/2``: the first selection compares against at most ``k-1`` remaining boxes,
    the second against ``k-2``, and so on. This is an ALGORITHMIC work bound and not a
    hardware worst-case execution time -- establishing the latter needs architectural
    analysis of the specific target, and the backend measurements behind this artifact
    show how far apart the two can be.
    """
    k = int(k)
    return 0 if k < 2 else k * (k - 1) // 2
