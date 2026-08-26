"""The mechanism end to end, on synthetic candidates. No dataset, no weights, no GPU.

Runs in a few seconds and shows the three things the defense rests on:

  1. suppression cost is set by the candidate count, not by the image;
  2. a global cap bounds it, and the bound is the one Eq. (6) states;
  3. the cap is chosen from the deadline rather than from a round number.

    python examples/demo.py
"""
from __future__ import annotations

import time

import numpy as np

from deadline_nms import (admit_topk, deadline_miss_ratio, greedy_nms,
                          pairwise_bound, quantile, select_k)

BUDGET_MS = 33.3          # a 30 Hz frame period, the screen for the suppression stage
IOU = 0.65


def candidates(n: int, rng: np.random.Generator, size: int = 1280):
    """``n`` scattered boxes, the geometry an inflation attack produces.

    Deliberately spread rather than clustered: candidates that do not overlap survive
    each other's suppression passes, so they are the expensive case, and the realised
    cost of NMS is a property of that geometry as much as of the count.
    """
    cx = rng.uniform(0, size, n)
    cy = rng.uniform(0, size, n)
    w = rng.uniform(8, 64, n)
    h = rng.uniform(8, 64, n)
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, rng.uniform(0.30, 0.99, n)


def suppression_ms(boxes, scores, k):
    keep = admit_topk(scores, k)
    t0 = time.perf_counter()
    greedy_nms(boxes[keep], scores[keep], IOU)
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    rng = np.random.default_rng(1337)

    print("candidate count drives suppression cost (uncapped)")
    print(f"{'candidates':>12}  {'NMS p95 (ms)':>13}  {'vs budget':>10}")
    for n in (250, 2_000, 10_000, 40_000):
        boxes, scores = candidates(n, rng)
        p95 = quantile([suppression_ms(boxes, scores, 0) for _ in range(5)])
        print(f"{n:>12,}  {p95:>13.2f}  {'MISS' if p95 > BUDGET_MS else 'ok':>10}")

    print("\nthe cap bounds it, whatever the input does")
    boxes, scores = candidates(40_000, rng)
    grid = (256, 512, 1024, 2048, 4096, 8192)
    sessions, cost = {}, {}
    print(f"{'K':>6}  {'NMS p95 (ms)':>13}  {'pair bound':>12}")
    for k in grid:
        sessions[k] = [suppression_ms(boxes, scores, k) for _ in range(5)]
        cost[k] = 0.0 if k >= 1024 else 0.02      # stand-in: accuracy falls away below
        print(f"{k:>6}  {quantile(sessions[k]):>13.2f}  {pairwise_bound(k):>12,}")

    chosen = select_k(grid, sessions, cost, BUDGET_MS, max_accuracy_cost=1e-3)
    print(f"\nlargest K meeting the {BUDGET_MS} ms screen and the accuracy limit: {chosen}")

    served = [suppression_ms(boxes, scores, 1024) for _ in range(40)]
    print(f"at K=1024: p95 {quantile(served):.2f} ms, "
          f"deadline-miss ratio {deadline_miss_ratio(served, BUDGET_MS):.3f}")

    print("\nNOTE: synthetic candidates on this machine. Absolute milliseconds are a "
          "property of the box geometry and the host, not portable numbers; the "
          "portable result is that the admitted count, and so the pairwise work, is "
          "bounded.")


if __name__ == "__main__":
    main()
