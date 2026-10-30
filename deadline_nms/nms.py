"""Greedy IoU non-maximum suppression, as the serving path runs it.

This is the reference the suppression-scope timings were taken on. It is a plain
single-threaded loop on purpose: it is what the deployed worker executes, and the
point of the backend comparison is that a faster implementation lowers the constant
while leaving the attacked candidate count free.

Since 2.1.0 suppression visits candidates in the same total order that
``admission.admit_topk`` uses to decide membership, ``(-score, index)``, so a tied
score never makes the kept set depend on the sorting algorithm. That shared order,
together with identical overlap arithmetic and the same output limit, is what makes
the prefix property hold: if NMS on the ``K`` admitted candidates already keeps
``limit`` detections, they are exactly the first ``limit`` detections of uncapped NMS.
"""
from __future__ import annotations

import numpy as np

__all__ = ["greedy_nms", "class_offset"]


def _as_scores(scores: np.ndarray, n: int) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.shape != (n,):
        raise ValueError("scores must have shape (n,)")
    # Negating unsigned integers wraps and negating the signed minimum overflows,
    # either of which silently reverses the order; booleans cannot be negated.
    if scores.dtype == bool or not (np.issubdtype(scores.dtype, np.integer)
                                    or np.issubdtype(scores.dtype, np.floating)):
        raise TypeError("scores must be a real integer or floating-point array")
    scores = scores.astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, *,
               classes: np.ndarray | None = None, limit: int = 0,
               order_ids: np.ndarray | None = None) -> list:
    """Kept indices for xyxy ``boxes``, in the order ``(-score, order_id)``.

    ``classes``: optional integer class per box; boxes of different classes never
    suppress each other. This compares classes explicitly, so unlike
    ``class_offset`` it makes no assumption about the coordinate range.

    ``limit``: stop once this many boxes are kept (0 = no limit), as a detector with a
    maximum detection count does. Stopping early returns the same first ``limit``
    indices as running to completion and truncating.

    ``order_ids``: optional distinct non-negative integers that break exact score ties.
    Pass the candidates' indices in the full candidate array (for example the output
    of ``admit_topk``) when ``boxes`` is a subset, so that ties are broken the same way
    with and without the cap. Defaults to the position in ``boxes``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    n = int(boxes.shape[0]) if boxes.ndim == 2 else -1
    if boxes.shape != (n, 4):
        raise ValueError("boxes must have shape (n, 4)")
    scores = _as_scores(scores, n)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError("boxes must be finite xyxy with x2 >= x1 and y2 >= y1")
    if not 0.0 <= float(iou_thres) <= 1.0:
        raise ValueError("iou_thres must lie in [0, 1]")
    if int(limit) != limit or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    ids = np.arange(n) if order_ids is None else np.asarray(order_ids)
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) \
            or np.unique(ids).size != n:
        raise ValueError("order_ids must be n distinct non-negative integers")
    if classes is not None:
        classes = np.asarray(classes)
        if classes.shape != (n,) or not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("classes must be an integer array of shape (n,)")

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.lexsort((ids, -scores))         # one total order, as in admit_topk
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if limit and len(keep) == limit:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        if classes is not None:
            iou = np.where(classes[rest] == classes[i], iou, 0.0)
        order = rest[iou <= iou_thres]
    return keep


def class_offset(boxes: np.ndarray, classes: np.ndarray, stride: float) -> np.ndarray:
    """Displace boxes per class so different classes never suppress each other.

    Adding ``class_id * stride`` to every coordinate puts each class in its own region
    of the plane, so one greedy pass behaves as one pass per class -- but only if
    ``stride`` exceeds the span of the coordinates. Detector boxes can extend past the
    input canvas (on a 1280-px input we observed -4.5 to 1296.9), so a stride equal to
    the canvas size plus one is not enough: two boxes of different classes at opposite
    corners then overlap after the shift and one suppresses the other. Since 2.1.0 a
    stride that does not exceed the span raises; ``greedy_nms(..., classes=...)``
    avoids the assumption altogether.
    """
    boxes = np.asarray(boxes)
    if boxes.size:
        span = float(boxes.max() - boxes.min())
        if not stride > span:
            raise ValueError(f"stride {stride} does not exceed the coordinate span {span}")
    return boxes + np.asarray(classes)[:, None].astype(boxes.dtype) * stride
