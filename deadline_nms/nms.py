"""Greedy IoU non-maximum suppression, as the serving path runs it.

This is the reference the suppression-scope timings were taken on. It is a plain
single-threaded loop on purpose: it is what the deployed worker executes, and the
point of the backend comparison is that a faster implementation lowers the constant
while leaving the attacked candidate count free.
"""
from __future__ import annotations

import numpy as np

__all__ = ["greedy_nms", "class_offset"]


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list:
    """Kept indices for xyxy ``boxes``, ordered by descending score."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def class_offset(boxes: np.ndarray, classes: np.ndarray, stride: float) -> np.ndarray:
    """Displace boxes per class so different classes never suppress each other.

    Adding ``class_id * stride`` to every coordinate puts each class in its own region
    of the plane, so one greedy pass behaves as one pass per class. ``stride`` must
    exceed the coordinate range.
    """
    return boxes + classes[:, None].astype(boxes.dtype) * stride
