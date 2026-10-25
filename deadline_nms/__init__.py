"""Bounded candidate admission for deadline-critical NMS-based object detection."""
from .admission import admit_topk, pairwise_bound
from .bounds import cp_lower, cp_upper, exceedance_bound, min_inputs_for_pass, verdict
from .metrics import deadline_miss_ratio, quantile, session_bootstrap_ci
from .nms import class_offset, greedy_nms
from .selection import select_k, select_k_certified, upper_confidence_limit

__version__ = "2.0.0"
__all__ = [
    "admit_topk", "pairwise_bound",
    "greedy_nms", "class_offset",
    "quantile", "deadline_miss_ratio", "session_bootstrap_ci",
    "select_k", "select_k_certified", "upper_confidence_limit",
    "cp_upper", "cp_lower", "exceedance_bound", "verdict", "min_inputs_for_pass",
]
