"""Shared ROI parsing, validation, and mask helpers.

This module provides a single, MATLAB-aligned ROI implementation used by OCV,
DVA, and ICA code paths. It keeps ROI behavior consistent across optimization
and reporting. The semantics mirror MATLAB's ``build_ROI_mask.m``:
single interval for scalar bounds, OR-combined two intervals for split bounds.

Supported ROI input formats
---------------------------
1. Single region (scalar bounds):
   ``roi_min=0.1, roi_max=0.9`` -> one interval ``[0.1, 0.9]``
2. Split regions (2-value sequences):
   ``roi_min=[0.05, 0.15], roi_max=(0.70, 0.95)`` ->
   two intervals ``[0.05, 0.15]`` OR ``[0.70, 0.95]``

Validation performed
--------------------
- Inputs must be numeric and finite
- Scalar mode requires ``roi_min <= roi_max``
- Split mode requires two values per interval and ordered ``[lower, upper]``
- Mixed modes (scalar + split) are rejected
- All bounds must lie in ``[0.0, 1.0]`` (SOC fraction, not percent)
- Overlapping split intervals warn: ``roi_min`` carries the first interval and
  ``roi_max`` the second, so a swapped pair silently covers the whole range

Main helpers
------------
- ``normalize_roi``: parse ROI into one or two validated intervals
- ``build_roi_mask``: build boolean mask on q/SOC using normalized intervals
- ``get_roi_outer_bounds``: get outer ``(min, max)`` bounds across intervals

Examples
--------
>>> import numpy as np
>>> q = np.linspace(0.0, 1.0, 11)
>>> build_roi_mask(q, 0.1, 0.9).sum()
9
>>> build_roi_mask(q, [0.0, 0.2], (0.8, 1.0)).sum()
6
>>> get_roi_outer_bounds([0.05, 0.15], [0.70, 0.95])
(0.05, 0.95)
"""

import warnings
from typing import TypeAlias

import numpy as np

# TypeAlias instead of a PEP 695 `type` statement: the alias form parses on
# every supported interpreter, and the `type` statement is what pinned the
# whole package to Python 3.12.
ROISpec: TypeAlias = float | tuple[float, float] | list[float] | np.ndarray
ROIIntervals: TypeAlias = tuple[tuple[float, float], ...]


def _try_parse_scalar(value: object) -> float | None:
    """Return scalar ROI value as float, otherwise None."""
    arr = np.asarray(value)
    if arr.ndim != 0:
        return None

    scalar = arr.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("ROI bounds must be numeric, not boolean")

    if isinstance(scalar, (int, float, np.integer, np.floating)):
        scalar_f = float(scalar)
        if not np.isfinite(scalar_f):
            raise ValueError("ROI bounds must be finite")
        return scalar_f

    return None


def _parse_interval(value: object, arg_name: str) -> tuple[float, float]:
    """Parse a 2-value interval-like ROI input."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg_name} must be scalar or a 2-value sequence") from exc

    if arr.size != 2:
        raise ValueError(f"{arg_name} must be scalar or a 2-value sequence")

    lo = float(arr[0])
    hi = float(arr[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{arg_name} must contain finite values")
    if lo > hi:
        raise ValueError(f"{arg_name} must be ordered as [lower, upper]")

    return lo, hi


def _check_roi_domain(
    intervals: ROIIntervals,
    *,
    min_name: str,
    max_name: str,
) -> None:
    """Reject ROI bounds outside the normalized SOC domain."""
    for lo, hi in intervals:
        if not 0.0 <= lo <= 1.0 or not 0.0 <= hi <= 1.0:
            raise ValueError(
                f"{min_name}/{max_name} must lie within [0.0, 1.0], got [{lo}, {hi}]. "
                "ROI bounds are SOC fractions (0-1), not percent."
            )


def _warn_on_overlap(
    intervals: ROIIntervals,
    *,
    min_name: str,
    max_name: str,
) -> None:
    """Warn when the two intervals of a split ROI overlap."""
    if len(intervals) != 2:
        return

    (first_lo, first_hi), (second_lo, second_hi) = intervals
    if first_lo <= second_hi and second_lo <= first_hi:
        warnings.warn(
            f"Split ROI intervals overlap: [{first_lo}, {first_hi}] and "
            f"[{second_lo}, {second_hi}]. {min_name} carries the first interval and "
            f"{max_name} the second, so a swapped pair covers the whole range.",
            UserWarning,
            stacklevel=2,
        )


def normalize_roi(
    roi_min: ROISpec,
    roi_max: ROISpec,
    *,
    min_name: str = "roi_min",
    max_name: str = "roi_max",
) -> ROIIntervals:
    """Normalize ROI input into one or two validated [lower, upper] intervals."""
    roi_min_scalar = _try_parse_scalar(roi_min)
    roi_max_scalar = _try_parse_scalar(roi_max)

    if roi_min_scalar is not None and roi_max_scalar is not None:
        if roi_min_scalar > roi_max_scalar:
            raise ValueError(f"For scalar ROI, {min_name} must be <= {max_name}")
        intervals: ROIIntervals = ((roi_min_scalar, roi_max_scalar),)
    elif (roi_min_scalar is None) != (roi_max_scalar is None):
        raise ValueError(
            f"{min_name} and {max_name} must both be scalars or both be 2-value sequences"
        )
    else:
        intervals = (_parse_interval(roi_min, min_name), _parse_interval(roi_max, max_name))

    _check_roi_domain(intervals, min_name=min_name, max_name=max_name)
    _warn_on_overlap(intervals, min_name=min_name, max_name=max_name)

    return intervals


def build_roi_mask(
    q: np.ndarray,
    roi_min: ROISpec,
    roi_max: ROISpec,
    *,
    min_name: str = "roi_min",
    max_name: str = "roi_max",
) -> np.ndarray:
    """Build boolean mask from scalar ROI or two split ROI intervals."""
    q = np.asarray(q).flatten()
    intervals = normalize_roi(roi_min, roi_max, min_name=min_name, max_name=max_name)

    mask = np.zeros(q.shape, dtype=bool)
    for lo, hi in intervals:
        mask |= (q >= lo) & (q <= hi)
    return mask


def get_roi_outer_bounds(
    roi_min: ROISpec,
    roi_max: ROISpec,
    *,
    min_name: str = "roi_min",
    max_name: str = "roi_max",
) -> tuple[float, float]:
    """Return outer [min, max] bounds across normalized ROI interval(s)."""
    intervals = normalize_roi(roi_min, roi_max, min_name=min_name, max_name=max_name)
    lower = min(lo for lo, _ in intervals)
    upper = max(hi for _, hi in intervals)
    return lower, upper
