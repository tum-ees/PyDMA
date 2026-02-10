"""
Differential Voltage Analysis (DVA) calculations.

This module provides functions for calculating DVA (dV/dQ) curves
from OCV data.
"""

import numpy as np
from typing import Tuple
from pydma.preprocessing.smoother import apply_filter


def calculate_dva(
    soc: np.ndarray,
    voltage: np.ndarray,
    n_points: int | None = None,
    smooth: bool = True,
    smooth_window: int = 30,
 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Differential Voltage Analysis (DVA) curve.

    DVA is defined as dV/dQ, showing how voltage changes with capacity.
    This is useful for identifying phase transitions in electrodes.

    Parameters
    ----------
    soc : np.ndarray
        State of charge values (0-1 or capacity in Ah).
    voltage : np.ndarray
        Voltage values.
    n_points : int
        Number of points in output DVA curve.
    smooth : bool
        Whether to smooth the output DVA.
    smooth_window : int
        Window size for smoothing.

    Returns
    -------
    tuple
        (q_dva, ocv_dva, dva) where:
        - q_dva: Uniformly sampled charge/SOC grid
        - ocv_dva: Interpolated voltage on that grid
        - dva: Differential voltage dV/dQ

    Notes
    -----
    This replicates MATLAB's calculate_DVA.m function.

    DIFFERENCE FROM MATLAB: Uses numpy gradient for cleaner derivative
    calculation, but produces equivalent results.

    Examples
    --------
    >>> q_dva, ocv_dva, dva = calculate_dva(soc, voltage)
    """
    soc = np.asarray(soc).flatten()
    voltage = np.asarray(voltage).flatten()

    # Remove NaN values
    valid = ~(np.isnan(soc) | np.isnan(voltage))
    soc = soc[valid]
    voltage = voltage[valid]

    # Ensure unique SOC values for interpolation
    soc_unique, unique_idx = np.unique(soc, return_index=True)
    voltage_unique = voltage[unique_idx]

    # Determine number of points (default: match input unique SOC length)
    if n_points is None:
        n_points = len(soc_unique)

    # Create uniform grid
    q_dva = np.linspace(soc_unique.min(), soc_unique.max(), n_points)

    # Interpolate voltage onto uniform grid
    ocv_dva = np.interp(q_dva, soc_unique, voltage_unique)

    # Calculate DVA (dV/dQ)
    dq = np.diff(q_dva)
    dv = np.diff(ocv_dva)

    # DIFFERENCE FROM MATLAB: MATLAB computes point-by-point in a loop
    # We use vectorized operations for efficiency
    dva = np.zeros(n_points)
    dva[:-1] = dv / dq
    dva[-1] = dva[-2]  # Set last value to previous (matching MATLAB)

    if smooth:
        dva = apply_filter(dva, method="lowess", window=smooth_window)

    return q_dva, ocv_dva, dva


def precompute_dva(
    soc: np.ndarray,
    voltage: np.ndarray,
    q0: float = 1.0,
) -> np.ndarray:
    """
    Precompute measured DVA for use in optimization.

    This version is optimized for repeated calls during fitting.

    Parameters
    ----------
    soc : np.ndarray
        SOC grid (assumed uniform 0-1).
    voltage : np.ndarray
        Voltage values on the SOC grid.
    q0 : float
        Reference capacity for normalization.

    Returns
    -------
    np.ndarray
        DVA values (dV/dQ * Q0) on the input grid.

    Notes
    -----
    This replicates the precompute_measured_DVA.m function exactly:
        for idx = 2:nQ
            dU = OCV(idx) - OCV(idx-1);
            dQ = Q(idx) - Q(idx-1);
            DVA_measured(idx-1) = (dU / dQ) * Q0;
        end
        DVA_measured = apply_filter(DVA_measured, 'filtermethod', 'sgolay');

    The Q0 multiplication normalizes DVA for comparison across cells
    with different capacities.
    """
    soc = np.asarray(soc).flatten()
    voltage = np.asarray(voltage).flatten()

    n = len(soc)
    dva = np.zeros(n)

    # Compute discrete derivative (MATLAB-compatible loop)
    for i in range(1, n):
        dv = voltage[i] - voltage[i - 1]
        dq = soc[i] - soc[i - 1]
        if abs(dq) > 1e-12:
            dva[i - 1] = (dv / dq) * q0
        else:
            dva[i - 1] = 0.0

    # Set last value to second-to-last (MATLAB doesn't explicitly do this, but it's implied)
    dva[-1] = dva[-2] if n > 1 else 0.0

    # Apply smoothing (MATLAB: apply_filter with 'sgolay', default window=50, order=3)
    dva = apply_filter(dva, method="sgolay", window=50, order=3)

    return dva
