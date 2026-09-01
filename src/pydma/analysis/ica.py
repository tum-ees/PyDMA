"""
Incremental Capacity Analysis (ICA) calculations.

This module provides functions for calculating ICA (dQ/dV) curves
from OCV data.
"""

import numpy as np

from pydma.analysis._grid import uniform_voltage_grid
from pydma.preprocessing.smoother import apply_filter, assure_non_zero_dv


def calculate_ica(
    soc: np.ndarray,
    voltage: np.ndarray,
    n_points: int | None = None,
    smooth: bool = True,
    smooth_window: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Incremental Capacity Analysis (ICA) curve.

    ICA is defined as dQ/dV, showing how capacity changes with voltage.
    This is useful for identifying phase transitions in electrodes.
    Peaks in ICA correspond to voltage plateaus.

    Parameters
    ----------
    soc : np.ndarray
        State of charge values (0-1 or capacity in Ah).
    voltage : np.ndarray
        Voltage values.
    n_points : int
        Number of points in output ICA curve.
    smooth : bool
        Whether to smooth the output ICA.
    smooth_window : int
        Window size for smoothing.

    Returns
    -------
    tuple
        (q_ica, ocv_ica, ica) where:
        - q_ica: Uniformly sampled charge/SOC grid
        - ocv_ica: Interpolated voltage on that grid
        - ica: Incremental capacity dQ/dV

    Notes
    -----
    This replicates MATLAB's calculate_ICA.m function.

    DIFFERENCE FROM MATLAB: Same algorithm, handles potential dV=0
    cases more explicitly.

    Examples
    --------
    >>> q_ica, ocv_ica, ica = calculate_ica(soc, voltage)
    """
    q_ica, ocv_ica, n_points = uniform_voltage_grid(soc, voltage, n_points)

    # Ensure non-zero dV for division
    ocv_ica = assure_non_zero_dv(ocv_ica)

    # Calculate ICA (dQ/dV)
    dq = np.diff(q_ica)
    dv = np.diff(ocv_ica)

    # DIFFERENCE FROM MATLAB: MATLAB computes point-by-point in a loop
    # We use vectorized operations for efficiency
    ica = np.zeros(n_points)
    ica[:-1] = np.where(np.abs(dv) > 1e-12, dq / dv, 0.0)
    ica[-1] = ica[-2]  # Set last value to previous

    if smooth:
        ica = apply_filter(ica, method="lowess", window=smooth_window)

    return q_ica, ocv_ica, ica


def precompute_ica(
    soc: np.ndarray,
    voltage: np.ndarray,
    q0: float = 1.0,
) -> np.ndarray:
    """
    Precompute measured ICA for use in optimization.

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
        ICA values (dQ/dV / Q0) on the input grid.

    Notes
    -----
    This replicates the precompute_measured_ICA.m function exactly:
        [~, ~, ICA_measured] = calculate_ICA(Q, OCV_interp);
        ICA_measured = ICA_measured / Q0;
        ICA_measured = apply_filter(ICA_measured, 'filtermethod', 'sgolay');

    The Q0 division normalizes ICA for comparison across cells
    with different capacities.

    Raises
    ------
    ValueError
        If the input is empty, or carries non-finite samples. Both would
        resolve to a curve of zeros rather than to a derivative.
    """
    soc = np.asarray(soc).flatten()
    voltage = np.asarray(voltage).flatten()

    n = len(soc)
    if n == 0:
        raise ValueError("precompute_ica requires at least one sample; got an empty array.")

    if not (np.all(np.isfinite(soc)) and np.all(np.isfinite(voltage))):
        raise ValueError(
            "precompute_ica requires finite soc/voltage; NaN/inf would silently "
            "resolve to 0.0 in the discrete derivative below."
        )

    # Ensure non-zero dV (MATLAB: assure_non_zero_dV is called in calculate_ICA)
    voltage = assure_non_zero_dv(voltage)

    ica = np.zeros(n)

    # Compute discrete derivative (MATLAB-compatible loop from calculate_ICA)
    for i in range(1, n):
        dq = soc[i] - soc[i - 1]
        dv = voltage[i] - voltage[i - 1]
        if abs(dv) > 1e-12:
            ica[i - 1] = dq / dv
        else:
            ica[i - 1] = 0.0

    # Set last value to second-to-last
    ica[-1] = ica[-2] if n > 1 else 0.0

    # Normalize by Q0 (MATLAB: ICA_measured = ICA_measured / Q0)
    ica = ica / q0

    # Apply smoothing (MATLAB: apply_filter with 'sgolay')
    ica = apply_filter(ica, method="sgolay", window=50, order=3)

    return ica
