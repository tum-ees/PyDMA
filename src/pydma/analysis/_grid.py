"""
Shared grid-preparation helper for DVA/ICA calculations.
"""

import warnings

import numpy as np


def uniform_voltage_grid(
    soc: np.ndarray,
    voltage: np.ndarray,
    n_points: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Resample (soc, voltage) onto a uniform, duplicate-free SOC grid.

    Shared preprocessing for `calculate_dva`/`calculate_ica`: drops NaN
    pairs, collapses duplicate SOC values (`np.unique` keeps the first
    occurrence), then interpolates voltage onto `n_points` uniformly
    spaced SOC samples.

    Parameters
    ----------
    soc : np.ndarray
        State of charge / capacity values.
    voltage : np.ndarray
        Voltage values aligned with `soc`.
    n_points : int, optional
        Number of points in the output grid. Defaults to the number of
        unique, finite SOC values.

    Returns
    -------
    tuple
        (q_grid, ocv_grid, n_points).

    Raises
    ------
    ValueError
        If fewer than 2 unique, finite SOC values remain, or if an
        explicitly requested `n_points` is below 2.
    """
    soc = np.asarray(soc).flatten()
    voltage = np.asarray(voltage).flatten()

    # Remove NaN values
    valid = ~(np.isnan(soc) | np.isnan(voltage))
    soc = soc[valid]
    voltage = voltage[valid]

    # Ensure unique SOC values for interpolation
    soc_unique, unique_idx = np.unique(soc, return_index=True)
    n_dropped = soc.size - soc_unique.size
    if n_dropped > 0:
        warnings.warn(f"Dropped {n_dropped} duplicate SOC point(s) before grid interpolation.")
    voltage_unique = voltage[unique_idx]

    if soc_unique.size < 2:
        raise ValueError(
            f"Need at least 2 unique, finite SOC points to build a grid, got {soc_unique.size}."
        )

    # Determine number of points (default: match input unique SOC length)
    if n_points is None:
        n_points = len(soc_unique)
    elif n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}.")

    # Create uniform grid
    q_grid = np.linspace(soc_unique.min(), soc_unique.max(), n_points)

    # Interpolate voltage onto uniform grid
    ocv_grid = np.interp(q_grid, soc_unique, voltage_unique)

    return q_grid, ocv_grid, n_points
