"""
Electrode inhomogeneity model.

This module provides functions for modeling electrode inhomogeneity effects
on the OCV curve using a Gaussian distribution of local SOCs.

The inhomogeneity model represents non-uniform SOC distribution across
the electrode, which causes voltage averaging effects.
"""

import numpy as np
from typing import Tuple, Optional
from functools import lru_cache


# Fixed parameters for inhomogeneity model
# DIFFERENCE FROM MATLAB: These values are fixed as specified in requirements
_INHOM_N_POINTS = 61
_INHOM_X_MIN = 0.5
_INHOM_X_MAX = 1.5


@lru_cache(maxsize=32)
def _get_inhomogeneity_weights(sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Gaussian weights for inhomogeneity model.

    The weights represent the SOC distribution across the electrode.
    Higher sigma means more inhomogeneous electrode.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian (inhomogeneity magnitude).

    Returns
    -------
    tuple
        (x, weights) where x is the SOC multiplier grid and weights
        are the normalized Gaussian weights.

    Notes
    -----
    DIFFERENCE FROM MATLAB: Same algorithm, using lru_cache for performance.
    The MATLAB code uses persistent variables for caching.
    """
    x = np.linspace(_INHOM_X_MIN, _INHOM_X_MAX, _INHOM_N_POINTS)
    mu = 1.0

    z = (x - mu) / sigma
    weights = np.exp(-0.5 * z ** 2)
    weights = weights / weights.sum()  # Normalize

    return x, weights


def calculate_inhomogeneity(
    soc: np.ndarray,
    voltage: np.ndarray,
    inhom_sigma: float,
) -> np.ndarray:
    """
    Apply inhomogeneity model to an electrode potential curve.

    Models SOC distribution across electrode as Gaussian with 61 points
    in range [0.5, 1.5]. The observed voltage at a given mean SOC is
    the weighted average of voltages at distributed local SOCs.

    Parameters
    ----------
    soc : np.ndarray
        SOC values (0-1).
    voltage : np.ndarray
        Voltage values corresponding to SOC.
    inhom_sigma : float
        Inhomogeneity magnitude (standard deviation).
        0 means no inhomogeneity, higher values mean more spread.

    Returns
    -------
    np.ndarray
        Voltage array with inhomogeneity effects applied.

    Notes
    -----
    This implements the equation:
    U_observed(SOC) = sum(weights[i] * U(SOC * x[i]))

    where x is the distribution of local SOC multipliers and weights
    are Gaussian weights centered at x=1.

    DIFFERENCE FROM MATLAB: Same mathematical model. The MATLAB code uses
    griddedInterpolant; we use numpy.interp for simplicity and compatibility.

    The inhomogeneity is zero at 0% full cell SOC and maximum at 100% SOC.
    This is implicitly handled by the SOC * x multiplication.

    Examples
    --------
    >>> soc = np.linspace(0, 1, 100)
    >>> voltage = 0.1 + 0.2 * soc
    >>> voltage_inhom = calculate_inhomogeneity(soc, voltage, 0.1)
    """
    # No inhomogeneity case
    if inhom_sigma <= 0:
        return voltage.copy()

    soc = np.asarray(soc).flatten()
    voltage = np.asarray(voltage).flatten()

    if len(soc) != len(voltage):
        raise ValueError(
            f"soc and voltage must have same length, got {len(soc)} and {len(voltage)}"
        )

    # Get weights for this sigma value (cached)
    x, weights = _get_inhomogeneity_weights(float(inhom_sigma))

    # Build query grid: each row is SOC values, each column is a different x multiplier
    # Xq[i, j] = soc[i] * x[j]
    Xq = np.outer(soc, x)

    # Interpolate voltage at all query points
    # Use linear interpolation with edge handling
    soc_min, soc_max = soc.min(), soc.max()

    # Interpolate each column
    voltage_at_xq = np.zeros_like(Xq)
    for j in range(len(x)):
        voltage_at_xq[:, j] = np.interp(Xq[:, j], soc, voltage)

    # Handle out-of-bounds: set ALL to U(end) matching MATLAB behavior
    # MATLAB (calculate_inhomogeneity.m:88-89):
    #   outMask = (Xq < socMin) | (Xq > socMax);
    #   E_OC_dist(outMask) = U(end);
    out_of_bounds = (Xq < soc_min) | (Xq > soc_max)
    voltage_at_xq[out_of_bounds] = voltage[-1]

    # Weighted average across x dimension
    voltage_mean = voltage_at_xq @ weights

    return voltage_mean


def calculate_inhomogeneity_for_electrode(
    electrode,
    inhom_sigma: float,
):
    """
    Apply inhomogeneity to an ElectrodeOCP object.

    Parameters
    ----------
    electrode : ElectrodeOCP
        Electrode OCP object.
    inhom_sigma : float
        Inhomogeneity magnitude.

    Returns
    -------
    ElectrodeOCP
        New electrode with inhomogeneity applied.
    """
    from pydma.electrodes.electrode import ElectrodeOCP

    if inhom_sigma <= 0:
        return electrode.copy()

    voltage_inhom = calculate_inhomogeneity(
        electrode.soc, electrode.voltage, inhom_sigma
    )

    return ElectrodeOCP(
        soc=electrode.soc.copy(),
        voltage=voltage_inhom,
        name=f"{electrode.name} (σ={inhom_sigma:.3f})",
        electrode_type=electrode.electrode_type,
        capacity=electrode.capacity,
        is_smoothed=electrode.is_smoothed,
    )


def get_inhomogeneity_distribution(sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the SOC distribution used in inhomogeneity model.

    This is useful for visualization and debugging.

    Parameters
    ----------
    sigma : float
        Inhomogeneity magnitude.

    Returns
    -------
    tuple
        (x_multipliers, weights) where x_multipliers are the SOC scaling
        factors (centered at 1) and weights are the Gaussian weights.
    """
    return _get_inhomogeneity_weights(sigma)
