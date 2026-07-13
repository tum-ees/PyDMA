"""
Electrode inhomogeneity model.

This module provides functions for modeling electrode inhomogeneity effects
on the OCV curve using a Gaussian distribution of local SOCs.

The inhomogeneity model represents non-uniform SOC distribution across
the electrode, which causes voltage averaging effects.
"""

from functools import lru_cache

import numpy as np

# Fixed parameters for inhomogeneity model
# DIFFERENCE FROM MATLAB: These values are fixed as specified in requirements
_INHOM_N_POINTS = 61
_INHOM_X_MIN = 0.5
_INHOM_X_MAX = 1.5


@lru_cache(maxsize=32)
def _get_inhomogeneity_weights(sigma: float) -> tuple[np.ndarray, np.ndarray]:
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
    weights = np.exp(-0.5 * z**2)
    weights = weights / weights.sum()  # Normalize

    return x, weights


def calculate_inhomogeneity(
    soc: np.ndarray,
    voltage: np.ndarray,
    inhom_sigma: float,
    inhom_offset_fraction: float = 0.0,
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
    inhom_offset_fraction : float, optional
        Fraction of the maximum spread already present at SOC = 0.
        ``0.0`` reproduces the original SOC-dependent behavior, while ``1.0``
        makes the spread SOC-independent. Matches MATLAB's
        ``inhomOffsetFraction`` argument.

    Returns
    -------
    np.ndarray
        Voltage array with inhomogeneity effects applied.

    Notes
    -----
    This implements the MATLAB equation:
    alpha_eff = inhom_offset_fraction + (1 - inhom_offset_fraction) * SOC
    U_observed(SOC) = sum(weights[i] * U(SOC + alpha_eff * (x[i] - 1)))

    where x is the distribution of local SOC multipliers and weights
    are Gaussian weights centered at x=1.

    DIFFERENCE FROM MATLAB: Same mathematical model. The MATLAB code uses
    griddedInterpolant; we use numpy.interp for simplicity and compatibility.

    With ``inhom_offset_fraction = 0`` the inhomogeneity is zero at 0% full
    cell SOC and maximum at 100% SOC. Positive offsets allow a finite
    fraction of the maximum spread already at SOC = 0.

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

    x_dev = x - 1.0
    alpha_eff = inhom_offset_fraction + (1.0 - inhom_offset_fraction) * soc
    x_query = soc[:, None] + alpha_eff[:, None] * x_dev[None, :]

    # Interpolate voltage at all query points
    # MATLAB uses griddedInterpolant(..., 'linear', 'nearest'), so clamp
    # out-of-range queries to the nearest boundary value.
    voltage_at_xq = np.zeros_like(x_query)
    for j in range(len(x)):
        voltage_at_xq[:, j] = np.interp(
            x_query[:, j],
            soc,
            voltage,
            left=voltage[0],
            right=voltage[-1],
        )

    # Weighted average across x dimension
    voltage_mean: np.ndarray = voltage_at_xq @ weights

    return voltage_mean


def calculate_inhomogeneity_for_electrode(
    electrode,
    inhom_sigma: float,
    inhom_offset_fraction: float = 0.0,
):
    """
    Apply inhomogeneity to an ElectrodeOCP object.

    Parameters
    ----------
    electrode : ElectrodeOCP
        Electrode OCP object.
    inhom_sigma : float
        Inhomogeneity magnitude.
    inhom_offset_fraction : float, optional
        Fraction of the maximum spread already present at SOC = 0.

    Returns
    -------
    ElectrodeOCP
        New electrode with inhomogeneity applied.
    """
    from pydma.electrodes.electrode import ElectrodeOCP

    if inhom_sigma <= 0:
        return electrode.copy()

    voltage_inhom = calculate_inhomogeneity(
        electrode.soc,
        electrode.voltage,
        inhom_sigma,
        inhom_offset_fraction=inhom_offset_fraction,
    )

    return ElectrodeOCP(
        soc=electrode.soc.copy(),
        voltage=voltage_inhom,
        name=f"{electrode.name} (σ={inhom_sigma:.3f})",
        electrode_type=electrode.electrode_type,
        capacity=electrode.capacity,
        is_smoothed=electrode.is_smoothed,
    )


def get_inhomogeneity_distribution(sigma: float) -> tuple[np.ndarray, np.ndarray]:
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
