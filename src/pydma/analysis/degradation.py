"""
Degradation mode calculations.

This module provides functions for calculating battery degradation modes:
- LLI: Loss of Lithium Inventory
- LAM_an: Loss of Active Material at Anode
- LAM_ca: Loss of Active Material at Cathode
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DegradationResult:
    """Container for degradation calculation results."""
    lam_anode: float
    lam_cathode: float
    lli: float
    lam_anode_blend1: float = 0.0
    lam_anode_blend2: float = 0.0
    lam_cathode_blend1: float = 0.0
    lam_cathode_blend2: float = 0.0


def calculate_degradation_modes(
    params: np.ndarray,
    capa_actual: float,
    capa_anode_init: float,
    capa_cathode_init: float,
    capa_inventory_init: float,
    gamma_an_blend2_init: float = 0.0,
    gamma_ca_blend2_init: float = 0.0,
    fit_reverse: bool = False,
) -> DegradationResult:
    """
    Calculate degradation modes from fitted parameters.

    Degradation modes quantify how much each component has degraded
    relative to the reference (first CU) state.

    Parameters
    ----------
    params : np.ndarray
        8-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca,
         gamma_blend2_an, gamma_blend2_ca, inhom_an, inhom_ca]
    capa_actual : float
        Actual cell capacity for this CU.
    capa_anode_init : float
        Initial anode capacity (from reference CU).
    capa_cathode_init : float
        Initial cathode capacity (from reference CU).
    capa_inventory_init : float
        Initial charge carrier inventory (from reference CU).
    gamma_an_blend2_init : float
        Initial anode blend2 fraction (from reference CU).
    gamma_ca_blend2_init : float
        Initial cathode blend2 fraction (from reference CU).
    fit_reverse : bool
        Whether fitting was done in reverse order.
        If True, adjusts degradation calculation for reverse reference.

    Returns
    -------
    DegradationResult
        Dataclass containing all degradation modes.

    Notes
    -----
    This replicates MATLAB's calculate_degradation_modes.m function.

    The degradation modes are calculated as:
    - capa_anode = alpha_an * capa_actual
    - capa_cathode = alpha_ca * capa_actual
    - LAM_an = (capa_anode_init - capa_anode) / capa_anode_init
    - LAM_ca = (capa_cathode_init - capa_cathode) / capa_cathode_init
    - capa_inventory = (alpha_ca + beta_ca - beta_an) * capa_actual
    - LLI = (capa_inventory_init - capa_inventory) / capa_inventory_init

    Examples
    --------
    >>> params = np.array([1.1, -0.05, 1.2, -0.1, 0.2, 0, 0.03, 0])
    >>> result = calculate_degradation_modes(params, 4.5, 5.0, 5.5, 5.2)
    >>> print(f"LAM_an: {result.lam_anode:.2%}")
    """
    params = np.asarray(params).flatten()

    # Ensure 8 elements (pad with zeros if needed)
    if len(params) < 8:
        full_params = np.zeros(8)
        full_params[:len(params)] = params
        params = full_params

    alpha_anode = params[0]
    beta_anode = params[1]
    alpha_cathode = params[2]
    beta_cathode = params[3]
    gamma_an_blend2 = params[4]
    gamma_ca_blend2 = params[5]

    # Compute current capacities
    capa_anode = alpha_anode * capa_actual
    capa_cathode = alpha_cathode * capa_actual

    # Initial sub-capacities based on reference gamma values
    capa_anode_blend2_init = capa_anode_init * gamma_an_blend2_init
    capa_anode_blend1_init = capa_anode_init * (1 - gamma_an_blend2_init)
    capa_cathode_blend2_init = capa_cathode_init * gamma_ca_blend2_init
    capa_cathode_blend1_init = capa_cathode_init * (1 - gamma_ca_blend2_init)

    # Current sub-capacities using optimized gamma values
    capa_anode_blend2 = capa_anode * gamma_an_blend2
    capa_anode_blend1 = capa_anode * (1 - gamma_an_blend2)
    capa_cathode_blend2 = capa_cathode * gamma_ca_blend2
    capa_cathode_blend1 = capa_cathode * (1 - gamma_ca_blend2)

    # Calculate loss for blend components separately
    lam_anode_blend2 = _safe_loss(capa_anode_blend2_init, capa_anode_blend2)
    lam_anode_blend1 = _safe_loss(capa_anode_blend1_init, capa_anode_blend1)
    lam_cathode_blend2 = _safe_loss(capa_cathode_blend2_init, capa_cathode_blend2)
    lam_cathode_blend1 = _safe_loss(capa_cathode_blend1_init, capa_cathode_blend1)

    # Overall anode and cathode degradation
    lam_anode = (capa_anode_init - capa_anode) / capa_anode_init
    lam_cathode = (capa_cathode_init - capa_cathode) / capa_cathode_init

    # Inventory loss calculation
    capa_inventory = (alpha_cathode + beta_cathode - beta_anode) * capa_actual
    lli = (capa_inventory_init - capa_inventory) / capa_inventory_init

    # Handle reverse fitting
    # DIFFERENCE FROM MATLAB: Same logic, but uses Python syntax
    if fit_reverse:
        lam_cathode = -lam_cathode * capa_cathode_init / capa_cathode if capa_cathode != 0 else 0
        lam_anode = -lam_anode * capa_anode_init / capa_anode if capa_anode != 0 else 0

        if capa_anode_blend1 != 0:
            lam_anode_blend1 = -lam_anode_blend1 * capa_anode_blend1_init / capa_anode_blend1
        if capa_anode_blend2 != 0:
            lam_anode_blend2 = -lam_anode_blend2 * capa_anode_blend2_init / capa_anode_blend2
        if capa_cathode_blend1 != 0:
            lam_cathode_blend1 = (
                -lam_cathode_blend1 * capa_cathode_blend1_init / capa_cathode_blend1
            )
        if capa_cathode_blend2 != 0:
            lam_cathode_blend2 = (
                -lam_cathode_blend2 * capa_cathode_blend2_init / capa_cathode_blend2
            )

    return DegradationResult(
        lam_anode=lam_anode,
        lam_cathode=lam_cathode,
        lli=lli,
        lam_anode_blend1=lam_anode_blend1,
        lam_anode_blend2=lam_anode_blend2,
        lam_cathode_blend1=lam_cathode_blend1,
        lam_cathode_blend2=lam_cathode_blend2,
    )


def _safe_loss(init_val: float, current_val: float) -> float:
    """
    Calculate loss avoiding divide-by-zero.

    Parameters
    ----------
    init_val : float
        Initial value.
    current_val : float
        Current value.

    Returns
    -------
    float
        Relative loss (init - current) / init, or 0 if init is 0.
    """
    if init_val == 0:
        return 0.0
    return (init_val - current_val) / init_val


def calculate_mse(
    measured: np.ndarray,
    calculated: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Masked mean squared error between two curves.

    Used by OCV/DVA/ICA fitting (objectives.py) and post-fit metrics (analyzer.py).
    For RMSE, take np.sqrt() of the result.
    """
    measured = np.asarray(measured).flatten()
    calculated = np.asarray(calculated).flatten()

    if mask is None:
        mask = np.ones(len(measured), dtype=bool)
    else:
        mask = np.asarray(mask).flatten()

    if mask.sum() == 0:
        return 0.0

    # MSE only over masked (ROI) points
    diff_sq = (calculated - measured) ** 2
    diff_sq[~mask] = 0.0
    return float(diff_sq.sum() / mask.sum())


def calculate_capacity_metrics(
    params: np.ndarray,
    capa_actual: float,
) -> Dict[str, float]:
    """
    Calculate capacity-related metrics from fitted parameters.

    Parameters
    ----------
    params : np.ndarray
        8-element parameter vector.
    capa_actual : float
        Actual cell capacity.

    Returns
    -------
    dict
        Dictionary containing:
        - capa_anode: Anode capacity
        - capa_cathode: Cathode capacity
        - capa_inventory: Charge carrier inventory
        - gamma_an_blend2: Anode blend2 fraction
        - gamma_ca_blend2: Cathode blend2 fraction
    """
    params = np.asarray(params).flatten()

    if len(params) < 8:
        full_params = np.zeros(8)
        full_params[:len(params)] = params
        params = full_params

    alpha_anode = params[0]
    beta_anode = params[1]
    alpha_cathode = params[2]
    beta_cathode = params[3]
    gamma_an_blend2 = params[4]
    gamma_ca_blend2 = params[5]

    capa_anode = alpha_anode * capa_actual
    capa_cathode = alpha_cathode * capa_actual
    capa_inventory = (alpha_cathode + beta_cathode - beta_anode) * capa_actual

    return {
        "capa_anode": capa_anode,
        "capa_cathode": capa_cathode,
        "capa_inventory": capa_inventory,
        "gamma_an_blend2": gamma_an_blend2,
        "gamma_ca_blend2": gamma_ca_blend2,
    }
