"""Objective functions for DMA optimization.

This module implements the cost functions used in the DMA optimization:
- fit_ocv: OCV fitting error
- fit_dva: DVA fitting error
- fit_ica: ICA fitting error
- combined_objective: Weighted combination of all objectives
- objective_with_penalty: Combined objective with penalty for constraint violations

The objective functions compare simulated half-cell curves (transformed to full-cell)
against measured full-cell data using MSE within a region of interest (ROI).
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from pydma.electrodes.electrode import ElectrodeOCP
from pydma.electrodes.blend import BlendElectrode
from pydma.electrodes.inhomogeneity import calculate_inhomogeneity
from pydma.analysis.dva import precompute_dva
from pydma.analysis.ica import precompute_ica
from pydma.analysis.degradation import calculate_mse
from pydma.preprocessing.smoother import apply_filter
from pydma.utils.roi import ROISpec, build_roi_mask


# Penalty scale factor (matches MATLAB's scale = 1e8)
PENALTY_SCALE = 1e8

@dataclass
class PreviousLAM:
    """Container for previous CU's LAM values used in penalty constraints.

    MATLAB Reference: objectiveWithPenalty in dma_core.m uses LAM_prev* variables
    to penalize physically implausible degradation between consecutive CUs.
    """
    lam_anode: Optional[float] = None
    lam_cathode: Optional[float] = None
    lam_anode_blend1: Optional[float] = None
    lam_anode_blend2: Optional[float] = None


@dataclass
class ReferenceData:
    """Reference data for degradation mode calculation.

    Contains initial capacities from the first CU (reference state).
    """
    capa_actual: float
    capa_anode_init: float
    capa_cathode_init: float
    capa_inventory_init: float
    gamma_an_blend2_init: float = 0.0
    gamma_ca_blend2_init: float = 0.0


@dataclass
class PenaltyConfig:
    """Configuration for penalty constraints.

    MATLAB Reference: These correspond to:
    - aAnodeLoss, aCathodeLoss, etc. (max gain, i.e., max capacity regeneration)
    - limitPositiveAnodeLoss, etc. (max loss per CU)
    """
    max_anode_gain: float = 0.01
    max_cathode_gain: float = 0.01
    max_anode_blend1_gain: float = 0.005
    max_anode_blend2_gain: float = 0.01
    max_anode_loss: float = 1.0
    max_cathode_loss: float = 1.0
    max_anode_blend1_loss: float = 1.0
    max_anode_blend2_loss: float = 1.0
    use_anode_blend: bool = False
    use_cathode_blend: bool = False


def apply_params_to_electrode(
    electrode: "ElectrodeOCP | BlendElectrode",
    alpha: float,
    beta: float,
    gamma_blend2: float = 0.0,
    inhom: float = 0.0,
    inhom_points: int = 61,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply transformation parameters to an electrode.

    Parameters
    ----------
    electrode : ElectrodeOCP or BlendElectrode
        The electrode to transform
    alpha : float
        SOC offset (shifts the curve horizontally)
    beta : float
        Capacity scaling factor (stretches/compresses the curve)
    gamma_blend2 : float, optional
        Blend weighting factor (only for BlendElectrode), by default 0.0
    inhom : float, optional
        Inhomogeneity factor (0 = no inhomogeneity), by default 0.0
    inhom_points : int, optional
        Number of points for inhomogeneity distribution, by default 61

    Returns
    -------
    tuple[NDArray, NDArray]
        Transformed (soc, voltage) arrays
    """
    # Handle blend electrode
    if isinstance(electrode, BlendElectrode):
        soc, voltage = electrode.get_blend_curve(gamma_blend2)
    else:
        soc = electrode.soc.copy()
        voltage = electrode.voltage.copy()

    # Apply inhomogeneity if enabled (MATLAB: calculate_inhomogeneity is applied BEFORE alpha-beta)
    # Note: calculate_inhomogeneity returns only voltage, not (soc, voltage)
    if abs(inhom) > 1e-10:
        voltage = calculate_inhomogeneity(soc, voltage, inhom)

    # Apply alpha-beta transformation (matches MATLAB): q = alpha * soc + beta
    soc_transformed = alpha * soc + beta

    return soc_transformed, voltage


def _interp1_linear_fill0(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    xq: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Linear interpolation with `0` outside bounds (MATLAB `interp1(..., 'linear', 0)`)."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    xq = np.asarray(xq).flatten()

    if len(x) == 0:
        return np.zeros_like(xq, dtype=np.float64)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    if len(x_unique) < 2:
        return np.zeros_like(xq, dtype=np.float64)

    return np.interp(xq, x_unique, y_unique, left=0.0, right=0.0)


def electrode_potential_on_q(
    electrode: "ElectrodeOCP | BlendElectrode",
    q: NDArray[np.floating],
    *,
    alpha: float,
    beta: float,
    gamma_blend2: float,
    inhom: float,
    inhom_points: int,
    is_blend: bool,
) -> NDArray[np.floating]:
    """Evaluate electrode potential on the full-cell Q/SOC grid (MATLAB-compatible)."""
    q = np.asarray(q, dtype=np.float64).flatten()

    soc_src, u_src = apply_params_to_electrode(
        electrode,
        alpha,
        beta,
        gamma_blend2=gamma_blend2 if is_blend else 0.0,
        inhom=inhom,
        inhom_points=inhom_points,
    )
    return _interp1_linear_fill0(soc_src, u_src, q)


def calculate_full_cell_ocv(
    anode_soc: NDArray[np.floating],
    anode_voltage: NDArray[np.floating],
    cathode_soc: NDArray[np.floating],
    cathode_voltage: NDArray[np.floating],
    n_points: int = 1000,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Calculate full-cell OCV from half-cell data.

    The full-cell voltage is V_fc = V_cathode - V_anode at each SOC.
    This function interpolates both electrodes to a common SOC grid
    and computes the full-cell voltage.

    Parameters
    ----------
    anode_soc : NDArray
        Anode SOC values
    anode_voltage : NDArray
        Anode voltage values
    cathode_soc : NDArray
        Cathode SOC values
    cathode_voltage : NDArray
        Cathode voltage values
    n_points : int, optional
        Number of points for the common SOC grid, by default 1000

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        (soc, full_cell_voltage, capacity) where capacity = soc * nominal_capacity
        The soc is the common grid, normalized to [0, 1]
    """
    # Find common SOC range (intersection of valid ranges)
    soc_min = max(np.min(anode_soc), np.min(cathode_soc))
    soc_max = min(np.max(anode_soc), np.max(cathode_soc))

    # Handle edge case where ranges don't overlap
    if soc_min >= soc_max:
        # Return minimal data
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 1.0]),
        )

    # Create common SOC grid
    soc_common = np.linspace(soc_min, soc_max, n_points)

    # Interpolate both electrodes to common grid
    anode_v_interp = np.interp(soc_common, anode_soc, anode_voltage)
    cathode_v_interp = np.interp(soc_common, cathode_soc, cathode_voltage)

    # Full-cell voltage is cathode - anode
    full_cell_voltage = cathode_v_interp - anode_v_interp

    # Normalize SOC to [0, 1] for the common range
    soc_normalized = (soc_common - soc_min) / (soc_max - soc_min)

    # Capacity is proportional to SOC (normalized)
    capacity = soc_normalized

    return soc_normalized, full_cell_voltage, capacity


def fit_ocv(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_voltage: NDArray[np.floating],
    q: NDArray[np.floating],
    roi_ocv_min: ROISpec,
    roi_ocv_max: ROISpec,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_points: int = 61,
) -> float:
    """Calculate OCV fitting error (MSE).

    Parameters
    ----------
    params : NDArray
        8-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca, inhom_an, inhom_ca]
    anode : ElectrodeOCP or BlendElectrode
        Anode electrode model
    cathode : ElectrodeOCP or BlendElectrode
        Cathode electrode model
    meas_voltage : NDArray
        Measured voltage values
    q : NDArray
        Full-cell SOC/Q grid (normalized).
    roi_ocv_min : float | tuple[float, float] | list[float] | np.ndarray
        Lower bound of OCV ROI (or first interval bounds).
    roi_ocv_max : float | tuple[float, float] | list[float] | np.ndarray
        Upper bound of OCV ROI (or second interval bounds).
    anode_is_blend : bool, optional
        Whether anode is a blend electrode
    cathode_is_blend : bool, optional
        Whether cathode is a blend electrode
    inhom_points : int, optional
        Number of points for inhomogeneity distribution

    Returns
    -------
    float
        Mean squared error (MSE) in the ROI, or large penalty if invalid
    """
    # Unpack parameters
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]

    try:
        q = np.asarray(q, dtype=np.float64).flatten()
        meas_voltage = np.asarray(meas_voltage, dtype=np.float64).flatten()

        anode_pot = electrode_potential_on_q(
            anode,
            q,
            alpha=alpha_an,
            beta=beta_an,
            gamma_blend2=gamma_blend2_an,
            inhom=inhom_an,
            inhom_points=inhom_points,
            is_blend=anode_is_blend,
        )

        cathode_pot = electrode_potential_on_q(
            cathode,
            q,
            alpha=alpha_ca,
            beta=beta_ca,
            gamma_blend2=gamma_blend2_ca,
            inhom=inhom_ca,
            inhom_points=inhom_points,
            is_blend=cathode_is_blend,
        )

        ocv_calc = cathode_pot - anode_pot

        # Build ROI mask on SOC/Q using shared ROI parsing+validation (utils.roi)
        roi_mask = build_roi_mask(q, roi_ocv_min, roi_ocv_max)

        if not np.any(roi_mask):
            return 1e6  # Large penalty if no valid points

        # MATLAB: Diff_OCV = sum((OCV_Calc - OCV_cell).^2 .* mask) / sum(mask)
        mse = calculate_mse(meas_voltage, ocv_calc, roi_mask)

        return float(mse)

    except Exception:
        return 1e6  # Large penalty for any errors


def _compute_discrete_dva(
    potential: NDArray[np.floating],
    q: NDArray[np.floating],
    q0: float,
) -> NDArray[np.floating]:
    """Compute discrete DVA (dU/dQ * Q0) matching MATLAB's fit_DVA.m.

    MATLAB computes DVA point-by-point in a loop:
        for idx = 2:nQ
            dU = potential(idx) - potential(idx-1)
            dQ = Q(idx) - Q(idx-1)
            dva(idx-1) = (dU / dQ) * Q0
        end

    This implementation uses vectorized numpy operations for efficiency.
    """
    n = len(q)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    du = np.diff(potential)
    dq = np.diff(q)
    dva = np.zeros(n, dtype=np.float64)
    dva[:-1] = np.where(np.abs(dq) > 1e-12, (du / dq) * q0, 0.0)
    dva[-1] = dva[-2]
    return dva


def fit_dva(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_dva: NDArray[np.floating],
    q: NDArray[np.floating],
    roi_mask: NDArray[np.bool_],
    q0: float,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_points: int = 61,
) -> float:
    """Calculate DVA fitting error (MSE) - MATLAB compatible.

    CRITICAL DIFFERENCE FROM ORIGINAL:
    MATLAB computes DVA SEPARATELY for anode and cathode, then subtracts:
        dva_anode(idx-1) = (dU_an / dQ) * Q0
        dva_cathode(idx-1) = (dU_cat / dQ) * Q0
        dva_sum = dva_cathode - dva_anode

    This is NOT the same as d(V_cat - V_an)/dQ due to smoothing!

    Parameters
    ----------
    params : NDArray
        8-element parameter vector
    anode : ElectrodeOCP or BlendElectrode
        Anode electrode model
    cathode : ElectrodeOCP or BlendElectrode
        Cathode electrode model
    meas_dva : NDArray
        Pre-computed measured DVA (dV/dQ * Q0)
    q : NDArray
        Full-cell SOC/Q grid (normalized).
    roi_mask : NDArray
        Pre-computed ROI mask on q.
    q0 : float
        SOC range scaling factor (MATLAB's Q0).
    anode_is_blend : bool, optional
        Whether anode is a blend electrode
    cathode_is_blend : bool, optional
        Whether cathode is a blend electrode
    inhom_points : int, optional
        Number of points for inhomogeneity distribution

    Returns
    -------
    float
        Mean squared error (MSE) of DVA in the ROI, normalized by ROI length
    """
    # Unpack parameters
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]

    try:
        q = np.asarray(q, dtype=np.float64).flatten()
        meas_dva = np.asarray(meas_dva, dtype=np.float64).flatten()
        roi_mask = np.asarray(roi_mask, dtype=bool).flatten()

        # Get electrode potentials on Q grid
        anode_pot = electrode_potential_on_q(
            anode,
            q,
            alpha=alpha_an,
            beta=beta_an,
            gamma_blend2=gamma_blend2_an,
            inhom=inhom_an,
            inhom_points=inhom_points,
            is_blend=anode_is_blend,
        )

        cathode_pot = electrode_potential_on_q(
            cathode,
            q,
            alpha=alpha_ca,
            beta=beta_ca,
            gamma_blend2=gamma_blend2_ca,
            inhom=inhom_ca,
            inhom_points=inhom_points,
            is_blend=cathode_is_blend,
        )

        # MATLAB-compatible: Compute DVA SEPARATELY for each electrode
        dva_anode = _compute_discrete_dva(anode_pot, q, q0)
        dva_cathode = _compute_discrete_dva(cathode_pot, q, q0)

        # DVA sum = cathode - anode (MATLAB: dva_sum = dva_cathode - dva_anode)
        dva_sum = dva_cathode - dva_anode

        # Apply smoothing (MATLAB: apply_filter with sgolay)
        dva_sum = apply_filter(dva_sum, method="sgolay", window=50, order=3)

        if not np.any(roi_mask):
            return 1e6

        # MATLAB: Diff_DVA = sum((dva_sum - dva_ocv).^2 .* mask) / sum(mask)
        mse = calculate_mse(meas_dva, dva_sum, roi_mask)

        return float(mse)

    except Exception:
        return 1e6


def _compute_discrete_ica(
    q: NDArray[np.floating],
    voltage: NDArray[np.floating],
    q0: float,
) -> NDArray[np.floating]:
    """Compute discrete ICA (dQ/dU / Q0) matching MATLAB's calculate_ICA + fit_ICA.

    MATLAB computes ICA point-by-point:
        for i = 2:numel(ICA)
            dU = OCV_ICA(i) - OCV_ICA(i-1)
            dQ = Q_ICA(i) - Q_ICA(i-1)
            ICA(i-1) = dQ / dU
        end
        ICA = ICA / Q0  % Normalization

    This implementation uses vectorized numpy operations for efficiency.
    """
    from pydma.preprocessing.smoother import assure_non_zero_dv

    n = len(q)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    # Ensure non-zero dV (MATLAB: assure_non_zero_dV)
    voltage = assure_non_zero_dv(voltage)

    dq = np.diff(q)
    dv = np.diff(voltage)
    ica = np.zeros(n, dtype=np.float64)
    ica[:-1] = np.where(np.abs(dv) > 1e-12, (dq / dv) / q0, 0.0)
    ica[-1] = ica[-2]
    return ica


def fit_ica(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_ica: NDArray[np.floating],
    q: NDArray[np.floating],
    roi_mask: NDArray[np.bool_],
    q0: float,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_points: int = 61,
) -> float:
    """Calculate ICA fitting error (MSE) - MATLAB compatible.

    MATLAB computes ICA from the modeled OCV (cathode - anode), then divides by Q0.

    Parameters
    ----------
    params : NDArray
        8-element parameter vector
    anode : ElectrodeOCP or BlendElectrode
        Anode electrode model
    cathode : ElectrodeOCP or BlendElectrode
        Cathode electrode model
    meas_ica : NDArray
        Pre-computed measured ICA (dQ/dV / Q0)
    q : NDArray
        Full-cell SOC/Q grid (normalized).
    roi_mask : NDArray
        Pre-computed ROI mask on q.
    q0 : float
        SOC range scaling factor (MATLAB's Q0).
    anode_is_blend : bool, optional
        Whether anode is a blend electrode
    cathode_is_blend : bool, optional
        Whether cathode is a blend electrode
    inhom_points : int, optional
        Number of points for inhomogeneity distribution

    Returns
    -------
    float
        Mean squared error (MSE) of ICA in the ROI, normalized by ROI length
    """
    # Unpack parameters
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]

    try:
        q = np.asarray(q, dtype=np.float64).flatten()
        meas_ica = np.asarray(meas_ica, dtype=np.float64).flatten()
        roi_mask = np.asarray(roi_mask, dtype=bool).flatten()

        anode_pot = electrode_potential_on_q(
            anode,
            q,
            alpha=alpha_an,
            beta=beta_an,
            gamma_blend2=gamma_blend2_an,
            inhom=inhom_an,
            inhom_points=inhom_points,
            is_blend=anode_is_blend,
        )

        cathode_pot = electrode_potential_on_q(
            cathode,
            q,
            alpha=alpha_ca,
            beta=beta_ca,
            gamma_blend2=gamma_blend2_ca,
            inhom=inhom_ca,
            inhom_points=inhom_points,
            is_blend=cathode_is_blend,
        )

        # MATLAB: OCV_sum = cathodePot - anodePot
        ocv_sum = cathode_pot - anode_pot

        # Compute ICA from modeled OCV
        ica_calc = _compute_discrete_ica(q, ocv_sum, q0)

        # Apply smoothing (MATLAB: apply_filter with sgolay)
        ica_calc = apply_filter(ica_calc, method="sgolay", window=50, order=3)

        if not np.any(roi_mask):
            return 1e6

        # MATLAB: Diff_ICA = sum((ICA_calc - ICA_OCV).^2 .* mask) / sum(mask)
        mse = calculate_mse(meas_ica, ica_calc, roi_mask)

        return float(mse)

    except Exception:
        return 1e6


def combined_objective(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_voltage: NDArray[np.floating],
    meas_dva: NDArray[np.floating],
    meas_ica: NDArray[np.floating],
    q: NDArray[np.floating],
    dva_roi_mask: NDArray[np.bool_],
    ica_roi_mask: NDArray[np.bool_],
    roi_ocv_min: ROISpec,
    roi_ocv_max: ROISpec,
    q0: float,
    w_ocv: float = 1.0,
    w_dva: float = 1.0,
    w_ica: float = 1.0,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_points: int = 61,
) -> float:
    """Combined objective function for DMA optimization.

    Computes weighted sum of OCV, DVA, and ICA fitting errors:
    cost = w_ocv * fit_ocv + w_dva * fit_dva + w_ica * fit_ica

    Parameters
    ----------
    params : NDArray
        8-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca, inhom_an, inhom_ca]
    anode : ElectrodeOCP or BlendElectrode
        Anode electrode model
    cathode : ElectrodeOCP or BlendElectrode
        Cathode electrode model
    meas_voltage : NDArray
        Measured voltage values
    meas_dva : NDArray
        Pre-computed measured DVA
    meas_ica : NDArray
        Pre-computed measured ICA
    q : NDArray
        Full-cell SOC/Q grid (normalized).
    dva_roi_mask : NDArray
        ROI mask for DVA term on q.
    ica_roi_mask : NDArray
        ROI mask for ICA term on q.
    roi_ocv_min : float | tuple[float, float] | list[float] | np.ndarray
        OCV ROI lower bound (or first interval bounds).
    roi_ocv_max : float | tuple[float, float] | list[float] | np.ndarray
        OCV ROI upper bound (or second interval bounds).
    q0 : float
        SOC range scaling factor (MATLAB's Q0).
    w_ocv : float, optional
        Weight for OCV fitting, by default 1.0
    w_dva : float, optional
        Weight for DVA fitting, by default 1.0
    w_ica : float, optional
        Weight for ICA fitting, by default 1.0
    anode_is_blend : bool, optional
        Whether anode is a blend electrode
    cathode_is_blend : bool, optional
        Whether cathode is a blend electrode
    inhom_points : int, optional
        Number of points for inhomogeneity distribution

    Returns
    -------
    float
        Weighted sum of MSE values
    """
    cost = 0.0

    # OCV contribution
    if w_ocv > 0:
        ocv_error = fit_ocv(
            params, anode, cathode,
            meas_voltage,
            q,
            roi_ocv_min,
            roi_ocv_max,
            anode_is_blend, cathode_is_blend,
            inhom_points,
        )
        cost += w_ocv * ocv_error

    # DVA contribution
    if w_dva > 0:
        dva_error = fit_dva(
            params, anode, cathode,
            meas_dva,
            q,
            dva_roi_mask,
            q0,
            anode_is_blend, cathode_is_blend,
            inhom_points,
        )
        cost += w_dva * dva_error

    # ICA contribution
    if w_ica > 0:
        ica_error = fit_ica(
            params, anode, cathode,
            meas_ica,
            q,
            ica_roi_mask,
            q0,
            anode_is_blend, cathode_is_blend,
            inhom_points,
        )
        cost += w_ica * ica_error

    return cost


def calculate_penalty(
    params: NDArray[np.floating],
    ref_data: ReferenceData,
    prev_lam: PreviousLAM,
    penalty_config: PenaltyConfig,
    fit_reverse: bool = False,
) -> float:
    """Calculate penalty for constraint violations.

    This implements MATLAB's objectiveWithPenalty penalty logic, which
    penalizes physically implausible degradation between consecutive CUs:
    - Capacity regeneration (LAM decrease) beyond max_*_gain
    - Excessive degradation (LAM increase) beyond max_*_loss

    Parameters
    ----------
    params : NDArray
        8-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca, inhom_an, inhom_ca]
    ref_data : ReferenceData
        Reference data containing initial capacities.
    prev_lam : PreviousLAM
        Previous CU's LAM values for comparison.
    penalty_config : PenaltyConfig
        Configuration for penalty constraints (max gain/loss values).
    fit_reverse : bool, optional
        Whether fitting is in reverse order.

    Returns
    -------
    float
        Total penalty value (0 if all constraints satisfied).

    Notes
    -----
    MATLAB Reference: objectiveWithPenalty in dma_core.m (lines 477-527)

    The penalty logic is:
        neg = (LAM_prev - max_gain) - LAM_current
        pos = LAM_current - (LAM_prev + max_loss)
        penalty = scale * max(neg, 0)^2 + scale * max(pos, 0)^2

    Where:
        - neg > 0 means capacity regenerated too much (LAM decreased)
        - pos > 0 means capacity degraded too much (LAM increased)
        - scale = 1e8 (PENALTY_SCALE constant)
    """
    from pydma.analysis.degradation import calculate_degradation_modes

    # Ensure params has 8 elements
    params = np.asarray(params).flatten()
    if len(params) < 8:
        full_params = np.zeros(8)
        full_params[:len(params)] = params
        params = full_params

    # Enforce zero blend fraction if blend not used
    if not penalty_config.use_anode_blend:
        params = params.copy()
        params[4] = 0.0
    if not penalty_config.use_cathode_blend:
        params = params.copy()
        params[5] = 0.0

    # Calculate current LAM values
    deg_result = calculate_degradation_modes(
        params,
        ref_data.capa_actual,
        ref_data.capa_anode_init,
        ref_data.capa_cathode_init,
        ref_data.capa_inventory_init,
        ref_data.gamma_an_blend2_init,
        ref_data.gamma_ca_blend2_init,
        fit_reverse,
    )

    lam_current_an = deg_result.lam_anode
    lam_current_cath = deg_result.lam_cathode
    lam_current_an_blend1 = deg_result.lam_anode_blend1
    lam_current_an_blend2 = deg_result.lam_anode_blend2

    penalty = 0.0

    # Anode penalty (MATLAB: lines 492-498)
    if prev_lam.lam_anode is not None:
        neg = (prev_lam.lam_anode - penalty_config.max_anode_gain) - lam_current_an
        pos = lam_current_an - (prev_lam.lam_anode + penalty_config.max_anode_loss)
        penalty += PENALTY_SCALE * max(neg, 0.0) ** 2 + PENALTY_SCALE * max(pos, 0.0) ** 2

    # Cathode penalty (MATLAB: lines 500-506)
    if prev_lam.lam_cathode is not None:
        neg = (prev_lam.lam_cathode - penalty_config.max_cathode_gain) - lam_current_cath
        pos = lam_current_cath - (prev_lam.lam_cathode + penalty_config.max_cathode_loss)
        penalty += PENALTY_SCALE * max(neg, 0.0) ** 2 + PENALTY_SCALE * max(pos, 0.0) ** 2

    # Anode blend1 penalty (MATLAB: lines 508-515)
    if prev_lam.lam_anode_blend1 is not None:
        neg = (
            prev_lam.lam_anode_blend1 - penalty_config.max_anode_blend1_gain
        ) - lam_current_an_blend1
        pos = lam_current_an_blend1 - (
            prev_lam.lam_anode_blend1 + penalty_config.max_anode_blend1_loss
        )
        penalty += PENALTY_SCALE * max(neg, 0.0) ** 2 + PENALTY_SCALE * max(pos, 0.0) ** 2

    # Anode blend2 penalty (MATLAB: lines 517-524)
    if prev_lam.lam_anode_blend2 is not None:
        neg = (
            prev_lam.lam_anode_blend2 - penalty_config.max_anode_blend2_gain
        ) - lam_current_an_blend2
        pos = lam_current_an_blend2 - (
            prev_lam.lam_anode_blend2 + penalty_config.max_anode_blend2_loss
        )
        penalty += PENALTY_SCALE * max(neg, 0.0) ** 2 + PENALTY_SCALE * max(pos, 0.0) ** 2

    return penalty


def objective_with_penalty(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_voltage: NDArray[np.floating],
    meas_dva: NDArray[np.floating],
    meas_ica: NDArray[np.floating],
    q: NDArray[np.floating],
    dva_roi_mask: NDArray[np.bool_],
    ica_roi_mask: NDArray[np.bool_],
    roi_ocv_min: ROISpec,
    roi_ocv_max: ROISpec,
    q0: float,
    w_ocv: float = 1.0,
    w_dva: float = 1.0,
    w_ica: float = 1.0,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_points: int = 61,
    ref_data: Optional[ReferenceData] = None,
    prev_lam: Optional[PreviousLAM] = None,
    penalty_config: Optional[PenaltyConfig] = None,
    fit_reverse: bool = False,
) -> float:
    """Combined objective function with penalty constraints for DMA optimization.

    This function combines the base fitting objective (OCV + DVA + ICA) with
    penalty constraints that enforce physically plausible degradation evolution.

    MATLAB Reference: objectiveWithPenalty in dma_core.m

    Parameters
    ----------
    params : NDArray
        8-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca, inhom_an, inhom_ca]
    anode : ElectrodeOCP or BlendElectrode
        Anode electrode model
    cathode : ElectrodeOCP or BlendElectrode
        Cathode electrode model
    meas_voltage : NDArray
        Measured voltage values
    meas_dva : NDArray
        Pre-computed measured DVA
    meas_ica : NDArray
        Pre-computed measured ICA
    q : NDArray
        Full-cell SOC/Q grid (normalized).
    dva_roi_mask : NDArray
        ROI mask for DVA term on q.
    ica_roi_mask : NDArray
        ROI mask for ICA term on q.
    roi_ocv_min : float | tuple[float, float] | list[float] | np.ndarray
        OCV ROI lower bound (or first interval bounds).
    roi_ocv_max : float | tuple[float, float] | list[float] | np.ndarray
        OCV ROI upper bound (or second interval bounds).
    q0 : float
        SOC range scaling factor (MATLAB's Q0).
    w_ocv : float, optional
        Weight for OCV fitting, by default 1.0
    w_dva : float, optional
        Weight for DVA fitting, by default 1.0
    w_ica : float, optional
        Weight for ICA fitting, by default 1.0
    anode_is_blend : bool, optional
        Whether anode is a blend electrode
    cathode_is_blend : bool, optional
        Whether cathode is a blend electrode
    inhom_points : int, optional
        Number of points for inhomogeneity distribution
    ref_data : ReferenceData, optional
        Reference data for degradation calculation (required for penalty).
    prev_lam : PreviousLAM, optional
        Previous CU's LAM values (None for first CU = no penalty).
    penalty_config : PenaltyConfig, optional
        Penalty constraint configuration.
    fit_reverse : bool, optional
        Whether fitting is in reverse order.

    Returns
    -------
    float
        Total cost = base_objective + penalty

    Notes
    -----
    The penalty is only applied if:
    1. ref_data is provided (needed for LAM calculation)
    2. prev_lam has at least one non-None value (not first CU)
    3. penalty_config is provided

    For the first CU (prev_lam is None or all values are None), no penalty is applied.
    """
    # Compute base objective
    base_cost = combined_objective(
        params, anode, cathode,
        meas_voltage, meas_dva, meas_ica,
        q, dva_roi_mask, ica_roi_mask,
        roi_ocv_min, roi_ocv_max, q0,
        w_ocv, w_dva, w_ica,
        anode_is_blend, cathode_is_blend,
        inhom_points,
    )

    # Add penalty if configured and not first CU
    penalty = 0.0
    if ref_data is not None and prev_lam is not None and penalty_config is not None:
        # Check if we have any previous LAM values (not first CU)
        has_prev = (
            prev_lam.lam_anode is not None or
            prev_lam.lam_cathode is not None or
            prev_lam.lam_anode_blend1 is not None or
            prev_lam.lam_anode_blend2 is not None
        )
        if has_prev:
            penalty = calculate_penalty(
                params, ref_data, prev_lam, penalty_config, fit_reverse
            )

    return base_cost + penalty
