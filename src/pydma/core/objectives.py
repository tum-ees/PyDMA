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

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pydma.analysis.degradation import calculate_mse
from pydma.electrodes.blend import BlendElectrode
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.electrodes.inhomogeneity import calculate_inhomogeneity
from pydma.preprocessing.smoother import apply_filter
from pydma.utils.roi import ROISpec, build_roi_mask

# Penalty scale factor (matches MATLAB's scale = 1e8)
PENALTY_SCALE = 1e8

# The fit functions are called once per population member per generation, so
# the diagnostic for a suppressed evaluation is emitted only for the first one.
_PENALTY_WARNING_ISSUED = False


def _warn_once_on_penalty(fit_name: str, reason: str) -> None:
    """Report the first evaluation that falls back to the 1e6 penalty."""
    global _PENALTY_WARNING_ISSUED
    if _PENALTY_WARNING_ISSUED:
        return

    _PENALTY_WARNING_ISSUED = True
    warnings.warn(
        f"{fit_name} returned the 1e6 penalty ({reason}). "
        "Further penalty fallbacks in this process stay silent.",
        UserWarning,
        stacklevel=3,
    )


@dataclass
class PreviousLAM:
    """Container for previous CU's LAM values used in penalty constraints.

    MATLAB Reference: objectiveWithPenalty in dma_core.m uses LAM_prev* variables
    to penalize physically implausible degradation between consecutive CUs.
    """

    lam_anode: float | None = None
    lam_cathode: float | None = None
    lam_anode_blend1: float | None = None
    lam_anode_blend2: float | None = None


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
    *,
    gamma_blend2: float = 0.0,
    inhom: float = 0.0,
    inhom_offset: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply transformation parameters to an electrode.

    Parameters
    ----------
    electrode : ElectrodeOCP or BlendElectrode
        The electrode to transform
    alpha : float
        SOC scaling factor (stretches/compresses the curve)
    beta : float
        SOC offset (shifts the curve horizontally)
    gamma_blend2 : float, optional
        Blend weighting factor (only for BlendElectrode), by default 0.0
    inhom : float, optional
        Inhomogeneity factor (0 = no inhomogeneity), by default 0.0
    inhom_offset : float, optional
        Fraction of the maximum inhomogeneity already present at SOC = 0.

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
        voltage = calculate_inhomogeneity(
            soc,
            voltage,
            inhom,
            inhom_offset_fraction=inhom_offset,
        )

    # Apply alpha-beta transformation (matches MATLAB): q = alpha * soc + beta
    soc_transformed = alpha * soc + beta

    return soc_transformed, voltage


def interp_linear_fill0(
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

    return np.asarray(np.interp(xq, x_unique, y_unique, left=0.0, right=0.0))


def electrode_potential_on_q(
    electrode: "ElectrodeOCP | BlendElectrode",
    q: NDArray[np.floating],
    *,
    alpha: float,
    beta: float,
    gamma_blend2: float,
    inhom: float,
    inhom_offset: float,
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
        inhom_offset=inhom_offset,
    )
    return interp_linear_fill0(soc_src, u_src, q)


def _precompute_electrode_potentials(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    q: NDArray[np.floating],
    *,
    anode_is_blend: bool,
    cathode_is_blend: bool,
    inhom_anode_offset: float,
    inhom_cathode_offset: float,
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    """Both electrode potentials on the full-cell grid for one parameter vector.

    The OCV, DVA and ICA terms each need the same two curves, so
    :func:`combined_objective` evaluates them here once and hands them to all
    three. A parameter vector the electrode transformation cannot handle
    yields ``(None, None)``, which sends every term back to its own guarded
    evaluation and therefore to its own penalty value.
    """
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]

    try:
        anode_pot = electrode_potential_on_q(
            anode,
            q,
            alpha=alpha_an,
            beta=beta_an,
            gamma_blend2=gamma_blend2_an,
            inhom=inhom_an,
            inhom_offset=inhom_anode_offset,
            is_blend=anode_is_blend,
        )

        cathode_pot = electrode_potential_on_q(
            cathode,
            q,
            alpha=alpha_ca,
            beta=beta_ca,
            gamma_blend2=gamma_blend2_ca,
            inhom=inhom_ca,
            inhom_offset=inhom_cathode_offset,
            is_blend=cathode_is_blend,
        )
    except (ValueError, IndexError, FloatingPointError):
        return None, None

    return anode_pot, cathode_pot


def fit_ocv(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_voltage: NDArray[np.floating],
    q: NDArray[np.floating],
    *,
    roi_ocv_min: ROISpec,
    roi_ocv_max: ROISpec,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_anode_offset: float = 0.0,
    inhom_cathode_offset: float = 0.0,
    precomputed_anode_pot: NDArray[np.floating] | None = None,
    precomputed_cathode_pot: NDArray[np.floating] | None = None,
    r_offset_sign: float = 1.0,
    pocv_current_a: float = 0.0,
) -> float:
    """Calculate OCV fitting error (MSE).

    Parameters
    ----------
    params : NDArray
        9-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca,
         inhom_an, inhom_ca, r_offset]
        A vector of 8 elements is read as a zero resistance offset.
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
    precomputed_anode_pot : NDArray, optional
        Anode potential on ``q`` for this parameter vector. ``None`` evaluates
        it here, which is what a caller outside :func:`combined_objective`
        wants.
    precomputed_cathode_pot : NDArray, optional
        Cathode potential on ``q`` for this parameter vector.
    r_offset_sign : float, optional
        Sign the resistance term enters with: ``+1`` for a charge pOCV, which
        lies above the OCV, ``-1`` for a discharge pOCV, which lies below it.
    pocv_current_a : float, optional
        Magnitude of the pOCV current in A. The model assumes it is constant
        over the CC pOCV, so ``R * I`` is one number for the whole curve.

    Returns
    -------
    float
        Mean squared error (MSE) in the ROI, or large penalty if invalid

    Raises
    ------
    ValueError
        If the ROI selects no point of the SOC grid. That is a configuration
        error, not a parameter that happens to be unusable.
    """
    # Unpack parameters
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]
    r_offset = params[8] if len(params) > 8 else 0.0

    # Build ROI mask on SOC/Q using shared ROI parsing+validation (utils.roi).
    # It only depends on the grid and the configuration, so it is checked
    # outside the parameter-error guard below.
    roi_mask = build_roi_mask(q, roi_ocv_min, roi_ocv_max)
    if not np.any(roi_mask):
        raise ValueError(
            f"OCV ROI selects no point of the SOC grid: roi_ocv_min={roi_ocv_min!r}, "
            f"roi_ocv_max={roi_ocv_max!r}."
        )

    try:
        q = np.asarray(q, dtype=np.float64).flatten()
        meas_voltage = np.asarray(meas_voltage, dtype=np.float64).flatten()

        if precomputed_anode_pot is None:
            anode_pot = electrode_potential_on_q(
                anode,
                q,
                alpha=alpha_an,
                beta=beta_an,
                gamma_blend2=gamma_blend2_an,
                inhom=inhom_an,
                inhom_offset=inhom_anode_offset,
                is_blend=anode_is_blend,
            )
        else:
            anode_pot = precomputed_anode_pot

        if precomputed_cathode_pot is None:
            cathode_pot = electrode_potential_on_q(
                cathode,
                q,
                alpha=alpha_ca,
                beta=beta_ca,
                gamma_blend2=gamma_blend2_ca,
                inhom=inhom_ca,
                inhom_offset=inhom_cathode_offset,
                is_blend=cathode_is_blend,
            )
        else:
            cathode_pot = precomputed_cathode_pot

        ocv_calc = cathode_pot - anode_pot

        # Series resistance: one constant over the whole curve, so it lifts the
        # level without touching the shape. A pOCV measured while charging sits
        # above the OCV and one measured while discharging sits below it, which
        # is what r_offset_sign carries. fit_dva and fit_ica differentiate the
        # curve and are blind to a constant by construction, so the term stays
        # out of them. The zero case is skipped rather than added, which keeps
        # a fit without the correction bit-identical.
        if r_offset != 0.0:
            ocv_calc = ocv_calc + r_offset_sign * r_offset * pocv_current_a

        # MATLAB: Diff_OCV = sum((OCV_Calc - OCV_cell).^2 .* mask) / sum(mask)
        mse = calculate_mse(meas_voltage, ocv_calc, roi_mask)

    except (ValueError, IndexError, FloatingPointError) as exc:
        _warn_once_on_penalty("fit_ocv", repr(exc))
        return 1e6

    if not np.isfinite(mse):
        _warn_once_on_penalty("fit_ocv", "non-finite objective")
        return 1e6

    return float(mse)


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
    *,
    roi_mask: NDArray[np.bool_],
    q0: float,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_anode_offset: float = 0.0,
    inhom_cathode_offset: float = 0.0,
    precomputed_anode_pot: NDArray[np.floating] | None = None,
    precomputed_cathode_pot: NDArray[np.floating] | None = None,
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
        9-element parameter vector. The resistance slot is not read here: a
        constant voltage term drops out of the derivative.
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
    precomputed_anode_pot : NDArray, optional
        Anode potential on ``q`` for this parameter vector. ``None`` evaluates
        it here.
    precomputed_cathode_pot : NDArray, optional
        Cathode potential on ``q`` for this parameter vector.

    Returns
    -------
    float
        Mean squared error (MSE) of DVA in the ROI, normalized by ROI length

    Raises
    ------
    ValueError
        If the ROI mask selects no point of the SOC grid. That is a
        configuration error, not a parameter that happens to be unusable.
    """
    # Unpack parameters
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]

    # The mask comes from the configuration, so it is checked outside the
    # parameter-error guard below.
    roi_mask = np.asarray(roi_mask, dtype=bool).flatten()
    if not np.any(roi_mask):
        raise ValueError(
            f"DVA ROI mask selects none of its {roi_mask.size} grid points: "
            "check roi_dva_min / roi_dva_max."
        )

    try:
        q = np.asarray(q, dtype=np.float64).flatten()
        meas_dva = np.asarray(meas_dva, dtype=np.float64).flatten()

        # Get electrode potentials on Q grid
        if precomputed_anode_pot is None:
            anode_pot = electrode_potential_on_q(
                anode,
                q,
                alpha=alpha_an,
                beta=beta_an,
                gamma_blend2=gamma_blend2_an,
                inhom=inhom_an,
                inhom_offset=inhom_anode_offset,
                is_blend=anode_is_blend,
            )
        else:
            anode_pot = precomputed_anode_pot

        if precomputed_cathode_pot is None:
            cathode_pot = electrode_potential_on_q(
                cathode,
                q,
                alpha=alpha_ca,
                beta=beta_ca,
                gamma_blend2=gamma_blend2_ca,
                inhom=inhom_ca,
                inhom_offset=inhom_cathode_offset,
                is_blend=cathode_is_blend,
            )
        else:
            cathode_pot = precomputed_cathode_pot

        # MATLAB-compatible: Compute DVA SEPARATELY for each electrode
        dva_anode = _compute_discrete_dva(anode_pot, q, q0)
        dva_cathode = _compute_discrete_dva(cathode_pot, q, q0)

        # DVA sum = cathode - anode (MATLAB: dva_sum = dva_cathode - dva_anode)
        dva_sum = dva_cathode - dva_anode

        # Apply smoothing (MATLAB: apply_filter with sgolay)
        dva_sum = apply_filter(dva_sum, method="sgolay", window=50, order=3)

        # MATLAB: Diff_DVA = sum((dva_sum - dva_ocv).^2 .* mask) / sum(mask)
        mse = calculate_mse(meas_dva, dva_sum, roi_mask)

    except (ValueError, IndexError, FloatingPointError) as exc:
        _warn_once_on_penalty("fit_dva", repr(exc))
        return 1e6

    if not np.isfinite(mse):
        _warn_once_on_penalty("fit_dva", "non-finite objective")
        return 1e6

    return float(mse)


def _compute_discrete_ica(
    q: NDArray[np.floating],
    voltage: NDArray[np.floating],
    q0: float,
) -> NDArray[np.floating]:
    """Compute discrete ICA (dQ/dU / Q0) from MATLAB's calculate_ICA + fit_ICA.

    MATLAB computes ICA point-by-point:
        for i = 2:numel(ICA)
            dU = OCV_ICA(i) - OCV_ICA(i-1)
            dQ = Q_ICA(i) - Q_ICA(i-1)
            ICA(i-1) = dQ / dU
        end
        ICA = ICA / Q0  % Normalization

    This implementation is vectorized, and departs from MATLAB on one point:
    the plateau handling below reports a zero-dU step as 0, whereas MATLAB
    spreads it with ``assure_non_zero_dV`` first. The measured side keeps
    MATLAB's treatment (see :func:`pydma.analysis.ica.precompute_ica`), so the
    two sides of the ICA term are not symmetric here. Aligning the measured
    side changes what the ICA term measures and is deliberately not done.
    """
    n = len(q)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    dq = np.diff(q)
    dv = np.diff(voltage)
    ica = np.zeros(n, dtype=np.float64)
    # A voltage plateau carries no ICA information, so it is reported as 0.
    # Spreading the plateau by an artificial 1e-10 dV would turn it into a
    # 1e7 spike instead. The division over the plateau is discarded by the
    # guard, so its floating-point flags are expected.
    with np.errstate(divide="ignore", invalid="ignore"):
        ica[:-1] = np.where(np.abs(dv) > 1e-12, (dq / dv) / q0, 0.0)
    ica[-1] = ica[-2]
    return ica


def fit_ica(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_ica: NDArray[np.floating],
    q: NDArray[np.floating],
    *,
    roi_mask: NDArray[np.bool_],
    q0: float,
    anode_is_blend: bool = False,
    cathode_is_blend: bool = False,
    inhom_anode_offset: float = 0.0,
    inhom_cathode_offset: float = 0.0,
    precomputed_anode_pot: NDArray[np.floating] | None = None,
    precomputed_cathode_pot: NDArray[np.floating] | None = None,
) -> float:
    """Calculate ICA fitting error (MSE) - MATLAB compatible.

    MATLAB computes ICA from the modeled OCV (cathode - anode), then divides by Q0.

    Parameters
    ----------
    params : NDArray
        9-element parameter vector. The resistance slot is not read here: a
        constant voltage term drops out of the derivative.
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
    precomputed_anode_pot : NDArray, optional
        Anode potential on ``q`` for this parameter vector. ``None`` evaluates
        it here.
    precomputed_cathode_pot : NDArray, optional
        Cathode potential on ``q`` for this parameter vector.

    Returns
    -------
    float
        Mean squared error (MSE) of ICA in the ROI, normalized by ROI length

    Raises
    ------
    ValueError
        If the ROI mask selects no point of the SOC grid. That is a
        configuration error, not a parameter that happens to be unusable.
    """
    # Unpack parameters
    alpha_an, beta_an, alpha_ca, beta_ca = params[:4]
    gamma_blend2_an, gamma_blend2_ca = params[4:6]
    inhom_an, inhom_ca = params[6:8]

    # The mask comes from the configuration, so it is checked outside the
    # parameter-error guard below.
    roi_mask = np.asarray(roi_mask, dtype=bool).flatten()
    if not np.any(roi_mask):
        raise ValueError(
            f"ICA ROI mask selects none of its {roi_mask.size} grid points: "
            "check roi_ica_min / roi_ica_max."
        )

    try:
        q = np.asarray(q, dtype=np.float64).flatten()
        meas_ica = np.asarray(meas_ica, dtype=np.float64).flatten()

        if precomputed_anode_pot is None:
            anode_pot = electrode_potential_on_q(
                anode,
                q,
                alpha=alpha_an,
                beta=beta_an,
                gamma_blend2=gamma_blend2_an,
                inhom=inhom_an,
                inhom_offset=inhom_anode_offset,
                is_blend=anode_is_blend,
            )
        else:
            anode_pot = precomputed_anode_pot

        if precomputed_cathode_pot is None:
            cathode_pot = electrode_potential_on_q(
                cathode,
                q,
                alpha=alpha_ca,
                beta=beta_ca,
                gamma_blend2=gamma_blend2_ca,
                inhom=inhom_ca,
                inhom_offset=inhom_cathode_offset,
                is_blend=cathode_is_blend,
            )
        else:
            cathode_pot = precomputed_cathode_pot

        # MATLAB: OCV_sum = cathodePot - anodePot
        ocv_sum = cathode_pot - anode_pot

        # Compute ICA from modeled OCV
        ica_calc = _compute_discrete_ica(q, ocv_sum, q0)

        # Apply smoothing (MATLAB: apply_filter with sgolay)
        ica_calc = apply_filter(ica_calc, method="sgolay", window=50, order=3)

        # MATLAB: Diff_ICA = sum((ICA_calc - ICA_OCV).^2 .* mask) / sum(mask)
        mse = calculate_mse(meas_ica, ica_calc, roi_mask)

    except (ValueError, IndexError, FloatingPointError) as exc:
        _warn_once_on_penalty("fit_ica", repr(exc))
        return 1e6

    if not np.isfinite(mse):
        _warn_once_on_penalty("fit_ica", "non-finite objective")
        return 1e6

    return float(mse)


def combined_objective(
    params: NDArray[np.floating],
    anode: "ElectrodeOCP | BlendElectrode",
    cathode: "ElectrodeOCP | BlendElectrode",
    meas_voltage: NDArray[np.floating],
    meas_dva: NDArray[np.floating],
    meas_ica: NDArray[np.floating],
    q: NDArray[np.floating],
    *,
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
    inhom_anode_offset: float = 0.0,
    inhom_cathode_offset: float = 0.0,
    r_offset_sign: float = 1.0,
    pocv_current_a: float = 0.0,
) -> float:
    """Combined objective function for DMA optimization.

    Computes weighted sum of OCV, DVA, and ICA fitting errors:
    cost = w_ocv * fit_ocv + w_dva * fit_dva + w_ica * fit_ica

    Parameters
    ----------
    params : NDArray
        9-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca,
         inhom_an, inhom_ca, r_offset]
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
    r_offset_sign : float, optional
        Sign of the resistance term, ``+1`` for charge and ``-1`` for
        discharge. Reaches the OCV term only.
    pocv_current_a : float, optional
        Magnitude of the pOCV current in A. Reaches the OCV term only.

    Returns
    -------
    float
        Weighted sum of MSE values
    """
    cost = 0.0

    # The three terms read the same two electrode potentials, so they are
    # evaluated once here and passed down. Each term keeps its own fallback
    # for the case where this evaluation cannot produce them.
    anode_pot: NDArray[np.floating] | None = None
    cathode_pot: NDArray[np.floating] | None = None
    if w_ocv > 0 or w_dva > 0 or w_ica > 0:
        anode_pot, cathode_pot = _precompute_electrode_potentials(
            params,
            anode,
            cathode,
            q,
            anode_is_blend=anode_is_blend,
            cathode_is_blend=cathode_is_blend,
            inhom_anode_offset=inhom_anode_offset,
            inhom_cathode_offset=inhom_cathode_offset,
        )

    # OCV contribution
    if w_ocv > 0:
        ocv_error = fit_ocv(
            params,
            anode=anode,
            cathode=cathode,
            meas_voltage=meas_voltage,
            q=q,
            roi_ocv_min=roi_ocv_min,
            roi_ocv_max=roi_ocv_max,
            anode_is_blend=anode_is_blend,
            cathode_is_blend=cathode_is_blend,
            inhom_anode_offset=inhom_anode_offset,
            inhom_cathode_offset=inhom_cathode_offset,
            precomputed_anode_pot=anode_pot,
            precomputed_cathode_pot=cathode_pot,
            r_offset_sign=r_offset_sign,
            pocv_current_a=pocv_current_a,
        )
        cost += w_ocv * ocv_error

    # DVA contribution
    if w_dva > 0:
        dva_error = fit_dva(
            params,
            anode=anode,
            cathode=cathode,
            meas_dva=meas_dva,
            q=q,
            roi_mask=dva_roi_mask,
            q0=q0,
            anode_is_blend=anode_is_blend,
            cathode_is_blend=cathode_is_blend,
            inhom_anode_offset=inhom_anode_offset,
            inhom_cathode_offset=inhom_cathode_offset,
            precomputed_anode_pot=anode_pot,
            precomputed_cathode_pot=cathode_pot,
        )
        cost += w_dva * dva_error

    # ICA contribution
    if w_ica > 0:
        ica_error = fit_ica(
            params,
            anode=anode,
            cathode=cathode,
            meas_ica=meas_ica,
            q=q,
            roi_mask=ica_roi_mask,
            q0=q0,
            anode_is_blend=anode_is_blend,
            cathode_is_blend=cathode_is_blend,
            inhom_anode_offset=inhom_anode_offset,
            inhom_cathode_offset=inhom_cathode_offset,
            precomputed_anode_pot=anode_pot,
            precomputed_cathode_pot=cathode_pot,
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
        9-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca,
         inhom_an, inhom_ca, r_offset]
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

    # Ensure params covers the whole layout. A longer vector keeps its extra
    # slots: the degradation calculation reads only [0..5], and truncating here
    # would silently drop the resistance slot from anything downstream.
    params = np.asarray(params).flatten()
    if len(params) < 9:
        full_params = np.zeros(9)
        full_params[: len(params)] = params
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
    *,
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
    inhom_anode_offset: float = 0.0,
    inhom_cathode_offset: float = 0.0,
    r_offset_sign: float = 1.0,
    pocv_current_a: float = 0.0,
    ref_data: ReferenceData | None = None,
    prev_lam: PreviousLAM | None = None,
    penalty_config: PenaltyConfig | None = None,
    fit_reverse: bool = False,
) -> float:
    """Combined objective function with penalty constraints for DMA optimization.

    This function combines the base fitting objective (OCV + DVA + ICA) with
    penalty constraints that enforce physically plausible degradation evolution.

    MATLAB Reference: objectiveWithPenalty in dma_core.m

    Parameters
    ----------
    params : NDArray
        9-element parameter vector:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an, gamma_blend2_ca,
         inhom_an, inhom_ca, r_offset]
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
    r_offset_sign : float, optional
        Sign of the resistance term, ``+1`` for charge and ``-1`` for
        discharge. Reaches the OCV term only.
    pocv_current_a : float, optional
        Magnitude of the pOCV current in A. Reaches the OCV term only.
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
        params,
        anode=anode,
        cathode=cathode,
        meas_voltage=meas_voltage,
        meas_dva=meas_dva,
        meas_ica=meas_ica,
        q=q,
        dva_roi_mask=dva_roi_mask,
        ica_roi_mask=ica_roi_mask,
        roi_ocv_min=roi_ocv_min,
        roi_ocv_max=roi_ocv_max,
        q0=q0,
        w_ocv=w_ocv,
        w_dva=w_dva,
        w_ica=w_ica,
        anode_is_blend=anode_is_blend,
        cathode_is_blend=cathode_is_blend,
        inhom_anode_offset=inhom_anode_offset,
        inhom_cathode_offset=inhom_cathode_offset,
        r_offset_sign=r_offset_sign,
        pocv_current_a=pocv_current_a,
    )

    # Add penalty if configured and not first CU
    penalty = 0.0
    if ref_data is not None and prev_lam is not None and penalty_config is not None:
        # Check if we have any previous LAM values (not first CU)
        has_prev = (
            prev_lam.lam_anode is not None
            or prev_lam.lam_cathode is not None
            or prev_lam.lam_anode_blend1 is not None
            or prev_lam.lam_anode_blend2 is not None
        )
        if has_prev:
            penalty = calculate_penalty(params, ref_data, prev_lam, penalty_config, fit_reverse)

    return base_cost + penalty
