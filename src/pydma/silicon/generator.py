"""Silicon curve generator.

This module implements the silicon curve extraction algorithm from
the MATLAB generateSiCurve.m function. It extracts an artificial
Si half-cell OCV curve from:
- A measured graphite-Si blend curve
- A pure graphite reference curve

The extraction formula is:
    Q_blend = γ·Q_Si + (1-γ)·Q_Gr
    →  Q_Si = (Q_blend - (1-γ)·Q_Gr) / γ

No GUI is provided - use the programmatic interface instead.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import scipy.io
from numpy.typing import NDArray

from pydma.preprocessing.smoother import smooth_lowess
from pydma.silicon.strict_sto import _collapse_plateaus, _pav_isotonic


@dataclass
class SiliconCurveResult:
    """Result of silicon curve extraction.

    Attributes
    ----------
    voltage : NDArray
        Voltage values [V]
    normalized_capacity : NDArray
        Normalized capacity values [0, 1]
    graphite_voltage : NDArray
        Graphite reference voltage used
    graphite_capacity : NDArray
        Graphite reference capacity used
    blend_voltage : NDArray
        Blend curve voltage
    blend_capacity : NDArray
        Blend curve capacity
    gamma_si : float
        Silicon fraction used
    q_si_raw_min, q_si_raw_max : float
        Range of the extracted silicon capacity before it is clipped to [0, 1].
    clipped_fraction : float
        Fraction of samples the clip moved, in [0, 1].
    """

    voltage: NDArray[np.floating]
    normalized_capacity: NDArray[np.floating]
    graphite_voltage: NDArray[np.floating]
    graphite_capacity: NDArray[np.floating]
    blend_voltage: NDArray[np.floating]
    blend_capacity: NDArray[np.floating]
    gamma_si: float
    q_si_raw_min: float = 0.0
    q_si_raw_max: float = 0.0
    clipped_fraction: float = 0.0

    def to_electrode_format(self) -> dict[str, NDArray[np.floating]]:
        """Convert to format suitable for ElectrodeOCP.

        Returns
        -------
        dict
            Dictionary with 'soc' and 'voltage' keys
        """
        return {
            "soc": self.normalized_capacity,
            "voltage": self.voltage,
        }

    def save(self, path: str | Path) -> None:
        """Save silicon curve to MAT file.

        Parameters
        ----------
        path : str or Path
            Output file path (will add .mat if needed)
        """
        path = Path(path)
        if path.suffix.lower() != ".mat":
            path = path.with_suffix(".mat")

        silicon_struct = {
            "voltage": self.voltage,
            "normalizedCapacity": self.normalized_capacity,
        }
        scipy.io.savemat(str(path), {"siliconStruct": silicon_struct})


def _as_curve(struct: Any) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """(voltage, normalizedCapacity) of a MATLAB struct that carries both."""
    return (
        np.asarray(struct.voltage, dtype=np.float64).flatten(),
        np.asarray(struct.normalizedCapacity, dtype=np.float64).flatten(),
    )


def _select_variable(
    candidates: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]],
    path: str | Path,
    variable_name: str | None,
    kind: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Pick one candidate variable, warning when the choice is not unique."""
    if variable_name is not None:
        if variable_name not in candidates:
            raise ValueError(
                f"Variable '{variable_name}' does not hold {kind} data in {path}; "
                f"candidates are {sorted(candidates)}."
            )
        return candidates[variable_name]

    if not candidates:
        raise ValueError(f"Could not find {kind} data in {path}")

    names = list(candidates)
    if len(names) > 1:
        warnings.warn(
            f"{path} holds {len(names)} variables with {kind} data ({names}); reading "
            f"'{names[0]}'. Pass variable_name to select one explicitly.",
            stacklevel=3,
        )
    return candidates[names[0]]


def load_ocp_data(
    path: str | Path,
    variable_name: str | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load OCP data from MAT file.

    Supports the standard OCV format with 'voltage' and 'normalizedCapacity'
    fields.

    Parameters
    ----------
    path : str or Path
        Path to MAT file
    variable_name : str, optional
        Name of the MAT variable to read. Without it the first matching
        variable is used, and a warning lists the alternatives when the file
        holds more than one.

    Returns
    -------
    tuple[NDArray, NDArray]
        (voltage, normalized_capacity) arrays
    """
    data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    candidates = {
        key: _as_curve(value)
        for key, value in data.items()
        if not key.startswith("_")
        and hasattr(value, "voltage")
        and hasattr(value, "normalizedCapacity")
    }
    return _select_variable(candidates, path, variable_name, "OCP")


def _as_blend_curve(value: Any) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
    """(voltage, normalizedCapacity) of one MAT variable, or None if it holds neither."""
    # Standard struct format
    if hasattr(value, "voltage") and hasattr(value, "normalizedCapacity"):
        return _as_curve(value)

    # Cell array with TestData
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # Scalar cell array
        inner = value.item()
        if hasattr(inner, "TestData"):
            return _as_curve(inner.TestData)
    elif isinstance(value, np.ndarray):
        # Array of structs
        for item in value.flat:
            if hasattr(item, "TestData"):
                return _as_curve(item.TestData)

    return None


def load_blend_data(
    path: str | Path,
    variable_name: str | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load blend data from MAT file.

    Supports both standard struct format and cell array with TestData.

    Parameters
    ----------
    path : str or Path
        Path to MAT file
    variable_name : str, optional
        Name of the MAT variable to read. Without it the first matching
        variable is used, and a warning lists the alternatives when the file
        holds more than one.

    Returns
    -------
    tuple[NDArray, NDArray]
        (voltage, normalized_capacity) arrays
    """
    data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    candidates = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        curve = _as_blend_curve(value)
        if curve is not None:
            candidates[key] = curve

    return _select_variable(candidates, path, variable_name, "blend")


def _smooth_unique(
    voltage: NDArray[np.floating],
    capacity: NDArray[np.floating],
    do_smooth: bool = True,
    window: int = 30,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply LOWESS smoothing and remove duplicate voltages.

    Parameters
    ----------
    voltage : NDArray
        Voltage values
    capacity : NDArray
        Capacity values
    do_smooth : bool
        Whether to apply smoothing
    window : int
        Window size for LOWESS

    Returns
    -------
    tuple[NDArray, NDArray]
        (voltage, capacity) with unique voltages
    """
    if do_smooth:
        # LOWESS smoothing
        frac = min(window / len(voltage), 0.3)  # Adaptive fraction
        voltage = smooth_lowess(voltage, frac=frac, it=0)

    # Remove duplicate voltages
    _, unique_idx = np.unique(voltage, return_index=True)
    unique_idx = np.sort(unique_idx)

    return voltage[unique_idx], capacity[unique_idx]


def _trim_and_renorm(
    voltage: NDArray[np.floating],
    capacity: NDArray[np.floating],
    v_min: float,
    v_max: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Trim to voltage range and renormalize capacity to [0, 1].

    Parameters
    ----------
    voltage : NDArray
        Voltage values
    capacity : NDArray
        Capacity values
    v_min : float
        Minimum voltage
    v_max : float
        Maximum voltage

    Returns
    -------
    tuple[NDArray, NDArray]
        (voltage, capacity) trimmed and renormalized

    Raises
    ------
    ValueError
        If the trimmed window carries a constant capacity. There is no
        normalisation for it, and passing the raw values on would leave a curve
        that silently is not on the [0, 1] scale the extraction assumes.
    """
    mask = (voltage >= v_min) & (voltage <= v_max)
    voltage = voltage[mask]
    capacity = capacity[mask]

    q_min = float(capacity.min()) if capacity.size else float("nan")
    q_max = float(capacity.max()) if capacity.size else float("nan")
    if not capacity.size or q_max <= q_min:
        raise ValueError(
            f"Capacity is constant at q = {q_min!r} inside the voltage window "
            f"[{v_min:.6g}, {v_max:.6g}] V ({capacity.size} sample(s)); there is no "
            "range to renormalise onto [0, 1]."
        )

    # Renormalize to [0, 1]
    capacity = (capacity - q_min) / (q_max - q_min)

    return voltage, capacity


def generate_si_curve(
    blend_path: str | Path | None = None,
    graphite_path: str | Path | None = None,
    blend_data: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    graphite_data: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    gamma_si: float = 0.245,
    filter_blend: bool = True,
    filter_graphite: bool = False,
    monotone_filter: bool = True,
    collapse_plateaus: bool = False,
    filter_input: bool | None = None,
) -> SiliconCurveResult:
    """Generate artificial silicon OCV curve from blend and graphite data.

    This implements the extraction formula:
        Q_Si = (Q_blend - (1-γ)·Q_Gr) / γ

    Either provide file paths or data arrays directly.

    Parameters
    ----------
    blend_path : str or Path, optional
        Path to blend OCP MAT file
    graphite_path : str or Path, optional
        Path to graphite reference MAT file
    blend_data : tuple[NDArray, NDArray], optional
        Blend (voltage, capacity) data directly
    graphite_data : tuple[NDArray, NDArray], optional
        Graphite (voltage, capacity) data directly
    gamma_si : float, optional
        Silicon fraction in blend (0 < γ < 1), by default 0.245
    filter_blend : bool, optional
        Whether to apply LOWESS smoothing to the measured blend curve,
        by default True. The blend is typically a noisy C/30 half-cell
        measurement that benefits from smoothing.
    filter_graphite : bool, optional
        Whether to apply LOWESS smoothing to the graphite reference,
        by default False. Graphite references are usually clean lookup
        tables; smoothing a uniformly-resampled table can distort the
        steep high-voltage tail and truncate the common voltage window.
    monotone_filter : bool, optional
        Whether to enforce monotonicity via isotonic regression (PAV),
        by default True
    filter_input : bool, optional
        Deprecated. If set, applies LOWESS to both curves (overrides
        ``filter_blend`` and ``filter_graphite``).

    Returns
    -------
    SiliconCurveResult
        Extracted silicon curve and intermediate data

    Raises
    ------
    ValueError
        If required data is not provided or gamma is invalid

    Examples
    --------
    >>> # From files
    >>> result = generate_si_curve(
    ...     blend_path='SiGr_blend.mat',
    ...     graphite_path='Gr_Lithiation_Rehm2026.mat',
    ...     gamma_si=0.245,
    ... )
    >>>
    >>> # From arrays
    >>> result = generate_si_curve(
    ...     blend_data=(blend_v, blend_q),
    ...     graphite_data=(gr_v, gr_q),
    ...     gamma_si=0.20,
    ... )
    >>>
    >>> # Save result
    >>> result.save('Si_extracted.mat')
    """
    # Validate gamma
    if not 0 < gamma_si < 1:
        raise ValueError(f"gamma_si must be between 0 and 1, got {gamma_si}")

    if filter_input is not None:
        warnings.warn(
            "filter_input is deprecated; use filter_blend and filter_graphite instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        filter_blend = filter_input
        filter_graphite = filter_input

    # Load data
    if blend_data is not None:
        blend_v, blend_q = blend_data
    elif blend_path is not None:
        blend_v, blend_q = load_blend_data(blend_path)
    else:
        raise ValueError("Either blend_path or blend_data must be provided")

    if graphite_data is not None:
        gr_v, gr_q = graphite_data
    elif graphite_path is not None:
        gr_v, gr_q = load_ocp_data(graphite_path)
    else:
        raise ValueError("Either graphite_path or graphite_data must be provided")

    # Ensure arrays
    blend_v = np.asarray(blend_v, dtype=np.float64).flatten()
    blend_q = np.asarray(blend_q, dtype=np.float64).flatten()
    gr_v = np.asarray(gr_v, dtype=np.float64).flatten()
    gr_q = np.asarray(gr_q, dtype=np.float64).flatten()

    # Smooth and remove duplicates
    gr_v, gr_q = _smooth_unique(gr_v, gr_q, filter_graphite)
    blend_v, blend_q = _smooth_unique(blend_v, blend_q, filter_blend)

    # Find common voltage window
    v_min = float(max(gr_v.min(), blend_v.min()))
    v_max = float(min(gr_v.max(), blend_v.max()))

    # Trim and renormalize
    gr_v, gr_q = _trim_and_renorm(gr_v, gr_q, v_min, v_max)
    blend_v, blend_q = _trim_and_renorm(blend_v, blend_q, v_min, v_max)

    # Sort by voltage ascending (required for np.interp)
    gr_order = np.argsort(gr_v)
    gr_v, gr_q = gr_v[gr_order], gr_q[gr_order]
    blend_order = np.argsort(blend_v)
    blend_v, blend_q = blend_v[blend_order], blend_q[blend_order]

    # Create common voltage grid over the conceptual pre-trim overlap window.
    # This mirrors MATLAB generate_si_ocp and avoids sampling-dependent endpoint
    # shifts that arise from using the first/last surviving data points after
    # _trim_and_renorm (those points are strictly inside [v_min, v_max] in
    # general, which produces a slightly inset grid that depends on the input
    # sampling rather than on the conceptual support boundary).
    n_points = max(len(gr_v), len(blend_v))
    v_common = np.linspace(v_min, v_max, n_points)

    # Interpolate to common grid
    q_gr = np.interp(v_common, gr_v, gr_q)
    q_blend = np.interp(v_common, blend_v, blend_q)

    # Drop the first grid point. It sits exactly on the lower window boundary,
    # where at least one of the two curves is pinned to its own edge sample.
    mask_first = np.zeros(len(v_common), dtype=bool)
    mask_first[0] = True
    mask_keep = ~mask_first

    v_common = v_common[mask_keep]
    q_gr = q_gr[mask_keep]
    q_blend = q_blend[mask_keep]

    # The subtraction below assumes both curves run their capacity in the same
    # direction over voltage. Mixing a lithiation reference with a delithiation
    # blend (or vice versa) still yields a smooth-looking silicon curve, so the
    # mismatch has to be caught here rather than left to the reader.
    slope_gr = float(np.polyfit(v_common, q_gr, 1)[0])
    slope_blend = float(np.polyfit(v_common, q_blend, 1)[0])
    if np.sign(slope_gr) != np.sign(slope_blend):
        raise ValueError(
            f"Graphite and blend run their capacity in opposite directions over "
            f"voltage: dQ_gr/dV = {slope_gr:.6g}, dQ_blend/dV = {slope_blend:.6g}. "
            "Both curves must be the same lithiation direction."
        )

    # Calculate silicon curve
    # Q_Si = (Q_blend - (1-γ)·Q_Gr) / γ
    q_si = (q_blend - (1 - gamma_si) * q_gr) / gamma_si

    # The clip below is silent, so record what it hides: how far the raw
    # extraction left [0, 1] and how much of the curve it moves. Both are
    # symptoms of a gamma_si that does not match the two input curves.
    q_si_raw_min = float(q_si.min())
    q_si_raw_max = float(q_si.max())
    clipped_fraction = float(np.mean((q_si < 0.0) | (q_si > 1.0)))
    excess = max(q_si_raw_max - 1.0, -q_si_raw_min, 0.0)
    if clipped_fraction > 0.01 and excess > 0.02:
        warnings.warn(
            f"Silicon extraction at gamma_si={gamma_si} leaves [0, 1] before clipping: "
            f"raw range [{q_si_raw_min:.4f}, {q_si_raw_max:.4f}], "
            f"{clipped_fraction:.1%} of the samples clipped. Check gamma_si and the "
            "graphite reference.",
            stacklevel=2,
        )

    # sum(diff(.)) telescopes to q_si[-1] - q_si[0], so this IS the endpoint
    # comparison — taken on the pre-clip array, where a clipped endpoint cannot
    # flip it.
    rises_with_voltage = float(np.sum(np.diff(q_si))) >= 0.0

    # Clip to [0, 1]
    q_si = np.clip(q_si, 0, 1)

    # Enforce monotonicity via isotonic regression (PAV)
    if monotone_filter:
        if rises_with_voltage:
            q_si = _pav_isotonic(q_si, "nondecreasing")
        else:
            q_si = _pav_isotonic(q_si, "nonincreasing")
        # Plateau-collapse is opt-in. Removing PAV plateau interiors makes the
        # curve a strict function of SOC (needed for downstream SOC->V
        # consumers) but under-resolves the low-V silicon plateau and breaks
        # the PyDMA discharge balancing fit (~21 mV vs ~3.34 mV native). Keep
        # this disabled for fitting; enable only for export paths that need
        # strict monotonicity in SOC.
        if collapse_plateaus:
            collapsed_v, collapsed_q = _collapse_plateaus(v_common, q_si)
            v_common = cast(NDArray[np.float64], collapsed_v)
            q_si = cast(NDArray[np.float64], collapsed_q)
            q_gr = np.interp(v_common, gr_v, gr_q)
            q_blend = np.interp(v_common, blend_v, blend_q)

    # Each voltage field gets its own copy: the three share one grid, and a
    # consumer editing one of them in place must not reach the other two.
    return SiliconCurveResult(
        voltage=v_common.copy(),
        normalized_capacity=q_si,
        graphite_voltage=v_common.copy(),
        graphite_capacity=q_gr,
        blend_voltage=v_common.copy(),
        blend_capacity=q_blend,
        gamma_si=gamma_si,
        q_si_raw_min=q_si_raw_min,
        q_si_raw_max=q_si_raw_max,
        clipped_fraction=clipped_fraction,
    )


# Convenience function for common graphite sources
GraphiteSource = Literal["Rehm2026", "Schmitt", "Rehm2025", "Hossain", "Wetjen"]
LithDirection = Literal["lithiation", "delithiation"]


def get_builtin_graphite_path(
    source: GraphiteSource = "Rehm2026",
    direction: LithDirection = "lithiation",
) -> Path | None:
    """Get path to built-in graphite reference data.

    Note: This function returns None if the built-in data is not
    available. The data files must be copied from the MATLAB
    InputData/Graphite folder.

    Parameters
    ----------
    source : str
        Graphite source: 'Rehm2026', 'Schmitt', 'Rehm2025', 'Hossain', 'Wetjen'
    direction : str
        'lithiation' or 'delithiation'

    Returns
    -------
    Path or None
        Path to MAT file if available, None otherwise
    """
    # Schmitt only has lithiation
    if source == "Schmitt" and direction == "delithiation":
        return None

    # Look for data in package directory
    import pydma

    package_dir = Path(pydma.__file__).parent
    data_dir = package_dir / "data" / "Graphite"

    direction_cap = direction.capitalize()
    source_cap = source.capitalize()
    filename = f"Gr_{direction_cap}_{source_cap}.mat"

    path = data_dir / filename
    if path.exists():
        return path

    return None
