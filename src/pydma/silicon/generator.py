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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import scipy.io

from pydma.preprocessing.smoother import smooth_lowess


@dataclass
class SiliconCurveParams:
    """Parameters for silicon curve generation.

    Attributes
    ----------
    gamma_si : float
        Silicon fraction in the blend (0 < gamma < 1).
        For P45B cells, approximately 0.245.
    filter_input : bool
        Whether to apply LOWESS smoothing to input data.
    monotone_filter : bool
        Whether to enforce monotonicity via isotonic regression (PAV).
    """

    gamma_si: float
    filter_input: bool = True
    monotone_filter: bool = True

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 < self.gamma_si < 1:
            raise ValueError(f"gamma_si must be between 0 and 1, got {self.gamma_si}")


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
    """

    voltage: NDArray[np.floating]
    normalized_capacity: NDArray[np.floating]
    graphite_voltage: NDArray[np.floating]
    graphite_capacity: NDArray[np.floating]
    blend_voltage: NDArray[np.floating]
    blend_capacity: NDArray[np.floating]
    gamma_si: float

    def to_electrode_format(self) -> dict[str, NDArray[np.floating]]:
        """Convert to format suitable for ElectrodeOCP.

        Returns
        -------
        dict
            Dictionary with 'soc' and 'voltage' keys
        """
        return {
            'soc': self.normalized_capacity,
            'voltage': self.voltage,
        }

    def save(self, path: str | Path) -> None:
        """Save silicon curve to MAT file.

        Parameters
        ----------
        path : str or Path
            Output file path (will add .mat if needed)
        """
        path = Path(path)
        if path.suffix.lower() != '.mat':
            path = path.with_suffix('.mat')

        silicon_struct = {
            'voltage': self.voltage,
            'normalizedCapacity': self.normalized_capacity,
        }
        scipy.io.savemat(str(path), {'siliconStruct': silicon_struct})


def load_ocp_data(
    path: str | Path,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load OCP data from MAT file.

    Supports the standard OCV format with 'voltage' and 'normalizedCapacity'
    fields.

    Parameters
    ----------
    path : str or Path
        Path to MAT file

    Returns
    -------
    tuple[NDArray, NDArray]
        (voltage, normalized_capacity) arrays
    """
    data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    # Find the struct (usually first non-private variable)
    for key, value in data.items():
        if key.startswith('_'):
            continue
        if hasattr(value, 'voltage') and hasattr(value, 'normalizedCapacity'):
            return (
                np.asarray(value.voltage, dtype=np.float64).flatten(),
                np.asarray(value.normalizedCapacity, dtype=np.float64).flatten(),
            )

    raise ValueError(f"Could not find OCP data in {path}")


def load_blend_data(
    path: str | Path,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load blend data from MAT file.

    Supports both standard struct format and cell array with TestData.

    Parameters
    ----------
    path : str or Path
        Path to MAT file

    Returns
    -------
    tuple[NDArray, NDArray]
        (voltage, normalized_capacity) arrays
    """
    data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    # Find the data (try different formats)
    for key, value in data.items():
        if key.startswith('_'):
            continue

        # Standard struct format
        if hasattr(value, 'voltage') and hasattr(value, 'normalizedCapacity'):
            return (
                np.asarray(value.voltage, dtype=np.float64).flatten(),
                np.asarray(value.normalizedCapacity, dtype=np.float64).flatten(),
            )

        # Cell array with TestData
        if isinstance(value, np.ndarray) and value.ndim == 0:
            # Scalar cell array
            inner = value.item()
            if hasattr(inner, 'TestData'):
                td = inner.TestData
                return (
                    np.asarray(td.voltage, dtype=np.float64).flatten(),
                    np.asarray(td.normalizedCapacity, dtype=np.float64).flatten(),
                )
        elif isinstance(value, np.ndarray):
            # Array of structs
            for item in value.flat:
                if hasattr(item, 'TestData'):
                    td = item.TestData
                    return (
                        np.asarray(td.voltage, dtype=np.float64).flatten(),
                        np.asarray(td.normalizedCapacity, dtype=np.float64).flatten(),
                    )

    raise ValueError(f"Could not find blend data in {path}")


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
    """
    mask = (voltage >= v_min) & (voltage <= v_max)
    voltage = voltage[mask]
    capacity = capacity[mask]

    # Renormalize to [0, 1]
    if capacity.max() > capacity.min():
        capacity = (capacity - capacity.min()) / (capacity.max() - capacity.min())

    return voltage, capacity


def _pav_isotonic(
    q_raw: NDArray[np.floating],
    direction: str = 'nondecreasing',
) -> NDArray[np.floating]:
    """Isotonic regression via Pool Adjacent Violators (PAV).

    Enforces monotonicity with minimal L2 change.

    Parameters
    ----------
    q_raw : NDArray
        Input capacity values.
    direction : str
        'nondecreasing' or 'nonincreasing'.

    Returns
    -------
    NDArray
        Monotone capacity values.
    """
    q = q_raw.copy()
    if direction == 'nonincreasing':
        q = -q
    n = len(q)
    val = np.zeros(n)
    sz = np.zeros(n, dtype=int)
    nb = 0
    for i in range(n):
        nb += 1
        val[nb - 1] = q[i]
        sz[nb - 1] = 1
        while nb > 1 and val[nb - 2] > val[nb - 1]:
            total = sz[nb - 2] + sz[nb - 1]
            val[nb - 2] = (val[nb - 2] * sz[nb - 2] + val[nb - 1] * sz[nb - 1]) / total
            sz[nb - 2] = total
            nb -= 1
    result = np.empty(n)
    idx = 0
    for k in range(nb):
        result[idx:idx + sz[k]] = val[k]
        idx += sz[k]
    if direction == 'nonincreasing':
        result = -result
    return result


def generate_si_curve(
    blend_path: str | Path | None = None,
    graphite_path: str | Path | None = None,
    blend_data: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    graphite_data: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    gamma_si: float = 0.245,
    filter_input: bool = True,
    monotone_filter: bool = True,
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
    filter_input : bool, optional
        Whether to apply LOWESS smoothing, by default True
    monotone_filter : bool, optional
        Whether to enforce monotonicity via isotonic regression (PAV),
        by default True

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
    gr_v, gr_q = _smooth_unique(gr_v, gr_q, filter_input)
    blend_v, blend_q = _smooth_unique(blend_v, blend_q, filter_input)

    # Find common voltage window
    v_min = max(gr_v.min(), blend_v.min())
    v_max = min(gr_v.max(), blend_v.max())

    # Trim and renormalize
    gr_v, gr_q = _trim_and_renorm(gr_v, gr_q, v_min, v_max)
    blend_v, blend_q = _trim_and_renorm(blend_v, blend_q, v_min, v_max)

    # Sort by voltage ascending (required for np.interp)
    gr_order = np.argsort(gr_v)
    gr_v, gr_q = gr_v[gr_order], gr_q[gr_order]
    blend_order = np.argsort(blend_v)
    blend_v, blend_q = blend_v[blend_order], blend_q[blend_order]

    # Create common voltage grid (use actual data range after trimming)
    v_lo = max(gr_v.min(), blend_v.min())
    v_hi = min(gr_v.max(), blend_v.max())
    n_points = max(len(gr_v), len(blend_v))
    v_common = np.linspace(v_lo, v_hi, n_points)

    # Interpolate to common grid
    q_gr = np.interp(v_common, gr_v, gr_q)
    q_blend = np.interp(v_common, blend_v, blend_q)

    # Remove flat regions at start
    mask_first = np.zeros(len(v_common), dtype=bool)
    mask_first[0] = True
    mask_flat = (q_gr == q_gr.min()) & (q_gr == q_gr.max())
    mask_keep = ~(mask_first | mask_flat)

    v_common = v_common[mask_keep]
    q_gr = q_gr[mask_keep]
    q_blend = q_blend[mask_keep]

    # Calculate silicon curve
    # Q_Si = (Q_blend - (1-γ)·Q_Gr) / γ
    q_si = (q_blend - (1 - gamma_si) * q_gr) / gamma_si

    # Clip to [0, 1]
    q_si = np.clip(q_si, 0, 1)

    # Enforce monotonicity via isotonic regression (PAV)
    if monotone_filter:
        if q_si[-1] < q_si[0]:
            q_si = _pav_isotonic(q_si, 'nonincreasing')
        else:
            q_si = _pav_isotonic(q_si, 'nondecreasing')

    return SiliconCurveResult(
        voltage=v_common,
        normalized_capacity=q_si,
        graphite_voltage=gr_v,
        graphite_capacity=gr_q,
        blend_voltage=blend_v,
        blend_capacity=blend_q,
        gamma_si=gamma_si,
    )


# Convenience function for common graphite sources
GraphiteSource = Literal['Rehm2026', 'Schmitt', 'Rehm2025', 'Hossain', 'Wetjen']
LithDirection = Literal['lithiation', 'delithiation']


def get_builtin_graphite_path(
    source: GraphiteSource = 'Rehm2026',
    direction: LithDirection = 'lithiation',
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
    if source == 'Schmitt' and direction == 'delithiation':
        return None

    # Look for data in package directory
    import pydma
    package_dir = Path(pydma.__file__).parent
    data_dir = package_dir / 'data' / 'Graphite'

    direction_cap = direction.capitalize()
    source_cap = source.capitalize()
    filename = f'Gr_{direction_cap}_{source_cap}.mat'

    path = data_dir / filename
    if path.exists():
        return path

    return None
