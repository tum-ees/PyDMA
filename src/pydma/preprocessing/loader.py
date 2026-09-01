"""
Data loading utilities with automatic column detection.

This module provides functions for loading electrode OCP and pOCV data
from various file formats with automatic column name detection.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pydma.electrodes.electrode import ElectrodeOCP

try:
    from scipy.io import loadmat

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Column name patterns for auto-detection
# DIFFERENCE FROM MATLAB: Python version has explicit column name patterns
# MATLAB uses specific field names like 'SOC', 'UCa', 'UAn', 'U'

SOC_COLUMN_PATTERNS = [
    "soc",
    "state_of_charge",
    "stateofcharge",
    "normalized_capacity",
    "normalizedcapacity",
    "norm_capacity",
    "normcapacity",
    "q_norm",
    "qnorm",
    "x",  # Often used for lithiation fraction
]

VOLTAGE_COLUMN_PATTERNS = [
    "voltage",
    "potential",
    "ocv",
    "ocp",
    "u",
    "v",
    "e",
    "uca",  # Cathode voltage
    "uan",  # Anode voltage
    "uocv",
    "e_ocv",
]

CAPACITY_COLUMN_PATTERNS = [
    "capacity",
    "capa",
    "q",
    "ah",
    "charge",
    "maxahstep",
    "max_ah_step",
]

FULLCELL_X_COLUMN_PATTERNS = [
    "ah_step",
    "max_ah_step",
    "maxahstep",
    "soc",
    "state_of_charge",
    "stateofcharge",
    "normalized_capacity",
    "normalizedcapacity",
    "norm_capacity",
    "normcapacity",
    "q_norm",
    "qnorm",
    "ah",
    "q",
    "capacity",
    "capa",
    "charge",
    "x",
]

FULLCELL_CAPACITY_COLUMN_PATTERNS = [
    "pocv_ch",
    "pocv_dch",
    "capacity",
    "capa",
    "charge_capacity",
    "discharge_capacity",
    "max_ah_step",
    "maxahstep",
    "ah",
    "q",
    "charge",
]

_DIRECTION_PATTERNS = {
    "charge": [
        "*pocv*ch*.csv",
        "*pocv*charge*.csv",
        "*pOCV*CH*.csv",
        "*pOCV*charge*.csv",
        "*charge*.csv",
        "*CH*.csv",
        "*pocv*.csv",
        "*pOCV*.csv",
    ],
    "discharge": [
        "*pocv*dch*.csv",
        "*pocv*discharge*.csv",
        "*pOCV*DCH*.csv",
        "*pOCV*discharge*.csv",
        "*discharge*.csv",
        "*DCH*.csv",
        "*pocv*.csv",
        "*pOCV*.csv",
    ],
}

# Field names a MATLAB aging-study table uses for the per-CU pOCV traces.
_EXPLICIT_DIRECTION_KEYS = {
    "charge": ["Testdata_pOCV_CH", "pOCV_CH", "charge", "Testdata_charge"],
    "discharge": ["Testdata_pOCV_DCH", "pOCV_DCH", "discharge", "Testdata_discharge"],
}


def _normalize_name(name: Any) -> str:
    """Normalize a key/column name for fuzzy matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_matching_name(names: list[str], patterns: list[str]) -> str | None:
    """Find the best matching name using exact match first, then safe substring match."""
    normalized_names = [(name, _normalize_name(name)) for name in names]

    for pattern in patterns:
        normalized_pattern = _normalize_name(pattern)
        for name, normalized_name in normalized_names:
            if normalized_name == normalized_pattern:
                return name

    for pattern in patterns:
        normalized_pattern = _normalize_name(pattern)
        if len(normalized_pattern) <= 1:
            continue
        for name, normalized_name in normalized_names:
            if normalized_pattern in normalized_name:
                return name

    return None


def _warn_if_percent_axis(values: np.ndarray, label: str) -> None:
    """Warn when an axis carries the signature of a 0..100 percent scale.

    That signature is narrow on purpose: the span sits within a few percent of
    100 and the axis starts at 0. A large-format cell has a capacity of 280 Ah
    and a percent axis has a span of 100, so a plain "span is large" test would
    keep flagging the cell and never the percentage.
    """
    span = float(values.max() - values.min())
    if 95.0 <= span <= 105.0 and abs(float(values.min())) <= 0.05 * span:
        warnings.warn(
            f"{label} spans {span:.4g} from {float(values.min()):.4g}, which looks "
            "like a 0..100 percent axis rather than an absolute capacity in Ah."
        )


def _extract_capacity_value(values: Any) -> float | None:
    """Extract a scalar capacity from a scalar, repeated series, or capacity trace."""
    arr = np.asarray(values, dtype=np.float64).flatten()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    span = float(arr.max() - arr.min())
    mean = float(arr.mean())
    # A relative span decides constancy: instrument noise on an otherwise
    # constant Ah column has span > 0 but must not be read as a capacity
    # trace, which np.allclose's default rtol would do for large means.
    is_constant = arr.size == 1 or (abs(span / mean) < 1e-3 if mean != 0 else span == 0)
    if is_constant:
        return float(arr[0])

    _warn_if_percent_axis(arr, "Capacity trace")
    if 1.0 < span <= 1.5:
        warnings.warn(
            f"Capacity trace spans {span:.3g}, which is ambiguous between a "
            "normalized 0..1 axis and a small absolute capacity in Ah."
        )

    if span > 0:
        return span
    return float(arr.max())


def _coerce_mat_container(data: Any) -> dict[str, Any] | None:
    """Convert MATLAB-loaded structs or records to plain Python dictionaries."""
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if not str(k).startswith("__")}

    fieldnames = getattr(data, "_fieldnames", None)
    if fieldnames is not None:
        return {str(name): getattr(data, name) for name in fieldnames}

    if isinstance(data, np.void) and data.dtype.names:
        return {str(name): data[name] for name in data.dtype.names}

    if isinstance(data, np.ndarray):
        if data.size == 1:
            return _coerce_mat_container(data.item())
        if data.dtype.names:
            return {str(name): np.asarray(data[name]).squeeze() for name in data.dtype.names}
        return None

    return None


def _iter_mat_candidates(data: Any):
    """Breadth-first traversal over nested MATLAB containers."""
    queue = [data]
    seen_ids: set[int] = set()

    while queue:
        current = queue.pop(0)
        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)
        yield current

        container = _coerce_mat_container(current)
        if container is not None:
            queue.extend(container.values())
            continue

        if isinstance(current, np.ndarray) and current.dtype == object:
            queue.extend(np.ravel(current).tolist())
        elif isinstance(current, (list, tuple)):
            queue.extend(list(current))


def _is_cu_table(container: dict[str, Any]) -> bool:
    """True when the column names identify a container as the CU table.

    The BFS walks every struct in the file, most of which are metadata rather
    than tables. Only the container this recognises is held to the
    uniform-column-length rule below.
    """
    keys = list(container.keys())
    if _find_matching_name(keys, ["CU", "cu"]) is None:
        return False
    return any(
        _find_matching_name(keys, patterns) is not None
        for patterns in _EXPLICIT_DIRECTION_KEYS.values()
    )


def _extract_table_rows(container: dict[str, Any], strict: bool = False) -> list[dict[str, Any]]:
    """Convert a MATLAB table-like dict of columns to a list of row dicts.

    Columns of differing length describe no table. Without ``strict`` that is
    reported as "not a table" so the search continues past a metadata struct;
    with it, on the container recognised as the CU table, it raises.
    """
    if not container:
        return []

    lengths = {}
    for key, value in container.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        lengths[key] = len(arr)

    if not lengths:
        return []

    row_count = max(lengths.values())
    if row_count <= 1:
        return []

    for key, length in lengths.items():
        if length != row_count:
            if not strict:
                return []
            raise ValueError(
                f"Column {key!r} has length {length}, expected {row_count} to match "
                "the table's other columns."
            )

    rows = []
    for idx in range(row_count):
        row: dict[str, Any] = {}
        for key, value in container.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                row[key] = arr.item()
            else:
                row[key] = arr[idx]
        rows.append(row)
    return rows


def _find_data_keys(mapping: dict[str, Any], electrode_type: str) -> tuple[str | None, str | None]:
    """Find SOC/capacity and voltage keys in a dict-like MATLAB container."""
    keys = list(mapping.keys())

    if electrode_type == "fullcell":
        x_key = _find_matching_name(keys, FULLCELL_X_COLUMN_PATTERNS)
        voltage_patterns = ["u", "voltage", "ocv", "uocv", "fullcellu", "cellvoltage"]
    elif electrode_type in ("cathode", "cathodeBlend2"):
        x_key = _find_matching_name(keys, SOC_COLUMN_PATTERNS)
        voltage_patterns = ["uca", "u_ca", "ucathode", "u", "voltage", "ocv"]
    else:
        x_key = _find_matching_name(keys, SOC_COLUMN_PATTERNS)
        voltage_patterns = ["uan", "u_an", "uanode", "u", "voltage", "ocv"]

    voltage_key = _find_matching_name(keys, voltage_patterns)
    return x_key, voltage_key


def _extract_fullcell_capacity_from_mapping(
    mapping: dict[str, Any], x_key: str | None
) -> float | None:
    """Extract total capacity from a full-cell data mapping when available."""
    keys = list(mapping.keys())
    capacity_key = _find_matching_name(keys, FULLCELL_CAPACITY_COLUMN_PATTERNS)
    if capacity_key is not None and capacity_key != x_key:
        return _extract_capacity_value(mapping[capacity_key])

    if x_key is None:
        return None

    x_values = np.asarray(mapping[x_key], dtype=np.float64).flatten()
    x_values = x_values[np.isfinite(x_values)]
    if x_values.size == 0:
        return None

    span = float(x_values.max() - x_values.min())
    if span > 1.0 + 1e-9:
        _warn_if_percent_axis(x_values, "Full-cell x-axis")
        if span <= 1.5:
            warnings.warn(
                f"Full-cell x-axis spans {span:.3g}, which is ambiguous between a "
                "normalized 0..1 axis and a small absolute capacity in Ah."
            )
        return span
    return None


def _coerce_cu_name(value: Any) -> str:
    """Normalize a CU identifier to the `CU<n>` style used by PyDMA."""
    arr = np.asarray(value).flatten()
    if arr.size == 0:
        raise ValueError("Empty CU identifier encountered in aging-study data.")
    if arr.size > 1:
        warnings.warn(f"CU identifier has {arr.size} elements; using the first one, {arr[0]!r}.")

    scalar = arr[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode()
    if isinstance(scalar, str):
        text = scalar.strip()
        return text if text.upper().startswith("CU") else f"CU{text}"

    if isinstance(scalar, (np.integer, int)):
        return f"CU{int(scalar)}"
    if isinstance(scalar, (np.floating, float)) and float(scalar).is_integer():
        return f"CU{int(scalar)}"

    return f"CU{scalar}"


def _cu_sort_key(value: str) -> tuple[int, str]:
    """Natural sort key for CU labels such as CU1, CU2, CU10."""
    normalized = _normalize_name(value)
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return (int(digits), normalized) if digits else (10**9, normalized)


def _extract_aging_study_from_mat(
    mat_data: dict[str, Any],
    direction: str,
    source: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, float | None]]:
    """Extract multi-CU pOCV data from a single MATLAB file."""
    results: dict[str, tuple[np.ndarray, np.ndarray, float | None]] = {}
    direction = direction.lower()

    for candidate in _iter_mat_candidates(mat_data):
        container = _coerce_mat_container(candidate)
        if container is None or not container:
            continue

        # Direct mapping: {CU1: <data>, CU2: <data>, ...}
        direct_cu_keys = []
        for key in container.keys():
            normalized_key = _normalize_name(key)
            if normalized_key.startswith("cu") and any(ch.isdigit() for ch in normalized_key[2:]):
                direct_cu_keys.append(key)
        if direct_cu_keys:
            parsed_any = False
            for cu_key in sorted(direct_cu_keys, key=_cu_sort_key):
                soc, voltage, capacity = _extract_soc_voltage_from_mat(
                    container[cu_key], "fullcell", source=source
                )
                results[_coerce_cu_name(cu_key)] = (soc, voltage, capacity)
                parsed_any = True
            if parsed_any:
                return results

        # Table-like dict of columns.
        rows = _extract_table_rows(container, strict=_is_cu_table(container))
        if rows:
            parsed_any = False
            for row in rows:
                row_container = _coerce_mat_container(row) or row
                cu_key_match = _find_matching_name(list(row_container.keys()), ["CU", "cu"])
                if cu_key_match is None:
                    continue
                cu_key = cu_key_match

                data_key = _find_matching_name(
                    list(row_container.keys()),
                    _EXPLICIT_DIRECTION_KEYS[direction],
                )
                if data_key is None:
                    continue

                soc, voltage, capacity = _extract_soc_voltage_from_mat(
                    row_container[data_key],
                    "fullcell",
                    source=source,
                )
                if capacity is None:
                    capacity = _extract_capacity_value(
                        row_container.get("pOCV_CH" if direction == "charge" else "pOCV_DCH")
                    )
                results[_coerce_cu_name(row_container[cu_key])] = (soc, voltage, capacity)
                parsed_any = True
            if parsed_any:
                return results

    raise ValueError(
        "Could not extract aging-study data from the MATLAB file. "
        "Expected either CU-keyed entries (e.g. CU1, CU2, ...) or a table-like "
        "structure with CU and Testdata_pOCV_* fields."
    )


def auto_detect_columns(
    df: pd.DataFrame,
    electrode_type: str = "anode",
) -> dict[str, str]:
    """
    Auto-detect SOC, voltage, and capacity columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with electrode data.
    electrode_type : str
        Type of electrode: 'anode', 'cathode', 'fullcell',
        'anodeBlend2', or 'cathodeBlend2'.

    Returns
    -------
    dict
        Dictionary with keys 'soc', 'voltage', 'capacity' (if found)
        mapping to column names.

    Raises
    ------
    ValueError
        If required columns cannot be detected.

    Notes
    -----
    DIFFERENCE FROM MATLAB: MATLAB's parse_data_input uses specific field
    names. This function provides flexible pattern matching.
    """
    columns = list(df.columns)

    result = {}

    # Detect x-axis column
    if electrode_type == "fullcell":
        x_patterns = FULLCELL_X_COLUMN_PATTERNS
    else:
        x_patterns = SOC_COLUMN_PATTERNS
    soc_col = _find_matching_name(columns, x_patterns)

    if not soc_col:
        raise ValueError(
            f"Could not detect SOC/capacity column. Looked for patterns: {x_patterns}. "
            f"Available columns: {list(df.columns)}"
        )
    result["soc"] = soc_col

    # Detect voltage column based on electrode type
    if electrode_type in ("cathode", "cathodeBlend2"):
        priority_patterns = ["uca", "u_ca", "ucathode", "u", "voltage", "ocv", "ocp"]
    elif electrode_type in ("anode", "anodeBlend2"):
        priority_patterns = ["uan", "u_an", "uanode", "u", "voltage", "ocv", "ocp"]
    else:  # fullcell
        priority_patterns = ["u", "voltage", "ocv", "ocp", "uocv", "full_cell_voltage"]
    voltage_col = _find_matching_name(columns, priority_patterns)

    if not voltage_col:
        raise ValueError(
            f"Could not detect voltage column. Looked for patterns: {priority_patterns}. "
            f"Available columns: {list(df.columns)}"
        )
    result["voltage"] = voltage_col

    # Detect capacity column (optional)
    capacity_patterns = (
        FULLCELL_CAPACITY_COLUMN_PATTERNS
        if electrode_type == "fullcell"
        else CAPACITY_COLUMN_PATTERNS
    )
    capacity_col = _find_matching_name(
        [col for col in columns if col not in {result["soc"], result["voltage"]}],
        capacity_patterns,
    )
    if capacity_col is not None:
        result["capacity"] = capacity_col

    return result


def _extract_soc_voltage_from_mat(
    data: Any, electrode_type: str, source: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Extract SOC and voltage from MATLAB .mat file structure.

    Parameters
    ----------
    data : dict
        Loaded .mat file data.
    electrode_type : str
        Type of electrode.
    source : str or Path, optional
        Originating file path, used only to identify the file in error messages.

    Returns
    -------
    tuple
        (soc, voltage, capacity) arrays.
    """
    top_level = _coerce_mat_container(data) or {"data": data}

    for candidate in _iter_mat_candidates(top_level):
        mapping = _coerce_mat_container(candidate)
        if mapping is None:
            continue

        if "TestData" in mapping:
            nested = _coerce_mat_container(mapping["TestData"])
            if nested is not None:
                mapping = nested

        x_key, voltage_key = _find_data_keys(mapping, electrode_type)
        if x_key is None or voltage_key is None:
            continue

        soc = np.asarray(mapping[x_key], dtype=np.float64).flatten()
        voltage = np.asarray(mapping[voltage_key], dtype=np.float64).flatten()
        if len(soc) != len(voltage):
            raise ValueError(
                f"SOC/capacity and voltage columns have mismatched lengths in "
                f"{source if source is not None else '<mat data>'}: "
                f"keys ({x_key!r}, {voltage_key!r}), len(soc)={len(soc)}, "
                f"len(voltage)={len(voltage)}."
            )
        if electrode_type == "fullcell":
            capacity = _extract_fullcell_capacity_from_mapping(mapping, x_key)
        else:
            cap_key = _find_matching_name(list(mapping.keys()), CAPACITY_COLUMN_PATTERNS)
            cap_val = mapping.get(cap_key) if cap_key is not None else None
            capacity = _extract_capacity_value(cap_val)
        return soc, voltage, capacity

    raise ValueError(
        f"Could not find SOC/capacity and voltage columns in .mat file. "
        f"Top-level keys: {list(top_level.keys())}"
    )


def load_ocp(
    filepath: str | Path,
    electrode_type: str = "anode",
    soc_col: str | None = None,
    voltage_col: str | None = None,
    capacity_col: str | None = None,
    smooth: bool = False,
    smooth_window: int = 30,
    csv_kwargs: dict | None = None,
) -> "ElectrodeOCP":
    """
    Load electrode OCP data from file.

    Supports CSV, Excel, and MATLAB .mat files with automatic
    column name detection.

    Parameters
    ----------
    filepath : str or Path
        Path to data file.
    electrode_type : str
        Type of electrode: 'anode', 'cathode', 'anodeBlend2', 'cathodeBlend2'.
    soc_col : str, optional
        Name of SOC column (auto-detected if not provided).
    voltage_col : str, optional
        Name of voltage column (auto-detected if not provided).
    capacity_col : str, optional
        Name of capacity column.
    smooth : bool
        Whether to smooth the data after loading.
    smooth_window : int
        Smoothing window size.
    csv_kwargs : dict, optional
        Extra keyword arguments passed through to ``pd.read_csv`` unchanged.
        For German-locale exports: ``csv_kwargs={"sep": ";", "decimal": ","}``.

    Returns
    -------
    ElectrodeOCP
        Loaded electrode OCP object.

    Examples
    --------
    >>> ocp = load_ocp("graphite_lithiation.csv", electrode_type="anode")
    >>> ocp = load_ocp("cathode.mat", electrode_type="cathode", smooth=True)
    """
    from pydma.electrodes.electrode import ElectrodeOCP

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.suffix.lower()
    name = filepath.stem

    # Load based on file type
    if ext == ".csv":
        df = pd.read_csv(filepath, **(csv_kwargs or {}))
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif ext == ".mat":
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files")

        mat_data = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)

        # Remove MATLAB metadata keys
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

        # Try to get the main data variable
        if len(mat_data) == 1:
            main_var = list(mat_data.values())[0]
        else:
            # Try common names
            for key in ["data", "TestData", "ocp", "electrode"]:
                if key in mat_data:
                    main_var = mat_data[key]
                    break
            else:
                main_var = mat_data

        soc, voltage, capacity = _extract_soc_voltage_from_mat(
            main_var, electrode_type, source=filepath
        )

        # Clean data - remove NaN, matching the CSV/Excel path below
        valid = ~(np.isnan(soc) | np.isnan(voltage))
        soc = soc[valid]
        voltage = voltage[valid]
        if len(soc) < 2:
            raise ValueError(f"Fewer than 2 valid (non-NaN) SOC/voltage points in {filepath}.")

        # Create electrode directly for .mat files
        base_type = electrode_type.replace("Blend2", "")
        ocp = ElectrodeOCP(
            soc=soc,
            voltage=voltage,
            name=name,
            electrode_type=base_type,
            capacity=capacity,
        )

        if smooth:
            ocp = ocp.smooth(window=smooth_window)

        return ocp
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # For CSV/Excel files, use column detection
    if soc_col is None or voltage_col is None:
        detected = auto_detect_columns(df, electrode_type)
        soc_col = soc_col or detected["soc"]
        voltage_col = voltage_col or detected["voltage"]
        capacity_col = capacity_col or detected.get("capacity")

    # Coerce to numeric first: text cells (e.g. from German-locale exports)
    # leave an object-dtype column that makes np.isnan raise a cryptic TypeError below.
    soc = pd.to_numeric(df[soc_col], errors="coerce").to_numpy(dtype=np.float64)
    voltage = pd.to_numeric(df[voltage_col], errors="coerce").to_numpy(dtype=np.float64)
    capacity = _extract_capacity_value(df[capacity_col].values) if capacity_col else None

    # Clean data - remove NaN
    valid = ~(np.isnan(soc) | np.isnan(voltage))
    soc = soc[valid]
    voltage = voltage[valid]
    if len(soc) < 2:
        raise ValueError(f"Fewer than 2 valid (non-NaN) SOC/voltage points in {filepath}.")

    base_type = electrode_type.replace("Blend2", "")
    ocp = ElectrodeOCP(
        soc=soc,
        voltage=voltage,
        name=name,
        electrode_type=base_type,
        capacity=capacity,
    )

    if smooth:
        ocp = ocp.smooth(window=smooth_window)

    return ocp


def load_pocv(
    filepath: str | Path,
    soc_col: str | None = None,
    voltage_col: str | None = None,
    capacity_col: str | None = None,
    smooth: bool = False,
    smooth_window: int = 30,
    csv_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Load pseudo-OCV (pOCV) data from file.

    Parameters
    ----------
    filepath : str or Path
        Path to data file.
    soc_col : str, optional
        Name of SOC column.
    voltage_col : str, optional
        Name of voltage column.
    capacity_col : str, optional
        Name of capacity column.
    smooth : bool
        Whether to smooth the data.
    smooth_window : int
        Smoothing window size.
    csv_kwargs : dict, optional
        Extra keyword arguments passed through to ``pd.read_csv`` unchanged.
        For German-locale exports: ``csv_kwargs={"sep": ";", "decimal": ","}``.

    Returns
    -------
    tuple
        (soc, voltage, capacity) where capacity is the total capacity in Ah when available.

    Examples
    --------
    >>> soc, voltage, capacity = load_pocv("cu1_pocv.csv")
    """
    from pydma.preprocessing.smoother import smooth_lowess

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = filepath.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(filepath, **(csv_kwargs or {}))
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif ext == ".mat":
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files")

        mat_data = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

        soc, voltage, capacity = _extract_soc_voltage_from_mat(
            mat_data, "fullcell", source=filepath
        )

        # Clean data - remove NaN, matching the CSV/Excel path below
        valid = ~(np.isnan(soc) | np.isnan(voltage))
        soc = soc[valid]
        voltage = voltage[valid]
        if len(soc) < 2:
            raise ValueError(f"Fewer than 2 valid (non-NaN) SOC/voltage points in {filepath}.")

        if smooth:
            voltage = smooth_lowess(voltage, soc, frac=smooth_window / len(soc))

        return soc, voltage, capacity
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Column detection for CSV/Excel
    if soc_col is None or voltage_col is None:
        detected = auto_detect_columns(df, "fullcell")
        soc_col = soc_col or detected["soc"]
        voltage_col = voltage_col or detected["voltage"]
        capacity_col = capacity_col or detected.get("capacity")

    # Coerce to numeric first: text cells (e.g. from German-locale exports)
    # leave an object-dtype column that makes np.isnan raise a cryptic TypeError below.
    soc = pd.to_numeric(df[soc_col], errors="coerce").to_numpy(dtype=np.float64)
    voltage = pd.to_numeric(df[voltage_col], errors="coerce").to_numpy(dtype=np.float64)

    # Clean data
    valid = ~(np.isnan(soc) | np.isnan(voltage))
    soc = soc[valid]
    voltage = voltage[valid]
    if len(soc) < 2:
        raise ValueError(f"Fewer than 2 valid (non-NaN) SOC/voltage points in {filepath}.")

    # Get capacity
    if capacity_col and capacity_col in df.columns:
        capacity = _extract_capacity_value(df[capacity_col].values)
    else:
        capacity = _extract_capacity_value(soc) if (np.nanmax(soc) - np.nanmin(soc)) > 1 else None

    if smooth:
        voltage = smooth_lowess(voltage, soc, frac=smooth_window / len(soc))

    return soc, voltage, capacity


_CU_FILE_EXTENSIONS = (".csv", ".mat", ".xlsx")


def _discover_cu_candidates(data_path: Path, cu_pattern: str) -> dict[int, Path]:
    """Find every CU directory/file that actually exists for a `{i}`-style pattern.

    Returns a mapping from numeric CU index to its resolved path. A CU
    directory takes precedence over a same-index file; among file
    candidates for the same index, the first matching extension in
    `_CU_FILE_EXTENSIONS` wins.

    Two names of the same kind can resolve to one index — ``CU1`` and ``CU01``
    both read as 1 — in which case only one of them is loaded and the
    collision is warned about.
    """
    if "{i}" not in cu_pattern:
        raise ValueError(f"cu_pattern must contain the '{{i}}' placeholder, got {cu_pattern!r}")
    prefix, suffix = cu_pattern.split("{i}", 1)

    def _index_from_name(name: str) -> int | None:
        if not (name.startswith(prefix) and name.endswith(suffix)):
            return None
        index_text = name[len(prefix) : len(name) - len(suffix)]
        return int(index_text) if index_text.isdigit() else None

    dir_candidates: dict[int, Path] = {}
    file_candidates: dict[int, dict[str, Path]] = {}

    def _warn_collision(index: int, kept: Path, dropped: Path) -> None:
        warnings.warn(
            f"CU index {index} is claimed by both '{dropped.name}' and "
            f"'{kept.name}'; only '{kept.name}' is loaded."
        )

    for entry in data_path.iterdir():
        if entry.is_dir():
            index = _index_from_name(entry.name)
            if index is not None:
                if index in dir_candidates:
                    _warn_collision(index, entry, dir_candidates[index])
                dir_candidates[index] = entry
        elif entry.suffix.lower() in _CU_FILE_EXTENSIONS:
            index = _index_from_name(entry.stem)
            if index is not None:
                by_ext = file_candidates.setdefault(index, {})
                ext = entry.suffix.lower()
                if ext in by_ext:
                    _warn_collision(index, entry, by_ext[ext])
                by_ext[ext] = entry

    candidates: dict[int, Path] = dict(dir_candidates)
    for index, by_ext in file_candidates.items():
        if index in candidates:
            continue
        for ext in _CU_FILE_EXTENSIONS:
            if ext in by_ext:
                candidates[index] = by_ext[ext]
                break

    return candidates


def load_aging_study(
    data_path: str | Path,
    direction: str = "charge",
    cu_pattern: str = "CU{i}",
    csv_kwargs: dict | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, float | None]]:
    """
    Load multiple pOCV measurements from an aging study.

    Parameters
    ----------
    data_path : str or Path
        Path to directory containing CU folders or a single file with all CUs.
    direction : str
        'charge' or 'discharge'.
    cu_pattern : str
        Pattern for CU folder/file names. Use {i} for CU index.
    csv_kwargs : dict, optional
        Extra keyword arguments passed through to ``pd.read_csv`` unchanged.
        For German-locale exports: ``csv_kwargs={"sep": ";", "decimal": ","}``.

    Returns
    -------
    dict
        Dictionary mapping CU names to (soc, voltage, capacity) tuples.

    Examples
    --------
    >>> data = load_aging_study("./aging_data/", direction="charge")
    >>> soc, voltage, capacity = data["CU1"]
    """
    data_path = Path(data_path)
    direction = direction.lower()
    if direction not in _DIRECTION_PATTERNS:
        raise ValueError(f"direction must be 'charge' or 'discharge', got {direction}")
    results = {}

    if data_path.is_file():
        # Single file containing all CUs (like MATLAB table format)
        if data_path.suffix == ".mat":
            if not HAS_SCIPY:
                raise ImportError("scipy is required to load .mat files")

            mat_data = loadmat(str(data_path), squeeze_me=True, struct_as_record=False)
            mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}
            return _extract_aging_study_from_mat(mat_data, direction=direction, source=data_path)
        raise NotImplementedError(
            "Loading all CUs from a single non-MATLAB file is not implemented. "
            "Please provide a directory of per-CU files or a MATLAB aging-study table."
        )

    # Directory with CU subfolders/files: collect every candidate that
    # actually exists instead of stopping at the first missing index.
    candidates = _discover_cu_candidates(data_path, cu_pattern)
    indices = sorted(candidates.keys())
    if indices:
        missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
        if missing:
            warnings.warn(
                f"Aging study at {data_path} is missing CU indices {missing} between "
                f"{cu_pattern.format(i=indices[0])} and {cu_pattern.format(i=indices[-1])}."
            )

    for index in indices:
        cu_name = cu_pattern.format(i=index)
        cu_path = candidates[index]

        if cu_path.is_dir():
            # Look for pOCV file in directory
            selected_files: list[Path] = []
            for pattern in _DIRECTION_PATTERNS[direction]:
                files = sorted(cu_path.glob(pattern))
                if direction == "charge":
                    # "charge" is a substring of "discharge". Exclude
                    # explicitly discharge-labelled files so selection does
                    # not depend on filesystem iteration order.
                    files = [
                        file
                        for file in files
                        if "discharge" not in file.stem.lower() and "dch" not in file.stem.lower()
                    ]
                else:
                    # Mirror the charge-side filter: a catch-all pattern can
                    # just as easily resolve to a charge-only file, so drop
                    # anything carrying an explicit charge-only marker. The
                    # "_ch" marker is read only where the stem says nothing
                    # about discharge, so a name such as
                    # "pocv_discharge_checkup" keeps its "_ch" from "_checkup"
                    # without being taken for a charge file.
                    files = [
                        file
                        for file in files
                        if not (
                            "charge" in file.stem.lower() and "discharge" not in file.stem.lower()
                        )
                        and not (
                            "_ch" in file.stem.lower()
                            and "dch" not in file.stem.lower()
                            and "discharge" not in file.stem.lower()
                        )
                    ]
                if files:
                    selected_files = files
                    break

            if not selected_files:
                warnings.warn(
                    f"No {direction} pOCV file found in {cu_path} "
                    f"(searched patterns: {_DIRECTION_PATTERNS[direction]}); skipping."
                )
                continue

            soc, voltage, capacity = load_pocv(selected_files[0], csv_kwargs=csv_kwargs)
            results[cu_name] = (soc, voltage, capacity)
        else:
            soc, voltage, capacity = load_pocv(cu_path, csv_kwargs=csv_kwargs)
            results[cu_name] = (soc, voltage, capacity)

    return results
