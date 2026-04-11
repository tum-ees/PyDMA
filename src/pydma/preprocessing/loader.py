"""
Data loading utilities with automatic column detection.

This module provides functions for loading electrode OCP and pOCV data
from various file formats with automatic column name detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
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


def _normalize_name(name: Any) -> str:
    """Normalize a key/column name for fuzzy matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_matching_name(names: List[str], patterns: List[str]) -> Optional[str]:
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


def _extract_capacity_value(values: Any) -> Optional[float]:
    """Extract a scalar capacity from a scalar, repeated series, or capacity trace."""
    arr = np.asarray(values, dtype=np.float64).flatten()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if arr.size == 1 or np.allclose(arr, arr[0]):
        return float(arr[0])

    span = float(arr.max() - arr.min())
    if span > 0:
        return span
    return float(arr.max())


def _coerce_mat_container(data: Any) -> Optional[Dict[str, Any]]:
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


def _extract_table_rows(container: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a MATLAB table-like dict of columns to a list of row dicts."""
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

    rows = []
    for idx in range(row_count):
        row: Dict[str, Any] = {}
        for key, value in container.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                row[key] = arr.item()
            elif len(arr) == row_count:
                row[key] = arr[idx]
            else:
                row[key] = value
        rows.append(row)
    return rows


def _find_data_keys(mapping: Dict[str, Any], electrode_type: str) -> Tuple[Optional[str], Optional[str]]:
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


def _extract_fullcell_capacity_from_mapping(mapping: Dict[str, Any], x_key: Optional[str]) -> Optional[float]:
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
        return span
    return None


def _coerce_cu_name(value: Any) -> str:
    """Normalize a CU identifier to the `CU<n>` style used by PyDMA."""
    arr = np.asarray(value).flatten()
    if arr.size == 0:
        raise ValueError("Empty CU identifier encountered in aging-study data.")

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
    mat_data: Dict[str, Any],
    direction: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]]:
    """Extract multi-CU pOCV data from a single MATLAB file."""
    results: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]] = {}
    direction = direction.lower()

    explicit_direction_keys = {
        "charge": ["Testdata_pOCV_CH", "pOCV_CH", "charge", "Testdata_charge"],
        "discharge": ["Testdata_pOCV_DCH", "pOCV_DCH", "discharge", "Testdata_discharge"],
    }

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
                soc, voltage, capacity = _extract_soc_voltage_from_mat(container[cu_key], "fullcell")
                results[_coerce_cu_name(cu_key)] = (soc, voltage, capacity)
                parsed_any = True
            if parsed_any:
                return results

        # Table-like dict of columns.
        rows = _extract_table_rows(container)
        if rows:
            parsed_any = False
            for row in rows:
                row_container = _coerce_mat_container(row) or row
                cu_key = _find_matching_name(list(row_container.keys()), ["CU", "cu"])
                if cu_key is None:
                    continue

                data_key = _find_matching_name(
                    list(row_container.keys()),
                    explicit_direction_keys[direction],
                )
                if data_key is None:
                    continue

                soc, voltage, capacity = _extract_soc_voltage_from_mat(
                    row_container[data_key],
                    "fullcell",
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
) -> Dict[str, str]:
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
        FULLCELL_CAPACITY_COLUMN_PATTERNS if electrode_type == "fullcell" else CAPACITY_COLUMN_PATTERNS
    )
    capacity_col = _find_matching_name(
        [col for col in columns if col not in {result["soc"], result["voltage"]}],
        capacity_patterns,
    )
    if capacity_col is not None:
        result["capacity"] = capacity_col

    return result


def _extract_soc_voltage_from_mat(
    data: Any, electrode_type: str
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Extract SOC and voltage from MATLAB .mat file structure.

    Parameters
    ----------
    data : dict
        Loaded .mat file data.
    electrode_type : str
        Type of electrode.

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
        capacity = (
            _extract_fullcell_capacity_from_mapping(mapping, x_key)
            if electrode_type == "fullcell"
            else _extract_capacity_value(
                mapping.get(_find_matching_name(list(mapping.keys()), CAPACITY_COLUMN_PATTERNS))
            )
        )
        return soc, voltage, capacity

    raise ValueError(
        f"Could not find SOC/capacity and voltage columns in .mat file. "
        f"Top-level keys: {list(top_level.keys())}"
    )


def load_ocp(
    filepath: Union[str, Path],
    electrode_type: str = "anode",
    soc_col: Optional[str] = None,
    voltage_col: Optional[str] = None,
    capacity_col: Optional[str] = None,
    smooth: bool = False,
    smooth_window: int = 30,
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
        df = pd.read_csv(filepath)
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

        soc, voltage, capacity = _extract_soc_voltage_from_mat(main_var, electrode_type)

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

    soc = df[soc_col].values
    voltage = df[voltage_col].values
    capacity = _extract_capacity_value(df[capacity_col].values) if capacity_col else None

    # Clean data - remove NaN
    valid = ~(np.isnan(soc) | np.isnan(voltage))
    soc = soc[valid]
    voltage = voltage[valid]

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
    filepath: Union[str, Path],
    soc_col: Optional[str] = None,
    voltage_col: Optional[str] = None,
    capacity_col: Optional[str] = None,
    smooth: bool = False,
    smooth_window: int = 30,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
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
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif ext == ".mat":
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files")

        mat_data = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

        soc, voltage, capacity = _extract_soc_voltage_from_mat(mat_data, "fullcell")

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

    soc = df[soc_col].values
    voltage = df[voltage_col].values

    # Clean data
    valid = ~(np.isnan(soc) | np.isnan(voltage))
    soc = soc[valid]
    voltage = voltage[valid]

    # Get capacity
    if capacity_col and capacity_col in df.columns:
        capacity = _extract_capacity_value(df[capacity_col].values)
    else:
        capacity = _extract_capacity_value(soc) if (np.nanmax(soc) - np.nanmin(soc)) > 1 else None

    if smooth:
        voltage = smooth_lowess(voltage, soc, frac=smooth_window / len(soc))

    return soc, voltage, capacity


def load_aging_study(
    data_path: Union[str, Path],
    direction: str = "charge",
    cu_pattern: str = "CU{i}",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float]]]:
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
            return _extract_aging_study_from_mat(mat_data, direction=direction)
        raise NotImplementedError(
            "Loading all CUs from a single non-MATLAB file is not implemented. "
            "Please provide a directory of per-CU files or a MATLAB aging-study table."
        )
    else:
        # Directory with CU subfolders
        i = 1
        while True:
            cu_name = cu_pattern.format(i=i)
            cu_path = data_path / cu_name

            if not cu_path.exists():
                # Try as file
                for ext in [".csv", ".mat", ".xlsx"]:
                    if (data_path / f"{cu_name}{ext}").exists():
                        cu_path = data_path / f"{cu_name}{ext}"
                        break
                else:
                    break

            if cu_path.is_dir():
                # Look for pOCV file in directory
                for pattern in _DIRECTION_PATTERNS[direction]:
                    files = list(cu_path.glob(pattern))
                    if files:
                        soc, voltage, capacity = load_pocv(files[0])
                        results[cu_name] = (soc, voltage, capacity)
                        break
            else:
                soc, voltage, capacity = load_pocv(cu_path)
                results[cu_name] = (soc, voltage, capacity)

            i += 1

    return results
