"""
Data loading utilities with automatic column detection.

This module provides functions for loading electrode OCP and pOCV data
from various file formats with automatic column name detection.
"""

from __future__ import annotations

import os
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
    columns = [c.lower() for c in df.columns]
    column_map = dict(zip(columns, df.columns))

    result = {}

    # Detect SOC column
    soc_col = None
    for pattern in SOC_COLUMN_PATTERNS:
        for col in columns:
            if pattern in col:
                soc_col = column_map[col]
                break
        if soc_col:
            break

    if not soc_col:
        raise ValueError(
            f"Could not detect SOC column. Looked for patterns: {SOC_COLUMN_PATTERNS}. "
            f"Available columns: {list(df.columns)}"
        )
    result["soc"] = soc_col

    # Detect voltage column based on electrode type
    voltage_col = None

    # Prioritize electrode-specific names
    if electrode_type in ("cathode", "cathodeBlend2"):
        priority_patterns = ["uca", "u_ca", "ucathode"] + VOLTAGE_COLUMN_PATTERNS
    elif electrode_type in ("anode", "anodeBlend2"):
        priority_patterns = ["uan", "u_an", "uanode"] + VOLTAGE_COLUMN_PATTERNS
    else:  # fullcell
        priority_patterns = VOLTAGE_COLUMN_PATTERNS

    for pattern in priority_patterns:
        for col in columns:
            if pattern in col:
                voltage_col = column_map[col]
                break
        if voltage_col:
            break

    if not voltage_col:
        raise ValueError(
            f"Could not detect voltage column. Looked for patterns: {priority_patterns}. "
            f"Available columns: {list(df.columns)}"
        )
    result["voltage"] = voltage_col

    # Detect capacity column (optional)
    for pattern in CAPACITY_COLUMN_PATTERNS:
        for col in columns:
            if pattern in col and col != result["soc"].lower():
                result["capacity"] = column_map[col]
                break
        if "capacity" in result:
            break

    return result


def _extract_soc_voltage_from_mat(
    data: dict, electrode_type: str
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
    # Try to find TestData sub-structure first
    if "TestData" in data:
        data = data["TestData"]

    # Handle nested structures from MATLAB
    if isinstance(data, np.ndarray) and data.dtype.names:
        # Structured array - convert to dict
        data = {name: data[name].flatten() for name in data.dtype.names}

    # Look for SOC column
    soc = None
    for key in data.keys():
        key_lower = key.lower()
        if any(p in key_lower for p in SOC_COLUMN_PATTERNS):
            soc = np.asarray(data[key]).flatten()
            break

    if soc is None:
        raise ValueError(f"Could not find SOC column in .mat file. Keys: {list(data.keys())}")

    # Look for voltage column based on electrode type
    voltage = None
    if electrode_type in ("cathode", "cathodeBlend2"):
        priority_keys = ["UCa", "U_Ca", "U"]
    elif electrode_type in ("anode", "anodeBlend2"):
        priority_keys = ["UAn", "U_An", "U"]
    else:
        priority_keys = ["U", "voltage", "OCV"]

    for key in priority_keys:
        if key in data:
            voltage = np.asarray(data[key]).flatten()
            break

    if voltage is None:
        # Fallback to pattern matching
        for key in data.keys():
            key_lower = key.lower()
            if any(p in key_lower for p in VOLTAGE_COLUMN_PATTERNS):
                voltage = np.asarray(data[key]).flatten()
                break

    if voltage is None:
        raise ValueError(f"Could not find voltage column in .mat file. Keys: {list(data.keys())}")

    # Look for capacity (optional)
    capacity = None
    for key in data.keys():
        key_lower = key.lower()
        if any(p in key_lower for p in CAPACITY_COLUMN_PATTERNS):
            cap_data = np.asarray(data[key]).flatten()
            if len(cap_data) == 1:
                capacity = float(cap_data[0])
            else:
                # Take max as capacity
                capacity = float(np.max(cap_data) - np.min(cap_data))
            break

    return soc, voltage, capacity


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

        soc, voltage, capacity = _extract_soc_voltage_from_mat(
            mat_data if isinstance(main_var, dict) else {"data": main_var},
            electrode_type
        )

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
    capacity = df[capacity_col].values.max() if capacity_col else None

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
) -> Tuple[np.ndarray, np.ndarray, float]:
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
        (soc, voltage, capacity) where capacity is the total capacity in Ah.

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

        return soc, voltage, capacity or 1.0
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
        capacity = df[capacity_col].max()
    else:
        # Estimate from SOC range
        capacity = soc.max() - soc.min()
        if capacity > 1:  # SOC might be in Ah, not normalized
            pass
        else:
            capacity = 1.0  # Assume normalized

    if smooth:
        voltage = smooth_lowess(voltage, soc, frac=smooth_window / len(soc))

    return soc, voltage, capacity


def load_aging_study(
    data_path: Union[str, Path],
    direction: str = "charge",
    cu_pattern: str = "CU{i}",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
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
    results = {}

    if data_path.is_file():
        # Single file containing all CUs (like MATLAB table format)
        if data_path.suffix == ".mat":
            if not HAS_SCIPY:
                raise ImportError("scipy is required to load .mat files")

            mat_data = loadmat(str(data_path), squeeze_me=True, struct_as_record=False)
            # This would need specific parsing based on file structure
            # For now, return empty - users should use load_pocv for each CU
            raise NotImplementedError(
                "Loading all CUs from single .mat file not yet implemented. "
                "Please load each CU individually using load_pocv()."
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
                for pattern in ["*pocv*.csv", "*pOCV*.csv", "*charge*.csv", "*discharge*.csv"]:
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
