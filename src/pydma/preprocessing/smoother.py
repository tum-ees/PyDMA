"""
Signal smoothing and filtering utilities.

This module provides functions for smoothing electrode data using
various filtering methods including LOWESS, Savitzky-Golay, and
moving average filters.
"""

import numpy as np
from typing import Optional, Literal, Union
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


def smooth_lowess(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    frac: float = 0.1,
    it: int = 3,
) -> np.ndarray:
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) filter.

    Parameters
    ----------
    y : np.ndarray
        Data to smooth.
    x : np.ndarray, optional
        X values for the data. If None, uses indices.
    frac : float
        Fraction of data to use for each local regression.
        Default: 0.1 (10% of data points).
    it : int
        Number of iterations for robustness.

    Returns
    -------
    np.ndarray
        Smoothed data.

    Notes
    -----
    DIFFERENCE FROM MATLAB: MATLAB uses smooth(y, n, 'lowess') where n is
    the span (number of points). We use statsmodels.lowess with a fraction.
    To convert: frac ≈ n / len(y).
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        # Fallback to Savitzky-Golay if statsmodels not available
        import warnings
        warnings.warn(
            "statsmodels not available, falling back to Savitzky-Golay filter"
        )
        window = max(3, int(frac * len(y)))
        if window % 2 == 0:
            window += 1
        return smooth_savgol(y, window)

    y = np.asarray(y).flatten()
    n = len(y)

    # Guard against trivial inputs (avoids divide-by-zero warnings in statsmodels)
    if n < 2:
        return y.copy()

    if x is None:
        x = np.arange(n, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).flatten()

    # Handle NaN values
    valid_mask = ~np.isnan(y)
    if not np.all(valid_mask):
        y_valid = y[valid_mask]
        x_valid = x[valid_mask]
        if len(y_valid) < 2:
            return y.copy()
        smoothed_valid = lowess(
            y_valid, x_valid, frac=frac, it=it, return_sorted=False
        )
        y_smooth = np.full_like(y, np.nan)
        y_smooth[valid_mask] = smoothed_valid
        return y_smooth

    # Ensure x is strictly increasing and unique.
    # Duplicate or unsorted x can produce invalid weights in LOWESS.
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    if np.any(np.diff(x_sorted) == 0):
        # Collapse duplicate x by averaging y at identical positions
        x_unique, inv = np.unique(x_sorted, return_inverse=True)
        y_accum = np.zeros_like(x_unique, dtype=np.float64)
        counts = np.zeros_like(x_unique, dtype=np.float64)
        for i, grp in enumerate(inv):
            y_accum[grp] += y_sorted[i]
            counts[grp] += 1.0
        y_sorted = y_accum / np.maximum(counts, 1.0)
        x_sorted = x_unique

    # Clamp frac to a sensible range (LOWESS expects 0 < frac <= 1)
    frac = float(frac)
    if frac <= 0:
        frac = min(1.0, 2.0 / max(len(x_sorted), 2))
    elif frac > 1:
        frac = 1.0

    # Apply LOWESS (sorted x)
    smoothed = lowess(y_sorted, x_sorted, frac=frac, it=it, return_sorted=False)

    # Map smoothed data back to original x grid
    smoothed = np.interp(x, x_sorted, smoothed)

    return smoothed


def smooth_savgol(
    y: np.ndarray,
    window_length: int = 51,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter.

    Parameters
    ----------
    y : np.ndarray
        Data to smooth.
    window_length : int
        Length of the filter window (must be odd).
    polyorder : int
        Order of polynomial to fit.

    Returns
    -------
    np.ndarray
        Smoothed data.

    Notes
    -----
    DIFFERENCE FROM MATLAB: MATLAB uses sgolayfilt(y, order, framelen).
    We use scipy.signal.savgol_filter which has the same algorithm.
    The window_length must be odd and greater than polyorder.
    """
    y = np.asarray(y).flatten()

    # Ensure window_length is valid (MATLAB-style: odd and >= order+1)
    window_length = int(window_length)
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 1
        if window_length % 2 == 0:
            window_length += 1

    # Handle NaN values
    valid_mask = ~np.isnan(y)
    if not np.all(valid_mask):
        y_valid = y[valid_mask]
        try:
            smoothed_valid = savgol_filter(y_valid, window_length, polyorder)
        except ValueError:
            # MATLAB behavior: if sgolayfilt fails, return original data
            return y.copy()
        y_smooth = np.full_like(y, np.nan)
        y_smooth[valid_mask] = smoothed_valid
        return y_smooth

    try:
        return savgol_filter(y, window_length, polyorder)
    except ValueError:
        # MATLAB behavior: if sgolayfilt fails, return original data
        return y.copy()


def smooth_moving_average(
    y: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Apply moving average filter.

    Parameters
    ----------
    y : np.ndarray
        Data to smooth.
    window : int
        Window size for moving average.

    Returns
    -------
    np.ndarray
        Smoothed data.

    Notes
    -----
    DIFFERENCE FROM MATLAB: MATLAB's smoothdata with 'movmean' option.
    We use scipy.ndimage.uniform_filter1d for efficiency.
    """
    y = np.asarray(y).flatten()
    window = max(1, int(window))

    return uniform_filter1d(y, size=window, mode="nearest")


def smooth_gaussian(
    y: np.ndarray,
    sigma: float = 10,
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter.

    Parameters
    ----------
    y : np.ndarray
        Data to smooth.
    sigma : float
        Standard deviation for Gaussian kernel.

    Returns
    -------
    np.ndarray
        Smoothed data.
    """
    from scipy.ndimage import gaussian_filter1d

    y = np.asarray(y).flatten()
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")


def apply_filter(
    data: np.ndarray,
    method: str = "sgolay",
    window: Optional[int] = None,
    window_frac: Optional[float] = None,
    order: int = 3,
    repeat: int = 1,
    fill_outliers: bool = False,
) -> np.ndarray:
    """
    Apply smoothing filter to data with flexible options.

    This is a unified interface matching MATLAB's apply_filter.m functionality.

    Parameters
    ----------
    data : np.ndarray
        Data to filter.
    method : str
        Filter method: 'sgolay', 'lowess', 'movmean', 'movmedian', 'gaussian'.
    window : int, optional
        Fixed window size. If None, uses default based on method.
    window_frac : float, optional
        Window size as fraction of data length.
        Takes precedence over fixed window.
    order : int
        Polynomial order for Savitzky-Golay filter.
    repeat : int
        Number of times to apply the filter.
    fill_outliers : bool
        Whether to fill outliers before filtering.

    Returns
    -------
    np.ndarray
        Filtered data.

    Notes
    -----
    DIFFERENCE FROM MATLAB: Simplified interface compared to MATLAB's
    varargin-based parsing. Same underlying algorithms.

    Examples
    --------
    >>> filtered = apply_filter(data, method='sgolay', window=51)
    >>> filtered = apply_filter(data, method='lowess', window_frac=0.1)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    # Default window sizes per method
    default_windows = {
        "sgolay": 50,
        "lowess": 30,
        "movmean": 10,
        "movmedian": 10,
        "gaussian": 10,
    }

    # Determine window size
    if window_frac is not None:
        window = max(3, int(window_frac * n))
    elif window is None:
        window = default_windows.get(method.lower(), 30)

    window = int(window)

    # Handle outliers if requested
    if fill_outliers:
        from scipy.stats import zscore
        z = zscore(data, nan_policy="omit")
        outliers = np.abs(z) > 3
        if np.any(outliers):
            # Linear interpolation for outliers
            valid = ~outliers
            x = np.arange(n)
            data = data.copy()
            data[outliers] = np.interp(x[outliers], x[valid], data[valid])

    # Apply filter
    result = data.copy()
    method = method.lower()

    for _ in range(repeat):
        if method == "sgolay" or method == "savgol":
            result = smooth_savgol(result, window, order)
        elif method == "lowess":
            frac = window / n if n > 0 else 0.1
            # MATLAB smooth(data, span, 'lowess') uses 0 robustness iterations
            result = smooth_lowess(result, frac=frac, it=0)
        elif method == "movmean":
            result = smooth_moving_average(result, window)
        elif method == "movmedian":
            from scipy.ndimage import median_filter
            result = median_filter(result, size=window, mode="nearest")
        elif method == "gaussian":
            result = smooth_gaussian(result, sigma=window / 3)
        else:
            raise ValueError(f"Unknown filter method: {method}")

    return result


def assure_non_zero_dv(voltage: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Ensure no consecutive voltage values are identical.

    This is needed for ICA calculations where dV=0 causes division errors.

    Parameters
    ----------
    voltage : np.ndarray
        Voltage array.
    eps : float
        Minimum voltage difference to enforce.

    Returns
    -------
    np.ndarray
        Voltage array with non-zero differences.

    Notes
    -----
    DIFFERENCE FROM MATLAB: Replicates assure_non_zero_dV.m behavior.
    """
    voltage = np.asarray(voltage).flatten().copy()
    n = len(voltage)

    for i in range(1, n):
        if abs(voltage[i] - voltage[i - 1]) < eps:
            # Add small offset
            voltage[i] = voltage[i - 1] + eps

    return voltage
