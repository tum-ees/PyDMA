"""
Electrode OCP (Open Circuit Potential) class.

This module provides the ElectrodeOCP class for handling electrode
open circuit potential data for anodes and cathodes.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
from scipy.interpolate import interp1d


@dataclass
class ElectrodeOCP:
    """
    Electrode Open Circuit Potential data container.

    This class stores and manipulates OCP data for a single electrode
    (anode or cathode). It provides interpolation, normalization,
    and smoothing functionality.

    Attributes
    ----------
    soc : np.ndarray
        State of charge values (normalized 0-1).
    voltage : np.ndarray
        Voltage/potential values in Volts.
    name : str
        Name/identifier for this electrode.
    electrode_type : str
        Type of electrode: 'anode' or 'cathode'.
    capacity : float, optional
        Electrode capacity in Ah (if known).
    is_smoothed : bool
        Whether the data has been smoothed.

    Examples
    --------
    >>> import numpy as np
    >>> soc = np.linspace(0, 1, 100)
    >>> voltage = 0.1 + 0.2 * soc  # Simple linear example
    >>> ocp = ElectrodeOCP(soc=soc, voltage=voltage, name="Graphite", electrode_type="anode")
    >>> v_at_50 = ocp.interpolate(0.5)
    """

    soc: np.ndarray
    voltage: np.ndarray
    name: str = ""
    electrode_type: str = "anode"  # 'anode' or 'cathode'
    capacity: Optional[float] = None
    is_smoothed: bool = False
    _interpolator: Optional[interp1d] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Validate and prepare data after initialization."""
        self.soc = np.asarray(self.soc).flatten()
        self.voltage = np.asarray(self.voltage).flatten()

        if len(self.soc) != len(self.voltage):
            raise ValueError(
                f"soc and voltage must have same length, "
                f"got {len(self.soc)} and {len(self.voltage)}"
            )

        if len(self.soc) < 2:
            raise ValueError("Need at least 2 data points")

        # Ensure SOC is increasing
        if self.soc[0] > self.soc[-1]:
            self.soc = np.flip(self.soc)
            self.voltage = np.flip(self.voltage)

        # Normalize SOC to 0-1 if not already
        soc_min, soc_max = self.soc.min(), self.soc.max()
        if abs(soc_max - soc_min) > 1e-10:
            if soc_min < -0.01 or soc_max > 1.01:
                # SOC is not in 0-1 range, normalize
                self.soc = (self.soc - soc_min) / (soc_max - soc_min)

        if self.electrode_type not in ("anode", "cathode"):
            raise ValueError(
                "electrode_type must be 'anode' or 'cathode', "
                f"got {self.electrode_type}"
            )

        # =============================================================================
        # VALIDATE AND AUTO-CORRECT ELECTRODE OCP CONVENTION
        # =============================================================================
        # Standard electrochemical convention for half-cell OCP:
        # - Voltage should DECREASE with increasing stoichiometry (lithium content)
        #   for BOTH anode and cathode materials.
        #
        # PyDMA internal convention:
        # - Anode: voltage DECREASES with increasing SOC (same as standard)
        # - Cathode: voltage INCREASES with increasing SOC (inverted for internal use)
        #
        # We validate and auto-correct to PyDMA's internal convention.
        # =============================================================================

        # Calculate average slope to check convention
        slope = np.polyfit(self.soc, self.voltage, 1)[0]

        # Anode: voltage should DECREASE with increasing SOC (lithiation)
        # If voltage[0] < voltage[-1], it's increasing → flip SOC
        if self.electrode_type == "anode":
            if slope > 0:  # Voltage is INCREASING with SOC (wrong convention)
                self.soc = 1.0 - self.soc
                # Restore increasing SOC order after transformation
                self.soc = np.flip(self.soc)
                self.voltage = np.flip(self.voltage)

        # Cathode: Standard convention has voltage DECREASING with stoichiometry
        # PyDMA expects voltage INCREASING with SOC (internal convention)
        # If slope < 0 (voltage decreasing with SOC), it's standard convention → flip to PyDMA
        if self.electrode_type == "cathode":
            if slope < 0:  # Voltage is DECREASING with SOC (standard convention)
                self.soc = 1.0 - self.soc
                # Restore increasing SOC order after transformation
                self.soc = np.flip(self.soc)
                self.voltage = np.flip(self.voltage)

        # Build interpolator
        self._build_interpolator()

    def _build_interpolator(self):
        """Build the interpolation function."""
        # DIFFERENCE FROM MATLAB: MATLAB uses interp1 with 'linear' and 0 for extrapolation
        # We use scipy interp1d with bounds_error=False and fill_value=0
        self._interpolator = interp1d(
            self.soc,
            self.voltage,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,  # Return 0 outside bounds, matching MATLAB behavior
        )

    def interpolate(self, soc_query: Union[float, np.ndarray]) -> np.ndarray:
        """
        Interpolate voltage at given SOC values.

        Parameters
        ----------
        soc_query : float or np.ndarray
            SOC value(s) at which to interpolate voltage.

        Returns
        -------
        np.ndarray
            Interpolated voltage value(s).

        Notes
        -----
        DIFFERENCE FROM MATLAB: MATLAB's interp1(x,y,xq,'linear',0) returns 0
        for values outside the range. We replicate this behavior.
        """
        return self._interpolator(soc_query)

    def get_potential_at_scaled_soc(
        self, soc: np.ndarray, alpha: float, beta: float
    ) -> np.ndarray:
        """
        Get potential at scaled SOC: alpha * soc + beta.

        This is the core operation for OCV reconstruction where:
        U_electrode(SOC) = U(alpha * SOC + beta)

        Parameters
        ----------
        soc : np.ndarray
            Full-cell SOC grid (typically 0-1).
        alpha : float
            Scaling factor (capacity ratio).
        beta : float
            Offset (SOC shift).

        Returns
        -------
        np.ndarray
            Electrode potential at scaled SOC values.
        """
        scaled_soc = alpha * soc + beta
        return self.interpolate(scaled_soc)

    def resample(self, n_points: int = 1000) -> "ElectrodeOCP":
        """
        Resample OCP to uniform SOC grid.

        Parameters
        ----------
        n_points : int
            Number of points in resampled data.

        Returns
        -------
        ElectrodeOCP
            New ElectrodeOCP with resampled data.
        """
        new_soc = np.linspace(0, 1, n_points)
        new_voltage = self.interpolate(new_soc)

        return ElectrodeOCP(
            soc=new_soc,
            voltage=new_voltage,
            name=self.name,
            electrode_type=self.electrode_type,
            capacity=self.capacity,
            is_smoothed=self.is_smoothed,
        )

    def smooth(self, window: int = 30, method: str = "lowess") -> "ElectrodeOCP":
        """
        Smooth the OCP data.

        Parameters
        ----------
        window : int
            Smoothing window size.
        method : str
            Smoothing method: 'lowess' or 'savgol'.

        Returns
        -------
        ElectrodeOCP
            New ElectrodeOCP with smoothed data.

        Notes
        -----
        DIFFERENCE FROM MATLAB: MATLAB uses smooth(y, n, 'lowess').
        We use statsmodels.lowess or scipy.savgol_filter.
        """
        from pydma.preprocessing.smoother import smooth_lowess, smooth_savgol

        if method == "lowess":
            smoothed_voltage = smooth_lowess(
                self.voltage,
                self.soc,
                frac=window / len(self.voltage),
            )
        elif method == "savgol":
            smoothed_voltage = smooth_savgol(self.voltage, window)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return ElectrodeOCP(
            soc=self.soc.copy(),
            voltage=smoothed_voltage,
            name=self.name,
            electrode_type=self.electrode_type,
            capacity=self.capacity,
            is_smoothed=True,
        )

    def get_voltage_range(self) -> Tuple[float, float]:
        """
        Get the voltage range of this electrode.

        Returns
        -------
        tuple
            (min_voltage, max_voltage)
        """
        return float(self.voltage.min()), float(self.voltage.max())

    def get_soc_at_voltage(self, voltage: float) -> Optional[float]:
        """
        Get approximate SOC at a given voltage.

        Parameters
        ----------
        voltage : float
            Voltage value.

        Returns
        -------
        float or None
            Approximate SOC, or None if voltage is out of range.

        Notes
        -----
        This is used for blend electrode calculations where we need
        to find Q(V) from V(Q) data.
        """
        v_min, v_max = self.get_voltage_range()
        if voltage < v_min or voltage > v_max:
            return None

        # Find closest voltage and interpolate
        idx = np.argmin(np.abs(self.voltage - voltage))
        return float(self.soc[idx])

    def copy(self) -> "ElectrodeOCP":
        """
        Create a copy of this electrode.

        Returns
        -------
        ElectrodeOCP
            Copy of this electrode.
        """
        return ElectrodeOCP(
            soc=self.soc.copy(),
            voltage=self.voltage.copy(),
            name=self.name,
            electrode_type=self.electrode_type,
            capacity=self.capacity,
            is_smoothed=self.is_smoothed,
        )

    def __len__(self) -> int:
        """Number of data points."""
        return len(self.soc)

    def __repr__(self) -> str:
        return (
            f"ElectrodeOCP(name='{self.name}', type='{self.electrode_type}', "
            f"points={len(self)}, V=[{self.voltage.min():.3f}, {self.voltage.max():.3f}])"
        )
