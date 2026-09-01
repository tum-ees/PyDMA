"""
Electrode OCP (Open Circuit Potential) class.

This module provides the ElectrodeOCP class for handling electrode
open circuit potential data for anodes and cathodes.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d


# eq=False: the numpy fields would make a generated __eq__ raise on the
# ambiguous truth value of an array comparison.
@dataclass(eq=False)
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
    capacity: float | None = None
    is_smoothed: bool = False
    _interpolator: interp1d | None = field(default=None, repr=False, compare=False)

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

        # Non-finite samples have to be rejected here rather than left to a later
        # check: every comparison against NaN is False, so a NaN slips past the
        # ordering, the normalisation and the orientation decision below and only
        # surfaces as a silently poisoned interpolant.
        for axis_name, values in (("soc", self.soc), ("voltage", self.voltage)):
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                i = int(bad[0])
                raise ValueError(
                    f"{axis_name} must be finite for electrode '{self.name}'; "
                    f"sample {i} is {float(values[i])!r} "
                    f"({int(bad.size)} non-finite sample(s) in total)."
                )

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

        if self.soc.min() < 0.0 or self.soc.max() > 1.0:
            warnings.warn(
                f"Electrode '{self.name}' keeps a SOC axis outside [0, 1]: "
                f"[{float(self.soc.min()):.6g}, {float(self.soc.max()):.6g}]. "
                "It is within the normalisation tolerance, so it is used as given.",
                stacklevel=2,
            )

        if self.electrode_type not in ("anode", "cathode"):
            raise ValueError(
                "electrode_type must be 'anode' or 'cathode', " f"got {self.electrode_type}"
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

        # The convention is a statement about the two ends of the curve, so it is
        # read off the endpoints of the SOC-sorted data. A least-squares slope
        # answers a different question and can be dominated by a plateau in the
        # middle of the curve.
        v_span = float(self.voltage[-1] - self.voltage[0])
        if abs(v_span) < 1e-3:
            raise ValueError(
                f"Cannot determine the OCP orientation of electrode '{self.name}': "
                f"the curve is flat, {float(self.voltage[0]):.6g} V at the lowest SOC "
                f"and {float(self.voltage[-1]):.6g} V at the highest differ by only "
                f"{v_span:.3g} V (at least 1e-3 V required)."
            )

        # Anode: voltage should DECREASE with increasing SOC (lithiation).
        # Cathode: PyDMA's internal convention has it INCREASE with SOC.
        if self.electrode_type == "anode":
            needs_mirror = v_span > 0
        else:
            needs_mirror = v_span < 0

        if needs_mirror:
            # Reflect about 0.5. The soc axis carries absolute stoichiometry, so
            # the reflection point is the stoichiometry range itself: mirroring a
            # partial-window curve about its own centre instead would keep the
            # window width but move every absolute sto value, which the fit does
            # not notice and every sto-referenced export gets wrong.
            self.soc = 1.0 - self.soc
            # Restore increasing SOC order after transformation
            self.soc = np.flip(self.soc)
            self.voltage = np.flip(self.voltage)

        # Build interpolator
        self._build_interpolator()

    def _build_interpolator(self):
        """Build the interpolation function.

        Queries outside the SOC support are clamped to the potentials at the two
        ends of the support. ``__post_init__`` leaves the SOC axis sorted
        ascending, so those are ``voltage[0]`` and ``voltage[-1]``.
        """
        self._interpolator = interp1d(
            self.soc,
            self.voltage,
            kind="linear",
            bounds_error=False,
            fill_value=(self.voltage[0], self.voltage[-1]),
        )

    def interpolate(self, soc_query: float | np.ndarray) -> np.ndarray:
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
        Queries outside the SOC support are clamped to the potentials at the two
        ends of the support rather than extrapolated.
        """
        if self._interpolator is None:
            raise RuntimeError(
                "ElectrodeOCP has no interpolator. __post_init__ builds one for every "
                "instance, so this signals an object that bypassed initialisation. "
                "Raised rather than asserted so the invariant survives python -O."
            )
        return np.asarray(self._interpolator(soc_query))

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
            Smoothing window size, in samples. Must be positive.
        method : str
            Smoothing method: 'lowess' or 'savgol'.

        Returns
        -------
        ElectrodeOCP
            New ElectrodeOCP with smoothed data.

        Raises
        ------
        ValueError
            If ``window`` is not positive, or ``method`` is unknown.

        Notes
        -----
        DIFFERENCE FROM MATLAB: MATLAB uses smooth(y, n, 'lowess').
        We use statsmodels.lowess or scipy.savgol_filter.
        """
        from pydma.preprocessing.smoother import smooth_lowess, smooth_savgol

        if window <= 0:
            raise ValueError(f"smooth() needs a positive window, got {window!r}.")

        if method == "lowess":
            # LOWESS needs a fraction in (0, 1], so a window wider than the curve
            # is capped rather than passed on.
            frac = min(window / len(self.voltage), 1.0)
            smoothed_voltage = smooth_lowess(
                self.voltage,
                self.soc,
                frac=frac,
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

    def get_voltage_range(self) -> tuple[float, float]:
        """
        Get the voltage range of this electrode.

        Returns
        -------
        tuple
            (min_voltage, max_voltage)
        """
        return float(self.voltage.min()), float(self.voltage.max())

    def get_soc_at_voltage(self, voltage: float) -> float | None:
        """
        Get the SOC at a given voltage by inverting the OCP curve.

        Parameters
        ----------
        voltage : float
            Voltage value.

        Returns
        -------
        float or None
            Linearly interpolated SOC, or None if the voltage lies outside the
            curve's voltage range.

        Raises
        ------
        ValueError
            If the voltage axis reverses direction. V -> SOC is then not a
            function and the inversion has no defined answer.

        Notes
        -----
        This is used for blend electrode calculations where we need
        to find Q(V) from V(Q) data.
        """
        v = np.asarray(self.voltage, dtype=float)
        signs = np.sign(np.diff(v))
        nonzero = signs[signs != 0]
        if nonzero.size and not bool(np.all(nonzero == nonzero[0])):
            first = int(np.flatnonzero(signs == -nonzero[0])[0])
            raise ValueError(
                f"Cannot invert electrode '{self.name}': its voltage axis is not "
                f"monotone, it reverses between samples {first} and {first + 1} "
                f"({float(v[first]):.6g} V -> {float(v[first + 1]):.6g} V)."
            )

        v_min, v_max = self.get_voltage_range()
        if voltage < v_min or voltage > v_max:
            return None

        if nonzero.size and nonzero[0] < 0:
            return float(np.interp(voltage, v[::-1], self.soc[::-1]))
        return float(np.interp(voltage, v, self.soc))

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
