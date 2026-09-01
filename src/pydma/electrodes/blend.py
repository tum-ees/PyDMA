"""
Blend electrode model for mixed active materials.

This module provides the BlendElectrode class for handling blended electrodes
such as Silicon-Graphite anodes where the OCP is a weighted combination
of two component materials.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from pydma.electrodes.electrode import ElectrodeOCP

# Raised rather than asserted so the invariant survives python -O.
_UNPREPARED_MSG = (
    "BlendElectrode common-window arrays are missing after _prepare_blend_data(); "
    "every code path builds them, so this signals a defect rather than bad input."
)


# eq=False: the numpy fields would make a generated __eq__ raise on the
# ambiguous truth value of an array comparison.
@dataclass(eq=False)
class BlendElectrode:
    """
    Blended electrode model combining two component materials.

    This class handles blended electrodes (e.g., Si-Gr anodes) where the
    total capacity is a weighted sum of two components:

    Q_blend(V) = gamma * Q_blend2(V) + (1 - gamma) * Q_blend1(V)

    The blending is done on a common voltage grid, not SOC.

    Meaning of gamma
    ----------------
    ``gamma`` weights the two components on the COMMON VOLTAGE WINDOW. Where
    the two components share one carrier, as they do along the
    :func:`~pydma.silicon.generator.generate_si_curve` path, it is the capacity
    share of blend2 in the blended electrode.

    Attributes
    ----------
    blend1 : ElectrodeOCP
        Primary component (e.g., Graphite).
    blend2 : ElectrodeOCP
        Secondary component (e.g., Silicon).
    electrode_type : str
        Type of electrode: 'anode' or 'cathode'.
    name : str
        Name for this blended electrode.

    Examples
    --------
    >>> graphite = ElectrodeOCP(soc_gr, v_gr, name="Graphite", electrode_type="anode")
    >>> silicon = ElectrodeOCP(soc_si, v_si, name="Silicon", electrode_type="anode")
    >>> blend = BlendElectrode(blend1=graphite, blend2=silicon, electrode_type="anode")
    >>> soc, voltage = blend.get_blend_curve(gamma=0.25)  # 25% silicon on the window
    """

    blend1: ElectrodeOCP
    blend2: ElectrodeOCP
    electrode_type: str = "anode"
    name: str = ""
    n_points: int | None = None
    _common_voltage: np.ndarray | None = field(default=None, repr=False)
    _q_blend1_interp: np.ndarray | None = field(default=None, repr=False)
    _q_blend2_interp: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Prepare blend electrode data after initialization."""
        if self.blend1.electrode_type != self.blend2.electrode_type:
            raise ValueError("blend1 and blend2 must have the same electrode_type")

        self.electrode_type = self.blend1.electrode_type

        if not self.name:
            self.name = f"Blend({self.blend1.name}-{self.blend2.name})"

        if self.n_points is None:
            self.n_points = 1000

        # Prepare common voltage grid and interpolated Q values
        self._prepare_blend_data()

    @staticmethod
    def _inversion_support(component: ElectrodeOCP, label: str) -> tuple[np.ndarray, np.ndarray]:
        """Sorted, tie-free (voltage, soc) support for one component's Q(V).

        Inverting V(Q) requires the component's voltage to walk in one direction
        along its own SOC axis. Where it reverses, several capacities share a
        voltage and the inversion silently keeps whichever one the sort happens
        to place first, so the reversal is rejected here instead.
        """
        v = np.asarray(component.voltage, dtype=float).ravel()
        q = np.asarray(component.soc, dtype=float).ravel()

        signs = np.sign(np.diff(v))
        nonzero = signs[signs != 0]
        if nonzero.size and not bool(np.all(nonzero == nonzero[0])):
            first = int(np.flatnonzero(signs == -nonzero[0])[0])
            raise ValueError(
                f"{label} ('{component.name}') has a non-monotone voltage axis: it "
                f"reverses between samples {first} and {first + 1} "
                f"({float(v[first]):.6g} V -> {float(v[first + 1]):.6g} V), so Q(V) "
                "is not a function and the blend inversion is undefined."
            )

        order = np.argsort(v, kind="stable")
        v_sorted = v[order]
        q_sorted = q[order]
        v_unique, unique_idx = np.unique(v_sorted, return_index=True)
        q_unique = q_sorted[unique_idx]
        if v_unique.size < 2:
            raise ValueError(
                f"{label} ('{component.name}') collapses to {int(v_unique.size)} distinct "
                "voltage(s); the blend inversion needs at least 2."
            )
        return v_unique, q_unique

    def _prepare_blend_data(self, n_points: int | None = None):
        """
        Prepare common voltage grid and Q interpolations.

        This sets up the data needed for blend calculations by:
        1. Finding common voltage range
        2. Creating Q(V) interpolators for both components
        3. Storing interpolated Q values on common voltage grid

        Parameters
        ----------
        n_points : int
            Number of points for common voltage grid.

        Notes
        -----
        DIFFERENCE FROM MATLAB: MATLAB creates commonVoltage in calculate_half_cell_data.m
        We do this in the class initialization for cleaner encapsulation.
        """
        # Get voltage ranges
        v1_min, v1_max = self.blend1.get_voltage_range()
        v2_min, v2_max = self.blend2.get_voltage_range()

        # Common voltage window (intersection of both ranges)
        v_min = max(v1_min, v2_min)
        v_max = min(v1_max, v2_max)

        if not (np.isfinite(v_min) and np.isfinite(v_max)):
            raise ValueError(
                f"Common voltage window is not finite: [{v_min}, {v_max}] from blend1 "
                f"[{v1_min}, {v1_max}] and blend2 [{v2_min}, {v2_max}]."
            )

        if v_min >= v_max:
            raise ValueError(
                f"No overlapping voltage range between blend1 [{v1_min:.3f}, {v1_max:.3f}] "
                f"and blend2 [{v2_min:.3f}, {v2_max:.3f}]"
            )

        if n_points is None:
            n_points = int(self.n_points) if self.n_points is not None else 1000

        self._common_voltage = np.linspace(v_min, v_max, n_points)

        # Create Q(V) interpolators (invert the V(Q) relationship). Queries off
        # the component's own voltage support are clamped to its end capacities;
        # a 0 fill would read as "empty electrode" instead.
        v1_support, q1_support = self._inversion_support(self.blend1, "blend1")
        q1_of_v = interp1d(
            v1_support,
            q1_support,
            kind="linear",
            bounds_error=False,
            fill_value=(float(q1_support[0]), float(q1_support[-1])),
            assume_sorted=True,
        )
        self._q_blend1_interp = q1_of_v(self._common_voltage)

        # For blend2
        v2_support, q2_support = self._inversion_support(self.blend2, "blend2")
        q2_of_v = interp1d(
            v2_support,
            q2_support,
            kind="linear",
            bounds_error=False,
            fill_value=(float(q2_support[0]), float(q2_support[-1])),
            assume_sorted=True,
        )
        self._q_blend2_interp = q2_of_v(self._common_voltage)

    def _ensure_prepared(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The common voltage grid and both Q interpolations, prepared if needed."""
        if self._common_voltage is None:
            self._prepare_blend_data()
        if (
            self._common_voltage is None
            or self._q_blend1_interp is None
            or self._q_blend2_interp is None
        ):
            raise RuntimeError(_UNPREPARED_MSG)
        return self._common_voltage, self._q_blend1_interp, self._q_blend2_interp

    def get_blend_curve(self, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate blended electrode curve for given blend2 fraction.

        This is the core blending operation:
        Q_blend(V) = gamma * Q_blend2(V) + (1 - gamma) * Q_blend1(V)

        The result is converted back to SOC vs Voltage format.

        Parameters
        ----------
        gamma : float
            Fraction of blend2 component (0-1).
            e.g., gamma=0.25 means 25% silicon, 75% graphite.

        Returns
        -------
        tuple
            (blend_soc, blend_voltage) arrays.

        Notes
        -----
        This replicates MATLAB's calculate_blend_curve.m function.
        The blending is done on a common voltage grid, then converted
        back to normalized SOC.
        """
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        common_voltage, q1_interp, q2_interp = self._ensure_prepared()

        # Weighted sum of capacities at each voltage
        # DIFFERENCE FROM MATLAB: Same algorithm, different implementation
        q_blend = gamma * q2_interp + (1 - gamma) * q1_interp

        # Normalize Q to 0-1 (SOC)
        q_min = q_blend.min()
        q_max = q_blend.max()

        if abs(q_max - q_min) < 1e-10:
            raise ValueError("Blend curve has no capacity range")

        q_norm = (q_blend - q_min) / (q_max - q_min)

        # Sort by normalized Q so we can invert to V(Q)
        sort_idx = np.argsort(q_norm)
        q_sorted = q_norm[sort_idx]
        v_sorted = common_voltage[sort_idx]

        # Create uniform SOC grid
        blend_soc = np.linspace(0, 1, len(q_sorted))

        # Interpolate voltage onto uniform SOC grid
        # MATLAB uses interp1(Q_sorted, V_sorted, blendSOC, 'linear', 'extrap')
        # so we use linear extrapolation at edges (not clamping)
        f = interp1d(q_sorted, v_sorted, kind="linear", fill_value="extrapolate")
        blend_voltage = f(blend_soc)

        return blend_soc, blend_voltage

    def get_component_stoichiometries(
        self, gamma: float, blend_soc: np.ndarray | float
    ) -> dict[str, np.ndarray]:
        """
        Evaluate component stoichiometries at fitted blend SOC positions.

        The fitted alpha/beta parameters operate on the normalized blend
        coordinate returned by :meth:`get_blend_curve`. This helper maps that
        coordinate back through the blended voltage curve and then reads the
        corresponding component SOC values on the same voltage grid.
        """
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        common_voltage, q1_interp, q2_interp = self._ensure_prepared()

        blend_soc_arr = np.asarray(blend_soc, dtype=float)

        q_blend = gamma * q2_interp + (1 - gamma) * q1_interp
        q_min = q_blend.min()
        q_max = q_blend.max()
        if abs(q_max - q_min) < 1e-10:
            raise ValueError("Blend curve has no capacity range")

        # PyDMA fits this normalized blend coordinate, not either component directly.
        q_norm = (q_blend - q_min) / (q_max - q_min)
        sort_idx = np.argsort(q_norm)
        q_sorted = q_norm[sort_idx]
        v_sorted = common_voltage[sort_idx]

        v_of_blend_soc = interp1d(
            q_sorted,
            v_sorted,
            kind="linear",
            fill_value="extrapolate",
        )
        # Component stoichiometries are recovered at the same physical voltage.
        voltage = np.asarray(v_of_blend_soc(blend_soc_arr), dtype=float)

        window_lo = float(common_voltage[0])
        window_hi = float(common_voltage[-1])
        outside = np.flatnonzero((voltage < window_lo) | (voltage > window_hi))
        if outside.size:
            warnings.warn(
                f"Blend SOC endpoint(s) map to "
                f"{np.array2string(voltage[outside], precision=4)} V, outside the common "
                f"voltage window [{window_lo:.4f}, {window_hi:.4f}] V of "
                f"'{self.name}'. The component stoichiometries reported there are "
                "clamped to the window edge.",
                stacklevel=2,
            )

        q1_of_v = interp1d(
            common_voltage,
            q1_interp,
            kind="linear",
            bounds_error=False,
            fill_value=(float(q1_interp[0]), float(q1_interp[-1])),
            assume_sorted=True,
        )
        q2_of_v = interp1d(
            common_voltage,
            q2_interp,
            kind="linear",
            bounds_error=False,
            fill_value=(float(q2_interp[0]), float(q2_interp[-1])),
            assume_sorted=True,
        )

        return {
            "blend_soc": blend_soc_arr,
            "voltage": voltage,
            "blend1_stoichiometry": q1_of_v(voltage),
            "blend2_stoichiometry": q2_of_v(voltage),
            "raw_blend_capacity": gamma * q2_of_v(voltage) + (1 - gamma) * q1_of_v(voltage),
            "raw_blend_min": np.asarray(q_min),
            "raw_blend_max": np.asarray(q_max),
        }

    def get_component_stoichiometry_window(
        self, gamma: float, blend_soc_window: tuple[float, float]
    ) -> dict[str, float]:
        """
        Return component stoichiometries at the two fitted blend endpoints.
        """
        values = self.get_component_stoichiometries(
            gamma, np.asarray(blend_soc_window, dtype=float)
        )
        return {
            "blend_sto_0soc": float(values["blend_soc"][0]),
            "blend_sto_100soc": float(values["blend_soc"][1]),
            "voltage_0soc": float(values["voltage"][0]),
            "voltage_100soc": float(values["voltage"][1]),
            "blend1_sto_0soc": float(values["blend1_stoichiometry"][0]),
            "blend1_sto_100soc": float(values["blend1_stoichiometry"][1]),
            "blend2_sto_0soc": float(values["blend2_stoichiometry"][0]),
            "blend2_sto_100soc": float(values["blend2_stoichiometry"][1]),
            "raw_blend_capacity_0soc": float(values["raw_blend_capacity"][0]),
            "raw_blend_capacity_100soc": float(values["raw_blend_capacity"][1]),
            "raw_blend_min": float(values["raw_blend_min"]),
            "raw_blend_max": float(values["raw_blend_max"]),
        }

    def get_blend_electrode(
        self, gamma: float, smooth: bool = False, window: int = 30
    ) -> ElectrodeOCP:
        """
        Get blended electrode as an ElectrodeOCP object.

        Parameters
        ----------
        gamma : float
            Fraction of blend2 component (0-1).
        smooth : bool
            Whether to smooth the resulting curve.
        window : int
            Smoothing window size.

        Returns
        -------
        ElectrodeOCP
            Blended electrode OCP object.
        """
        soc, voltage = self.get_blend_curve(gamma)

        ocp = ElectrodeOCP(
            soc=soc,
            voltage=voltage,
            name=f"{self.name} (γ={gamma:.3f})",
            electrode_type=self.electrode_type,
            is_smoothed=False,
        )

        if smooth:
            ocp = ocp.smooth(window=window)

        return ocp

    @property
    def common_voltage(self) -> np.ndarray:
        """Get the common voltage grid."""
        return self._ensure_prepared()[0]

    @property
    def q_blend1_interp(self) -> np.ndarray:
        """Get interpolated Q values for blend1 on common voltage grid."""
        return self._ensure_prepared()[1]

    @property
    def q_blend2_interp(self) -> np.ndarray:
        """Get interpolated Q values for blend2 on common voltage grid."""
        return self._ensure_prepared()[2]

    def get_single_component(self, component: int = 1) -> ElectrodeOCP:
        """
        Get single component electrode (for non-blend mode).

        Parameters
        ----------
        component : int
            Which component to return (1 or 2).

        Returns
        -------
        ElectrodeOCP
            The specified component electrode.
        """
        if component == 1:
            return self.blend1
        elif component == 2:
            return self.blend2
        else:
            raise ValueError(f"component must be 1 or 2, got {component}")

    def get_blended_ocp(self, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """Alias for get_blend_curve for API compatibility."""
        return self.get_blend_curve(gamma)

    def __repr__(self) -> str:
        return (
            f"BlendElectrode(name='{self.name}', type='{self.electrode_type}', "
            f"blend1='{self.blend1.name}', blend2='{self.blend2.name}')"
        )
