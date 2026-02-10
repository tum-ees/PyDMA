"""
Blend electrode model for mixed active materials.

This module provides the BlendElectrode class for handling blended electrodes
such as Silicon-Graphite anodes where the OCP is a weighted combination
of two component materials.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d

from pydma.electrodes.electrode import ElectrodeOCP


@dataclass
class BlendElectrode:
    """
    Blended electrode model combining two component materials.

    This class handles blended electrodes (e.g., Si-Gr anodes) where the
    total capacity is a weighted sum of two components:

    Q_blend(V) = gamma * Q_blend2(V) + (1 - gamma) * Q_blend1(V)

    The blending is done on a common voltage grid, not SOC.

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
    >>> soc, voltage = blend.get_blend_curve(gamma=0.25)  # 25% silicon
    """

    blend1: ElectrodeOCP
    blend2: ElectrodeOCP
    electrode_type: str = "anode"
    name: str = ""
    n_points: Optional[int] = None
    _common_voltage: Optional[np.ndarray] = field(default=None, repr=False)
    _q_blend1_interp: Optional[np.ndarray] = field(default=None, repr=False)
    _q_blend2_interp: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Prepare blend electrode data after initialization."""
        if self.blend1.electrode_type != self.blend2.electrode_type:
            raise ValueError(
                "blend1 and blend2 must have the same electrode_type"
            )

        self.electrode_type = self.blend1.electrode_type

        if not self.name:
            self.name = f"Blend({self.blend1.name}-{self.blend2.name})"

        if self.n_points is None:
            self.n_points = 1000

        # Prepare common voltage grid and interpolated Q values
        self._prepare_blend_data()

    def _prepare_blend_data(self, n_points: Optional[int] = None):
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

        if v_min >= v_max:
            raise ValueError(
                f"No overlapping voltage range between blend1 [{v1_min:.3f}, {v1_max:.3f}] "
                f"and blend2 [{v2_min:.3f}, {v2_max:.3f}]"
            )

        if n_points is None:
            n_points = int(self.n_points) if self.n_points is not None else 1000

        self._common_voltage = np.linspace(v_min, v_max, n_points)

        # Create Q(V) interpolators (invert the V(Q) relationship)
        # For blend1
        q1_of_v = interp1d(
            self.blend1.voltage,
            self.blend1.soc,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._q_blend1_interp = q1_of_v(self._common_voltage)

        # For blend2
        q2_of_v = interp1d(
            self.blend2.voltage,
            self.blend2.soc,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._q_blend2_interp = q2_of_v(self._common_voltage)

    def get_blend_curve(self, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
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

        if self._common_voltage is None:
            self._prepare_blend_data()

        # Weighted sum of capacities at each voltage
        # DIFFERENCE FROM MATLAB: Same algorithm, different implementation
        q_blend = gamma * self._q_blend2_interp + (1 - gamma) * self._q_blend1_interp

        # Normalize Q to 0-1 (SOC)
        q_min = q_blend.min()
        q_max = q_blend.max()

        if abs(q_max - q_min) < 1e-10:
            raise ValueError("Blend curve has no capacity range")

        q_norm = (q_blend - q_min) / (q_max - q_min)

        # Sort by normalized Q so we can invert to V(Q)
        sort_idx = np.argsort(q_norm)
        q_sorted = q_norm[sort_idx]
        v_sorted = self._common_voltage[sort_idx]

        # Create uniform SOC grid
        blend_soc = np.linspace(0, 1, len(q_sorted))

        # Interpolate voltage onto uniform SOC grid
        # MATLAB uses interp1(Q_sorted, V_sorted, blendSOC, 'linear', 'extrap')
        # so we use linear extrapolation at edges (not clamping)
        f = interp1d(q_sorted, v_sorted, kind='linear', fill_value='extrapolate')
        blend_voltage = f(blend_soc)

        return blend_soc, blend_voltage

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
        if self._common_voltage is None:
            self._prepare_blend_data()
        return self._common_voltage

    @property
    def q_blend1_interp(self) -> np.ndarray:
        """Get interpolated Q values for blend1 on common voltage grid."""
        if self._q_blend1_interp is None:
            self._prepare_blend_data()
        return self._q_blend1_interp

    @property
    def q_blend2_interp(self) -> np.ndarray:
        """Get interpolated Q values for blend2 on common voltage grid."""
        if self._q_blend2_interp is None:
            self._prepare_blend_data()
        return self._q_blend2_interp

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

    def get_blended_ocp(self, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for get_blend_curve for API compatibility."""
        return self.get_blend_curve(gamma)

    def __repr__(self) -> str:
        return (
            f"BlendElectrode(name='{self.name}', type='{self.electrode_type}', "
            f"blend1='{self.blend1.name}', blend2='{self.blend2.name}')"
        )
