"""
Configuration dataclass for DMA analysis.

This module defines the DMAConfig dataclass which holds all configuration
parameters for the degradation mode analysis.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List, Union
import numpy as np
from pydma.utils.roi import ROISpec, get_roi_outer_bounds


@dataclass
class DMAConfig:
    """
    Configuration for Degradation Mode Analysis.

    This dataclass holds all parameters needed to configure a DMA run.
    Parameters are organized by category with sensible defaults.

    Attributes
    ----------
    direction : str
        Direction of pOCV measurement: 'charge' or 'discharge'.
    data_length : int
        Number of points to resample data to. Default: 1000.
    smoothing_points : int
        Window size for LOWESS smoothing of input curves. Default: 30.

    weight_ocv : float
        Weight for OCV fitting term in cost function. Default: 100.
    weight_dva : float
        Weight for DVA fitting term. Default: 1.
    weight_ica : float
        Weight for ICA fitting term. Default: 0.

    roi_ocv_min : float or 2-value sequence
        Lower bound(s) of OCV fitting region. Default: 0.0.
    roi_ocv_max : float or 2-value sequence
        Upper bound(s) of OCV fitting region. Default: 1.0.
    roi_dva_min : float or 2-value sequence
        Lower bound of DVA fitting region. Default: 0.1.
    roi_dva_max : float or 2-value sequence
        Upper bound of DVA fitting region. Default: 0.9.
    roi_ica_min : float or 2-value sequence
        Lower bound of ICA fitting region. Default: 0.13.
    roi_ica_max : float or 2-value sequence
        Upper bound of ICA fitting region. Default: 0.9.

    lower_bounds : tuple
        Lower bounds for [alpha_an, beta_an, alpha_ca, beta_ca].
        Default: (0.8, -1.0, 0.8, -1.0).
    upper_bounds : tuple
        Upper bounds for [alpha_an, beta_an, alpha_ca, beta_ca].
        Default: (2.0, 0.2, 2.0, 0.2).

    use_anode_blend : bool
        Enable anode blend electrode model. Default: False.
    gamma_anode_blend2_init : float
        Initial guess for anode blend2 fraction. Default: 0.25.
    gamma_anode_blend2_upper : float
        Upper bound for anode blend2 fraction. Default: 0.30.

    use_cathode_blend : bool
        Enable cathode blend electrode model. Default: False.
    gamma_cathode_blend2_init : float
        Initial guess for cathode blend2 fraction. Default: 0.5.
    gamma_cathode_blend2_upper : float
        Upper bound for cathode blend2 fraction. Default: 1.0.

    allow_anode_inhomogeneity : bool
        Enable anode inhomogeneity modeling. Default: False.
    allow_cathode_inhomogeneity : bool
        Enable cathode inhomogeneity modeling. Default: False.
    allow_first_cycle_inhomogeneity : bool
        Allow inhomogeneity for the very first CU. Default: True.
    max_inhomogeneity : float or tuple
        Maximum allowed inhomogeneity for (anode, cathode). Default: 0.3.
    max_inhomogeneity_delta : float or tuple
        Maximum inhomogeneity increase per CU. Default: 0.1.

    max_anode_gain : float
        Maximum allowed anode capacity gain per CU. Default: 0.01.
    max_cathode_gain : float
        Maximum allowed cathode capacity gain per CU. Default: 0.01.
    max_anode_blend1_gain : float
        Maximum anode blend1 gain per CU. Default: 0.005.
    max_anode_blend2_gain : float
        Maximum anode blend2 gain per CU. Default: 0.01.
    max_anode_loss : float
        Maximum anode loss per CU. Default: 1.0.
    max_cathode_loss : float
        Maximum cathode loss per CU. Default: 1.0.
    max_anode_blend1_loss : float
        Maximum anode blend1 loss per CU. Default: 1.0.
    max_anode_blend2_loss : float
        Maximum anode blend2 loss per CU. Default: 1.0.

    req_accepted : int
        Required number of accepted solutions per CU. Default: 3.
    max_tries_overall : int
        Maximum optimization attempts per CU. Default: 10.
    rmse_threshold : float
        RMSE threshold for accepting a solution (in Volts). Default: 0.01.
    print_progress : bool
        Print intermediate optimization results after each run. Default: True.

    speed_preset : str
        Optimization speed preset: 'fast', 'medium', or 'thorough'.
        Default: 'thorough'.
    algorithm : str
        Optimization algorithm. Default: 'differential_evolution'.
    workers : int
        Number of parallel workers for differential evolution.
        Default: 1 (single-threaded). Set to -1 for all available CPUs.
        Note: workers > 1 uses 'deferred' updating which may need more iterations.
    popsize : int
        Target population size (MATLAB GA equivalent). For SciPy's
        differential_evolution this is converted to a multiplier internally.

    Notes
    -----
    **LFP Cells:** Use ``DMAConfig.lfp_preset()`` or see inline comments below.

    Examples
    --------
    >>> config = DMAConfig(direction="charge", weight_ocv=100, weight_dva=1)
    >>> config.use_anode_blend = True
    >>> config.gamma_anode_blend2_upper = 0.30

    >>> # For LFP cells, use the preset:
    >>> config = DMAConfig.lfp_preset()
    >>> config.weight_ocv
    10.0
    """

    # Data processing
    direction: str = "charge"  # 'charge' or 'discharge'
    data_length: int = 1000
    smoothing_points: int = 30

    # Cost function weights
    # LFP? -> we recommend weight_ocv / weight_dva = 10 / 3
    weight_ocv: float = 100.0
    weight_dva: float = 1.0
    weight_ica: float = 0.0

    # Region of interest (ROI) for fitting
    # =========================================================================
    # All ROI parameters (OCV, DVA, ICA) support two formats:
    #
    # 1. SINGLE REGION: Use scalar values for min and max
    #    Example: roi_dva_min=0.1, roi_dva_max=0.9
    #    -> Fits in the range [0.1, 0.9] (i.e., 10-90% SOC)
    #
    # 2. TWO REGIONS: Use 2-value sequence-like bounds for min and max
    #    Example: roi_dva_min=(0.05, 0.15), roi_dva_max=(0.85, 0.95)
    #    -> Fits in TWO separate regions: [0.05, 0.15] OR [0.85, 0.95]
    #    This is useful for LFP cells where you want to avoid the flat plateau.
    #
    # LFP Recommendation:
    #   - OCV: Use two regions to fit only high and low SOC (e.g., 0-15% and 85-100%)
    #     roi_ocv_min=(0.0, 0.15), roi_ocv_max=(0.85, 1.0)
    #   - DVA: Single middle region (e.g., 10-90% SOC)
    #     roi_dva_min=0.1, roi_dva_max=0.9
    # =========================================================================
    roi_ocv_min: ROISpec = 0.0
    roi_ocv_max: ROISpec = 1.0
    roi_dva_min: ROISpec = 0.1
    roi_dva_max: ROISpec = 0.9
    roi_ica_min: ROISpec = 0.13
    roi_ica_max: ROISpec = 0.9

    # Parameter bounds for [alpha_an, beta_an, alpha_ca, beta_ca]
    # MATLAB Reference: main_DMA.m lines 196-197
    # Note: MATLAB comments mention wider defaults (0.8 to 2.0), but actual code uses tighter bounds
    lower_bounds: Tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)
    upper_bounds: Tuple[float, float, float, float] = (2.0, 0.0, 2.1, 0.0)

    # Anode blend settings
    use_anode_blend: bool = False
    gamma_anode_blend2_init: float = 0.25
    gamma_anode_blend2_upper: float = 0.30

    # Cathode blend settings
    use_cathode_blend: bool = False
    gamma_cathode_blend2_init: float = 0.5
    gamma_cathode_blend2_upper: float = 1.0

    # Inhomogeneity settings
    # Do not use cathodeInhomogeneity for LFP cells!
    allow_anode_inhomogeneity: bool = False
    allow_cathode_inhomogeneity: bool = False
    allow_first_cycle_inhomogeneity: bool = True
    max_inhomogeneity: Union[float, Tuple[float, float]] = 0.3
    max_inhomogeneity_delta: Union[float, Tuple[float, float]] = 0.1

    # Constraint settings (max gain/loss per CU)
    max_anode_gain: float = 0.01
    max_cathode_gain: float = 0.01
    max_anode_blend1_gain: float = 0.005
    max_anode_blend2_gain: float = 0.01
    max_anode_loss: float = 1.0
    max_cathode_loss: float = 1.0
    max_anode_blend1_loss: float = 1.0
    max_anode_blend2_loss: float = 1.0

    # Optimization control
    # RMSE is calculated both in fit region (ROI) and full range (0-100% SOC)
    req_accepted: int = 3
    max_tries_overall: int = 10
    rmse_threshold: float = 0.01  # Applied to rmse_fit_region
    print_progress: bool = True

    # Solver settings
    speed_preset: str = "thorough"  # 'fast', 'medium', or 'thorough'
    algorithm: str = "differential_evolution"
    workers: int = 1  # Number of parallel workers for DE (1 = single-threaded)

    # Plotting labels
    label_cathode: str = "Cathode"
    label_anode: str = "Anode"
    label_anode_blend1: str = "An-blend1"
    label_anode_blend2: str = "An-blend2"
    label_cathode_blend1: str = "Ca-blend1"
    label_cathode_blend2: str = "Ca-blend2"
    label_charge_carrier_inv: str = "Charge-carrier-inv"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if self.direction not in ("charge", "discharge"):
            raise ValueError(f"direction must be 'charge' or 'discharge', got {self.direction}")

        if self.workers < -1 or self.workers == 0:
            raise ValueError(f"workers must be -1 (all CPUs) or >= 1, got {self.workers}")

        if self.data_length < 100:
            raise ValueError(f"data_length must be >= 100, got {self.data_length}")

        if self.smoothing_points < 1:
            raise ValueError(f"smoothing_points must be >= 1, got {self.smoothing_points}")

        if self.weight_ocv < 0 or self.weight_dva < 0 or self.weight_ica < 0:
            raise ValueError("Cost function weights must be non-negative")

        if self.rmse_threshold <= 0:
            raise ValueError(f"rmse_threshold must be positive, got {self.rmse_threshold}")

        if self.req_accepted < 1:
            raise ValueError(f"req_accepted must be >= 1, got {self.req_accepted}")

        if self.max_tries_overall < self.req_accepted:
            raise ValueError("max_tries_overall must be >= req_accepted")

        if self.speed_preset not in ("fast", "medium", "thorough"):
            raise ValueError(
                f"speed_preset must be 'fast', 'medium', or 'thorough', got {self.speed_preset}"
            )

    def get_solver_options(self) -> dict:
        """
        Get solver options based on speed preset.

        Returns
        -------
        dict
            Dictionary of solver options for scipy.optimize.differential_evolution.

        Notes
        -----
        DIFFERENCE FROM MATLAB: MATLAB uses 'ga' with PopulationSize=500 as default.
        We use differential_evolution which is similar in behavior.
        The high population size is critical for good results.

        When workers > 1, DE uses parallel evaluation of the objective function.
        Note: workers > 1 requires updating='deferred' which may need more iterations.
        """
        # DIFFERENCE FROM MATLAB: We use differential_evolution instead of ga()
        # but maintain similar population sizes for equivalent exploration
        preset_options = {
            "fast": {
                "popsize": 30,
                "maxiter": 50,
                "tol": 1e-4,
                "mutation": (0.5, 1.0),
                "recombination": 0.8,
                "polish": True,
            },
            "medium": {
                # ~30% of MATLAB: 150 * 100 = 15,000 evaluations
                "popsize": 150,
                "maxiter": 100,
                "tol": 1e-7,
                "mutation": (0.5, 1.0),
                "recombination": 0.8,
                "polish": True,
            },
            "thorough": {
                # Matches MATLAB exactly: PopulationSize=500, MaxGenerations=100
                "popsize": 500,
                "maxiter": 100,
                "tol": 1e-8,
                "mutation": (0.5, 1.0),
                "recombination": 0.8,
                "polish": True,
            },
        }
        opts = preset_options[self.speed_preset]
        # Use instance workers setting (allows parallelization when workers > 1)
        opts["workers"] = self.workers
        return opts

    def get_inhomogeneity_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get inhomogeneity bounds as (anode, cathode) tuples.

        Returns
        -------
        tuple
            ((anode_max, anode_delta), (cathode_max, cathode_delta))
        """
        if isinstance(self.max_inhomogeneity, (int, float)):
            max_an = max_ca = float(self.max_inhomogeneity)
        else:
            max_an, max_ca = self.max_inhomogeneity

        if isinstance(self.max_inhomogeneity_delta, (int, float)):
            delta_an = delta_ca = float(self.max_inhomogeneity_delta)
        else:
            delta_an, delta_ca = self.max_inhomogeneity_delta

        return ((max_an, delta_an), (max_ca, delta_ca))

    def get_active_param_mask(self) -> List[bool]:
        """
        Get mask indicating which of the 8 parameters are active.

        Returns
        -------
        list
            8-element boolean list where True means parameter is active.

        Notes
        -----
        Parameters are in order:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an,
         gamma_blend2_ca, inhom_an, inhom_ca]
        """
        return [
            True,  # alpha_an - always active
            True,  # beta_an - always active
            True,  # alpha_ca - always active
            True,  # beta_ca - always active
            self.use_anode_blend,  # gamma_blend2_an
            self.use_cathode_blend,  # gamma_blend2_ca
            self.allow_anode_inhomogeneity,  # inhom_an
            self.allow_cathode_inhomogeneity,  # inhom_ca
        ]

    def get_full_bounds(
        self, inhom_an_prev: Optional[float] = None, inhom_ca_prev: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full 8-element lower and upper bounds arrays.

        Parameters
        ----------
        inhom_an_prev : float, optional
            Previous anode inhomogeneity value for constraining bounds.
        inhom_ca_prev : float, optional
            Previous cathode inhomogeneity value for constraining bounds.

        Returns
        -------
        tuple
            (lower_bounds, upper_bounds) as numpy arrays of length 8.
        """
        lb = np.zeros(8)
        ub = np.zeros(8)

        # Base 4 parameters
        lb[0:4] = self.lower_bounds
        ub[0:4] = self.upper_bounds

        # Anode blend gamma
        if self.use_anode_blend:
            lb[4] = 0.02
            ub[4] = self.gamma_anode_blend2_upper
        else:
            lb[4] = 0.0
            ub[4] = 0.0

        # Cathode blend gamma
        if self.use_cathode_blend:
            lb[5] = 0.02
            ub[5] = self.gamma_cathode_blend2_upper
        else:
            lb[5] = 0.0
            ub[5] = 0.0

        # Inhomogeneity bounds
        (max_an, delta_an), (max_ca, delta_ca) = self.get_inhomogeneity_bounds()

        if self.allow_anode_inhomogeneity:
            lb[6] = 0.0
            if inhom_an_prev is None:
                ub[6] = max_an
            else:
                ub[6] = min(max_an, inhom_an_prev + delta_an)
        else:
            lb[6] = 0.0
            ub[6] = 0.0

        if self.allow_cathode_inhomogeneity:
            lb[7] = 0.0
            if inhom_ca_prev is None:
                ub[7] = max_ca
            else:
                ub[7] = min(max_ca, inhom_ca_prev + delta_ca)
        else:
            lb[7] = 0.0
            ub[7] = 0.0

        return lb, ub

    def calculate_roi_bounds(self) -> Tuple[float, float]:
        """
        Calculate the overall ROI bounds considering all active fitting methods.

        Returns
        -------
        tuple
            (lowest_roi, highest_roi) bounds.
        """
        roi_mins = []
        roi_maxs = []

        # Shared ROI parser/validator also handles split interval inputs.
        if self.weight_ocv > 0:
            roi_lo, roi_hi = get_roi_outer_bounds(
                self.roi_ocv_min,
                self.roi_ocv_max,
                min_name="roi_ocv_min",
                max_name="roi_ocv_max",
            )
            roi_mins.append(roi_lo)
            roi_maxs.append(roi_hi)

        if self.weight_dva > 0:
            roi_lo, roi_hi = get_roi_outer_bounds(
                self.roi_dva_min,
                self.roi_dva_max,
                min_name="roi_dva_min",
                max_name="roi_dva_max",
            )
            roi_mins.append(roi_lo)
            roi_maxs.append(roi_hi)

        if self.weight_ica > 0:
            roi_lo, roi_hi = get_roi_outer_bounds(
                self.roi_ica_min,
                self.roi_ica_max,
                min_name="roi_ica_min",
                max_name="roi_ica_max",
            )
            roi_mins.append(roi_lo)
            roi_maxs.append(roi_hi)

        if not roi_mins:
            return (0.0, 1.0)

        return (min(roi_mins), max(roi_maxs))

    @property
    def enable_inhomogeneity(self) -> bool:
        """Whether inhomogeneity is enabled for either electrode."""
        return self.allow_anode_inhomogeneity or self.allow_cathode_inhomogeneity

    @property
    def filter_type(self) -> Optional[str]:
        """Filter type for pre-smoothing raw OCV data.

        MATLAB-compatible: Uses LOWESS filter with smoothing_points window
        (default 30) applied to raw OCV before resampling and DVA computation.

        This matches MATLAB's calculate_full_cell_data.m:
            fcU_smooth = smooth(fcU_raw, smoothingPoints, 'lowess');
        """
        return "lowess"  # MATLAB-compatible default

    @property
    def filter_kwargs(self) -> dict:
        """Filter keyword arguments for pre-smoothing.

        Returns LOWESS filter settings matching MATLAB behavior:
        - window: smoothing_points (default 30)
        """
        return {"window": self.smoothing_points}

    @property
    def random_seed(self) -> Optional[int]:
        """Random seed for optimization."""
        return None  # Use random seed by default

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds as list of (min, max) tuples for the optimizer."""
        lb, ub = self.get_full_bounds()
        return list(zip(lb, ub))

    @classmethod
    def lfp_preset(cls, **kwargs) -> "DMAConfig":
        """Create a DMAConfig optimized for LFP (Lithium Iron Phosphate) cells.

        LFP cells have a flat voltage plateau in the middle SOC region, requiring:
        - Different OCV/DVA weight ratio (10:3 instead of default 100:1)
        - Split ROI for OCV to avoid the flat middle region
        - Disabled cathode inhomogeneity (not meaningful for LFP)

        These recommendations match the MATLAB implementation (main_DMA.m lines 158-169, 213).

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to DMAConfig constructor.
            Use these to override defaults or add other settings.

        Returns
        -------
        DMAConfig
            Configuration optimized for LFP cells.

        Examples
        --------
        >>> config = DMAConfig.lfp_preset()
        >>> config.weight_ocv
        10.0
        >>> config.weight_dva
        3.0

        >>> # With custom overrides
        >>> config = DMAConfig.lfp_preset(rmse_threshold=0.02, workers=-1)
        """
        defaults = dict(
            # Weight ratio 10:3 instead of default 100:1
            weight_ocv=10.0,
            weight_dva=3.0,
            # Split ROI for OCV: 0-15% and 85-100% SOC (avoid flat middle)
            roi_ocv_min=(0.0, 0.85),
            roi_ocv_max=(0.15, 1.0),
            # Standard DVA ROI
            roi_dva_min=0.10,
            roi_dva_max=0.90,
            # Disable cathode inhomogeneity (not meaningful for LFP)
            allow_cathode_inhomogeneity=False,
        )
        # User kwargs override defaults
        defaults.update(kwargs)
        return cls(**defaults)
