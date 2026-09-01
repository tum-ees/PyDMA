"""
Configuration dataclass for DMA analysis.

This module defines the DMAConfig dataclass which holds all configuration
parameters for the degradation mode analysis.
"""

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

import numpy as np

from pydma.utils.roi import ROISpec

SUPPORTED_ALGORITHMS = {"differential_evolution"}

# Methods implemented by preprocessing.smoother.apply_filter.
SUPPORTED_FILTER_TYPES = {"sgolay", "savgol", "lowess", "movmean", "movmedian", "gaussian"}


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
    filter_type : str
        Smoothing method applied to the raw full-cell OCV before resampling.
        One of 'lowess', 'sgolay', 'savgol', 'movmean', 'movmedian',
        'gaussian'. Default: 'lowess'.

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
        Default: (1.0, -1.0, 1.0, -1.0).
    upper_bounds : tuple
        Upper bounds for [alpha_an, beta_an, alpha_ca, beta_ca].
        Default: (2.0, 0.0, 2.1, 0.0). Capping both beta at 0 encodes the
        assumption that each electrode starts at sto >= 0 for SOC = 0.

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
    inhom_anode_offset : float
        Fraction of maximum anode inhomogeneity already present at SOC = 0.
        Default: 0.0.
    inhom_cathode_offset : float
        Fraction of maximum cathode inhomogeneity already present at SOC = 0.
        Default: 0.0.

    allow_resistance_offset : bool
        Fit a series resistance that lifts the reconstructed full-cell OCV by
        ``sign(direction) * R * |pocv_current_a|``. Default: False, which pins
        parameter slot 9 to zero. Enabling it requires ``pocv_current_a``.
        Every assignment re-validates the whole configuration, so on an existing
        instance set ``pocv_current_a`` first and ``allow_resistance_offset``
        second; the other order validates an intermediate state that has the
        flag on without a current and raises.
    pocv_current_a : float
        Magnitude of the current the pOCV was measured at, in A. Only its
        absolute value enters the model; the sign comes from ``direction``.
        Default: 0.0.
    resistance_offset_limit_ohm_ah : float
        Capacity-normalized ceiling for the fitted resistance, in Ohm*Ah. The
        bound is ``limit / capa_actual``, so at a common C-rate every cell of a
        study gets the same voltage headroom regardless of its capacity.
        Default: 0.25.
    allow_negative_resistance_offset : bool
        Open the resistance bounds symmetrically to ``[-r_max, r_max]``. A
        negative resistance is unphysical, so this is a diagnostic setting for
        checking whether the offset absorbs a level error of the other sign.
        Default: False.

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
    # MATLAB calculate_full_cell_data.m smooths the raw pOCV with
    # smooth(fcU_raw, smoothingPoints, 'lowess') before resampling.
    filter_type: str = "lowess"

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
    lower_bounds: tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)
    upper_bounds: tuple[float, float, float, float] = (2.0, 0.0, 2.1, 0.0)

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
    max_inhomogeneity: float | tuple[float, float] = 0.3
    max_inhomogeneity_delta: float | tuple[float, float] = 0.1
    inhom_anode_offset: float = 0.0
    inhom_cathode_offset: float = 0.0

    # Series-resistance correction of the reconstructed full-cell OCV.
    # Off by default; parameter slot 9 is pinned to zero while it is off.
    # Set pocv_current_a together with the flag: enabling the flag without a
    # current is rejected, because R and the current only ever appear as their
    # product.
    allow_resistance_offset: bool = False
    pocv_current_a: float = 0.0
    resistance_offset_limit_ohm_ah: float = 0.25
    allow_negative_resistance_offset: bool = False

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
    # None = nondeterministic. When set, the i-th multistart run uses seed+i
    # (see core.optimizer.DMAOptimizer.run), so a single config seed
    # locks the whole multi-run sequence.
    random_seed: int | None = None

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
        object.__setattr__(self, "_revalidate_on_assignment", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign a declared field and re-validate the whole configuration.

        Names that are not declared fields are rejected so that a misspelled
        setting (``config.direciton = 'discharge'``) fails loudly instead of
        being silently ignored by the rest of the code.

        A value the re-validation rejects is rolled back before the exception
        leaves this method, so a caught exception leaves the configuration on
        the value it had. The rollback covers every exception type, not just
        ``ValueError``: ``config.algorithm = None`` fails inside ``.lower()``
        with an ``AttributeError``, and leaving that assignment standing would
        hand back a configuration nothing can read.
        """
        if not name.startswith("_") and name not in self.__dataclass_fields__:
            close = get_close_matches(name, self.__dataclass_fields__, n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise AttributeError(f"DMAConfig has no field {name!r}.{hint}")

        # The flag is unset until __post_init__ ran, so field assignments made
        # by the generated __init__ are not validated one at a time. Clearing
        # it around _validate also covers the field assignments _validate
        # itself makes.
        if name.startswith("_") or not getattr(self, "_revalidate_on_assignment", False):
            object.__setattr__(self, name, value)
            return

        previous = getattr(self, name)
        object.__setattr__(self, name, value)
        object.__setattr__(self, "_revalidate_on_assignment", False)
        try:
            self._validate()
        except Exception:
            object.__setattr__(self, name, previous)
            raise
        finally:
            object.__setattr__(self, "_revalidate_on_assignment", True)

    def _validate(self):
        """Validate configuration values."""
        if self.direction not in ("charge", "discharge"):
            raise ValueError(f"direction must be 'charge' or 'discharge', got {self.direction}")

        normalized_algorithm = self.algorithm.lower().replace("-", "_").replace(" ", "_")
        if normalized_algorithm in {"de", "differentialevolution"}:
            normalized_algorithm = "differential_evolution"
        if normalized_algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(SUPPORTED_ALGORITHMS)}, got {self.algorithm}"
            )
        self.algorithm = normalized_algorithm

        if self.workers < -1 or self.workers == 0:
            raise ValueError(f"workers must be -1 (all CPUs) or >= 1, got {self.workers}")

        if self.data_length < 100:
            raise ValueError(f"data_length must be >= 100, got {self.data_length}")

        if self.smoothing_points < 1:
            raise ValueError(f"smoothing_points must be >= 1, got {self.smoothing_points}")

        if not isinstance(self.filter_type, str) or (
            self.filter_type.lower() not in SUPPORTED_FILTER_TYPES
        ):
            raise ValueError(
                f"filter_type must be one of {sorted(SUPPORTED_FILTER_TYPES)}, "
                f"got {self.filter_type!r}"
            )

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

        lower = np.asarray(self.lower_bounds, dtype=float).flatten()
        upper = np.asarray(self.upper_bounds, dtype=float).flatten()
        if lower.size != 4 or upper.size != 4:
            raise ValueError(
                "lower_bounds and upper_bounds must each hold 4 values "
                "[alpha_an, beta_an, alpha_ca, beta_ca], got "
                f"{lower.size} and {upper.size}"
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError(
                f"lower_bounds and upper_bounds must be finite, got {self.lower_bounds} "
                f"and {self.upper_bounds}"
            )
        if np.any(lower >= upper):
            idx = int(np.argmax(lower >= upper))
            raise ValueError(
                f"lower_bounds[{idx}] must be below upper_bounds[{idx}], "
                f"got {lower[idx]} and {upper[idx]}"
            )

        if not 0.0 < self.gamma_anode_blend2_upper <= 1.0:
            raise ValueError(
                "gamma_anode_blend2_upper must be within (0, 1], got "
                f"{self.gamma_anode_blend2_upper}"
            )

        if not 0.0 < self.gamma_cathode_blend2_upper <= 1.0:
            raise ValueError(
                "gamma_cathode_blend2_upper must be within (0, 1], got "
                f"{self.gamma_cathode_blend2_upper}"
            )

        if not 0.0 <= self.gamma_anode_blend2_init <= self.gamma_anode_blend2_upper:
            raise ValueError("gamma_anode_blend2_init must be within [0, gamma_anode_blend2_upper]")

        if not 0.0 <= self.gamma_cathode_blend2_init <= self.gamma_cathode_blend2_upper:
            raise ValueError(
                "gamma_cathode_blend2_init must be within [0, gamma_cathode_blend2_upper]"
            )

        for limit_name in (
            "max_anode_gain",
            "max_cathode_gain",
            "max_anode_blend1_gain",
            "max_anode_blend2_gain",
            "max_anode_loss",
            "max_cathode_loss",
            "max_anode_blend1_loss",
            "max_anode_blend2_loss",
        ):
            limit_value = getattr(self, limit_name)
            if limit_value < 0:
                raise ValueError(f"{limit_name} must be >= 0, got {limit_value}")

        (max_inhom_an, _), (max_inhom_ca, _) = self.get_inhomogeneity_bounds()
        if not 0.0 <= max_inhom_an <= 1.0 or not 0.0 <= max_inhom_ca <= 1.0:
            raise ValueError(
                f"max_inhomogeneity must be within [0, 1], got {self.max_inhomogeneity}"
            )

        if not 0.0 <= self.inhom_anode_offset <= 1.0:
            raise ValueError("inhom_anode_offset must be within [0, 1]")

        if not 0.0 <= self.inhom_cathode_offset <= 1.0:
            raise ValueError("inhom_cathode_offset must be within [0, 1]")

        limit = float(self.resistance_offset_limit_ohm_ah)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                "resistance_offset_limit_ohm_ah must be finite and positive, got "
                f"{self.resistance_offset_limit_ohm_ah}"
            )

        current = float(self.pocv_current_a)
        if not np.isfinite(current) or current < 0:
            raise ValueError(
                f"pocv_current_a is a magnitude in A: finite and >= 0, got {self.pocv_current_a}"
            )

        if self.allow_resistance_offset and current <= 0:
            raise ValueError(
                "allow_resistance_offset needs pocv_current_a > 0: without a current "
                "the resistance is not identifiable, because the model only ever sees "
                "the product R * I."
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
        opts = preset_options.get(self.speed_preset)
        if opts is None:
            raise ValueError(
                f"speed_preset must be one of {sorted(preset_options)}, got {self.speed_preset}"
            )
        # Use instance workers setting (allows parallelization when workers > 1)
        opts["workers"] = self.workers
        return opts

    def get_inhomogeneity_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
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

    def get_active_param_mask(self) -> list[bool]:
        """
        Get mask indicating which of the 9 parameters are active.

        Returns
        -------
        list
            9-element boolean list where True means parameter is active.

        Notes
        -----
        Parameters are in order:
        [alpha_an, beta_an, alpha_ca, beta_ca, gamma_blend2_an,
         gamma_blend2_ca, inhom_an, inhom_ca, r_offset]
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
            self.allow_resistance_offset,  # r_offset
        ]

    def _resistance_offset_bounds(self, capa_actual: float | None) -> tuple[float, float]:
        """Bounds for the resistance slot, in Ohm.

        The configured limit is capacity-normalized (Ohm*Ah), so the ceiling is
        ``limit / capa_actual``. At a fixed C-rate the current scales with the
        capacity, which makes ``r_max * I`` the same voltage headroom for every
        cell of a study.
        """
        if not self.allow_resistance_offset:
            return 0.0, 0.0

        if capa_actual is None:
            raise ValueError(
                "allow_resistance_offset is set, so capa_actual is required to turn "
                "resistance_offset_limit_ohm_ah into a resistance bound."
            )
        capa = float(capa_actual)
        if not np.isfinite(capa) or capa <= 0:
            raise ValueError(f"capa_actual must be finite and positive, got {capa_actual}")

        r_max = float(self.resistance_offset_limit_ohm_ah) / capa
        return (-r_max if self.allow_negative_resistance_offset else 0.0), r_max

    def get_full_bounds(
        self,
        inhom_an_prev: float | None = None,
        inhom_ca_prev: float | None = None,
        capa_actual: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get full 9-element lower and upper bounds arrays.

        Parameters
        ----------
        inhom_an_prev : float, optional
            Previous anode inhomogeneity value for constraining bounds.
        inhom_ca_prev : float, optional
            Previous cathode inhomogeneity value for constraining bounds.
        capa_actual : float, optional
            Cell capacity of this check-up in Ah. Required when
            ``allow_resistance_offset`` is set, because the resistance bound is
            derived from the capacity-normalized limit.

        Returns
        -------
        tuple
            (lower_bounds, upper_bounds) as numpy arrays of length 9.

        Raises
        ------
        ValueError
            If ``allow_resistance_offset`` is set and ``capa_actual`` is
            missing or not a positive, finite capacity.
        """
        lb = np.zeros(9)
        ub = np.zeros(9)

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

        # Series resistance
        lb[8], ub[8] = self._resistance_offset_bounds(capa_actual)

        return lb, ub

    def get_initial_guess(
        self,
        inhom_an_prev: float | None = None,
        inhom_ca_prev: float | None = None,
        capa_actual: float | None = None,
    ) -> np.ndarray:
        """Get a MATLAB-style initial guess vector clipped to active bounds.

        ``capa_actual`` is required when ``allow_resistance_offset`` is set; the
        resistance slot then starts at the midpoint of its bounds.
        """
        if self.use_anode_blend:
            init = np.array([1.05, -0.005, 1.1, -0.01, 0.0, 0.0, 0.03, 0.03, 0.0], dtype=float)
        else:
            init = np.array([1.2, 0.0, 1.1, -0.1, 0.0, 0.0, 0.03, 0.03, 0.0], dtype=float)

        if self.use_anode_blend:
            init[4] = self.gamma_anode_blend2_init
        if self.use_cathode_blend:
            init[5] = self.gamma_cathode_blend2_init
        if not self.allow_anode_inhomogeneity:
            init[6] = 0.0
        if not self.allow_cathode_inhomogeneity:
            init[7] = 0.0

        lb, ub = self.get_full_bounds(
            inhom_an_prev=inhom_an_prev,
            inhom_ca_prev=inhom_ca_prev,
            capa_actual=capa_actual,
        )
        init[8] = 0.5 * (lb[8] + ub[8])
        return np.asarray(np.clip(init, lb, ub))

    @property
    def enable_inhomogeneity(self) -> bool:
        """Whether inhomogeneity is enabled for either electrode."""
        return self.allow_anode_inhomogeneity or self.allow_cathode_inhomogeneity

    @property
    def filter_kwargs(self) -> dict:
        """Filter keyword arguments for pre-smoothing.

        Returns LOWESS filter settings matching MATLAB behavior:
        - window: smoothing_points (default 30)
        """
        return {"window": self.smoothing_points}

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
        defaults: dict[str, Any] = dict(
            # Weight ratio 10:3 instead of default 100:1
            weight_ocv=10.0,
            weight_dva=3.0,
            # Split ROI for OCV: roi_ocv_min holds the first interval (0-15% SOC),
            # roi_ocv_max the second (85-100% SOC), avoiding the flat middle
            roi_ocv_min=(0.0, 0.15),
            roi_ocv_max=(0.85, 1.0),
            # Standard DVA ROI
            roi_dva_min=0.10,
            roi_dva_max=0.90,
            # Disable cathode inhomogeneity (not meaningful for LFP)
            allow_cathode_inhomogeneity=False,
        )
        # User kwargs override defaults
        defaults.update(kwargs)
        return cls(**defaults)
