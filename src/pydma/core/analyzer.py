"""Main DMA Analyzer class.

This module provides the DMAAnalyzer class, which is the main entry point
for performing Degradation Mode Analysis. It orchestrates:
- Loading and validating electrode data
- Pre-computing measured DVA/ICA
- Running the optimization
- Computing degradation modes from fitted parameters
- Storing and returning results

Example
-------
>>> from pydma import DMAAnalyzer, DMAConfig, load_ocp
>>>
>>> # Load electrode data
>>> anode = load_ocp('anode_ocp.csv')
>>> cathode = load_ocp('cathode_ocp.csv')
>>>
>>> # Create analyzer
>>> config = DMAConfig(speed_preset='medium')
>>> analyzer = DMAAnalyzer(config)
>>>
>>> # Run analysis
>>> result = analyzer.analyze(
...     anode=anode,
...     cathode=cathode,
...     measured_capacity=cap,
...     measured_voltage=volt,
... )
>>>
>>> # Access results
>>> print(f"LLI: {result.degradation_modes.lli:.2%}")
"""

from typing import Callable, Any
from functools import partial
import warnings
import numpy as np
from numpy.typing import NDArray

from pydma.utils.dma_config import DMAConfig
from pydma.utils.results import (
    DMAResult,
    FittedParams,
    DegradationModes,
    ReferenceData,
)
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.electrodes.blend import BlendElectrode
from pydma.analysis.dva import calculate_dva, precompute_dva
from pydma.analysis.ica import calculate_ica, precompute_ica
from pydma.analysis.degradation import calculate_degradation_modes, calculate_mse
from pydma.utils.roi import build_roi_mask
from pydma.core.objectives import (
    combined_objective,
    objective_with_penalty,
    calculate_full_cell_ocv,
    fit_ocv,
    PreviousLAM,
    ReferenceData as ObjectiveRefData,
    PenaltyConfig,
    _interp1_linear_fill0,
)
from pydma.core.optimizer import DMAOptimizer, MultiRunResult


class DMAAnalyzer:
    """Main class for Degradation Mode Analysis.

    DMAAnalyzer orchestrates the complete DMA workflow:
    1. Set up electrode models (single or blend)
    2. Pre-compute measured DVA and ICA
    3. Configure and run the optimizer
    4. Compute degradation modes from fitted parameters
    5. Return comprehensive results

    Parameters
    ----------
    config : DMAConfig, optional
        Configuration object. If not provided, uses default settings.
    anode : ElectrodeOCP or BlendElectrode, optional
        Anode electrode model. Can also be set via set_anode().
    cathode : ElectrodeOCP or BlendElectrode, optional
        Cathode electrode model. Can also be set via set_cathode().
    reference_capacity : float, optional
        Reference capacity for degradation mode calculations.
        Typically the initial/fresh cell capacity.

    Attributes
    ----------
    config : DMAConfig
        Configuration object
    anode : ElectrodeOCP or BlendElectrode or None
        Anode electrode model
    cathode : ElectrodeOCP or BlendElectrode or None
        Cathode electrode model
    reference_capacity : float or None
        Reference capacity for degradation modes

    Examples
    --------
    >>> analyzer = DMAAnalyzer(DMAConfig(speed_preset='medium'))
    >>> analyzer.set_anode(my_anode)
    >>> analyzer.set_cathode(my_cathode)
    >>> result = analyzer.analyze(measured_capacity, measured_voltage)
    """

    def __init__(
        self,
        config: DMAConfig | None = None,
        anode: "ElectrodeOCP | BlendElectrode | None" = None,
        cathode: "ElectrodeOCP | BlendElectrode | None" = None,
        reference_capacity: float | None = None,
    ):
        self.config = config or DMAConfig()
        self.anode = anode
        self.cathode = cathode
        self.reference_capacity = reference_capacity
        self.reference_data: ReferenceData | None = None

        # State
        self._last_result: DMAResult | None = None
        self._previous_lam: PreviousLAM | None = None  # For penalty constraints between CUs
        self._previous_inhom_an: float | None = None  # For inhomogeneity delta constraint
        self._previous_inhom_ca: float | None = None  # For inhomogeneity delta constraint
        self._is_first_cu: bool = True  # Track if this is the first CU
        self._capacity_history: list[float] = []  # Track capacity across CUs for warning
        self._normalized_soc_warning_issued: bool = False  # Only warn once

    def set_anode(
        self,
        electrode: "ElectrodeOCP | BlendElectrode",
    ) -> "DMAAnalyzer":
        """Set the anode electrode model.

        Parameters
        ----------
        electrode : ElectrodeOCP or BlendElectrode
            Anode electrode model

        Returns
        -------
        DMAAnalyzer
            Self for method chaining
        """
        self.anode = electrode
        return self

    def set_cathode(
        self,
        electrode: "ElectrodeOCP | BlendElectrode",
    ) -> "DMAAnalyzer":
        """Set the cathode electrode model.

        Parameters
        ----------
        electrode : ElectrodeOCP or BlendElectrode
            Cathode electrode model

        Returns
        -------
        DMAAnalyzer
            Self for method chaining
        """
        self.cathode = electrode
        return self

    def set_reference_capacity(self, capacity: float) -> "DMAAnalyzer":
        """Set the reference capacity for degradation calculations.

        Parameters
        ----------
        capacity : float
            Reference (initial) capacity in Ah or mAh

        Returns
        -------
        DMAAnalyzer
            Self for method chaining
        """
        self.reference_capacity = capacity
        return self

    def _check_normalized_soc_warning(self) -> None:
        """Check if input appears to be normalized SOC and warn about LAM accuracy.

        If we have multiple CUs with capacity values all close to 1.0, this suggests
        the user is providing normalized SOC data instead of actual Ah capacity.
        In this case, LAM calculations will be incorrect because they won't account
        for the actual capacity fade between CUs.
        """
        if self._normalized_soc_warning_issued:
            return  # Only warn once

        if len(self._capacity_history) < 2:
            return  # Need at least 2 CUs to detect the pattern

        # Check if all capacities are close to 1.0 (normalized SOC)
        all_near_one = all(0.5 < cap < 1.5 for cap in self._capacity_history)

        # Check if capacities are suspiciously constant (no variation)
        if len(self._capacity_history) >= 2:
            cap_range = max(self._capacity_history) - min(self._capacity_history)
            capacities_constant = cap_range < 0.01  # Less than 1% variation
        else:
            capacities_constant = False

        if all_near_one and capacities_constant:
            self._normalized_soc_warning_issued = True
            warnings.warn(
                "Detected constant capacity (~1.0) across multiple CUs. "
                "This suggests normalized SOC data is being used instead of actual Ah capacity. "
                "LAM calculations may be INCORRECT because they won't reflect "
                "actual capacity fade. "
                "To fix: Either provide measured_capacity in Ah (not normalized SOC), "
                "or set reference_capacity to the actual cell capacity in Ah.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def anode_is_blend(self) -> bool:
        """Check if anode is a blend electrode."""
        return isinstance(self.anode, BlendElectrode)

    @property
    def cathode_is_blend(self) -> bool:
        """Check if cathode is a blend electrode."""
        return isinstance(self.cathode, BlendElectrode)

    def _validate_inputs(
        self,
        measured_capacity: NDArray[np.floating] | None = None,
        measured_voltage: NDArray[np.floating] | None = None,
    ) -> None:
        """Validate that all required inputs are set."""
        if self.anode is None:
            raise ValueError(
                "Anode electrode not set. Use set_anode() or pass anode to analyze()."
            )
        if self.cathode is None:
            raise ValueError(
                "Cathode electrode not set. Use set_cathode() or pass cathode to analyze()."
            )
        if measured_capacity is None or measured_voltage is None:
            raise ValueError("Measured capacity and voltage must be provided.")
        if len(measured_capacity) != len(measured_voltage):
            raise ValueError(
                f"Capacity and voltage arrays must have same length. "
                f"Got {len(measured_capacity)} and {len(measured_voltage)}."
            )
        if len(measured_capacity) < 10:
            raise ValueError(
                f"Need at least 10 data points. Got {len(measured_capacity)}."
            )

    def _prepare_electrode(
        self,
        electrode: ElectrodeOCP | BlendElectrode,
    ) -> ElectrodeOCP | BlendElectrode:
        """Prepare electrode data to match MATLAB preprocessing.

        MATLAB behavior:
        - Smooth half-cell curves with LOWESS (smoothingPoints)
        - Resample to uniform SOC grid (dataLength)
        - For blends, build common voltage grid with dataLength points
        """
        if isinstance(electrode, BlendElectrode):
            # MATLAB: blend1 is treated like a normal electrode (monotonic),
            # blend2 allows non-monotonic SOC for silicon-type curves.
            blend1 = self._prepare_blend_component(electrode.blend1, drop_decreasing=True)
            blend2 = self._prepare_blend_component(electrode.blend2, drop_decreasing=False)
            return BlendElectrode(
                blend1=blend1,
                blend2=blend2,
                electrode_type=electrode.electrode_type,
                name=electrode.name,
                n_points=self.config.data_length,
            )

        # ElectrodeOCP path
        return self._prepare_single_electrode(electrode)

    @staticmethod
    def _interp_linear_extrap(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        xq: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Linear interpolation with linear extrapolation at the ends (MATLAB interp1 'extrap')."""
        yq = np.interp(xq, x, y)
        if len(x) < 2:
            return yq

        dx_start = x[1] - x[0]
        dx_end = x[-1] - x[-2]
        slope_start = (y[1] - y[0]) / dx_start if abs(dx_start) > 1e-12 else 0.0
        slope_end = (y[-1] - y[-2]) / dx_end if abs(dx_end) > 1e-12 else 0.0

        mask_low = xq < x[0]
        mask_high = xq > x[-1]
        if np.any(mask_low):
            yq[mask_low] = y[0] + slope_start * (xq[mask_low] - x[0])
        if np.any(mask_high):
            yq[mask_high] = y[-1] + slope_end * (xq[mask_high] - x[-1])
        return yq

    @staticmethod
    def _ensure_monotonic(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        *,
        drop_decreasing: bool,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Ensure SOC/capacity is increasing; drop decreasing points when requested."""
        if x.size == 0:
            return x, y

        # Flip if needed so x starts low and ends high
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]

        if not drop_decreasing:
            return x, y

        # Drop points that would make x decrease
        keep = np.zeros_like(x, dtype=bool)
        last = -np.inf
        for i, val in enumerate(x):
            if np.isfinite(val) and val >= last:
                keep[i] = True
                last = val
        return x[keep], y[keep]

    def _prepare_single_electrode(self, electrode: ElectrodeOCP) -> ElectrodeOCP:
        """Prepare non-blend electrode to match MATLAB preprocessing."""
        soc = np.asarray(electrode.soc, dtype=np.float64).flatten()
        voltage = np.asarray(electrode.voltage, dtype=np.float64).flatten()

        # MATLAB behavior: enforce non-decreasing SOC for non-blend curves
        soc, voltage = self._ensure_monotonic(soc, voltage, drop_decreasing=True)

        # Smooth voltage (LOWESS) using index-based smoothing (MATLAB smooth)
        from pydma.preprocessing.smoother import apply_filter
        voltage = apply_filter(
            voltage,
            method="lowess",
            window=self.config.smoothing_points,
        )

        # Resample to uniform SOC grid with linear extrapolation
        soc_uniform = np.linspace(0.0, 1.0, self.config.data_length)
        if len(soc) >= 2:
            sort_idx = np.argsort(soc)
            soc_sorted = soc[sort_idx]
            volt_sorted = voltage[sort_idx]
            volt_uniform = self._interp_linear_extrap(soc_sorted, volt_sorted, soc_uniform)
        else:
            volt_uniform = np.full_like(soc_uniform, voltage[0] if len(voltage) else 0.0)

        return ElectrodeOCP(
            soc=soc_uniform,
            voltage=volt_uniform,
            name=electrode.name,
            electrode_type=electrode.electrode_type,
            capacity=electrode.capacity,
            is_smoothed=True,
        )

    def _prepare_blend_component(
        self,
        electrode: ElectrodeOCP,
        *,
        drop_decreasing: bool,
    ) -> ElectrodeOCP:
        """Prepare blend component to match MATLAB preprocessing.

        MATLAB blend path smooths voltage but keeps original SOC points
        for Q(V) inversion; it does not resample before blending.
        """
        soc = np.asarray(electrode.soc, dtype=np.float64).flatten()
        voltage = np.asarray(electrode.voltage, dtype=np.float64).flatten()

        # Flip if needed; optionally drop decreasing points (blend2 keeps them)
        soc, voltage = self._ensure_monotonic(
            soc,
            voltage,
            drop_decreasing=drop_decreasing,
        )

        # For blend components, keep SOC points (no resampling)
        from pydma.preprocessing.smoother import apply_filter
        voltage = apply_filter(
            voltage,
            method="lowess",
            window=self.config.smoothing_points,
        )

        return ElectrodeOCP(
            soc=soc,
            voltage=voltage,
            name=electrode.name,
            electrode_type=electrode.electrode_type,
            capacity=electrode.capacity,
            is_smoothed=True,
        )

    def _validate_fullcell_ocv_convention(
        self,
        capacity: NDArray[np.floating],
        voltage: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], bool]:
        """Validate and auto-correct full-cell OCV convention.

        Full-cell OCV convention: Voltage should INCREASE with increasing SOC.
        If not, the SOC axis is inverted (1 - SOC) to correct the convention.

        Parameters
        ----------
        capacity : NDArray
            Capacity/SOC values (should be in 0-1 range)
        voltage : NDArray
            Voltage values

        Returns
        -------
        tuple
            (capacity, voltage, was_corrected) - corrected arrays and correction flag
        """
        # First, ensure capacity is normalized to 0-1 range
        cap_min = capacity.min()
        cap_max = capacity.max()
        cap_span = cap_max - cap_min

        if cap_span > 0:
            cap_normalized = (capacity - cap_min) / cap_span
        else:
            cap_normalized = capacity.copy()

        # Calculate average slope using linear regression
        slope = np.polyfit(cap_normalized, voltage, 1)[0]

        if slope > 0:
            # Convention is correct: voltage INCREASES with SOC
            return cap_normalized, voltage.copy(), False
        else:
            # Convention violated: voltage DECREASES with SOC
            # Fix by inverting the SOC axis: SOC_new = 1 - SOC_old
            # This keeps voltage values the same but flips their relationship to SOC
            cap_corrected = 1.0 - cap_normalized
            return cap_corrected, voltage.copy(), True

    def _prepare_measured_data(
        self,
        measured_capacity: NDArray[np.floating],
        measured_voltage: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], float, float]:
        """Prepare full-cell measured data to match MATLAB preprocessing.

        MATLAB behavior (calculate_full_cell_data.m):
        - Smooth voltage with LOWESS (smoothingPoints)
        - Resample to uniform SOC grid (dataLength)
        - Q0 is raw SOC/capacity span (not normalized)

        Also validates that voltage INCREASES with increasing SOC (full-cell convention).
        """
        meas_capacity = np.asarray(measured_capacity, dtype=np.float64)
        meas_voltage = np.asarray(measured_voltage, dtype=np.float64)

        # Remove NaNs
        valid = ~(np.isnan(meas_capacity) | np.isnan(meas_voltage))
        meas_capacity = meas_capacity[valid]
        meas_voltage = meas_voltage[valid]

        if meas_capacity.size < 2:
            raise ValueError("Measured capacity must contain at least 2 valid points.")

        # MATLAB behavior: enforce non-decreasing capacity for full-cell data
        meas_capacity, meas_voltage = self._ensure_monotonic(
            meas_capacity,
            meas_voltage,
            drop_decreasing=True,
        )

        # Calculate cap_span BEFORE normalizing capacity
        # This preserves the actual Ah span for downstream degradation calculations
        cap_span = float(meas_capacity.max() - meas_capacity.min())

        # =============================================================================
        # VALIDATE FULL-CELL OCV CONVENTION
        # =============================================================================
        # Full-cell OCV must have: voltage INCREASES with increasing SOC
        # If not, auto-correct by inverting SOC axis and warn the user
        meas_capacity, meas_voltage, was_corrected = self._validate_fullcell_ocv_convention(
            meas_capacity,
            meas_voltage,
        )
        if cap_span <= 0:
            raise ValueError("Measured capacity must span a non-zero range.")

        # meas_capacity is already normalized to [0, 1] by
        # _validate_fullcell_ocv_convention.
        # No need to re-normalize here. cap_span preserves the original
        # Ah span for LAM calculations.
        q_raw = meas_capacity.copy()

        # Apply LOWESS smoothing to raw voltage (MATLAB-compatible)
        from pydma.preprocessing.smoother import apply_filter
        meas_voltage = apply_filter(
            meas_voltage,
            method=self.config.filter_type,
            **self.config.filter_kwargs,
        )

        # Resample to uniform SOC grid
        q_uniform = np.linspace(0.0, 1.0, self.config.data_length)

        # If duplicate q values exist, fall back to unique (MATLAB interp1 retry)
        q_sorted_idx = np.argsort(q_raw)
        q_sorted = q_raw[q_sorted_idx]
        v_sorted = meas_voltage[q_sorted_idx]
        if np.any(np.diff(q_sorted) == 0):
            q_unique, unique_idx = np.unique(q_sorted, return_index=True)
            v_unique = v_sorted[unique_idx]
            meas_voltage_uniform = self._interp_linear_extrap(q_unique, v_unique, q_uniform)
        else:
            meas_voltage_uniform = self._interp_linear_extrap(q_sorted, v_sorted, q_uniform)

        q0 = float(q_raw.max() - q_raw.min())

        return q_uniform, meas_voltage_uniform, q0, cap_span

    def _create_objective(
        self,
        q: NDArray[np.floating],
        meas_voltage: NDArray[np.floating],
        meas_dva: NDArray[np.floating],
        meas_ica: NDArray[np.floating],
        dva_roi_mask: NDArray[np.bool_],
        ica_roi_mask: NDArray[np.bool_],
        q0: float,
        actual_capacity: float | None = None,
        anode: ElectrodeOCP | BlendElectrode | None = None,
        cathode: ElectrodeOCP | BlendElectrode | None = None,
        anode_is_blend: bool | None = None,
        cathode_is_blend: bool | None = None,
    ) -> Callable[[NDArray[np.floating]], float]:
        """Create the objective function for optimization.

        Uses objective_with_penalty when:
        1. We have reference data (not first CU)
        2. We have previous LAM values to compare against
        3. Penalty config can be created from DMAConfig

        Otherwise uses combined_objective (no penalty).
        """
        # Check if we should use penalty constraints
        # Penalty is applied when we have previous LAM values (not first CU)
        use_penalty = (
            self.reference_data is not None
            and self._previous_lam is not None
            and actual_capacity is not None
        )

        if use_penalty:
            # Create penalty config from DMAConfig
            penalty_config = PenaltyConfig(
                max_anode_gain=self.config.max_anode_gain,
                max_cathode_gain=self.config.max_cathode_gain,
                max_anode_blend1_gain=self.config.max_anode_blend1_gain,
                max_anode_blend2_gain=self.config.max_anode_blend2_gain,
                max_anode_loss=self.config.max_anode_loss,
                max_cathode_loss=self.config.max_cathode_loss,
                max_anode_blend1_loss=self.config.max_anode_blend1_loss,
                max_anode_blend2_loss=self.config.max_anode_blend2_loss,
                use_anode_blend=self.config.use_anode_blend,
                use_cathode_blend=self.config.use_cathode_blend,
            )

            # Create reference data for objective function
            ref_data = ObjectiveRefData(
                capa_actual=actual_capacity,
                capa_anode_init=self.reference_data.capa_anode_init,
                capa_cathode_init=self.reference_data.capa_cathode_init,
                capa_inventory_init=self.reference_data.capa_inventory_init,
                gamma_an_blend2_init=self.reference_data.gamma_an_blend2_init,
                gamma_ca_blend2_init=self.reference_data.gamma_ca_blend2_init,
            )

            return partial(
                objective_with_penalty,
                anode=anode or self.anode,
                cathode=cathode or self.cathode,
                meas_voltage=meas_voltage,
                meas_dva=meas_dva,
                meas_ica=meas_ica,
                q=q,
                dva_roi_mask=dva_roi_mask,
                ica_roi_mask=ica_roi_mask,
                roi_ocv_min=self.config.roi_ocv_min,
                roi_ocv_max=self.config.roi_ocv_max,
                q0=q0,
                w_ocv=self.config.weight_ocv,
                w_dva=self.config.weight_dva,
                w_ica=self.config.weight_ica,
                anode_is_blend=(
                    anode_is_blend if anode_is_blend is not None else self.anode_is_blend
                ),
                cathode_is_blend=(
                    cathode_is_blend
                    if cathode_is_blend is not None
                    else self.cathode_is_blend
                ),
                inhom_points=61,
                ref_data=ref_data,
                prev_lam=self._previous_lam,
                penalty_config=penalty_config,
            )
        else:
            # First CU or no penalty config - use base objective
            return partial(
                combined_objective,
                anode=anode or self.anode,
                cathode=cathode or self.cathode,
                meas_voltage=meas_voltage,
                meas_dva=meas_dva,
                meas_ica=meas_ica,
                q=q,
                dva_roi_mask=dva_roi_mask,
                ica_roi_mask=ica_roi_mask,
                roi_ocv_min=self.config.roi_ocv_min,
                roi_ocv_max=self.config.roi_ocv_max,
                q0=q0,
                w_ocv=self.config.weight_ocv,
                w_dva=self.config.weight_dva,
                w_ica=self.config.weight_ica,
                anode_is_blend=(
                    anode_is_blend if anode_is_blend is not None else self.anode_is_blend
                ),
                cathode_is_blend=(
                    cathode_is_blend
                    if cathode_is_blend is not None
                    else self.cathode_is_blend
                ),
                inhom_points=61,  # Fixed as per MATLAB implementation
            )

    def analyze(
        self,
        measured_capacity: NDArray[np.floating] | None = None,
        measured_voltage: NDArray[np.floating] | None = None,
        anode: "ElectrodeOCP | BlendElectrode | None" = None,
        cathode: "ElectrodeOCP | BlendElectrode | None" = None,
        reference_capacity: float | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
        **kwargs: Any,
    ) -> DMAResult:
        """Perform Degradation Mode Analysis.

        This is the main entry point for running DMA. It:
        1. Validates inputs
        2. Pre-computes DVA and ICA from measured data
        3. Runs multi-run optimization
        4. Computes degradation modes
        5. Returns comprehensive results

        Parameters
        ----------
        measured_capacity : NDArray
            Measured capacity values (Ah or mAh)
        measured_voltage : NDArray
            Measured voltage values (V)
        anode : ElectrodeOCP or BlendElectrode, optional
            Anode electrode model (overrides set_anode())
        cathode : ElectrodeOCP or BlendElectrode, optional
            Cathode electrode model (overrides set_cathode())
        reference_capacity : float, optional
            Reference capacity for degradation calculations
            (overrides set_reference_capacity())
        progress_callback : Callable, optional
            Called after each optimization run with
            (accepted_count, rejected_count, run_number)
        **kwargs : Any
            Additional arguments passed to optimizer

        Returns
        -------
        DMAResult
            Comprehensive results including fitted parameters,
            degradation modes, and fit quality metrics

        Raises
        ------
        ValueError
            If required inputs are missing or invalid

        Examples
        --------
        >>> result = analyzer.analyze(
        ...     measured_capacity=cap_data,
        ...     measured_voltage=volt_data,
        ... )
        >>> print(f"LLI: {result.degradation_modes.lli:.2%}")
        >>> print(f"LAM_an: {result.degradation_modes.lam_an:.2%}")
        """
        # Update electrodes if provided
        if anode is not None:
            self.anode = anode
        if cathode is not None:
            self.cathode = cathode
        if reference_capacity is not None:
            self.reference_capacity = reference_capacity

        # Validate inputs
        self._validate_inputs(measured_capacity, measured_voltage)

        # Prepare measured data to match MATLAB preprocessing
        q, meas_voltage, q0, cap_span = self._prepare_measured_data(
            measured_capacity,
            measured_voltage,
        )

        # Prepare electrodes to match MATLAB preprocessing
        anode = self._prepare_electrode(self.anode)
        cathode = self._prepare_electrode(self.cathode)
        anode_is_blend = isinstance(anode, BlendElectrode)
        cathode_is_blend = isinstance(cathode, BlendElectrode)

        # Pre-compute DVA and ICA only when needed (MATLAB-compatible)
        # Skip computation when weight is 0 to save time
        if self.config.weight_dva > 0:
            meas_dva = precompute_dva(q, meas_voltage, q0=q0)
        else:
            meas_dva = np.zeros_like(q)  # Placeholder, won't be used

        if self.config.weight_ica > 0:
            meas_ica = precompute_ica(q, meas_voltage, q0=q0)
        else:
            meas_ica = np.zeros_like(q)  # Placeholder, won't be used

        # Display DVA/ICA: no Q0 normalisation, lowess smoothing.
        # Matches MATLAB dma_core.m lines 412-418 where calculate_DVA /
        # calculate_ICA are used for *both* measured and reconstructed
        # display curves (without Q0).  The Q0-normalised versions above
        # are kept only for the optimisation objective.
        _, _, display_dva_measured = calculate_dva(q, meas_voltage)
        _, _, display_ica_measured = calculate_ica(q, meas_voltage)

        # ROI masks (SOC-based) via shared ROI parser/validator (utils.roi)
        dva_roi_mask = build_roi_mask(q, self.config.roi_dva_min, self.config.roi_dva_max)
        ica_roi_mask = build_roi_mask(q, self.config.roi_ica_min, self.config.roi_ica_max)

        # Create objective function (with penalty constraints if not first CU)
        objective = self._create_objective(
            q, meas_voltage, meas_dva, meas_ica, dva_roi_mask, ica_roi_mask, q0,
            actual_capacity=cap_span,
            anode=anode,
            cathode=cathode,
            anode_is_blend=anode_is_blend,
            cathode_is_blend=cathode_is_blend,
        )

        # RMSE used for acceptance should be OCV RMSE in Volts, not sqrt(weighted-cost).
        ocv_mse_fn = partial(
            fit_ocv,
            anode=anode,
            cathode=cathode,
            meas_voltage=meas_voltage,
            q=q,
            roi_ocv_min=self.config.roi_ocv_min,
            roi_ocv_max=self.config.roi_ocv_max,
            anode_is_blend=anode_is_blend,
            cathode_is_blend=cathode_is_blend,
            inhom_points=61,
        )

        def rmse_fn(x: NDArray[np.floating]) -> float:
            return float(np.sqrt(ocv_mse_fn(x)))

        # Create and run optimizer
        # Get bounds with inhomogeneity constraints based on CU state
        # MATLAB behavior: allow_first_cycle_inhomogeneity controls first CU,
        # subsequent CUs are constrained by max_inhomogeneity_delta
        if self._is_first_cu and not self.config.allow_first_cycle_inhomogeneity:
            # First CU with inhomogeneity disabled - force inhom bounds to 0
            lb, ub = self.config.get_full_bounds(inhom_an_prev=0.0, inhom_ca_prev=0.0)
            # Override to exactly 0 (no delta allowed from 0)
            lb[6] = 0.0
            ub[6] = 0.0
            lb[7] = 0.0
            ub[7] = 0.0
            bounds = list(zip(lb, ub))
        else:
            # Pass previous inhom values for delta constraint (None for first CU = full range)
            lb, ub = self.config.get_full_bounds(
                inhom_an_prev=self._previous_inhom_an,
                inhom_ca_prev=self._previous_inhom_ca,
            )
            bounds = list(zip(lb, ub))

        optimizer = DMAOptimizer(self.config, objective, bounds, rmse_fn=rmse_fn)

        opt_result: MultiRunResult = optimizer.run(
            progress_callback=progress_callback,
            **kwargs,
        )

        # Extract fitted parameters
        params = opt_result.best_params
        fitted_params = FittedParams(
            alpha_an=params[0],
            beta_an=params[1],
            alpha_ca=params[2],
            beta_ca=params[3],
            gamma_blend2_an=params[4] if self.anode_is_blend else None,
            gamma_blend2_ca=params[5] if self.cathode_is_blend else None,
            inhom_an=params[6] if self.config.enable_inhomogeneity else None,
            inhom_ca=params[7] if self.config.enable_inhomogeneity else None,
        )

        # Compute degradation modes
        actual_capacity = cap_span

        # Convert fitted params to parameter array for degradation
        # calculation. The degradation function expects:
        # params, capa_actual, capa_anode_init, capa_cathode_init,
        # capa_inventory_init.
        param_array = np.array([
            fitted_params.alpha_an,
            fitted_params.beta_an,
            fitted_params.alpha_ca,
            fitted_params.beta_ca,
            fitted_params.gamma_blend2_an or 0.0,
            fitted_params.gamma_blend2_ca or 0.0,
            fitted_params.inhom_an or 0.0,
            fitted_params.inhom_ca or 0.0,
        ])

        # MATLAB behavior: first CU defines reference capacities from fitted params
        if self.reference_data is None:
            self.reference_data = ReferenceData(
                capa_anode_init=fitted_params.alpha_an * actual_capacity,
                capa_cathode_init=fitted_params.alpha_ca * actual_capacity,
                capa_inventory_init=(
                    fitted_params.alpha_ca + fitted_params.beta_ca - fitted_params.beta_an
                )
                * actual_capacity,
                gamma_an_blend2_init=float(fitted_params.gamma_blend2_an or 0.0),
                gamma_ca_blend2_init=float(fitted_params.gamma_blend2_ca or 0.0),
                reference_capacity=actual_capacity,
            )

        deg_result = calculate_degradation_modes(
            params=param_array,
            capa_actual=actual_capacity,
            capa_anode_init=self.reference_data.capa_anode_init,
            capa_cathode_init=self.reference_data.capa_cathode_init,
            capa_inventory_init=self.reference_data.capa_inventory_init,
            gamma_an_blend2_init=self.reference_data.gamma_an_blend2_init,
            gamma_ca_blend2_init=self.reference_data.gamma_ca_blend2_init,
        )

        # Capacity loss based on measured capacity (relative to reference)
        ref_capacity = self.reference_capacity or self.reference_data.reference_capacity
        if ref_capacity and ref_capacity != 0:
            capacity_loss = (ref_capacity - actual_capacity) / ref_capacity
        else:
            capacity_loss = 0.0

        # Convert to DegradationModes dataclass (include blend components)
        degradation_modes = DegradationModes(
            lli=deg_result.lli,
            lam_anode=deg_result.lam_anode,
            lam_cathode=deg_result.lam_cathode,
            capacity_loss=capacity_loss,
            lam_anode_blend1=deg_result.lam_anode_blend1,
            lam_anode_blend2=deg_result.lam_anode_blend2,
            lam_cathode_blend1=deg_result.lam_cathode_blend1,
            lam_cathode_blend2=deg_result.lam_cathode_blend2,
        )

        # Store current LAM values for penalty constraints in next CU analysis
        # MATLAB: objectiveWithPenalty uses LAM_prev* from previous CU
        self._previous_lam = PreviousLAM(
            lam_anode=deg_result.lam_anode,
            lam_cathode=deg_result.lam_cathode,
            lam_anode_blend1=deg_result.lam_anode_blend1 if self.config.use_anode_blend else None,
            lam_anode_blend2=deg_result.lam_anode_blend2 if self.config.use_anode_blend else None,
        )

        # Store current inhomogeneity values for delta constraint in next CU
        # MATLAB: subsequent CUs are constrained by max_inhomogeneity_delta
        self._previous_inhom_an = fitted_params.inhom_an
        self._previous_inhom_ca = fitted_params.inhom_ca
        self._is_first_cu = False  # No longer first CU after this analysis

        # Store reference data (first CU) for downstream use
        reference_data = self.reference_data

        # Compute fit quality metrics
        fit_ocv_mse = opt_result.best_cost / 3.0  # Approximate split
        fit_dva_mse = opt_result.best_cost / 3.0
        fit_ica_mse = opt_result.best_cost / 3.0

        # Calculate RMSE values matching MATLAB (calculate_RMSE.m)
        # Compute reconstructed OCV for RMSE calculation
        sim_curves = self.compute_simulated_curves(fitted_params, n_points=len(q))
        reconstructed_voltage = np.interp(q, sim_curves['capacity'], sim_curves['voltage'])

        # RMSE in fit region (ROI mask from shared ROI utilities)
        ocv_roi_mask = build_roi_mask(q, self.config.roi_ocv_min, self.config.roi_ocv_max)
        if not np.any(ocv_roi_mask):
            ocv_roi_mask = np.ones(len(q), dtype=bool)
        rmse_fit_region = np.sqrt(calculate_mse(meas_voltage, reconstructed_voltage, ocv_roi_mask))

        # RMSE over full range (no mask = all points)
        rmse_full_range = np.sqrt(calculate_mse(meas_voltage, reconstructed_voltage))

        # DVA RMSE over full range (both sides use calculate_dva-style,
        # no Q0, so the comparison is on consistent scales)
        reconstructed_dva_on_q = np.interp(q, sim_curves['capacity'], sim_curves['dva'])
        rmse_dva = np.sqrt(calculate_mse(display_dva_measured, reconstructed_dva_on_q))

        # Create result
        result = DMAResult(
            fitted_params=fitted_params,
            degradation_modes=degradation_modes,
            reference_data=reference_data,
            cost=opt_result.best_cost,
            rmse=opt_result.best_rmse,
            rmse_fit_region=rmse_fit_region,
            rmse_full_range=rmse_full_range,
            rmse_dva=rmse_dva,
            fit_ocv_mse=fit_ocv_mse,
            fit_dva_mse=fit_dva_mse,
            fit_ica_mse=fit_ica_mse,
            n_accepted_runs=opt_result.n_accepted,
            n_total_runs=opt_result.n_total,
            measured_capacity=q,
            measured_voltage=meas_voltage,
            soc_measured=q,
            ocv_measured=meas_voltage,
            measured_dva=display_dva_measured,
            measured_ica=display_ica_measured,
            dva_q_measured=q,
            dva_measured=display_dva_measured,
            dva_q_reconstructed=sim_curves['capacity'],
            dva_reconstructed=sim_curves['dva'],
            ica_q_measured=q,
            ica_measured=display_ica_measured,
            ica_q_reconstructed=sim_curves['capacity'],
            ica_reconstructed=sim_curves['ica'],
            soc_reconstructed=sim_curves['capacity'],
            ocv_reconstructed=sim_curves['voltage'],
            anode_soc=sim_curves['anode_soc'],
            anode_potential=sim_curves['anode_voltage'],
            cathode_soc=sim_curves['cathode_soc'],
            cathode_potential=sim_curves['cathode_voltage'],
            capacity=actual_capacity,
        )

        # Store for later access
        self._last_result = result

        # Track capacity history and warn if it looks like normalized SOC is being used
        self._capacity_history.append(actual_capacity)
        self._check_normalized_soc_warning()

        return result

    def compute_simulated_curves(
        self,
        params: FittedParams | None = None,
        n_points: int = 1000,
    ) -> dict[str, NDArray[np.floating]]:
        """Compute simulated full-cell curves from fitted parameters.

        Parameters
        ----------
        params : FittedParams, optional
            Fitted parameters. If not provided, uses last analysis result.
        n_points : int, optional
            Number of points for the curves, by default 1000

        Returns
        -------
        dict
            Dictionary with keys:
            - 'soc': SOC values
            - 'voltage': Full-cell voltage
            - 'capacity': Capacity values
            - 'dva': DVA (dV/dQ)
            - 'ica': ICA (dQ/dV)
            - 'anode_soc': Reconstructed anode SOC grid
            - 'anode_voltage': Reconstructed anode potential
            - 'cathode_soc': Reconstructed cathode SOC grid
            - 'cathode_voltage': Reconstructed cathode potential
        """
        if params is None:
            if self._last_result is None:
                raise ValueError("No parameters provided and no previous analysis result.")
            params = self._last_result.fitted_params

        if self.anode is None or self.cathode is None:
            raise ValueError("Electrodes not set.")

        from pydma.core.objectives import apply_params_to_electrode

        # Preprocess electrodes the same way as during fitting
        anode_prep = self._prepare_electrode(self.anode)
        cathode_prep = self._prepare_electrode(self.cathode)

        # Apply transformations
        anode_soc, anode_v = apply_params_to_electrode(
            anode_prep,
            params.alpha_an,
            params.beta_an,
            gamma_blend2=params.gamma_blend2_an or 0.0,
            inhom=params.inhom_an or 0.0,
        )

        cathode_soc, cathode_v = apply_params_to_electrode(
            cathode_prep,
            params.alpha_ca,
            params.beta_ca,
            gamma_blend2=params.gamma_blend2_ca or 0.0,
            inhom=params.inhom_ca or 0.0,
        )

        # MATLAB-matching reconstruction: fixed [0,1] grid with 0-fill outside
        # electrode range (matches dma_core.m lines 406-410).
        # Previously used calculate_full_cell_ocv() which normalised the
        # electrode intersection range to [0,1], misaligning with the
        # measured [0,1] grid.
        recon_soc = np.linspace(0.0, 1.0, n_points)
        anode_v_recon = _interp1_linear_fill0(anode_soc, anode_v, recon_soc)
        cathode_v_recon = _interp1_linear_fill0(
            cathode_soc, cathode_v, recon_soc
        )
        fc_voltage = cathode_v_recon - anode_v_recon

        # DVA and ICA from reconstructed full-cell
        # (matches MATLAB dma_core.m lines 412-426)
        _, _, dva = calculate_dva(recon_soc, fc_voltage)
        _, _, ica = calculate_ica(recon_soc, fc_voltage)

        return {
            'soc': recon_soc,
            'voltage': fc_voltage,
            'capacity': recon_soc,
            'dva': dva,
            'ica': ica,
            'anode_soc': anode_soc,
            'anode_voltage': anode_v,
            'cathode_soc': cathode_soc,
            'cathode_voltage': cathode_v,
        }

    def compare_with_reference(
        self,
        reference_params: FittedParams,
        current_params: FittedParams | None = None,
        reference_capacity: float | None = None,
        current_capacity: float | None = None,
    ) -> DegradationModes:
        """Compare current state with a reference state.

        Useful for computing degradation modes relative to a specific
        reference point (e.g., BOL state).

        Parameters
        ----------
        reference_params : FittedParams
            Parameters from reference measurement
        current_params : FittedParams, optional
            Current parameters. If not provided, uses last analysis result.
        reference_capacity : float, optional
            Reference capacity. If not provided, uses stored value.
        current_capacity : float, optional
            Measured capacity for current state. If provided, used to compute
            measured capacity loss.

        Returns
        -------
        DegradationModes
            Degradation modes relative to reference
        """
        if current_params is None:
            if self._last_result is None:
                raise ValueError("No current parameters and no previous analysis.")
            current_params = self._last_result.fitted_params

        if reference_capacity is None:
            reference_capacity = self.reference_capacity
        if reference_capacity is None:
            raise ValueError("Reference capacity not set.")

        # MATLAB calculate_degradation_modes.m uses capa_act (the *current*
        # CU capacity) for current-state sub-capacities and the reference
        # capacity for initial-state sub-capacities.
        if current_capacity is None:
            raise ValueError(
                "current_capacity is required for compare_with_reference. "
                "Pass the measured capacity of the current CU."
            )
        capa_for_current = current_capacity

        # LLI: capa_inventory = (alpha_ca + beta_ca - beta_an) * capa_act
        capa_inventory_init = (
            reference_params.alpha_ca + reference_params.beta_ca - reference_params.beta_an
        ) * reference_capacity
        capa_inventory_current = (
            current_params.alpha_ca + current_params.beta_ca - current_params.beta_an
        ) * capa_for_current
        lli = (capa_inventory_init - capa_inventory_current) / capa_inventory_init

        # LAM: capa_anode = alpha_anode * capa_act
        capa_anode_init = reference_params.alpha_an * reference_capacity
        capa_cathode_init = reference_params.alpha_ca * reference_capacity
        capa_anode_current = current_params.alpha_an * capa_for_current
        capa_cathode_current = current_params.alpha_ca * capa_for_current
        lam_an = (
            (capa_anode_init - capa_anode_current) / capa_anode_init
            if capa_anode_init != 0 else 0.0
        )
        lam_ca = (
            (capa_cathode_init - capa_cathode_current) / capa_cathode_init
            if capa_cathode_init != 0 else 0.0
        )

        # Compute measured capacity loss when available
        if current_capacity is not None:
            capacity_loss = 1.0 - (current_capacity / reference_capacity)
        else:
            capacity_loss = 0.0

        return DegradationModes(
            lli=lli,
            lam_an=lam_an,
            lam_ca=lam_ca,
            capacity_loss=capacity_loss,
        )

    @property
    def last_result(self) -> DMAResult | None:
        """Get the result from the last analysis."""
        return self._last_result

    def reset_state(self) -> "DMAAnalyzer":
        """Reset the analyzer state for a new aging study.

        Clears reference data, previous LAM values, and inhomogeneity state,
        so the next analyze() call will be treated as the first CU
        (no penalty constraints, no inhomogeneity delta constraints).

        Returns
        -------
        DMAAnalyzer
            Self for method chaining.
        """
        self.reference_data = None
        self._previous_lam = None
        self._previous_inhom_an = None
        self._previous_inhom_ca = None
        self._is_first_cu = True
        self._last_result = None
        return self
