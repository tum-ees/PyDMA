"""
Result dataclasses for DMA analysis.

This module defines dataclasses for storing DMA analysis results,
including single-CU results and multi-CU aging study results.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _longest_increasing_run(values: np.ndarray) -> tuple[int, int]:
    """Half-open index range of the longest strictly increasing run in ``values``.

    Ties break a run: two equal samples carry no direction, so a segment that
    contains them cannot be inverted.
    """
    n = int(values.size)
    if n < 2:
        return 0, n

    breaks = np.flatnonzero(~(np.diff(values) > 0))
    starts = np.concatenate((np.zeros(1, dtype=np.intp), breaks + 1))
    stops = np.concatenate((breaks + 1, np.array([n], dtype=np.intp)))
    best = int(np.argmax(stops - starts))
    return int(starts[best]), int(stops[best])


@dataclass
class DegradationModes:
    """
    Degradation mode values for a single check-up.

    Attributes
    ----------
    lam_anode : float
        Loss of Active Material at Anode (relative to reference).
    lam_cathode : float
        Loss of Active Material at Cathode (relative to reference).
    lli : float
        Loss of Lithium Inventory (relative to reference).
    capacity_loss : float
        Measured capacity loss (relative to reference capacity).
    lam_anode_blend1 : float, optional
        LAM for anode blend1 component (e.g., graphite in Si-Gr).
    lam_anode_blend2 : float, optional
        LAM for anode blend2 component (e.g., silicon in Si-Gr).
    lam_cathode_blend1 : float, optional
        LAM for cathode blend1 component.
    lam_cathode_blend2 : float, optional
        LAM for cathode blend2 component.
    """

    lam_anode: float
    lam_cathode: float
    lli: float
    capacity_loss: float = 0.0
    lam_anode_blend1: float = 0.0
    lam_anode_blend2: float = 0.0
    lam_cathode_blend1: float = 0.0
    lam_cathode_blend2: float = 0.0

    @property
    def lam_an(self) -> float:
        """Alias for lam_anode."""
        return self.lam_anode

    @property
    def lam_ca(self) -> float:
        """Alias for lam_cathode."""
        return self.lam_cathode

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "lam_anode": self.lam_anode,
            "lam_cathode": self.lam_cathode,
            "lli": self.lli,
            "capacity_loss": self.capacity_loss,
            "lam_anode_blend1": self.lam_anode_blend1,
            "lam_anode_blend2": self.lam_anode_blend2,
            "lam_cathode_blend1": self.lam_cathode_blend1,
            "lam_cathode_blend2": self.lam_cathode_blend2,
        }

    def summary(self) -> str:
        """
        Get a formatted summary of degradation modes.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "Degradation Modes:",
            (
                f"  LLI={self.lli:.2%}, "
                f"LAM_an={self.lam_anode:.2%}, "
                f"LAM_ca={self.lam_cathode:.2%}, "
                f"CapLoss={self.capacity_loss:.2%}"
            ),
        ]

        an_b1 = self.lam_anode_blend1 or 0.0
        an_b2 = self.lam_anode_blend2 or 0.0
        ca_b1 = self.lam_cathode_blend1 or 0.0
        ca_b2 = self.lam_cathode_blend2 or 0.0

        if an_b1 != 0.0 or an_b2 != 0.0:
            lines.append(f"  Anode blend: b1={an_b1:.2%}, b2={an_b2:.2%}")
        if ca_b1 != 0.0 or ca_b2 != 0.0:
            lines.append(f"  Cathode blend: b1={ca_b1:.2%}, b2={ca_b2:.2%}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation showing summary."""
        return self.summary()


@dataclass
class FittedParams:
    """
    Fitted parameters from optimization.

    Attributes
    ----------
    alpha_an : float
        Anode scaling / capacity ratio.
    beta_an : float
        Anode offset / SOC shift.
    alpha_ca : float
        Cathode scaling / capacity ratio.
    beta_ca : float
        Cathode offset / SOC shift.
    gamma_blend2_an : float
        Anode blend2 fraction (0 if disabled).
    gamma_blend2_ca : float
        Cathode blend2 fraction (0 if disabled).
    inhom_an : float
        Anode inhomogeneity magnitude.
    inhom_ca : float
        Cathode inhomogeneity magnitude.
    r_offset_ohm : float, optional
        Fitted series resistance in Ohm. ``None`` means the fit did not carry
        the resistance correction, which is the default.
    """

    alpha_an: float
    beta_an: float
    alpha_ca: float
    beta_ca: float
    gamma_blend2_an: float | None = 0.0
    gamma_blend2_ca: float | None = 0.0
    inhom_an: float | None = 0.0
    inhom_ca: float | None = 0.0
    r_offset_ohm: float | None = None

    # ============================================================
    # Derived stoichiometry properties for PyBaMM users
    # ============================================================

    @property
    def utilization_an(self) -> float:
        """
        Fraction of anode electrode capacity used by the cell (1/alpha_an).

        A value of 0.95 means 95% of the anode's theoretical capacity
        is utilized in the cell's operating window. A zero ``alpha_an``
        describes an electrode of unbounded capacity and returns ``inf``.
        """
        return 1.0 / self.alpha_an if self.alpha_an != 0 else float("inf")

    @property
    def utilization_ca(self) -> float:
        """
        Fraction of cathode electrode capacity used by the cell (1/alpha_ca).

        A value of 0.90 means 90% of the cathode's theoretical capacity
        is utilized in the cell's operating window. A zero ``alpha_ca``
        describes an electrode of unbounded capacity and returns ``inf``.
        """
        return 1.0 / self.alpha_ca if self.alpha_ca != 0 else float("inf")

    @property
    def sto_init_an(self) -> float:
        """
        Initial stoichiometry for anode at 0% cell SOC (-beta_an/alpha_an).

        This equals c_init/c_max for the anode in PyBaMM terms:
            c_init = sto_init_an × c_max

        Example: sto_init_an = 0.015 means at 0% cell SOC,
        c_init_an = 0.015 × c_max_an (1.5% of maximum concentration).
        """
        return -self.beta_an / self.alpha_an

    @property
    def sto_init_ca(self) -> float:
        """
        Initial stoichiometry (lithiation x) for cathode at 0% cell SOC.

        This equals c_init/c_max for the cathode in PyBaMM terms:
            c_init = sto_init_ca × c_max

        At 0% cell SOC, cathode is highly lithiated (discharged state).
        Example: sto_init_ca = 0.91 means c_init_ca = 0.91 × c_max_ca
        """
        # The fit places the cathode on its delithiation axis; the position there
        # at 0% cell SOC is the same -beta/alpha the anode uses. Lithiation is its
        # complement.
        sto_delith_at_0 = -self.beta_ca / self.alpha_ca
        return 1.0 - sto_delith_at_0

    @property
    def sto_window_an(self) -> tuple[float, float]:
        """
        Anode stoichiometry window as (sto_at_0%_SOC, sto_at_100%_SOC).

        Returns c/c_max ratios at cell SOC boundaries (PyBaMM convention):
        - First value: c/c_max at 0% cell SOC (discharged)
        - Second value: c/c_max at 100% cell SOC (charged)

        For anodes: first < second (anode lithiates when cell charges)
        Example: (0.01, 0.95) means:
        - At 0% SOC: c_an = 0.01 × c_max_an
        - At 100% SOC: c_an = 0.95 × c_max_an
        """
        sto_at_0 = self.sto_init_an
        sto_at_100 = self.sto_init_an + self.utilization_an
        return (sto_at_0, sto_at_100)

    @property
    def sto_window_ca(self) -> tuple[float, float]:
        """
        Cathode stoichiometry window as (x_at_0%_SOC, x_at_100%_SOC).

        Returns lithiation fraction x (= c/c_max) at cell SOC boundaries:
        - First value: x at 0% cell SOC (discharged, highly lithiated)
        - Second value: x at 100% cell SOC (charged, delithiated)

        For cathodes: first > second (cathode delithiates when cell charges)
        Example: (0.91, 0.02) means:
        - At 0% SOC: c_ca = 0.91 × c_max_ca (91% lithiated)
        - At 100% SOC: c_ca = 0.02 × c_max_ca (2% lithiated)
        """
        x_at_0 = self.sto_init_ca  # Already converted to lithiation x
        x_at_100 = x_at_0 - self.utilization_ca  # Lithiation decreases as cell charges
        return (x_at_0, x_at_100)

    def sto_window_an_per_phase(
        self, blend_electrode, gamma: float | None = None
    ) -> dict[str, Any]:
        """
        Return graphite/silicon stoichiometry windows for a blend anode.

        ``sto_window_an`` always remains the fitted normalized blend coordinate.
        For blend electrodes, physical component stoichiometries require the
        ``BlendElectrode`` used during fitting, because the mapping is
        voltage-based:

            blend sto -> blend voltage -> blend1/blend2 stoichiometry

        Parameters
        ----------
        blend_electrode : BlendElectrode
            The blend anode used for the fit.
        gamma : float, optional
            Blend2 fraction. Defaults to ``gamma_blend2_an``.

        Returns
        -------
        dict
            Component stoichiometry window and raw blend normalization values.
        """
        if gamma is None:
            gamma = self.gamma_blend2_an
        if gamma is None:
            raise ValueError("gamma_blend2_an is not set for this fit")
        if not hasattr(blend_electrode, "get_component_stoichiometry_window"):
            raise TypeError("blend_electrode must provide get_component_stoichiometry_window")
        # Keep sto_window_an unchanged; this explicit helper returns phase windows.
        result: dict[str, Any] = blend_electrode.get_component_stoichiometry_window(
            float(gamma), self.sto_window_an
        )
        return result

    def with_voltage_anchored_windows(
        self,
        soc_recon: np.ndarray,
        v_recon: np.ndarray,
        v_min: float,
        v_max: float,
        tol: float = 1e-6,
    ) -> "FittedParams":
        """
        Return a NEW :class:`FittedParams` whose ``sto_window_*`` properties
        are anchored to the input cell voltage cutoffs ``(v_min, v_max)``
        rather than to PyDMA's normalised SOC = [0, 1] boundaries.

        This is a post-processing/export helper. It does not represent the
        optimiser output and should not be used for DMA fit quality, degradation
        calculations, or reference-state tracking. Use it only when a downstream
        consumer needs a voltage-bounded parameterisation (for example PyBaMM
        c_max / c_init export).

        The fit's (alpha, beta) define the SOC → sto mapping, but the
        endpoints at SOC = 0 and SOC = 1 are determined by the optimiser's
        free extrapolation past the measured voltage range. The returned
        instance carries re-derived (alpha, beta) such that:

            ``V_cell(s_anc = 0) = v_min``  and  ``V_cell(s_anc = 1) = v_max``

        Numerically, given the reconstructed OCV curve ``v_recon(soc_recon)``
        evaluated with the raw fit:

            ``s_low  = soc such that v_recon(soc) = v_min``
            ``s_high = soc such that v_recon(soc) = v_max``
            ``α_anc  = α_raw / (s_high − s_low)``
            ``β_anc  = (β_raw − s_low) / (s_high − s_low)``

        The same (s_low, s_high) is used for both anode and cathode because
        cell voltage cutoffs are properties of the full cell — both
        electrodes hit their respective endpoints at the same cell SOC.

        Parameters
        ----------
        soc_recon, v_recon : array-like
            Reconstructed OCV curve sampled across the fit's SOC range.
            Both must be 1-D and the same length.
        v_min, v_max : float
            Voltage cutoffs from the input pseudo-OCV.
        tol : float, optional
            Tolerance allowed when checking that ``v_min``/``v_max`` lie
            inside the reconstructed range. Defaults to 1e-6 V.

        Returns
        -------
        FittedParams
            New instance with anchored alpha/beta. ``gamma_blend2_*``,
            ``inhom_*`` and ``r_offset_ohm`` are copied through unchanged; the
            resistance term shifts the voltage level, not the SOC window the
            anchoring rescales.

        Raises
        ------
        ValueError
            If the reconstructed arrays are empty, the requested voltages lie
            outside the reconstructed range so anchoring would extrapolate,
            or the implied window shrink factor is non-positive.
        """
        soc = np.asarray(soc_recon, dtype=float)
        v = np.asarray(v_recon, dtype=float)
        if soc.size == 0 or v.size == 0:
            raise ValueError(
                "with_voltage_anchored_windows() needs non-empty soc_recon " "and v_recon arrays."
            )
        if soc.shape != v.shape:
            raise ValueError(f"soc_recon shape {soc.shape} does not match v_recon shape {v.shape}.")

        order = np.argsort(v)
        v_sorted = v[order]
        soc_sorted = soc[order]
        v_lo, v_hi = float(v_sorted[0]), float(v_sorted[-1])
        if v_min < v_lo - tol or v_max > v_hi + tol:
            raise ValueError(
                f"Requested voltage cutoffs [{v_min:.4f}, {v_max:.4f}] V lie "
                f"outside the reconstructed OCV range [{v_lo:.4f}, {v_hi:.4f}] V; "
                "anchoring would require extrapolation."
            )

        s_low = float(np.interp(v_min, v_sorted, soc_sorted))
        s_high = float(np.interp(v_max, v_sorted, soc_sorted))
        delta_s = s_high - s_low
        if delta_s <= 0:
            raise ValueError(
                f"Anchoring failed: s_high - s_low = {delta_s} (needs to be > 0). "
                f"Got s_low={s_low}, s_high={s_high}."
            )

        return FittedParams(
            alpha_an=self.alpha_an / delta_s,
            beta_an=(self.beta_an - s_low) / delta_s,
            alpha_ca=self.alpha_ca / delta_s,
            beta_ca=(self.beta_ca - s_low) / delta_s,
            gamma_blend2_an=self.gamma_blend2_an,
            gamma_blend2_ca=self.gamma_blend2_ca,
            inhom_an=self.inhom_an,
            inhom_ca=self.inhom_ca,
            r_offset_ohm=self.r_offset_ohm,
        )

    @classmethod
    def from_array(cls, params: np.ndarray) -> "FittedParams":
        """
        Create from a parameter array of 4 to 9 elements.

        Parameters
        ----------
        params : np.ndarray
            Array in standard parameter order. Slots the array does not reach
            are filled with 0.0, except the resistance offset: a vector that
            stops short of the ninth slot has no resistance to report and
            leaves ``r_offset_ohm`` at ``None``, while a ninth component
            becomes a float. :meth:`to_array` writes that ``None`` back as
            0.0, which leaves the reconstruction untouched either way.

        Returns
        -------
        FittedParams
            Instance populated from array.

        Raises
        ------
        ValueError
            If the array holds fewer than 4 or more than 9 elements. A longer
            array carries a slot this layout has no name for, which is a
            mismatch rather than something to truncate.
        """
        params = np.asarray(params).flatten()
        if len(params) < 4:
            raise ValueError(f"params must have at least 4 elements, got {len(params)}")
        if len(params) > 9:
            raise ValueError(f"params must have at most 9 elements, got {len(params)}")

        # Pad to 9 elements if needed
        full_params = np.zeros(9)
        full_params[: len(params)] = params

        return cls(
            alpha_an=full_params[0],
            beta_an=full_params[1],
            alpha_ca=full_params[2],
            beta_ca=full_params[3],
            gamma_blend2_an=full_params[4],
            gamma_blend2_ca=full_params[5],
            inhom_an=full_params[6],
            inhom_ca=full_params[7],
            r_offset_ohm=float(params[8]) if len(params) > 8 else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FittedParams":
        """
        Create from a :meth:`to_dict` mapping.

        The derived entries ``to_dict`` adds (``utilization_*``,
        ``sto_init_*``, ``sto_window_*``) are recomputed from the stored
        alpha/beta and are therefore ignored here, which makes
        ``FittedParams.from_dict(p.to_dict())`` a faithful round trip.

        Parameters
        ----------
        data : dict
            Mapping holding at least ``alpha_an``, ``beta_an``, ``alpha_ca``
            and ``beta_ca``.

        Returns
        -------
        FittedParams
            Instance populated from the mapping.
        """
        missing = [k for k in ("alpha_an", "beta_an", "alpha_ca", "beta_ca") if k not in data]
        if missing:
            raise ValueError(f"FittedParams.from_dict is missing required key(s): {missing}.")

        return cls(
            alpha_an=float(data["alpha_an"]),
            beta_an=float(data["beta_an"]),
            alpha_ca=float(data["alpha_ca"]),
            beta_ca=float(data["beta_ca"]),
            gamma_blend2_an=data.get("gamma_blend2_an", 0.0),
            gamma_blend2_ca=data.get("gamma_blend2_ca", 0.0),
            inhom_an=data.get("inhom_an", 0.0),
            inhom_ca=data.get("inhom_ca", 0.0),
            r_offset_ohm=data.get("r_offset_ohm"),
        )

    def to_array(self) -> np.ndarray:
        """
        Convert to 9-element parameter array.

        Returns
        -------
        np.ndarray
            9-element array in standard parameter order.

        Notes
        -----
        The array has no place for "not fitted": a ``None`` gamma, inhom or
        resistance serialises as ``0.0``, so a round trip through the array
        turns "blend disabled" into "blend fitted at zero". Use :meth:`to_dict`
        where that distinction matters. A zero in the resistance slot leaves
        the reconstruction untouched, so that particular collapse is harmless.
        """
        return np.array(
            [
                self.alpha_an,
                self.beta_an,
                self.alpha_ca,
                self.beta_ca,
                self.gamma_blend2_an if self.gamma_blend2_an is not None else 0.0,
                self.gamma_blend2_ca if self.gamma_blend2_ca is not None else 0.0,
                self.inhom_an if self.inhom_an is not None else 0.0,
                self.inhom_ca if self.inhom_ca is not None else 0.0,
                self.r_offset_ohm if self.r_offset_ohm is not None else 0.0,
            ],
            dtype=float,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including derived stoichiometry values."""
        return {
            "alpha_an": self.alpha_an,
            "beta_an": self.beta_an,
            "alpha_ca": self.alpha_ca,
            "beta_ca": self.beta_ca,
            "gamma_blend2_an": self.gamma_blend2_an,
            "gamma_blend2_ca": self.gamma_blend2_ca,
            "inhom_an": self.inhom_an,
            "inhom_ca": self.inhom_ca,
            "r_offset_ohm": self.r_offset_ohm,
            # Derived stoichiometry values for PyBaMM users
            "utilization_an": self.utilization_an,
            "utilization_ca": self.utilization_ca,
            "sto_init_an": self.sto_init_an,
            "sto_init_ca": self.sto_init_ca,
            "sto_window_an": self.sto_window_an,
            "sto_window_ca": self.sto_window_ca,
        }

    def summary(self) -> str:
        """
        Get a formatted summary of fitted parameters and derived stoichiometry.

        Returns
        -------
        str
            Multi-line summary string with raw parameters and PyBaMM-compatible values.
        """
        lines = [
            "Fitted Parameters:",
            f"  Anode:   alpha={self.alpha_an:.4f}, beta={self.beta_an:.4f}",
            f"           utilization={self.utilization_an:.2%}, "
            f"sto_window=[{self.sto_window_an[0]:.3f}, {self.sto_window_an[1]:.3f}]",
            f"  Cathode: alpha={self.alpha_ca:.4f}, beta={self.beta_ca:.4f}",
            f"           utilization={self.utilization_ca:.2%}, "
            f"sto_window=[{self.sto_window_ca[0]:.3f}, {self.sto_window_ca[1]:.3f}]",
        ]
        gamma_an = self.gamma_blend2_an or 0.0
        gamma_ca = self.gamma_blend2_ca or 0.0
        if gamma_an > 0:
            lines.append(f"  Blend:   gamma_an={gamma_an:.4f}")
        if gamma_ca > 0:
            lines.append(f"           gamma_ca={gamma_ca:.4f}")
        inhom_an = self.inhom_an or 0.0
        inhom_ca = self.inhom_ca or 0.0
        if inhom_an > 0 or inhom_ca > 0:
            lines.append(f"  Inhom:   an={inhom_an:.4f}, ca={inhom_ca:.4f}")
        if self.r_offset_ohm is not None:
            lines.append(f"  R:       {self.r_offset_ohm * 1000.0:.2f} mOhm")
        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation showing summary."""
        return self.summary()


@dataclass
class DMAResult:
    """
    Result of DMA analysis for a single check-up (CU).

    Attributes
    ----------
    cu_name : str
        Name/identifier of the check-up (e.g., 'CU1', 'CU2').
    fitted_params : FittedParams
        Fitted parameters from optimization.
    degradation_modes : DegradationModes
        Calculated degradation modes.

    soc_measured : np.ndarray
        SOC grid for measured pOCV.
    ocv_measured : np.ndarray
        Measured pOCV values.
    soc_reconstructed : np.ndarray
        SOC grid for reconstructed OCV.
    ocv_reconstructed : np.ndarray
        Reconstructed OCV values.

    dva_q_measured : np.ndarray
        Charge grid for measured DVA.
    dva_measured : np.ndarray
        Measured DVA values.
    dva_q_reconstructed : np.ndarray
        Charge grid for reconstructed DVA.
    dva_reconstructed : np.ndarray
        Reconstructed DVA values.

    ica_q_measured : np.ndarray
        Charge grid for measured ICA.
    ica_measured : np.ndarray
        Measured ICA values.
    ica_q_reconstructed : np.ndarray
        Charge grid for reconstructed ICA.
    ica_reconstructed : np.ndarray
        Reconstructed ICA values.

    anode_soc : np.ndarray
        SOC grid for anode potential.
    anode_potential : np.ndarray
        Anode potential values.
    cathode_soc : np.ndarray
        SOC grid for cathode potential.
    cathode_potential : np.ndarray
        Cathode potential values.

    capacity : float
        Actual cell capacity for this CU.
    rmse_fit_region : float
        RMSE in the fitting region.
    rmse_full_range : float
        RMSE over full 0-100% SOC range.
    rmse_dva : float
        RMSE between measured and reconstructed DVA.

    is_accepted : bool
        Whether the solution was below RMSE threshold.
    status : str
        Status string ('accepted', 'rejected_above_threshold', etc.).
    algorithm : str
        Algorithm used for optimization.

    config_snapshot : dict
        The ``DMAConfig`` this fit ran with, as a plain dict. A result read
        back months later carries the settings that produced it.
    param_std : np.ndarray, optional
        Standard deviation of the parameters over the accepted runs, ``None``
        when no run met the threshold. With ``req_accepted = 1`` there is one
        accepted run and the entries are all zero.
    """

    cu_name: str = ""
    fitted_params: FittedParams = field(default_factory=lambda: FittedParams(1.0, 0.0, 1.0, 0.0))
    degradation_modes: DegradationModes = field(
        default_factory=lambda: DegradationModes(0.0, 0.0, 0.0)
    )

    # Optional reference data (from first CU)
    reference_data: "ReferenceData | None" = None

    # Measured and reconstructed OCV (optional - may be set later)
    soc_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    ocv_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    soc_reconstructed: np.ndarray = field(default_factory=lambda: np.array([]))
    ocv_reconstructed: np.ndarray = field(default_factory=lambda: np.array([]))

    # DVA curves
    dva_q_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    dva_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    dva_q_reconstructed: np.ndarray = field(default_factory=lambda: np.array([]))
    dva_reconstructed: np.ndarray = field(default_factory=lambda: np.array([]))

    # ICA curves
    ica_q_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    ica_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    ica_q_reconstructed: np.ndarray = field(default_factory=lambda: np.array([]))
    ica_reconstructed: np.ndarray = field(default_factory=lambda: np.array([]))

    # Electrode potentials
    anode_soc: np.ndarray = field(default_factory=lambda: np.array([]))
    anode_potential: np.ndarray = field(default_factory=lambda: np.array([]))
    cathode_soc: np.ndarray = field(default_factory=lambda: np.array([]))
    cathode_potential: np.ndarray = field(default_factory=lambda: np.array([]))

    # Capacity and error metrics
    capacity: float = 0.0
    rmse_fit_region: float = 0.0
    rmse_full_range: float = 0.0
    rmse_dva: float = 0.0

    # New / analyzer-facing fields (ensure compatibility)
    cost: float = 0.0
    rmse: float = 0.0
    fit_ocv_mse: float = 0.0
    fit_dva_mse: float = 0.0
    fit_ica_mse: float = 0.0

    # Optimization run counts
    n_accepted_runs: int = 0
    n_total_runs: int = 0

    # Measured arrays stored for later plotting/inspection
    # measured_capacity is SOC normalized to [0, 1] (MATLAB Q_cell convention)
    measured_capacity: np.ndarray = field(default_factory=lambda: np.array([]))
    measured_voltage: np.ndarray = field(default_factory=lambda: np.array([]))
    measured_dva: np.ndarray = field(default_factory=lambda: np.array([]))
    measured_ica: np.ndarray = field(default_factory=lambda: np.array([]))

    # Status
    is_accepted: bool = True
    status: str = "accepted"
    algorithm: str = "differential_evolution"

    # Fit provenance
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    param_std: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary containing all result data.

        Notes
        -----
        The measured and reconstructed curves stay out of the dictionary, as
        they do for every other array on this class: they are one array per
        grid point and belong in the pickle, not in a summary record. The
        nine-element ``param_std`` is provenance rather than a curve, so it is
        carried as a list, and a run without accepted runs keeps its ``None``.
        """
        return {
            "cu_name": self.cu_name,
            "fitted_params": self.fitted_params.to_dict(),
            "degradation_modes": self.degradation_modes.to_dict(),
            "capacity": self.capacity,
            "rmse_fit_region": self.rmse_fit_region,
            "rmse_full_range": self.rmse_full_range,
            "rmse_dva": self.rmse_dva,
            "cost": self.cost,
            "rmse": self.rmse,
            "n_accepted_runs": self.n_accepted_runs,
            "n_total_runs": self.n_total_runs,
            "is_accepted": self.is_accepted,
            "status": self.status,
            "algorithm": self.algorithm,
            "config_snapshot": dict(self.config_snapshot),
            "param_std": None if self.param_std is None else np.asarray(self.param_std).tolist(),
        }

    # ============================================================
    # Voltage-anchored stoichiometry windows
    # ============================================================
    def voltage_anchored_windows(
        self,
        v_min: float | None = None,
        v_max: float | None = None,
        on_out_of_range: str = "raise",
    ) -> dict[str, Any]:
        """
        Re-anchor stoichiometry windows so endpoints correspond to the voltage
        cutoffs of the measured pseudo-OCV (e.g. 2.5 V / 4.2 V) rather than to
        PyDMA's normalized SOC = [0, 1] boundaries.

        This is a post-processing helper for exporting voltage-bounded windows.
        It does not change ``self.fitted_params`` and is not part of the DMA
        optimisation itself. The raw fit remains the source of truth for fit
        quality metrics, degradation calculations, and reference tracking.

        The fitted alpha/beta define the cell voltage as a smooth function of a
        normalized SOC parameter ``s ∈ [0, 1]``. PyDMA's ``sto_window_*``
        properties report the stoichiometry endpoints at ``s = 0`` and
        ``s = 1`` — but those endpoints can extrapolate past the input data's
        voltage range. Downstream simulators (e.g. PyBaMM) bounded by physical
        voltage cutoffs only ever traverse the *inner* sto interval where the
        reconstructed cell voltage actually lies in [v_min, v_max], so PyDMA's
        nominal windows over-report electrode utilization and inflate ``c_max``.

        This helper inverts the reconstructed OCV to find the SOC values
        ``s_low`` and ``s_high`` where the fit's V_cell hits ``v_min`` and
        ``v_max``, then derives the voltage-anchored stoichiometry windows
        from those SOC values. The accompanying anchored utilization /
        alpha values are what a PyBaMM consumer should use to size c_max so
        that the simulator delivers the measured nominal capacity over the
        voltage-bounded window without an external ``window_scale`` patch.

        Only the longest stretch on which the reconstruction rises strictly with
        SOC is inverted. A cell OCV is strictly increasing in SOC, so anything
        flat or falling in the reconstruction is an artefact — typically the
        electrode interpolants running into their fill values near the ends —
        and inverting across it would map one voltage to several SOC values.
        ``reconstructed_v_min`` / ``reconstructed_v_max`` are the two ends of
        that stretch, not of the whole array.

        The inversion runs on the reconstruction as fitted, so an active
        ``r_offset_ohm`` is part of the curve being inverted. It is a pure
        level term bounded by ``resistance_offset_limit_ohm_ah`` times the
        C-rate: at the default 0.25 Ohm*Ah and C/20 it reaches 12.5 mV, and it
        moves the anchored SOC endpoints by whatever that is worth on the local
        slope of the curve.

        Parameters
        ----------
        v_min : float, optional
            Lower voltage cutoff. Defaults to the lower end of the invertible
            reconstruction.
        v_max : float, optional
            Upper voltage cutoff. Defaults to its upper end. The default is
            read off the reconstruction rather than the measurement because the
            reconstruction ends inside the measured voltage range on a real
            fit, so the measured cutoffs would ask for extrapolation.
        on_out_of_range : {"raise", "clip"}, default "raise"
            Policy when a requested cutoff lies outside the reconstructed OCV
            range. ``"raise"`` preserves the strict historical behavior and
            refuses to extrapolate. ``"clip"`` anchors to the intersection of
            the requested voltage window and the reconstructed OCV range.

        Returns
        -------
        dict
            Keys:
              - ``v_min``, ``v_max``: voltage cutoffs used for anchoring
              - ``soc_low``, ``soc_high``: normalized SOC values where the
                reconstructed V_cell crosses v_min and v_max
              - ``sto_window_an``: (sto_an at v_min, sto_an at v_max)
              - ``sto_window_ca``: (sto_ca at v_min, sto_ca at v_max)
              - ``utilization_an``, ``utilization_ca``: anchored window widths
              - ``alpha_an``, ``alpha_ca``: 1 / utilization_anchored
              - ``window_shrink_factor``: ``soc_high - soc_low`` (1.0 = no
                shrink; <1 means PyDMA's fit window over-extrapolates the
                voltage-bounded range)

        Raises
        ------
        ValueError
            If reconstructed OCV / SOC arrays are empty, if the strictly
            increasing stretch covers less than half of the reconstruction, or
            if the requested voltage cutoffs lie outside that stretch so
            inversion would extrapolate.
        """
        soc = np.asarray(self.soc_reconstructed, dtype=float)
        v = np.asarray(self.ocv_reconstructed, dtype=float)
        if soc.size == 0 or v.size == 0:
            raise ValueError(
                "voltage_anchored_windows() requires soc_reconstructed and "
                "ocv_reconstructed to be populated. Run the analysis first."
            )

        # Invert only where the reconstruction actually rises with SOC.
        order = np.argsort(soc, kind="stable")
        soc_by_soc = soc[order]
        v_by_soc = v[order]
        start, stop = _longest_increasing_run(v_by_soc)
        n_used = stop - start
        if n_used < 0.5 * int(v_by_soc.size):
            raise ValueError(
                f"The reconstructed OCV rises strictly with SOC on only {n_used} of "
                f"{int(v_by_soc.size)} samples (SOC [{float(soc_by_soc[start]):.4f}, "
                f"{float(soc_by_soc[stop - 1]):.4f}], V [{float(v_by_soc[start]):.4f}, "
                f"{float(v_by_soc[stop - 1]):.4f}] V); anchoring needs at least half "
                "of the curve."
            )

        v_seg = v_by_soc[start:stop]
        soc_seg = soc_by_soc[start:stop]
        v_sorted, unique_idx = np.unique(v_seg, return_index=True)
        soc_sorted = soc_seg[unique_idx]

        v_lo, v_hi = float(v_sorted[0]), float(v_sorted[-1])
        # The defaults anchor to the invertible stretch itself, which is the
        # widest window this method can serve without extrapolating.
        requested_v_min = v_lo if v_min is None else float(v_min)
        requested_v_max = v_hi if v_max is None else float(v_max)
        v_min, v_max = requested_v_min, requested_v_max
        if on_out_of_range not in {"raise", "clip"}:
            raise ValueError(
                "on_out_of_range must be either 'raise' or 'clip', " f"got {on_out_of_range!r}."
            )
        if requested_v_min > requested_v_max:
            raise ValueError(
                f"Requested voltage cutoffs [{requested_v_min:.4f}, "
                f"{requested_v_max:.4f}] V are not ordered."
            )

        clipped_low = requested_v_min < v_lo - 1e-9
        clipped_high = requested_v_max > v_hi + 1e-9
        if clipped_low or clipped_high:
            if on_out_of_range == "raise":
                raise ValueError(
                    f"Requested voltage cutoffs [{requested_v_min:.4f}, "
                    f"{requested_v_max:.4f}] V lie outside the reconstructed "
                    f"OCV range [{v_lo:.4f}, {v_hi:.4f}] V; anchoring would "
                    "require extrapolation."
                )
            v_min = max(requested_v_min, v_lo)
            v_max = min(requested_v_max, v_hi)
        if v_min > v_max:
            raise ValueError(
                f"Requested voltage cutoffs [{requested_v_min:.4f}, "
                f"{requested_v_max:.4f}] V do not overlap the reconstructed "
                f"OCV range [{v_lo:.4f}, {v_hi:.4f}] V."
            )

        soc_low = float(np.interp(v_min, v_sorted, soc_sorted))
        soc_high = float(np.interp(v_max, v_sorted, soc_sorted))

        p = self.fitted_params
        sto_an_low = p.sto_init_an + soc_low * p.utilization_an
        sto_an_high = p.sto_init_an + soc_high * p.utilization_an
        sto_ca_low = p.sto_init_ca - soc_low * p.utilization_ca
        sto_ca_high = p.sto_init_ca - soc_high * p.utilization_ca

        util_an = sto_an_high - sto_an_low
        util_ca = sto_ca_low - sto_ca_high

        return {
            "v_min": v_min,
            "v_max": v_max,
            "requested_v_min": requested_v_min,
            "requested_v_max": requested_v_max,
            "reconstructed_v_min": v_lo,
            "reconstructed_v_max": v_hi,
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "on_out_of_range": on_out_of_range,
            "soc_low": soc_low,
            "soc_high": soc_high,
            "sto_window_an": (sto_an_low, sto_an_high),
            "sto_window_ca": (sto_ca_low, sto_ca_high),
            "utilization_an": util_an,
            "utilization_ca": util_ca,
            "alpha_an": 1.0 / util_an if util_an != 0 else float("inf"),
            "alpha_ca": 1.0 / util_ca if util_ca != 0 else float("inf"),
            "window_shrink_factor": soc_high - soc_low,
        }

    @staticmethod
    def _interp_voltage_on_sto(
        electrode: Any,
        sto: np.ndarray,
        *,
        gamma: float | None = None,
        allow_extrapolation: bool = False,
        label: str = "electrode",
        cathode_convention: str = "lithiation",
    ) -> np.ndarray:
        """Evaluate an electrode/tuple/callable voltage model on stoichiometry.

        ``cathode_convention`` says which axis the supplied OCP is tabulated on.
        ``"lithiation"`` reads it at ``sto`` (PyDMA's own convention, in which
        ``sto_window_ca`` is a lithiation fraction x). ``"delithiation"`` reads
        it at ``1 - sto``, for a cathode OCP tabulated against the delithiated
        fraction. The setting is meaningless for an anode, which is why it
        defaults to the identity.
        """
        if cathode_convention not in {"lithiation", "delithiation"}:
            raise ValueError(
                "cathode_convention must be either 'lithiation' or 'delithiation', "
                f"got {cathode_convention!r}."
            )
        sto = np.asarray(sto, dtype=float)
        if cathode_convention == "delithiation":
            sto = 1.0 - sto

        if callable(electrode):
            return np.asarray(electrode(sto), dtype=float)

        if hasattr(electrode, "get_blend_curve"):
            if gamma is None:
                raise ValueError(f"{label} is a blend electrode; gamma is required.")
            x, y = electrode.get_blend_curve(float(gamma))
        elif isinstance(electrode, (tuple, list)) and len(electrode) == 2:
            x, y = electrode
        elif hasattr(electrode, "soc") and hasattr(electrode, "voltage"):
            x, y = electrode.soc, electrode.voltage
        else:
            raise TypeError(
                f"{label} must be a callable, (sto, voltage) tuple, ElectrodeOCP, "
                "or BlendElectrode-like object."
            )

        x = np.asarray(x, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        if x.size < 2 or y.size < 2 or x.shape != y.shape:
            raise ValueError(f"{label} OCP arrays must be non-empty and same-shaped.")

        order = np.argsort(x, kind="stable")
        x_sorted = x[order]
        y_sorted = y[order]
        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]

        if not allow_extrapolation:
            x_min, x_max = float(x_unique[0]), float(x_unique[-1])
            if np.nanmin(sto) < x_min - 1e-9 or np.nanmax(sto) > x_max + 1e-9:
                raise ValueError(
                    f"{label} stoichiometry query [{np.nanmin(sto):.6g}, "
                    f"{np.nanmax(sto):.6g}] lies outside OCP domain "
                    f"[{x_min:.6g}, {x_max:.6g}]."
                )
            return np.asarray(np.interp(sto, x_unique, y_unique))

        yq = np.asarray(np.interp(sto, x_unique, y_unique))
        left = sto < x_unique[0]
        right = sto > x_unique[-1]
        if np.any(left):
            slope = (y_unique[1] - y_unique[0]) / (x_unique[1] - x_unique[0])
            yq[left] = y_unique[0] + slope * (sto[left] - x_unique[0])
        if np.any(right):
            slope = (y_unique[-1] - y_unique[-2]) / (x_unique[-1] - x_unique[-2])
            yq[right] = y_unique[-1] + slope * (sto[right] - x_unique[-1])
        return yq

    def voltage_anchored_windows_from_ocp_model(
        self,
        anode_ocp: Any,
        cathode_ocp: Any,
        v_min: float | None = None,
        v_max: float | None = None,
        on_out_of_range: str = "raise",
        gamma_an: float | None = None,
        gamma_ca: float | None = None,
        allow_extrapolation: bool = False,
        n_points: int = 4001,
        cathode_convention: str = "lithiation",
    ) -> dict[str, Any]:
        """
        Voltage-anchor the export window using a supplied OCP voltage model.

        This is the PyBaMM-consistent post-processing path: the raw DMA fit is
        kept unchanged, but the exported endpoint is moved along the fitted
        lithium trajectory until the supplied OCP model satisfies the requested
        full-cell voltage cutoffs.

        The fitted trajectory is exactly:

            ``sto_an(s) = sto_an_raw_0 + s * utilization_an_raw``
            ``sto_ca(s) = sto_ca_raw_0 - s * utilization_ca_raw``

        for ``s in [0, 1]``. The helper evaluates
        ``U_cell_export(s) = U_cathode(sto_ca(s)) - U_anode(sto_an(s))`` and
        inverts that curve for ``v_min`` / ``v_max``. It returns a dictionary
        with the same core keys as :meth:`voltage_anchored_windows`, plus
        metadata showing that this is a post-processed export view.

        Parameters
        ----------
        anode_ocp, cathode_ocp
            Voltage models used for export. Each can be a callable, a
            ``(sto, voltage)`` tuple, an ``ElectrodeOCP``, or a
            ``BlendElectrode``-like object. Blend electrodes require the
            corresponding ``gamma_*`` value.
        v_min, v_max : float, optional
            Requested lower/upper full-cell voltage cutoffs. Defaults to the
            min/max measured voltage.
        on_out_of_range : {"raise", "clip"}, default "raise"
            Policy when requested cutoffs are outside the OCP-model voltage
            range over the fitted ``s in [0, 1]`` trajectory.
        gamma_an, gamma_ca : float, optional
            Blend2 fractions for blend anode/cathode OCP models. Defaults to
            the fitted ``gamma_blend2_*`` values.
        allow_extrapolation : bool, default False
            Whether to linearly extrapolate OCPs outside their stoichiometry
            data range. Keep this False for PyBaMM export unless you
            intentionally want extrapolation.
        n_points : int, default 4001
            Number of samples used to build the export voltage curve.
        cathode_convention : {"lithiation", "delithiation"}, default "lithiation"
            Axis the supplied ``cathode_ocp`` is tabulated on. PyDMA's fitted
            ``sto_ca`` is a lithiation fraction, so the default reads the model
            at ``sto_ca``; ``"delithiation"`` reads it at ``1 - sto_ca``.

        Raises
        ------
        ValueError
            If the anchored window has a non-positive width on either
            electrode. On a cathode OCP that is tabulated the other way round,
            the exported window comes out reversed, so this is where a wrong
            ``cathode_convention`` surfaces.
        """
        if on_out_of_range not in {"raise", "clip"}:
            raise ValueError(
                "on_out_of_range must be either 'raise' or 'clip', " f"got {on_out_of_range!r}."
            )
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        meas_v = np.asarray(self.measured_voltage, dtype=float)
        if v_min is None:
            if meas_v.size == 0:
                raise ValueError("v_min not provided and measured_voltage is empty.")
            v_min = float(np.min(meas_v))
        if v_max is None:
            if meas_v.size == 0:
                raise ValueError("v_max not provided and measured_voltage is empty.")
            v_max = float(np.max(meas_v))

        requested_v_min = float(v_min)
        requested_v_max = float(v_max)
        if requested_v_min > requested_v_max:
            raise ValueError(
                f"Requested voltage cutoffs [{requested_v_min:.4f}, "
                f"{requested_v_max:.4f}] V are not ordered."
            )

        p = self.fitted_params
        if gamma_an is None:
            gamma_an = p.gamma_blend2_an
        if gamma_ca is None:
            gamma_ca = p.gamma_blend2_ca

        s = np.linspace(0.0, 1.0, int(n_points))
        sto_an = p.sto_init_an + s * p.utilization_an
        sto_ca = p.sto_init_ca - s * p.utilization_ca
        u_an = self._interp_voltage_on_sto(
            anode_ocp,
            sto_an,
            gamma=gamma_an,
            allow_extrapolation=allow_extrapolation,
            label="anode_ocp",
        )
        u_ca = self._interp_voltage_on_sto(
            cathode_ocp,
            sto_ca,
            gamma=gamma_ca,
            allow_extrapolation=allow_extrapolation,
            label="cathode_ocp",
            cathode_convention=cathode_convention,
        )
        v_model = u_ca - u_an

        order = np.argsort(v_model)
        v_sorted = v_model[order]
        s_sorted = s[order]
        v_unique, unique_idx = np.unique(v_sorted, return_index=True)
        s_unique = s_sorted[unique_idx]

        v_lo, v_hi = float(v_unique[0]), float(v_unique[-1])
        clipped_low = requested_v_min < v_lo - 1e-9
        clipped_high = requested_v_max > v_hi + 1e-9
        if clipped_low or clipped_high:
            if on_out_of_range == "raise":
                raise ValueError(
                    f"Requested voltage cutoffs [{requested_v_min:.4f}, "
                    f"{requested_v_max:.4f}] V lie outside the OCP-model "
                    f"range [{v_lo:.4f}, {v_hi:.4f}] V over fitted s=[0, 1]."
                )
            v_min = max(requested_v_min, v_lo)
            v_max = min(requested_v_max, v_hi)
        if v_min > v_max:
            raise ValueError(
                f"Requested voltage cutoffs [{requested_v_min:.4f}, "
                f"{requested_v_max:.4f}] V do not overlap the OCP-model "
                f"range [{v_lo:.4f}, {v_hi:.4f}] V."
            )

        soc_low = float(np.interp(v_min, v_unique, s_unique))
        soc_high = float(np.interp(v_max, v_unique, s_unique))

        sto_an_low = p.sto_init_an + soc_low * p.utilization_an
        sto_an_high = p.sto_init_an + soc_high * p.utilization_an
        sto_ca_low = p.sto_init_ca - soc_low * p.utilization_ca
        sto_ca_high = p.sto_init_ca - soc_high * p.utilization_ca

        util_an = sto_an_high - sto_an_low
        util_ca = sto_ca_low - sto_ca_high

        if util_an <= 0 or util_ca <= 0:
            raise ValueError(
                f"Voltage anchoring produced a window of non-positive width "
                f"(anode {util_an:.6g}, cathode {util_ca:.6g}) between "
                f"{v_min:.4f} V and {v_max:.4f} V. The cell voltage of the supplied "
                f"OCP models does not rise along the fitted trajectory. With "
                f"cathode_convention={cathode_convention!r} in use, a cathode OCP "
                "tabulated on the other axis is the first thing to check."
            )

        out: dict[str, Any] = {
            "v_min": v_min,
            "v_max": v_max,
            "requested_v_min": requested_v_min,
            "requested_v_max": requested_v_max,
            "reconstructed_v_min": v_lo,
            "reconstructed_v_max": v_hi,
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "on_out_of_range": on_out_of_range,
            "soc_low": soc_low,
            "soc_high": soc_high,
            "sto_window_an": (sto_an_low, sto_an_high),
            "sto_window_ca": (sto_ca_low, sto_ca_high),
            "utilization_an": util_an,
            "utilization_ca": util_ca,
            "alpha_an": 1.0 / util_an if util_an != 0 else float("inf"),
            "alpha_ca": 1.0 / util_ca if util_ca != 0 else float("inf"),
            "window_shrink_factor": soc_high - soc_low,
            "voltage_model": "ocp_model",
            "post_processed_only": True,
            "fit_is_unchanged": True,
            "allow_extrapolation": bool(allow_extrapolation),
            "cathode_convention": cathode_convention,
            "raw_fitted_params": p.to_dict(),
            "low_voltage_endpoint": {
                "voltage": v_min,
                "cell_soc": 0.0,
                "fit_soc": soc_low,
                "sto_an": sto_an_low,
                "sto_ca": sto_ca_low,
            },
            "high_voltage_endpoint": {
                "voltage": v_max,
                "cell_soc": 1.0,
                "fit_soc": soc_high,
                "sto_an": sto_an_high,
                "sto_ca": sto_ca_high,
            },
        }

        # A blend electrode without gamma cannot reach this point: the evaluation
        # above raises on it. Checked rather than asserted so the invariant
        # survives python -O.
        if hasattr(anode_ocp, "get_component_stoichiometry_window"):
            if gamma_an is None:
                raise RuntimeError("Blend anode reached the phase window without a gamma.")
            out["anode_phase_window"] = anode_ocp.get_component_stoichiometry_window(
                float(gamma_an), out["sto_window_an"]
            )
        if hasattr(cathode_ocp, "get_component_stoichiometry_window"):
            if gamma_ca is None:
                raise RuntimeError("Blend cathode reached the phase window without a gamma.")
            out["cathode_phase_window"] = cathode_ocp.get_component_stoichiometry_window(
                float(gamma_ca), out["sto_window_ca"]
            )

        return out

    def voltage_anchored_blend_phase_export_from_ocp_model(
        self,
        anode_blend: Any,
        cathode_ocp: Any,
        c_max_blend: float | None = None,
        eps_total: float | None = None,
        eps_blend1: float | None = None,
        eps_blend2: float | None = None,
        v_min: float | None = None,
        v_max: float | None = None,
        on_out_of_range: str = "raise",
        gamma_an: float | None = None,
        allow_extrapolation: bool = False,
        n_points: int = 4001,
        cathode_convention: str = "lithiation",
    ) -> dict[str, Any]:
        """
        Export a voltage-anchored blend window as per-phase PyBaMM values.

        This is a post-processing helper for downstream simulators. It keeps the
        raw DMA fit unchanged, voltage-anchors the fitted blend window with
        :meth:`voltage_anchored_windows_from_ocp_model`, maps that window to the
        blend components, and computes phase ``c_max`` / ``c_init`` values that
        include PyDMA's raw blend-capacity normalization.

        PyDMA's ``BlendElectrode`` first computes

            ``q_raw = (1 - gamma) * q_blend1 + gamma * q_blend2``

        on the common voltage window, then normalizes it to the fitted blend SOC
        coordinate. Therefore per-phase capacities must be divided by
        ``raw_blend_max - raw_blend_min``. Omitting that factor underestimates
        phase capacity whenever the component OCPs do not each span their full
        ``[0, 1]`` stoichiometry range on the common voltage window.

        Parameters
        ----------
        anode_blend
            Blend electrode used for the fitted anode.
        cathode_ocp
            Cathode OCP model used for voltage anchoring.
        c_max_blend : float, optional
            Voltage-anchored blended-anode maximum concentration. If omitted,
            the helper returns only normalized windows and conversion factors.
        eps_total : float, optional
            Total negative-electrode active-material volume fraction.
        eps_blend1, eps_blend2 : float, optional
            Component active-material volume fractions. If all three volume
            fractions are supplied, per-phase ``c_max / c_max_blend`` factors
            are returned even when ``c_max_blend`` itself is omitted.
        v_min, v_max, on_out_of_range, gamma_an, allow_extrapolation, n_points,
        cathode_convention
            Passed to :meth:`voltage_anchored_windows_from_ocp_model`.

        Returns
        -------
        dict
            Corrected per-phase stoichiometry windows, ``c_max`` values,
            initial concentrations, raw blend normalization metadata, and a
            capacity consistency check. The raw fit is not modified.

            ``raw_capacity_share_blend1`` and ``raw_capacity_share_blend2`` are
            the two components' contributions to the blend's raw capacity swing:
            they sum to ``raw_blend_capacity_delta`` in raw blend units, not to
            1. Divide by that delta for fractions.

            ``phase_capacity_ratio_to_target`` is a numerical round-trip check:
            it recombines the per-phase ``c_max`` values back into the blended
            capacity and reports the ratio to the blended target, so it detects
            an arithmetic slip in this conversion. It is 1 by construction
            whenever the conversion is consistent and says nothing about whether
            the underlying volume fractions describe the real cell.
        """
        if not hasattr(anode_blend, "get_component_stoichiometry_window"):
            raise TypeError("anode_blend must provide get_component_stoichiometry_window.")

        if c_max_blend is not None:
            c_max_blend = float(c_max_blend)
            if c_max_blend <= 0:
                raise ValueError(f"c_max_blend must be positive, got {c_max_blend}.")

        # One gate for the three volume fractions: they are only usable together,
        # and every block below reads them from this single tuple.
        eps: tuple[float, float, float] | None
        if eps_total is None or eps_blend1 is None or eps_blend2 is None:
            if any(v is not None for v in (eps_total, eps_blend1, eps_blend2)):
                raise ValueError("eps_total, eps_blend1, and eps_blend2 must be supplied together.")
            eps = None
        else:
            eps = (float(eps_total), float(eps_blend1), float(eps_blend2))
            if min(eps) <= 0:
                raise ValueError("eps_total, eps_blend1, and eps_blend2 must all be positive.")

        if gamma_an is None:
            gamma_an = self.fitted_params.gamma_blend2_an
        if gamma_an is None:
            raise ValueError(
                "gamma_an was not provided and fitted_params.gamma_blend2_an " "is not set."
            )
        gamma_an = float(gamma_an)
        if gamma_an < 0 or gamma_an > 1:
            raise ValueError(f"gamma_an must be in [0, 1], got {gamma_an}.")

        export = self.voltage_anchored_windows_from_ocp_model(
            anode_ocp=anode_blend,
            cathode_ocp=cathode_ocp,
            v_min=v_min,
            v_max=v_max,
            on_out_of_range=on_out_of_range,
            gamma_an=gamma_an,
            allow_extrapolation=allow_extrapolation,
            n_points=n_points,
            cathode_convention=cathode_convention,
        )
        if "anode_phase_window" not in export:
            raise ValueError("Voltage export did not produce an anode phase window.")
        window = export["anode_phase_window"]

        raw_blend_min = float(window["raw_blend_min"])
        raw_blend_max = float(window["raw_blend_max"])
        raw_blend_range = raw_blend_max - raw_blend_min
        if raw_blend_range <= 0:
            raise ValueError(f"Blend raw capacity range must be positive, got {raw_blend_range}.")

        sto1_low = float(window["blend1_sto_0soc"])
        sto1_high = float(window["blend1_sto_100soc"])
        sto2_low = float(window["blend2_sto_0soc"])
        sto2_high = float(window["blend2_sto_100soc"])

        util1 = sto1_high - sto1_low
        util2 = sto2_high - sto2_low
        raw_delta = float(window["raw_blend_capacity_100soc"]) - float(
            window["raw_blend_capacity_0soc"]
        )
        blend_util = export["sto_window_an"][1] - export["sto_window_an"][0]

        out: dict[str, Any] = {
            "voltage_anchor": export,
            "phase_window": window,
            "post_processed_only": True,
            "fit_is_unchanged": True,
            "gamma_blend2": gamma_an,
            "c_max_blend": c_max_blend,
            "eps_total": eps[0] if eps is not None else None,
            "eps_blend1": eps[1] if eps is not None else None,
            "eps_blend2": eps[2] if eps is not None else None,
            "raw_blend_min": raw_blend_min,
            "raw_blend_max": raw_blend_max,
            "raw_blend_range": raw_blend_range,
            "raw_blend_range_correction_factor": 1.0 / raw_blend_range,
            "raw_blend_capacity_delta": raw_delta,
            "blend_utilization": blend_util,
            "blend_utilization_from_raw_range": raw_delta / raw_blend_range,
            "raw_capacity_share_blend1": (1.0 - gamma_an) * util1,
            "raw_capacity_share_blend2": gamma_an * util2,
            "sto_blend1_0soc": sto1_low,
            "sto_blend1_100soc": sto1_high,
            "sto_blend2_0soc": sto2_low,
            "sto_blend2_100soc": sto2_high,
            "utilization_blend1": util1,
            "utilization_blend2": util2,
            "voltage_anode_0soc": float(window["voltage_0soc"]),
            "voltage_anode_100soc": float(window["voltage_100soc"]),
        }

        if eps is not None:
            eps_all, eps_1, eps_2 = eps
            c_max_blend1_per_blend = (1.0 - gamma_an) * eps_all / (eps_1 * raw_blend_range)
            c_max_blend2_per_blend = gamma_an * eps_all / (eps_2 * raw_blend_range)
            out.update(
                {
                    "c_max_blend1_per_c_max_blend": c_max_blend1_per_blend,
                    "c_max_blend2_per_c_max_blend": c_max_blend2_per_blend,
                }
            )

            if c_max_blend is not None:
                c_max_blend1 = c_max_blend * c_max_blend1_per_blend
                c_max_blend2 = c_max_blend * c_max_blend2_per_blend
                phase_capacity_delta = eps_1 * c_max_blend1 * util1 + eps_2 * c_max_blend2 * util2
                target_capacity_delta = eps_all * c_max_blend * blend_util
                capacity_ratio = (
                    phase_capacity_delta / target_capacity_delta
                    if target_capacity_delta != 0
                    else float("nan")
                )
                out.update(
                    {
                        "c_max_blend1": c_max_blend1,
                        "c_max_blend2": c_max_blend2,
                        "c_init_blend1_0soc": sto1_low * c_max_blend1,
                        "c_init_blend1_100soc": sto1_high * c_max_blend1,
                        "c_init_blend2_0soc": sto2_low * c_max_blend2,
                        "c_init_blend2_100soc": sto2_high * c_max_blend2,
                        "phase_capacity_delta_mol_m3_electrode": phase_capacity_delta,
                        "target_capacity_delta_mol_m3_electrode": target_capacity_delta,
                        "phase_capacity_ratio_to_target": capacity_ratio,
                    }
                )

        return out

    # Backwards-compatibility aliases for code expecting older attribute names
    @property
    def params(self) -> FittedParams:
        """Alias for `fitted_params` (legacy name `params`)."""
        return self.fitted_params

    @property
    def degradation(self) -> DegradationModes:
        """Alias for `degradation_modes` (legacy name `degradation`)."""
        return self.degradation_modes


@dataclass
class ReferenceData:
    """
    Reference data from the first CU for calculating relative degradation.

    Attributes
    ----------
    capa_anode_init : float
        Initial anode capacity.
    capa_cathode_init : float
        Initial cathode capacity.
    capa_inventory_init : float
        Initial charge carrier inventory.
    gamma_an_blend2_init : float
        Initial anode blend2 fraction.
    gamma_ca_blend2_init : float
        Initial cathode blend2 fraction.
    reference_capacity : float
        Reference (initial) cell capacity in Ah.
    """

    capa_anode_init: float
    capa_cathode_init: float
    capa_inventory_init: float
    gamma_an_blend2_init: float = 0.0
    gamma_ca_blend2_init: float = 0.0
    reference_capacity: float = 0.0

    @classmethod
    def from_result(cls, result: DMAResult) -> "ReferenceData":
        """
        Create reference data from a DMA result (typically first CU).

        Parameters
        ----------
        result : DMAResult
            Result from the reference CU.

        Returns
        -------
        ReferenceData
            Reference data extracted from the result.
        """
        params = result.params
        capa = result.capacity

        capa_anode = params.alpha_an * capa
        capa_cathode = params.alpha_ca * capa
        capa_inventory = (params.alpha_ca + params.beta_ca - params.beta_an) * capa

        return cls(
            capa_anode_init=capa_anode,
            capa_cathode_init=capa_cathode,
            capa_inventory_init=capa_inventory,
            gamma_an_blend2_init=float(params.gamma_blend2_an or 0.0),
            gamma_ca_blend2_init=float(params.gamma_blend2_ca or 0.0),
            reference_capacity=capa,
        )


@dataclass
class AgingStudyResults:
    """
    Results for a complete aging study with multiple check-ups.

    Attributes
    ----------
    results : Dict[str, DMAResult]
        Dictionary of results keyed by CU name.
    reference_data : ReferenceData
        Reference data from first CU.
    cu_labels : List[str]
        Ordered list of CU names.
    efc_values : List[float | None]
        Equivalent full cycles or CU numbers for each check-up, one entry per
        entry in ``cu_labels``. ``None`` where a check-up was added without one.
    """

    results: dict[str, DMAResult] = field(default_factory=dict)
    reference_data: ReferenceData | None = None
    cu_labels: list[str] = field(default_factory=list)
    efc_values: list[float | None] = field(default_factory=list)

    def __getitem__(self, key: str) -> DMAResult:
        """Access result by CU name."""
        return self.results[key]

    def __len__(self) -> int:
        """Number of check-ups."""
        return len(self.results)

    def __iter__(self):
        """Iterate over CU names in order."""
        return iter(self.cu_labels)

    @property
    def cycle_numbers(self) -> list[float | None]:
        """Alias for efc_values (backwards compatibility)."""
        return self.efc_values

    def add_result(self, result: DMAResult, efc: float | None = None):
        """
        Add a result to the study.

        ``efc_values`` keeps one entry per CU, ``None`` included. Appending only
        the known values would silently misalign every later EFC with the wrong
        check-up as soon as one CU comes without one.

        Re-adding a CU replaces its result. Its EFC survives that unless a new
        one is passed, so refitting a check-up does not cost the study its
        x-axis.

        Parameters
        ----------
        result : DMAResult
            Result to add.
        efc : float, optional
            EFC value for this CU.
        """
        self.results[result.cu_name] = result
        if result.cu_name in self.cu_labels:
            if efc is not None:
                self.efc_values[self.cu_labels.index(result.cu_name)] = efc
        else:
            self.cu_labels.append(result.cu_name)
            self.efc_values.append(efc)

        # Set reference data from first CU
        if self.reference_data is None:
            self.reference_data = ReferenceData.from_result(result)

    def plot_degradation_modes(self, **kwargs):
        """Plot degradation modes evolution over aging.

        Convenience method that delegates to :func:`pydma.plot_aging_study`.

        Parameters
        ----------
        **kwargs
            Passed through to :func:`~pydma.visualization.plots.plot_aging_study`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        from pydma.visualization.plots import plot_aging_study

        return plot_aging_study(self, **kwargs)

    def get_degradation_dataframe(self) -> pd.DataFrame:
        """
        Get degradation modes as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with degradation modes for each CU.
        """
        data = []
        for cu_name in self.cu_labels:
            result = self.results[cu_name]
            row = {
                "CU": cu_name,
                "LAM_anode": result.degradation.lam_anode,
                "LAM_cathode": result.degradation.lam_cathode,
                "LLI": result.degradation.lli,
                "Capacity_Loss": result.degradation.capacity_loss,
                "LAM_anode_blend1": result.degradation.lam_anode_blend1,
                "LAM_anode_blend2": result.degradation.lam_anode_blend2,
                "LAM_cathode_blend1": result.degradation.lam_cathode_blend1,
                "LAM_cathode_blend2": result.degradation.lam_cathode_blend2,
                "Capacity": result.capacity,
                "RMSE_fit": result.rmse_fit_region,
                "is_accepted": result.is_accepted,
            }
            data.append(row)

        df = pd.DataFrame(data)
        efc_column = self._efc_column(len(df))
        if efc_column is not None:
            df.insert(1, "EFC", efc_column)
        return df

    def _efc_column(self, n_rows: int) -> list[float] | None:
        """EFC values as a DataFrame column, or None when the study has none.

        Check-ups added without an EFC hold a None placeholder; those become NaN
        so the column stays aligned with the rows.
        """
        if len(self.efc_values) != n_rows:
            return None
        if not any(value is not None for value in self.efc_values):
            return None
        return [float("nan") if value is None else float(value) for value in self.efc_values]

    def get_params_dataframe(self) -> pd.DataFrame:
        """
        Get fitted parameters as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with parameters for each CU.
        """
        data = []
        for cu_name in self.cu_labels:
            result = self.results[cu_name]
            row = {"CU": cu_name, **result.params.to_dict()}
            data.append(row)

        df = pd.DataFrame(data)
        efc_column = self._efc_column(len(df))
        if efc_column is not None:
            df.insert(1, "EFC", efc_column)
        return df

    def to_pickle(self, filepath: "str | os.PathLike[str]"):
        """
        Save results to pickle file.

        Parameters
        ----------
        filepath : str or path-like
            Path to save file. Missing parent directories are created.
        """
        import pickle

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filepath: "str | os.PathLike[str]") -> "AgingStudyResults":
        """
        Load results from pickle file.

        Unpickling runs code embedded in the file, so only load pickles from a
        source you trust — typically one this class wrote itself.

        Parameters
        ----------
        filepath : str or path-like
            Path to pickle file.

        Returns
        -------
        AgingStudyResults
            Loaded results.
        """
        import pickle

        with open(filepath, "rb") as f:
            loaded: "AgingStudyResults" = pickle.load(f)
        return loaded

    def to_csv(self, filepath: "str | os.PathLike[str]"):
        """
        Save degradation results to CSV file.

        Parameters
        ----------
        filepath : str or path-like
            Path to save file.
        """
        df = self.get_degradation_dataframe()
        df.to_csv(filepath, index=False)

    def summary(self) -> str:
        """
        Get a summary string of the results.

        Returns
        -------
        str
            Summary of the aging study results.
        """
        lines = [
            f"Aging Study Results: {len(self)} check-ups",
            "",
            "Degradation Summary:",
        ]

        if self.results:
            first_cu = self.cu_labels[0]
            last_cu = self.cu_labels[-1]

            first_result = self.results[first_cu]
            last_result = self.results[last_cu]

            lines.extend(
                [
                    f"  {first_cu}: LAM_an={first_result.degradation.lam_anode:.2%}, "
                    f"LAM_ca={first_result.degradation.lam_cathode:.2%}, "
                    f"LLI={first_result.degradation.lli:.2%}",
                    f"  {last_cu}: LAM_an={last_result.degradation.lam_anode:.2%}, "
                    f"LAM_ca={last_result.degradation.lam_cathode:.2%}, "
                    f"LLI={last_result.degradation.lli:.2%}",
                ]
            )

            # Count accepted solutions
            n_accepted = sum(1 for r in self.results.values() if r.is_accepted)
            lines.append(f"\nAccepted solutions: {n_accepted}/{len(self)}")

        return "\n".join(lines)
