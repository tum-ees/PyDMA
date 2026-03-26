"""
Result dataclasses for DMA analysis.

This module defines dataclasses for storing DMA analysis results,
including single-CU results and multi-CU aging study results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


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

    def to_dict(self) -> Dict[str, float]:
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
    """

    alpha_an: float
    beta_an: float
    alpha_ca: float
    beta_ca: float
    gamma_blend2_an: float = 0.0
    gamma_blend2_ca: float = 0.0
    inhom_an: float = 0.0
    inhom_ca: float = 0.0

    # ============================================================
    # Derived stoichiometry properties for PyBaMM users
    # ============================================================

    @property
    def utilization_an(self) -> float:
        """
        Fraction of anode electrode capacity used by the cell (1/alpha_an).

        A value of 0.95 means 95% of the anode's theoretical capacity
        is utilized in the cell's operating window.
        """
        return 1.0 / self.alpha_an

    @property
    def utilization_ca(self) -> float:
        """
        Fraction of cathode electrode capacity used by the cell (1/alpha_ca).

        A value of 0.90 means 90% of the cathode's theoretical capacity
        is utilized in the cell's operating window.
        """
        return 1.0 / self.alpha_ca

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
        # Convert from delithiation curve position to lithiation fraction
        delith_position = self.beta_ca / self.alpha_ca
        return 1.0 + delith_position

    @property
    def sto_window_an(self) -> Tuple[float, float]:
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
    def sto_window_ca(self) -> Tuple[float, float]:
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

    @classmethod
    def from_array(cls, params: np.ndarray) -> "FittedParams":
        """
        Create from 8-element parameter array.

        Parameters
        ----------
        params : np.ndarray
            8-element array in standard parameter order.

        Returns
        -------
        FittedParams
            Instance populated from array.
        """
        params = np.asarray(params).flatten()
        if len(params) < 4:
            raise ValueError(f"params must have at least 4 elements, got {len(params)}")

        # Pad to 8 elements if needed
        full_params = np.zeros(8)
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
        )

    def to_array(self) -> np.ndarray:
        """
        Convert to 8-element parameter array.

        Returns
        -------
        np.ndarray
            8-element array in standard parameter order.
        """
        return np.array(
            [
                self.alpha_an,
                self.beta_an,
                self.alpha_ca,
                self.beta_ca,
                self.gamma_blend2_an,
                self.gamma_blend2_ca,
                self.inhom_an,
                self.inhom_ca,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
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
    """

    cu_name: str = ""
    fitted_params: FittedParams = field(default_factory=lambda: FittedParams(1.0, 0.0, 1.0, 0.0))
    degradation_modes: DegradationModes = field(
        default_factory=lambda: DegradationModes(0.0, 0.0, 0.0)
    )

    # Optional reference data (from first CU)
    reference_data: Optional["ReferenceData"] = None

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary containing all result data.
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
        }

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
            gamma_an_blend2_init=params.gamma_blend2_an,
            gamma_ca_blend2_init=params.gamma_blend2_ca,
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
    efc_values : List[float]
        Equivalent full cycles or CU numbers for each check-up.
    is_cyclic : bool
        Whether this is a cyclic (True) or calendar (False) aging study.
    fit_reverse : bool
        Whether fitting was performed in reverse order.
    """

    results: Dict[str, DMAResult] = field(default_factory=dict)
    reference_data: Optional[ReferenceData] = None
    cu_labels: List[str] = field(default_factory=list)
    efc_values: List[float] = field(default_factory=list)
    is_cyclic: bool = True
    fit_reverse: bool = False

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
    def cycle_numbers(self) -> List[float]:
        """Alias for efc_values (backwards compatibility)."""
        return self.efc_values

    def add_result(self, result: DMAResult, efc: Optional[float] = None):
        """
        Add a result to the study.

        Parameters
        ----------
        result : DMAResult
            Result to add.
        efc : float, optional
            EFC value for this CU.
        """
        self.results[result.cu_name] = result
        if result.cu_name not in self.cu_labels:
            self.cu_labels.append(result.cu_name)
        if efc is not None:
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
        if self.efc_values and len(self.efc_values) == len(df):
            df.insert(1, "EFC", self.efc_values)
        return df

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
        if self.efc_values and len(self.efc_values) == len(df):
            df.insert(1, "EFC", self.efc_values)
        return df

    def to_pickle(self, filepath: str):
        """
        Save results to pickle file.

        Parameters
        ----------
        filepath : str
            Path to save file.
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filepath: str) -> "AgingStudyResults":
        """
        Load results from pickle file.

        Parameters
        ----------
        filepath : str
            Path to pickle file.

        Returns
        -------
        AgingStudyResults
            Loaded results.
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def to_csv(self, filepath: str):
        """
        Save degradation results to CSV file.

        Parameters
        ----------
        filepath : str
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
            f"Study type: {'Cyclic' if self.is_cyclic else 'Calendar'}",
            f"Fit direction: {'Reverse' if self.fit_reverse else 'Forward'}",
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
