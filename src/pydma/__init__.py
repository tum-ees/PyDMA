"""
PyDMA - Battery Degradation Mode Analysis

Python implementation of TUM-EES DegradationModeAnalysis framework for
analyzing battery degradation through OCV, DVA, and ICA fitting.

This package quantifies three degradation mechanisms:
- LLI: Loss of Lithium Inventory (charge carrier loss)
- LAM_an: Loss of Active Material at Anode
- LAM_ca: Loss of Active Material at Cathode

Example
-------
>>> import pydma
>>> from pydma import DMAAnalyzer, DMAConfig
>>>
>>> # Load electrode data
>>> anode_ocp = pydma.load_ocp("anode.csv")
>>> cathode_ocp = pydma.load_ocp("cathode.csv")
>>>
>>> # Configure and run analysis
>>> config = DMAConfig(direction="charge")
>>> analyzer = DMAAnalyzer(config)
>>> analyzer.set_anode(anode_ocp)
>>> analyzer.set_cathode(cathode_ocp)
>>> result = analyzer.analyze(measured_capacity, measured_voltage)
"""

from pydma._version import __version__

__author__ = "TUM-EES"
__email__ = "mathias.rehm@tum.de"

from pydma.analysis.degradation import calculate_degradation_modes

# Analysis functions
from pydma.analysis.dva import calculate_dva
from pydma.analysis.ica import calculate_ica

# Core classes
from pydma.core.analyzer import DMAAnalyzer
from pydma.core.optimizer import DMAOptimizer
from pydma.electrodes.blend import BlendElectrode

# Electrode classes
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.electrodes.inhomogeneity import calculate_inhomogeneity

# Data loading
from pydma.preprocessing.loader import auto_detect_columns, load_aging_study, load_ocp, load_pocv

# Preprocessing
from pydma.preprocessing.smoother import apply_filter, smooth_lowess, smooth_savgol

# Silicon curve generation
from pydma.silicon.generator import SiliconCurveParams, generate_si_curve

# Electrode balancing export (PyDMA fit + cell geometry -> c_max / c_init).
# Simulator-agnostic; ElectrodeBalancing.pybamm_overrides() is the only
# PyBaMM-flavoured convenience.
from pydma.utils.balancing import (
    CellGeometry,
    ElectrodeBalancing,
    derive_balancing,
    derive_balancing_from_result,
)
from pydma.utils.dma_config import DMAConfig
from pydma.utils.results import (
    AgingStudyResults,
    DegradationModes,
    DMAResult,
    FittedParams,
    ReferenceData,
)
from pydma.utils.roi import build_roi_mask

# Visualization
from pydma.visualization.plots import (
    DMAPlotter,
    plot_aging_study,
    plot_degradation_modes,
    plot_dma_result,
    plot_dva_comparison,
    plot_ica_comparison,
    plot_ocv_comparison,
    plot_ocv_model_param_show,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core classes
    "DMAAnalyzer",
    "DMAOptimizer",
    "DMAConfig",
    "DMAResult",
    "AgingStudyResults",
    "DegradationModes",
    "FittedParams",
    "ReferenceData",
    # Electrode classes
    "ElectrodeOCP",
    "BlendElectrode",
    "calculate_inhomogeneity",
    # Data loading
    "load_ocp",
    "load_pocv",
    "load_aging_study",
    "auto_detect_columns",
    # Preprocessing
    "smooth_lowess",
    "smooth_savgol",
    "apply_filter",
    # Analysis functions
    "calculate_dva",
    "calculate_ica",
    "calculate_degradation_modes",
    "build_roi_mask",
    # Visualization
    "plot_ocv_model_param_show",
    "plot_dma_result",
    "plot_degradation_modes",
    "plot_ocv_comparison",
    "plot_dva_comparison",
    "plot_ica_comparison",
    "plot_aging_study",
    "DMAPlotter",
    # Silicon
    "generate_si_curve",
    "SiliconCurveParams",
    # Electrode balancing export
    "CellGeometry",
    "ElectrodeBalancing",
    "derive_balancing",
    "derive_balancing_from_result",
]
