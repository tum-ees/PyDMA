"""Utility classes and functions."""

from pydma.utils.balancing import (
    CellGeometry,
    ElectrodeBalancing,
    apply_aging,
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
from pydma.utils.roi import ROISpec, build_roi_mask, get_roi_outer_bounds, normalize_roi

__all__ = [
    "DMAConfig",
    "DMAResult",
    "FittedParams",
    "DegradationModes",
    "ReferenceData",
    "AgingStudyResults",
    "ROISpec",
    "normalize_roi",
    "build_roi_mask",
    "get_roi_outer_bounds",
    "CellGeometry",
    "ElectrodeBalancing",
    "derive_balancing",
    "derive_balancing_from_result",
    "apply_aging",
]
