"""Silicon module package."""

from pydma.silicon.generator import SiliconCurveResult, generate_si_curve
from pydma.silicon.strict_sto import pchip_resample_for_pybamm

__all__ = [
    "generate_si_curve",
    "SiliconCurveResult",
    "pchip_resample_for_pybamm",
]
