"""Tests for the electrode models and the electrode/config agreement check.

Two failure modes are covered. A non-finite sample in an OCP table has to be
rejected at construction, because every comparison against NaN is false and it
would otherwise pass every later check and poison the interpolant. And an
electrode model that disagrees with the blend flags in ``DMAConfig`` has to be
rejected before the fit: the flags decide which parameter slots the optimizer
opens, so a mismatch fits a different cell than the configuration describes.
"""

import numpy as np
import pytest

from pydma.core.analyzer import DMAAnalyzer
from pydma.electrodes.blend import BlendElectrode
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.utils.dma_config import DMAConfig

# ---------------------------------------------------------------------------
# Two linear components with deliberately different voltage windows
# ---------------------------------------------------------------------------
#
# blend1 runs 0.50 V -> 0.05 V across its full sto axis, blend2 runs
# 0.40 V -> 0.10 V, so their common voltage window [0.10, 0.40] V cuts each
# component differently.
_B1_V_HI, _B1_V_LO = 0.50, 0.05
_B2_V_HI, _B2_V_LO = 0.40, 0.10


def _linear_anode(v_hi: float, v_lo: float, name: str, n: int = 501) -> ElectrodeOCP:
    """An anode whose potential falls linearly from ``v_hi`` to ``v_lo``.

    Falling with sto is PyDMA's anode convention, so ``__post_init__`` leaves
    the curve as given instead of mirroring it.
    """
    soc = np.linspace(0.0, 1.0, n)
    return ElectrodeOCP(
        soc=soc,
        voltage=v_hi - (v_hi - v_lo) * soc,
        name=name,
        electrode_type="anode",
    )


def _simple_anode() -> ElectrodeOCP:
    return _linear_anode(0.30, 0.05, "anode", n=60)


def _simple_cathode() -> ElectrodeOCP:
    soc = np.linspace(0.0, 1.0, 60)
    return ElectrodeOCP(
        soc=soc,
        voltage=3.4 + 0.8 * soc,  # rising with sto: PyDMA's internal cathode convention
        name="cathode",
        electrode_type="cathode",
    )


def _blend_anode() -> BlendElectrode:
    return BlendElectrode(
        blend1=_linear_anode(_B1_V_HI, _B1_V_LO, "blend1"),
        blend2=_linear_anode(_B2_V_HI, _B2_V_LO, "blend2"),
        electrode_type="anode",
        n_points=2001,
    )


# ---------------------------------------------------------------------------
# Non-finite OCP samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["voltage", "soc"], ids=["voltage", "soc"])
def test_non_finite_ocp_sample_is_rejected_and_the_index_is_named(axis):
    """The message names the axis and the first offending sample, so the user
    can find it in a table of thousands of rows."""
    soc = np.linspace(0.0, 1.0, 11)
    voltage = 0.30 - 0.25 * soc
    if axis == "voltage":
        voltage = voltage.copy()
        voltage[3] = np.nan
    else:
        soc = soc.copy()
        soc[3] = np.nan

    with pytest.raises(ValueError, match=rf"{axis} must be finite.*sample 3 is nan"):
        ElectrodeOCP(soc=soc, voltage=voltage, name="broken", electrode_type="anode")


# ---------------------------------------------------------------------------
# Electrode model vs. blend flags
# ---------------------------------------------------------------------------

_MEASURED_Q = np.linspace(0.0, 1.0, 60)
_MEASURED_V = 3.6 + 0.8 * _MEASURED_Q


def _fast_config(**overrides) -> DMAConfig:
    """Small grids: the mismatch is raised before any fitting starts, so the
    only cost that matters here is the electrode preparation."""
    settings = dict(
        data_length=100,
        smoothing_points=1,
        req_accepted=1,
        max_tries_overall=1,
        print_progress=False,
        speed_preset="fast",
    )
    settings.update(overrides)
    return DMAConfig(**settings)


def test_blend_anode_with_the_blend_flag_off_is_rejected():
    analyzer = DMAAnalyzer(
        config=_fast_config(use_anode_blend=False),
        anode=_blend_anode(),
        cathode=_simple_cathode(),
    )

    with pytest.raises(
        ValueError, match="anode is a BlendElectrode but config.use_anode_blend is False"
    ):
        analyzer.analyze(
            measured_capacity=_MEASURED_Q,
            measured_voltage=_MEASURED_V,
            actual_capacity=4.2,
        )


def test_single_component_anode_with_the_blend_flag_on_is_rejected():
    analyzer = DMAAnalyzer(
        config=_fast_config(use_anode_blend=True),
        anode=_simple_anode(),
        cathode=_simple_cathode(),
    )

    with pytest.raises(
        ValueError, match="config.use_anode_blend is True but anode is not a BlendElectrode"
    ):
        analyzer.analyze(
            measured_capacity=_MEASURED_Q,
            measured_voltage=_MEASURED_V,
            actual_capacity=4.2,
        )
