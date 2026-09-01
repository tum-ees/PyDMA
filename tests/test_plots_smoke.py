"""Smoke tests for ``pydma.visualization.plots``.

These do not compare pixels. They pin the three things a plotting bug can
break without anyone noticing: the axes have to enclose the data that was
handed in (a hard-coded limit crops the curve silently) — for the DVA panel,
whose two ends diverge, the enclosed range is the middle 10-90 % of the x span
rather than the whole curve — a plot call has to
leave the global ``plt.rcParams`` exactly as it found them (the style is
applied through ``plt.rc_context``, so it must not leak into the caller's
session), and the axis labels have to follow the explicit ``x_is_soc`` flag
rather than the range heuristic it exists to override.

Every figure is closed again: a loop over many CUs otherwise fills the pyplot
registry, which is exactly what the plot docstrings warn about.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pydma.utils.results import AgingStudyResults, DegradationModes, DMAResult
from pydma.visualization.plots import (
    plot_aging_study,
    plot_degradation_modes,
    plot_dma_result,
    plot_dva_comparison,
    plot_ica_comparison,
    plot_ocv_comparison,
)

# Headless: never pick up an interactive backend from the environment.
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_leftover_figures():
    """Each test closes its own figures; this is the safety net behind them."""
    yield
    plt.close("all")


# Small synthetic curves. The DVA and ICA values stay positive so the lower
# axis limit is comparable too (plot_dva_comparison floors its lower limit at
# zero, which is only below the data for a non-negative curve).
Q = np.linspace(0.0, 1.0, 21)
V = 3.0 + 1.2 * Q
DVA = 0.5 + 0.3 * np.sin(np.pi * Q)
ICA = 1.0 + 0.4 * np.cos(np.pi * Q)


def _degradation_modes() -> DegradationModes:
    """Eight distinct, non-zero modes, so a mislabelled bar cannot pass."""
    return DegradationModes(
        lam_anode=0.12,
        lam_cathode=0.07,
        lli=0.05,
        capacity_loss=0.09,
        lam_anode_blend1=0.03,
        lam_anode_blend2=0.15,
        lam_cathode_blend1=0.02,
        lam_cathode_blend2=0.04,
    )


def _result(cu_name: str = "CU1", rmse: float = 0.004, capacity: float = 4.2) -> DMAResult:
    return DMAResult(
        cu_name=cu_name,
        degradation_modes=_degradation_modes(),
        capacity=capacity,
        rmse=rmse,
        measured_capacity=Q,
        measured_voltage=V,
        measured_dva=DVA,
        measured_ica=ICA,
    )


# ---------------------------------------------------------------------------
# (a) The axes enclose the data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_fn, x, y",
    [
        (plot_ocv_comparison, Q, V),
        (plot_ica_comparison, V, ICA),
    ],
    ids=["ocv", "ica"],
)
def test_comparison_axes_enclose_the_data(plot_fn, x, y):
    """The returned axes carry a figure and show the whole curve.

    A hard-coded limit -- a leftover 0..1 y-range, say -- crops the curve
    without raising anything, so the limits are compared against the data
    that was handed in.
    """
    ax = plot_fn(x, y)
    try:
        assert isinstance(ax.figure, Figure)

        y_lo, y_hi = ax.get_ylim()
        assert y_hi >= float(y.max())
        assert y_lo <= float(y.min())

        x_lo, x_hi = ax.get_xlim()
        assert x_hi >= float(x.max())
        assert x_lo <= float(x.min())
    finally:
        plt.close(ax.figure)


_INTERIOR = (Q >= 0.1) & (Q <= 0.9)


def test_dva_axis_covers_the_interior_and_not_the_edge_divergence():
    """The DVA panel caps its y-axis rather than scaling to the global peak:
    the two ends of a DVA diverge, and letting them set the scale flattens the
    staging features in between. The cap gives way to the peak inside the
    middle 10-90 % of the x span, and to nothing outside it.
    """
    spiked = DVA.copy()
    spiked[0] = 40.0  # edge divergence, outside the interior

    ax = plot_dva_comparison(Q, spiked)
    try:
        y_lo, y_hi = ax.get_ylim()
        assert y_hi >= float(spiked[_INTERIOR].max())
        assert y_hi < float(spiked.max()), "the edge spike must not set the scale"
        assert y_lo <= float(spiked[_INTERIOR].min())
    finally:
        plt.close(ax.figure)

    tall = DVA + 5.0  # interior peak above the default cap

    ax = plot_dva_comparison(Q, tall)
    try:
        assert ax.get_ylim()[1] >= float(tall[_INTERIOR].max())
    finally:
        plt.close(ax.figure)


def test_degradation_bars_carry_the_modes_and_fit_inside_the_axes():
    """Each bar is its mode in percent, in the documented panel order, and the
    y-limits cover all of them."""
    modes = _degradation_modes()
    ax = plot_degradation_modes(modes, show_anode_blend=True, show_cathode_blend=True)
    try:
        expected = [
            modes.lli * 100,
            modes.lam_anode * 100,
            modes.lam_anode_blend1 * 100,
            modes.lam_anode_blend2 * 100,
            modes.lam_cathode * 100,
            modes.lam_cathode_blend1 * 100,
            modes.lam_cathode_blend2 * 100,
            modes.capacity_loss * 100,
        ]
        heights = [bar.get_height() for bar in ax.containers[0]]
        np.testing.assert_allclose(heights, expected, rtol=1e-12)

        y_lo, y_hi = ax.get_ylim()
        assert y_hi >= max(expected)
        assert y_lo <= min(expected)
    finally:
        plt.close(ax.figure)


def test_dma_result_builds_four_panels_that_enclose_their_data():
    fig = plot_dma_result(_result())
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4
        ocv_ax, dva_ax, ica_ax, _ = fig.axes

        assert ocv_ax.get_ylim()[1] >= float(V.max())
        assert ocv_ax.get_ylim()[0] <= float(V.min())
        assert dva_ax.get_ylim()[1] >= float(DVA[_INTERIOR].max())
        assert ica_ax.get_ylim()[1] >= float(ICA.max())
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# (b) No rcParams leak
# ---------------------------------------------------------------------------

_PLOT_CALLS = {
    "ocv": lambda: plot_ocv_comparison(Q, V).figure,
    "dva": lambda: plot_dva_comparison(Q, DVA).figure,
    "ica": lambda: plot_ica_comparison(V, ICA).figure,
    "degradation": lambda: plot_degradation_modes(_degradation_modes()).figure,
    "dma_result": lambda: plot_dma_result(_result()),
}

# Exactly the three keys ``_setup_style`` overrides (to 10, True and 1.5), so a
# style applied globally instead of through ``plt.rc_context`` overwrites them.
_SENTINEL_RCPARAMS = {"font.size": 7.5, "axes.grid": False, "lines.linewidth": 0.25}


@pytest.mark.parametrize("name", sorted(_PLOT_CALLS))
def test_plot_calls_leave_the_global_rcparams_untouched(name):
    """A full before/after snapshot of ``plt.rcParams`` around one plot call.

    The plot style is applied inside ``plt.rc_context`` so the caller's own
    matplotlib settings survive the call. The sentinels above are the very
    values the style overrides, so dropping the context manager flips them and
    both assertions fail.
    """
    with matplotlib.rc_context(_SENTINEL_RCPARAMS):
        before = dict(plt.rcParams)
        figure = _PLOT_CALLS[name]()
        plt.close(figure)
        after = dict(plt.rcParams)
        survivors = {key: plt.rcParams[key] for key in _SENTINEL_RCPARAMS}

    assert after == before
    assert survivors == _SENTINEL_RCPARAMS


# ---------------------------------------------------------------------------
# (c) x_is_soc overrides the range heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_fn, y",
    [(plot_ocv_comparison, V), (plot_dva_comparison, DVA)],
    ids=["ocv", "dva"],
)
def test_x_is_soc_pins_the_axis_label_against_the_range_heuristic(plot_fn, y):
    """Both flag values are exercised against the range they contradict.

    ``Q`` lies inside [0, 1] and would be read as SOC, so ``x_is_soc=False``
    has to produce the capacity label; ``4.2 * Q`` leaves [0, 1] and would be
    read as a capacity, so ``x_is_soc=True`` has to produce the SOC label.
    Ignoring the flag therefore fails in one direction or the other.
    """
    ax = plot_fn(Q, y, x_is_soc=False)
    try:
        assert ax.get_xlabel() == r"$Q$ / Ah"
    finally:
        plt.close(ax.figure)

    ax = plot_fn(4.2 * Q, y, x_is_soc=True)
    try:
        assert ax.get_xlabel() == r"SOC / -"
    finally:
        plt.close(ax.figure)


# ---------------------------------------------------------------------------
# (d) plot_aging_study x-axis alignment
# ---------------------------------------------------------------------------


def _study(efc_values) -> AgingStudyResults:
    study = AgingStudyResults()
    for index, efc in enumerate(efc_values):
        study.add_result(
            _result(
                cu_name=f"CU{index + 1}",
                rmse=0.003 + 0.001 * index,
                capacity=4.2 - 0.1 * index,
            ),
            efc=efc,
        )
    return study


def test_aging_study_uses_the_known_efc_values_as_the_x_axis():
    """Positive control for the fallback below: with every EFC known, the
    x-axis is the EFC list and nothing is warned about."""
    study = _study([0.0, 250.0, 500.0])

    fig = plot_aging_study(study)
    try:
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [0.0, 250.0, 500.0])
    finally:
        plt.close(fig)


def test_aging_study_warns_and_falls_back_when_no_efc_is_known():
    """All EFCs missing: the x-axis becomes the 1-based check-up index, and
    the substitution is announced rather than done silently."""
    study = _study([None, None, None])
    assert study.efc_values == [None, None, None]

    with pytest.warns(UserWarning, match="falling back to the RPT-index x-axis"):
        fig = plot_aging_study(study)
    try:
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1, 2, 3])
    finally:
        plt.close(fig)


def test_aging_study_refuses_an_efc_list_that_does_not_match_the_cus():
    """A shorter EFC list would silently label CU2 with CU1's cycle count."""
    study = _study([0.0, 250.0, 500.0])
    study.efc_values = [0.0]

    with pytest.raises(ValueError, match="efc_values has 1 entries but 3 CUs"):
        plot_aging_study(study)


def test_aging_study_treats_an_empty_efc_list_like_all_missing_values():
    """An empty list carries as much EFC information as a list of Nones, so it
    takes the same fallback rather than reading as a length mismatch."""
    study = _study([0.0, 250.0, 500.0])
    study.efc_values = []

    with pytest.warns(UserWarning, match="falling back to the RPT-index x-axis"):
        fig = plot_aging_study(study)
    try:
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1, 2, 3])
    finally:
        plt.close(fig)


def test_re_adding_a_cu_without_an_efc_keeps_the_stored_one():
    """Refitting a check-up replaces its result. Its EFC is not part of that,
    so it survives unless a new one is passed."""
    study = _study([0.0, 250.0, 500.0])

    study.add_result(_result(cu_name="CU2", rmse=0.009))

    assert study.efc_values == [0.0, 250.0, 500.0]
    assert study.results["CU2"].rmse == 0.009
