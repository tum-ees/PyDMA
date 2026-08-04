"""Regression tests for `pydma.silicon.strict_sto._collapse_plateaus`.

These pin the 1.1.2 plateau-collapse contract: the shift applied to a plateau's
two surviving endpoints is bounded by a quarter of the distance to the
neighbouring plateau levels, a plateau sitting on the capacity-range boundary is
shifted inward only, levels closer than eight ulps are pooled into one plateau,
and the output is never clamped back into the range. Together those guarantee a
strictly monotone, tie-free, in-range result, so no consumer has to deduplicate
and no voltage support is silently lost.

`_legacy_collapse_plateaus` below is a frozen copy of the pre-1.1.2
implementation. It is the contrast reference: it documents the exact input class
that used to lose data, and it pins that the new code reproduces the old
arithmetic bit-for-bit wherever the plateau levels are well separated.
"""

import numpy as np
import pytest

from pydma.silicon.strict_sto import _collapse_plateaus, pchip_resample_for_pybamm

# ---------------------------------------------------------------------------
# Frozen pre-1.1.2 reference implementation
# ---------------------------------------------------------------------------


def _legacy_collapse_plateaus(voltage, capacity, eps=None):
    """Verbatim pre-1.1.2 `_collapse_plateaus`: fixed shift, endpoint clamping,
    min-step sweep and a final clip back into the capacity range."""
    n = len(capacity)
    if n < 2:
        return voltage, capacity

    direction = 1 if capacity[-1] >= capacity[0] else -1

    q_min = float(capacity.min())
    q_max = float(capacity.max())
    q_range = q_max - q_min

    if eps is None:
        eps = q_range * 1e-5 if q_range > 0 else 1e-9
    eps = max(float(eps), np.finfo(np.float64).eps)

    q_arr = capacity.astype(np.float64)
    q_clean = np.maximum.accumulate(q_arr) if direction > 0 else np.minimum.accumulate(q_arr)

    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and q_clean[j + 1] == q_clean[i]:
            j += 1
        if j > i + 1:
            keep[i + 1 : j] = False
        i = j + 1

    v_out = voltage[keep].copy()
    q_out = q_clean[keep].astype(np.float64, copy=True)

    m = len(q_out)
    k = 0
    while k < m - 1:
        if q_out[k] == q_out[k + 1]:
            q_out[k] = max(q_min, min(q_max, q_out[k] - direction * eps))
            q_out[k + 1] = max(q_min, min(q_max, q_out[k + 1] + direction * eps))
            k += 2
        else:
            k += 1

    min_step = max(eps * 0.1, np.finfo(np.float64).eps)
    if direction > 0:
        for k in range(1, m):
            if q_out[k] <= q_out[k - 1]:
                q_out[k] = q_out[k - 1] + min_step
    else:
        for k in range(1, m):
            if q_out[k] >= q_out[k - 1]:
                q_out[k] = q_out[k - 1] - min_step
    q_out = np.clip(q_out, q_min, q_max)

    return v_out, q_out


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Two plateau LEVELS 1e-6 apart -- a tenth of the 1e-5 tie-breaking shift --
# right next to the q = 0 saturation boundary. This is the configuration that
# arises naturally where the clip-to-[0, 1] step in the silicon generator
# saturates the curve.
TRIGGER_Q = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1e-6, 1e-6, 0.0, 0.0, 0.0])
TRIGGER_V = np.array([0.30, 0.28, 0.26, 0.20, 0.18, 0.12, 0.11, 0.10, 0.09, 0.08])

# Plateau levels far enough apart that the quarter-gap bound never bites, so
# the global 1e-5 shift applies everywhere and the new code must reproduce the
# old arithmetic exactly.
SEPARATED_Q_UP = np.array([0.0, 0.0, 0.25, 0.25, 0.6, 0.6, 1.0, 1.0])
SEPARATED_V_UP = np.array([0.40, 0.38, 0.30, 0.28, 0.22, 0.20, 0.14, 0.12])
SEPARATED_Q_DOWN = np.array([1.0, 1.0, 0.6, 0.6, 0.25, 0.25, 0.0, 0.0])
SEPARATED_V_DOWN = np.array([0.12, 0.14, 0.20, 0.22, 0.28, 0.30, 0.38, 0.40])


def _dedup_mask(sto):
    """The duplicate-sto mask `pchip_resample_for_pybamm` applies internally."""
    order = np.argsort(sto, kind="stable")
    sto_sorted = sto[order]
    keep = np.concatenate(([True], np.diff(sto_sorted) > 0))
    return order, keep


# ---------------------------------------------------------------------------
# The failure the fix removes
# ---------------------------------------------------------------------------


def test_legacy_collapse_silently_loses_voltage_support():
    """Pre-1.1.2 contrast: on TRIGGER_Q the two shifts cross, the sweep pushes
    past the capacity range, the final clip puts the samples back onto q = 0 as
    exact ties, and the downstream dedup drops them together with their
    voltages -- with no exception anywhere.
    """
    v_leg, q_leg = _legacy_collapse_plateaus(TRIGGER_V, TRIGGER_Q)

    ties = int(np.sum(np.diff(q_leg) == 0.0))
    assert ties == 2, f"expected the legacy code to produce 2 exact ties, got {ties}"
    assert np.sum(q_leg == 0.0) == 3, "the ties should sit on the q = 0 boundary"

    order, keep = _dedup_mask(q_leg)
    dropped_v = v_leg[order][~keep]
    assert dropped_v.size == 2, "the dedup should silently drop the 2 tied samples"
    # Voltage support really is lost: those voltages are gone from the table.
    for v_lost in dropped_v:
        assert v_lost not in set(v_leg[order][keep].tolist())

    # And the silent part: the resample accepts the damaged curve without a word.
    sto_grid, v_grid = pchip_resample_for_pybamm(v_leg, q_leg, n_points=11)
    assert sto_grid.size == 11
    assert np.all(np.isfinite(v_grid))


def test_fixed_collapse_is_clean_on_the_trigger_input():
    """Post-fix: the same input yields a strictly monotone, tie-free, in-range
    curve that keeps every plateau endpoint voltage."""
    v_out, q_out = _collapse_plateaus(TRIGGER_V, TRIGGER_Q)

    assert np.all(np.diff(q_out) < 0.0), "output must be strictly decreasing"
    assert int(np.sum(np.diff(q_out) == 0.0)) == 0, "output must be tie-free"
    assert q_out.min() >= TRIGGER_Q.min()
    assert q_out.max() <= TRIGGER_Q.max()

    # Voltage support preserved: both endpoints of every plateau survive, so the
    # overall voltage span is unchanged.
    assert float(v_out.min()) == float(TRIGGER_V.min())
    assert float(v_out.max()) == float(TRIGGER_V.max())

    # No deduplication needed any more.
    _, keep = _dedup_mask(q_out)
    assert bool(keep.all()), "collapsed output must need no dedup"


def test_collapsed_output_survives_pchip_resample_without_dedup():
    """The collapsed curve is a valid PCHIP abscissa as-is."""
    v_out, q_out = _collapse_plateaus(TRIGGER_V, TRIGGER_Q)
    sto_grid, v_grid = pchip_resample_for_pybamm(v_out, q_out, n_points=101)

    assert sto_grid.size == 101
    assert np.all(np.diff(sto_grid) > 0.0)
    assert np.all(np.isfinite(v_grid))
    np.testing.assert_allclose(sto_grid[0], q_out.min(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(sto_grid[-1], q_out.max(), atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Eight-ulp level pooling, both directions
# ---------------------------------------------------------------------------


def test_ulp_close_levels_are_pooled_rising():
    """Two plateau levels two ulps apart are one level at machine precision, so
    their runs merge and the merged run keeps the first and the last sample of
    the whole group instead of being shifted apart by an unrepresentable step.
    """
    u = float(np.spacing(0.5))
    q = np.array([0.0, 0.0, 0.5, 0.5, 0.5 + 2 * u, 0.5 + 2 * u, 1.0, 1.0])
    v = np.array([0.40, 0.38, 0.30, 0.28, 0.26, 0.24, 0.20, 0.18])

    v_out, q_out = _collapse_plateaus(v, q)

    # 4 raw runs pooled to 3 -> 6 emitted samples, not 8.
    assert q_out.size == 6, f"expected the two ulp-close levels to pool, got {q_out.size} samples"
    assert np.all(np.diff(q_out) > 0.0)
    np.testing.assert_array_equal(v_out, np.array([v[0], v[1], v[2], v[5], v[6], v[7]]))
    np.testing.assert_array_equal(
        q_out, np.array([0.0, 1e-5, 0.5 - 1e-5, 0.5 + 1e-5, 1.0 - 1e-5, 1.0])
    )


def test_ulp_close_levels_are_pooled_falling():
    """Same as the rising case for a falling curve.

    This is the regression test for the ulp sign trap: the falling branch mirrors
    the curve, so every working level is negative, and `np.spacing` of a negative
    argument is negative. Measuring the ulp on the raw value instead of on its
    magnitude makes the pooling comparison unsatisfiable, the two levels get
    shifted apart by an unrepresentable step, and the result ties or leaves the
    range. Measuring on the magnitude keeps this symmetric with the rising case.
    """
    u = float(np.spacing(0.5))
    q = np.array([1.0, 1.0, 0.5 + 2 * u, 0.5 + 2 * u, 0.5, 0.5, 0.0, 0.0])
    v = np.array([0.18, 0.20, 0.24, 0.26, 0.28, 0.30, 0.38, 0.40])

    v_out, q_out = _collapse_plateaus(v, q)

    assert q_out.size == 6, f"expected the two ulp-close levels to pool, got {q_out.size} samples"
    assert np.all(np.diff(q_out) < 0.0)
    np.testing.assert_array_equal(v_out, np.array([v[0], v[1], v[2], v[5], v[6], v[7]]))
    assert float(q_out[0]) == 1.0
    assert float(q_out[-1]) == 0.0


# ---------------------------------------------------------------------------
# Inward-only boundary shift, no clamping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "v_in, q_in",
    [
        (SEPARATED_V_UP, SEPARATED_Q_UP),
        (SEPARATED_V_DOWN, SEPARATED_Q_DOWN),
    ],
    ids=["rising", "falling"],
)
def test_boundary_runs_shift_inward_only(v_in, q_in):
    """A plateau sitting exactly on the range boundary keeps its boundary sample
    at the exact boundary value and moves only its partner into the range, so
    the output never has to be clamped back."""
    v_out, q_out = _collapse_plateaus(v_in, q_in)

    assert float(q_out[0]) == float(q_in[0]), "first boundary sample must keep its exact value"
    assert float(q_out[-1]) == float(q_in[-1]), "last boundary sample must keep its exact value"
    assert float(q_out.min()) >= float(q_in.min()), "output must not leave the input range"
    assert float(q_out.max()) <= float(q_in.max()), "output must not leave the input range"
    # The partner of each boundary sample moved strictly inward.
    assert q_out[1] != q_out[0]
    assert q_out[-2] != q_out[-1]
    assert v_out.size == q_out.size == 8


# ---------------------------------------------------------------------------
# No-op where the old arithmetic was already correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "v_in, q_in",
    [
        (SEPARATED_V_UP, SEPARATED_Q_UP),
        (SEPARATED_V_DOWN, SEPARATED_Q_DOWN),
    ],
    ids=["rising", "falling"],
)
def test_well_separated_levels_reproduce_legacy_arithmetic(v_in, q_in):
    """Where the plateau levels are well separated the quarter-gap bound never
    bites and the inward-only boundary rule lands on the same values the old
    clamp did, so the fix must be bit-identical to the pre-1.1.2 output."""
    v_leg, q_leg = _legacy_collapse_plateaus(v_in, q_in)
    v_new, q_new = _collapse_plateaus(v_in, q_in)

    np.testing.assert_array_equal(v_new, v_leg)
    np.testing.assert_array_equal(q_new, q_leg)


# ---------------------------------------------------------------------------
# Input contracts as exceptions, not asserts
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="same number of samples"):
        _collapse_plateaus(np.array([0.3, 0.2, 0.1]), np.array([0.0, 0.5]))


def test_constant_capacity_raises_value_error():
    """A constant curve has no range to collapse onto. That is a bad input, so
    it must surface as ValueError rather than as an internal post-condition."""
    with pytest.raises(ValueError, match="non-degenerate capacity range"):
        _collapse_plateaus(np.array([0.3, 0.2, 0.1]), np.array([0.4, 0.4, 0.4]))


@pytest.mark.parametrize(
    "bad",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "inf", "-inf"],
)
def test_non_finite_capacity_raises_value_error(bad):
    """Non-finite capacity must be rejected at entry.

    Every comparison against NaN is False, so before this check a NaN slipped
    past the degeneracy guard, the repair sweep and both post-conditions and the
    caller silently received a NaN-poisoned curve.
    """
    with pytest.raises(ValueError, match="capacity must be finite"):
        _collapse_plateaus(np.array([0.3, 0.2, 0.1]), np.array([0.0, bad, 1.0]))


def test_non_finite_voltage_raises_value_error():
    with pytest.raises(ValueError, match="voltage must be finite"):
        _collapse_plateaus(np.array([0.3, np.nan, 0.1]), np.array([0.0, 0.5, 1.0]))


def test_non_finite_eps_raises_value_error():
    with pytest.raises(ValueError, match="eps must be finite"):
        _collapse_plateaus(SEPARATED_V_UP, SEPARATED_Q_UP, eps=np.nan)


def test_capacity_range_of_one_ulp_raises_value_error():
    """A range about one ulp wide is degenerate one step out from constant.

    All levels pool into a single run and the machine-epsilon shift floor is
    then wider than the range itself, so the inward shift would step past the
    lower bound. Before this check that surfaced as a RuntimeError, which
    claimed a defect in the collapse rather than naming the unusable input.
    """
    nxt = np.nextafter(0.5, 1.0)
    with pytest.raises(ValueError, match="wider than the smallest"):
        _collapse_plateaus(np.array([0.40, 0.38, 0.30, 0.28]), np.array([0.5, 0.5, nxt, nxt]))
    with pytest.raises(ValueError, match="wider than the smallest"):
        _collapse_plateaus(np.array([0.40, 0.30]), np.array([0.5, nxt]))


def test_narrow_but_usable_capacity_range_still_collapses():
    """The guard must reject only what is genuinely unusable. A range that is
    narrow yet wider than the shift floor still produces a valid curve."""
    q = np.array([1.0, 1.0, 1.0 + 1e-14, 1.0 + 1e-14])
    v = np.array([0.40, 0.38, 0.30, 0.28])

    v_out, q_out = _collapse_plateaus(v, q)

    assert np.all(np.diff(q_out) > 0.0)
    assert float(q_out[0]) >= float(q.min())
    assert float(q_out[-1]) <= float(q.max())
    assert v_out.size == q_out.size


def test_short_input_passes_through():
    v = np.array([0.3])
    q = np.array([0.5])
    v_out, q_out = _collapse_plateaus(v, q)
    np.testing.assert_array_equal(v_out, v)
    np.testing.assert_array_equal(q_out, q)


def test_explicit_eps_is_still_bounded_by_the_quarter_gap():
    """An oversized explicit eps must not push a plateau endpoint across its
    neighbouring level."""
    q = np.array([0.0, 0.0, 0.5, 0.5, 0.500004, 0.500004, 1.0, 1.0])
    v = np.array([0.40, 0.38, 0.30, 0.28, 0.26, 0.24, 0.20, 0.18])

    _, q_out = _collapse_plateaus(v, q, eps=1.0)

    assert np.all(np.diff(q_out) > 0.0)
    assert float(q_out.min()) >= 0.0
    assert float(q_out.max()) <= 1.0
    # Quarter of the 4e-6 gap caps the shift for the two middle runs, so each
    # plateau spans half that gap rather than the requested eps.
    np.testing.assert_allclose(float(q_out[3] - q_out[2]), 0.5 * 4e-6, rtol=1e-9)
    # The two shifted plateaus still cannot meet.
    assert float(q_out[3]) < float(q_out[4])
