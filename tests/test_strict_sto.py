"""Regression tests for `pydma.silicon.strict_sto`.

The bulk of this file covers `_collapse_plateaus`; `_pav_isotonic` and the
input contract of `pchip_resample_for_pybamm` are covered at the end.

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

from pydma.silicon.generator import generate_si_curve
from pydma.silicon.strict_sto import _collapse_plateaus, _pav_isotonic, pchip_resample_for_pybamm

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
    # These fixtures run V upward with sto, so the endpoint snap does not apply.
    sto_grid, v_grid = pchip_resample_for_pybamm(v_leg, q_leg, n_points=11, snap_endpoint=False)
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
    sto_grid, v_grid = pchip_resample_for_pybamm(v_out, q_out, n_points=101, snap_endpoint=False)

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


def test_small_magnitude_narrow_range_is_not_rejected():
    """Small-magnitude data must not be judged against an absolute threshold.

    This range is 1e-16 wide, which is billions of ulps at a magnitude of 1e-10
    and perfectly resolvable. An absolute machine-epsilon span guard rejected it
    as degenerate, because 1e-16 is smaller than 2.22e-16 in absolute terms
    while being enormous in local ulps.
    """
    lo, hi = 1e-10, 1.000001e-10
    assert (hi - lo) / float(np.spacing(lo)) > 1e9, "test input is not actually resolvable"
    v = np.array([0.40, 0.38, 0.30, 0.28])
    q = np.array([lo, lo, hi, hi])

    v_out, q_out = _collapse_plateaus(v, q)

    assert np.all(np.diff(q_out) > 0.0)
    assert float(q_out[0]) == lo
    assert float(q_out[-1]) == hi
    assert float(v_out[0]) == float(v[0])
    assert float(v_out[-1]) == float(v[-1])


def test_overflowing_capacity_range_raises_value_error():
    """Every sample finite, yet their range is not.

    Before this check the overflowing range poisoned every derived shift and the
    function returned [-1e308, inf, nan, 1e308] without a word, because a NaN
    makes the monotonicity comparison false and so slipped past the
    post-conditions.
    """
    q = np.array([-1e308, -1e308, 1e308, 1e308])
    assert bool(np.all(np.isfinite(q))), "input samples must themselves be finite"
    with np.errstate(over="ignore"):
        assert not np.isfinite(q.max() - q.min()), "test input must overflow its range"

    with pytest.raises(ValueError, match="capacity range overflows"):
        _collapse_plateaus(np.array([0.40, 0.38, 0.30, 0.28]), q)


def _ulp_steps(base, k):
    """``base`` advanced by ``k`` representable steps upward.

    Stepping toward ``+inf`` rather than toward a fixed value keeps this correct
    for bases on either side of 1.0.
    """
    out = base
    for _ in range(k):
        out = np.nextafter(out, np.inf)
    return float(out)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("base", [0.5, 1.0, 2.0], ids=["base0.5", "base1.0", "base2.0"])
@pytest.mark.parametrize("rising", [True, False], ids=["rising", "falling"])
def test_few_ulp_capacity_range_keeps_both_exact_edges(k, base, rising):
    """A range of one to eight ulps pools into a single run, and that run must
    report both exact range edges. Every span in this matrix is valid input.

    Two separate defects lived here. The pooled level sits at the upper end of
    the range, so the ``level == q_work_min`` branch never fired and the
    opposite edge was shifted off its exact value and silently lost: rising
    curves lost their minimum, falling curves lost their maximum. Separately, an
    absolute machine-epsilon span guard rejected the narrowest of these ranges
    outright, before the single-run mapping could handle them correctly.
    """
    lo = base
    hi = _ulp_steps(lo, k)
    if rising:
        v = np.array([0.40, 0.38, 0.30, 0.28])
        q = np.array([lo, lo, hi, hi])
    else:
        v = np.array([0.28, 0.30, 0.38, 0.40])
        q = np.array([hi, hi, lo, lo])

    v_out, q_out = _collapse_plateaus(v, q)

    # Exact equality, not merely "inside the range".
    assert float(q_out.min()) == lo, f"lower edge lost: {q_out.min()!r} != {lo!r}"
    assert float(q_out.max()) == hi, f"upper edge lost: {q_out.max()!r} != {hi!r}"
    assert float(q_out[0]) == float(q[0]), "first sample must keep its exact capacity"
    assert float(q_out[-1]) == float(q[-1]), "last sample must keep its exact capacity"
    if rising:
        assert np.all(np.diff(q_out) > 0.0)
    else:
        assert np.all(np.diff(q_out) < 0.0)
    # Voltage support: the run's first and last voltage survive.
    assert float(v_out[0]) == float(v[0])
    assert float(v_out[-1]) == float(v[-1])


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


# ---------------------------------------------------------------------------
# _pav_isotonic: pooled means worked out by hand
# ---------------------------------------------------------------------------
#
# PAV walks left to right and pools a block with its predecessor whenever the
# block mean would violate the requested direction. The three cases below were
# traced by hand; the pooled values are written out as constants so nothing on
# the expected side comes from the function under test.


def test_pav_pools_a_single_violating_pair_to_their_mean():
    """[1, 3, 2, 4] non-decreasing.

    3 and 2 violate, so they pool to (3 + 2) / 2 = 2.5; 1 is already below
    that and 4 above it, so both stay. A sign error in the direction handling
    would return the input unchanged or pool the wrong pair.
    """
    result = _pav_isotonic(np.array([1.0, 3.0, 2.0, 4.0]), "nondecreasing")

    np.testing.assert_allclose(result, np.array([1.0, 2.5, 2.5, 4.0]), rtol=0.0, atol=0.0)
    assert np.all(np.diff(result) >= 0.0)
    # Isotonic regression preserves the sum: the pooling only redistributes.
    assert float(result.sum()) == pytest.approx(1.0 + 3.0 + 2.0 + 4.0, rel=1e-12)


def test_pav_mirrors_the_pooling_for_a_nonincreasing_direction():
    """[4, 2, 3, 1] non-increasing.

    The direction flag negates the input, so the violating pair is 2 and 3;
    they pool to 2.5 and the sign flips back. Measuring the violation on the
    raw values instead would leave the sequence untouched.
    """
    result = _pav_isotonic(np.array([4.0, 2.0, 3.0, 1.0]), "nonincreasing")

    np.testing.assert_allclose(result, np.array([4.0, 2.5, 2.5, 1.0]), rtol=0.0, atol=0.0)
    assert np.all(np.diff(result) <= 0.0)
    assert float(result.sum()) == pytest.approx(4.0 + 2.0 + 3.0 + 1.0, rel=1e-12)


def test_pav_grows_a_pool_to_three_members():
    """[1, 5, 3, 2, 6] non-decreasing.

    5 and 3 pool to 4 first. 2 then violates that block, so the block absorbs
    it as a size-weighted mean, (4 * 2 + 2 * 1) / 3 = 10 / 3, and NOT as the
    unweighted (4 + 2) / 2 = 3. This is the case an off-by-one in the block
    sizes gets wrong.
    """
    pooled = 10.0 / 3.0
    result = _pav_isotonic(np.array([1.0, 5.0, 3.0, 2.0, 6.0]), "nondecreasing")

    np.testing.assert_allclose(result, np.array([1.0, pooled, pooled, pooled, 6.0]), rtol=1e-15)
    assert np.all(np.diff(result) >= 0.0)
    assert float(result.sum()) == pytest.approx(1.0 + 5.0 + 3.0 + 2.0 + 6.0, rel=1e-12)


def test_pav_leaves_an_already_monotone_sequence_alone():
    """Nothing violates, so nothing pools."""
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    np.testing.assert_array_equal(_pav_isotonic(values, "nondecreasing"), values)


# ---------------------------------------------------------------------------
# generate_si_curve's monotone filter, end to end
# ---------------------------------------------------------------------------

# A graphite reference that rises linearly with voltage, and a blend built so
# the extracted silicon curve carries three deliberate dips. With gamma = 0.5
# the extraction Q_Si = (Q_blend - (1 - gamma) Q_Gr) / gamma inverts the
# construction exactly, so Q_Si is the wobble below and its three descending
# flanks are the non-monotone stretches the filter has to remove.
_SI_V_LO, _SI_V_HI, _SI_N = 0.05, 0.50, 61
_SI_GAMMA = 0.5
_SI_V = np.linspace(_SI_V_LO, _SI_V_HI, _SI_N)
_SI_T = (_SI_V - _SI_V_LO) / (_SI_V_HI - _SI_V_LO)
# Amplitude 0.08 against a slope of 1 over t: d/dt = 1 + 0.48 pi cos(6 pi t)
# reaches -0.5, so the wobble genuinely reverses rather than merely flattening.
_SI_WOBBLE = _SI_T + 0.08 * np.sin(6.0 * np.pi * _SI_T)
_SI_Q_GR_RISING = _SI_T
_SI_Q_BLEND_RISING = _SI_GAMMA * _SI_WOBBLE + (1.0 - _SI_GAMMA) * _SI_T


def _contiguous_runs(indices):
    """Group a sorted index array into (first, last) contiguous runs."""
    runs = []
    for index in indices.tolist():
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return [(first, last) for first, last in runs]


@pytest.mark.parametrize("rising", [True, False], ids=["rising", "falling"])
def test_generate_si_curve_monotone_filter_only_touches_the_violations(rising):
    """The default ``monotone_filter=True`` returns a monotone sto axis, and
    it differs from the unfiltered curve exactly around the constructed dips.

    Two things are pinned. The output really is monotone in the direction the
    curve runs -- a PAV direction picked from the wrong end would leave the
    dips in place. And every stretch where the two results differ contains an
    actual violation of the unfiltered curve, so the filter is not reshaping
    parts of the curve that were already fine.
    """
    q_gr = _SI_Q_GR_RISING if rising else 1.0 - _SI_Q_GR_RISING
    q_blend = _SI_Q_BLEND_RISING if rising else 1.0 - _SI_Q_BLEND_RISING

    filtered = generate_si_curve(
        blend_data=(_SI_V, q_blend), graphite_data=(_SI_V, q_gr), gamma_si=_SI_GAMMA
    ).normalized_capacity
    unfiltered = generate_si_curve(
        blend_data=(_SI_V, q_blend),
        graphite_data=(_SI_V, q_gr),
        gamma_si=_SI_GAMMA,
        monotone_filter=False,
    ).normalized_capacity

    steps = np.diff(unfiltered)
    violations = np.flatnonzero(steps < 0.0) if rising else np.flatnonzero(steps > 0.0)
    assert violations.size >= 3, "the fixture must actually be non-monotone"

    if rising:
        assert np.all(np.diff(filtered) >= 0.0), "filtered curve must not fall"
    else:
        assert np.all(np.diff(filtered) <= 0.0), "filtered curve must not rise"

    changed = np.flatnonzero(filtered != unfiltered)
    assert changed.size > 0, "the filter has to change something on this input"

    # A PAV pool is a contiguous block built around at least one violation, so
    # no run of changed samples may sit in an already-monotone stretch.
    violating_samples = set(violations.tolist()) | set((violations + 1).tolist())
    for first, last in _contiguous_runs(changed):
        assert violating_samples & set(
            range(first, last + 1)
        ), f"samples {first}..{last} changed although no violation touches them"

    # Each violating pair really was pooled away.
    for index in violations:
        assert filtered[index] != unfiltered[index] or filtered[index + 1] != unfiltered[index + 1]


# ---------------------------------------------------------------------------
# pchip_resample_for_pybamm: input contracts as exceptions
# ---------------------------------------------------------------------------

# A short falling V(sto) curve: the direction snap_endpoint assumes.
_RESAMPLE_V = np.array([0.30, 0.20, 0.10])
_RESAMPLE_STO = np.array([0.0, 0.5, 1.0])


def test_resample_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="must have identical shape"):
        pchip_resample_for_pybamm(np.array([0.30, 0.20, 0.10]), np.array([0.0, 1.0]))


def test_resample_needs_at_least_two_samples():
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        pchip_resample_for_pybamm(np.array([0.30]), np.array([0.5]))


@pytest.mark.parametrize(
    "n_points",
    [1, 0, -5, 2.5, "11"],
    ids=["one", "zero", "negative", "float", "string"],
)
def test_resample_rejects_a_grid_that_is_not_an_integer_of_at_least_two(n_points):
    with pytest.raises(ValueError, match="n_points must be an integer >= 2"):
        pchip_resample_for_pybamm(_RESAMPLE_V, _RESAMPLE_STO, n_points=n_points)


def test_resample_rejects_a_curve_that_is_degenerate_after_dedup():
    """Every sample shares one sto value, so the dedup leaves a single point
    and there is no abscissa to interpolate along."""
    with pytest.raises(ValueError, match="fewer than 2 unique sto values"):
        pchip_resample_for_pybamm(_RESAMPLE_V, np.array([0.5, 0.5, 0.5]))


def test_resample_refuses_to_snap_the_endpoint_of_a_rising_curve():
    """``snap_endpoint`` writes min(voltage) into the LAST grid point, which is
    the highest-voltage end of a rising V(sto). Snapping there would replace the
    curve's maximum with its minimum, so the rising direction is rejected. The
    same curve passes with ``snap_endpoint=False``.
    """
    rising_v = _RESAMPLE_V[::-1]

    with pytest.raises(ValueError, match="rises instead"):
        pchip_resample_for_pybamm(rising_v, _RESAMPLE_STO)

    sto_grid, v_grid = pchip_resample_for_pybamm(rising_v, _RESAMPLE_STO, snap_endpoint=False)
    assert float(v_grid[-1]) == pytest.approx(float(rising_v.max()))
    assert sto_grid.size == v_grid.size


def test_resample_accepts_a_numpy_integer_grid_size():
    """A grid size computed with numpy arrives as np.int64, which is not an
    ``int`` instance. The check is on ``numbers.Integral`` for exactly this.
    """
    sto_grid, v_grid = pchip_resample_for_pybamm(_RESAMPLE_V, _RESAMPLE_STO, n_points=np.int64(11))

    assert sto_grid.size == 11
    assert v_grid.size == 11
    assert np.all(np.diff(sto_grid) > 0.0)
    np.testing.assert_allclose(sto_grid[0], _RESAMPLE_STO.min(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(sto_grid[-1], _RESAMPLE_STO.max(), atol=0.0, rtol=0.0)
    # snap_endpoint defaults to True: the lowest raw voltage is written back.
    assert float(v_grid[-1]) == float(_RESAMPLE_V.min())
