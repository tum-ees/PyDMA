"""Regression tests for `pydma.silicon.generator.generate_si_curve`.

These pin the behaviour after the pre-trim common-voltage-window fix that
aligns PyDMA with the MATLAB DMA tool's `generate_si_ocp`.

The fix replaces the post-trim recomputation of the linspace endpoints

    v_lo = max(gr_v.min(), blend_v.min())   # POST-TRIM (old behaviour)
    v_hi = min(gr_v.max(), blend_v.max())

with the conceptual pre-trim window

    v_min = max(gr_v.min(), blend_v.min())   # PRE-TRIM (new behaviour)
    v_max = min(gr_v.max(), blend_v.max())

so the linspace covers the full conceptual support intersection
[v_min, v_max] rather than an inset range whose endpoints depend on input
sampling.

The last group covers the three guards around the extraction: the two input
curves have to run their capacity in the same direction, a silicon curve that
leaves [0, 1] before the clip has to say so, and a MAT file holding several
candidate variables has to name them instead of silently reading the first.
"""

import warnings

import numpy as np
import pytest
import scipy.io

from pydma.silicon.generator import generate_si_curve, load_blend_data, load_ocp_data


def _trimmed_len(v, vmin, vmax):
    return int(np.sum((v >= vmin) & (v <= vmax)))


def test_pretrim_endpoints_used_for_common_grid():
    """v_common[-1] is exactly v_max (the conceptual pre-trim upper boundary),
    and v_common[0] sits one linspace step above v_min -- proving the linspace
    endpoints are computed from the pre-trim window. With the old post-trim
    behaviour, the last point would be inset below v_max (it would equal the
    last sample of whichever input got trimmed at its upper end).
    """
    gamma = 0.25
    # gr is wider than blend on the top end so post-trim upper sample is
    # blend.max() = 1.100 EXACTLY -- that part is sampling-coincidental.
    # gr is wider on the bottom too: gr's first kept sample after trim is
    # strictly INSIDE (0.030, ...) because gr's grid doesn't land exactly
    # at 0.030. So the OLD behaviour would have started the grid at
    # ~0.0318, not at 0.030.
    gr_v = np.linspace(0.020, 1.200, 501)
    gr_q = np.linspace(0.0, 1.0, 501)
    blend_v = np.linspace(0.030, 1.100, 401)
    blend_q = np.linspace(0.0, 1.0, 401)

    v_min_pre = max(gr_v.min(), blend_v.min())
    v_max_pre = min(gr_v.max(), blend_v.max())

    res = generate_si_curve(
        graphite_data=(gr_v, gr_q),
        blend_data=(blend_v, blend_q),
        gamma_si=gamma,
        filter_blend=False,
        filter_graphite=False,
        monotone_filter=False,
    )
    v_common = np.asarray(res.voltage, dtype=float)

    # n_points = max(len(gr_after_trim), len(blend_after_trim))
    n_points = max(
        _trimmed_len(gr_v, v_min_pre, v_max_pre), _trimmed_len(blend_v, v_min_pre, v_max_pre)
    )
    expected_step = (v_max_pre - v_min_pre) / (n_points - 1)

    # After mask_first drops index 0, the FIRST surviving entry equals
    # v_min + 1*step (the second linspace sample).
    np.testing.assert_allclose(
        v_common[0],
        v_min_pre + expected_step,
        atol=1e-12,
        err_msg="v_common[0] should be v_min + one linspace step " "(pre-trim behaviour)",
    )
    # The LAST surviving entry is exactly v_max.
    np.testing.assert_allclose(
        v_common[-1],
        v_max_pre,
        atol=1e-12,
        err_msg="v_common[-1] should be exactly v_max " "(pre-trim behaviour)",
    )


def test_offset_blend_grid_starts_at_v_min():
    """Two grids whose post-trim first samples are INSIDE (v_min, v_max).
    The new behaviour starts the linspace exactly at v_min; the old behaviour
    would have shifted it to the first surviving sample, which is strictly
    above v_min here.
    """
    gamma = 0.20
    n_gr, n_bl = 501, 401
    # blend_v[0] = 0.040 + 1e-5, just above gr's nearest sample
    blend_v = np.linspace(0.040 + 1e-5, 0.960 - 1e-5, n_bl)
    blend_q = np.linspace(0.0, 1.0, n_bl)
    gr_v = np.linspace(0.000, 1.000, n_gr)
    gr_q = np.linspace(0.0, 1.0, n_gr)

    v_min_pre = max(gr_v.min(), blend_v.min())  # = blend_v[0] = 0.040 + 1e-5
    v_max_pre = min(gr_v.max(), blend_v.max())  # = blend_v[-1]

    res = generate_si_curve(
        graphite_data=(gr_v, gr_q),
        blend_data=(blend_v, blend_q),
        gamma_si=gamma,
        filter_blend=False,
        filter_graphite=False,
        monotone_filter=False,
    )
    v_common = np.asarray(res.voltage, dtype=float)

    # In this construction, gr_v has NO sample exactly at v_min_pre = 0.040 + 1e-5
    # (gr's grid is at integer multiples of 0.002). So gr's first surviving
    # post-trim sample is gr_v[20] = 0.040, which is BELOW v_min_pre -- so
    # post-trim mask drops it. The first gr sample IN [v_min_pre, v_max_pre]
    # is gr_v[21] = 0.042. Under the OLD behaviour, the grid would start at
    # max(0.042, blend_v[0]) = 0.042. Under the NEW behaviour, the grid
    # starts at v_min_pre = 0.040 + 1e-5.
    n_points = max(
        _trimmed_len(gr_v, v_min_pre, v_max_pre), _trimmed_len(blend_v, v_min_pre, v_max_pre)
    )
    step = (v_max_pre - v_min_pre) / (n_points - 1)

    # First retained point = v_min_pre + step (NEW behaviour: linspace starts
    # exactly at v_min_pre, the conceptual lower boundary).
    np.testing.assert_allclose(
        v_common[0],
        v_min_pre + step,
        atol=1e-12,
        err_msg="grid should start at v_min_pre (not at the " "first surviving post-trim sample)",
    )

    # Compute what the OLD behaviour WOULD have produced and confirm it
    # differs from the new one by more than 1e-9 (so the assertion above
    # genuinely distinguishes the two implementations on this input).
    gr_post_trim_min = gr_v[(gr_v >= v_min_pre) & (gr_v <= v_max_pre)].min()
    bl_post_trim_min = blend_v[(blend_v >= v_min_pre) & (blend_v <= v_max_pre)].min()
    old_v_lo = max(gr_post_trim_min, bl_post_trim_min)
    old_v_hi = min(
        gr_v[(gr_v >= v_min_pre) & (gr_v <= v_max_pre)].max(),
        blend_v[(blend_v >= v_min_pre) & (blend_v <= v_max_pre)].max(),
    )
    old_step = (old_v_hi - old_v_lo) / (n_points - 1)
    old_first = old_v_lo + old_step
    assert abs(old_first - (v_min_pre + step)) > 1e-9, (
        f"Test input is degenerate: OLD and NEW first-points coincide "
        f"({old_first} vs {v_min_pre + step}); choose offsets that separate them."
    )

    # Last retained point = v_max
    np.testing.assert_allclose(v_common[-1], v_max_pre, atol=1e-12)


def test_linspace_step_is_pretrim_window_width_over_n_minus_1():
    """The linspace step is exactly (v_max - v_min)/(n_points-1), where v_min
    and v_max come from the PRE-trim window. This pins both endpoints AND
    the spacing in one test, on a third independent input configuration
    (gr wider than blend on BOTH ends, no offset shenanigans).
    """
    gamma = 0.20
    # gr strictly wider than blend on both ends -> v_min = blend.min(),
    # v_max = blend.max(). All blend samples survive trim; gr loses samples
    # at both ends but its trimmed min is >= v_min (and in general > v_min
    # because gr's grid won't align exactly to blend.min()).
    blend_v = np.linspace(0.050, 1.200, 401)
    blend_q = np.linspace(0.0, 1.0, 401)
    gr_v = np.linspace(0.000, 1.300, 501)
    gr_q = np.linspace(0.0, 1.0, 501)

    v_min_pre = max(gr_v.min(), blend_v.min())  # = blend_v.min() = 0.050
    v_max_pre = min(gr_v.max(), blend_v.max())  # = blend_v.max() = 1.200

    res = generate_si_curve(
        graphite_data=(gr_v, gr_q),
        blend_data=(blend_v, blend_q),
        gamma_si=gamma,
        filter_blend=False,
        filter_graphite=False,
        monotone_filter=False,
    )
    v_common = np.asarray(res.voltage, dtype=float)

    n_points = max(
        _trimmed_len(gr_v, v_min_pre, v_max_pre), _trimmed_len(blend_v, v_min_pre, v_max_pre)
    )
    expected_step = (v_max_pre - v_min_pre) / (n_points - 1)

    # Step (from the first two retained samples) equals the pre-trim
    # window width / (n_points - 1) exactly.
    actual_step = float(v_common[1] - v_common[0])
    np.testing.assert_allclose(
        actual_step,
        expected_step,
        atol=1e-12,
        err_msg="linspace step must be pre-trim window / (n-1)",
    )

    # First retained = v_min + 1*step, last retained = v_max.
    np.testing.assert_allclose(v_common[0], v_min_pre + expected_step, atol=1e-12)
    np.testing.assert_allclose(v_common[-1], v_max_pre, atol=1e-12)

    # The step is also distinguishable from what the old behaviour would
    # have produced (gr's first trimmed sample > v_min_pre, so old window
    # is narrower and old_step != expected_step on this input).
    gr_post_trim_min = gr_v[(gr_v >= v_min_pre) & (gr_v <= v_max_pre)].min()
    gr_post_trim_max = gr_v[(gr_v >= v_min_pre) & (gr_v <= v_max_pre)].max()
    old_step = (min(gr_post_trim_max, blend_v.max()) - max(gr_post_trim_min, blend_v.min())) / (
        n_points - 1
    )
    assert (
        abs(old_step - expected_step) > 1e-9
    ), f"degenerate input: old/new step coincide ({old_step} vs {expected_step})"


# ---------------------------------------------------------------------------
# Guards around the extraction
# ---------------------------------------------------------------------------

_GUARD_V = np.linspace(0.05, 0.50, 401)


def test_opposite_capacity_directions_are_rejected():
    """A lithiation graphite reference against a delithiation blend still
    subtracts to a smooth-looking silicon curve, so the mismatch has to be
    caught. The same graphite against a same-direction blend goes through.
    """
    rising = np.linspace(0.0, 1.0, _GUARD_V.size)

    with pytest.raises(ValueError, match="opposite directions over"):
        generate_si_curve(
            graphite_data=(_GUARD_V, rising),
            blend_data=(_GUARD_V, rising[::-1]),
            gamma_si=0.25,
            filter_blend=False,
            filter_graphite=False,
            monotone_filter=False,
        )

    result = generate_si_curve(
        graphite_data=(_GUARD_V, rising),
        blend_data=(_GUARD_V, rising),
        gamma_si=0.25,
        filter_blend=False,
        filter_graphite=False,
        monotone_filter=False,
    )
    assert result.voltage.size > 0


def test_a_silicon_extraction_that_leaves_the_unit_interval_is_reported():
    """The clip to [0, 1] is silent, so a gamma_si that does not match the two
    curves has to surface as a warning. Both thresholds have to be crossed:
    a share of the samples AND a margin beyond the interval.
    """
    rising = np.linspace(0.0, 1.0, _GUARD_V.size)

    with pytest.warns(UserWarning, match=r"leaves \[0, 1\] before clipping"):
        result = generate_si_curve(
            graphite_data=(_GUARD_V, rising),
            blend_data=(_GUARD_V, np.sqrt(rising)),
            gamma_si=0.10,
            filter_blend=False,
            filter_graphite=False,
            monotone_filter=False,
        )

    assert result.clipped_fraction > 0.01
    assert result.q_si_raw_max > 1.02
    assert float(result.normalized_capacity.max()) <= 1.0


def _write_two_curve_mat(path, keys):
    """A MAT file carrying one voltage/normalizedCapacity struct per key."""
    voltage = np.linspace(0.05, 0.50, 32)
    scipy.io.savemat(
        str(path),
        {
            key: {
                "voltage": voltage,
                "normalizedCapacity": np.linspace(0.0, 1.0, voltage.size) + offset,
            }
            for offset, key in enumerate(keys)
        },
    )


@pytest.mark.parametrize(
    "loader, kind",
    [(load_ocp_data, "OCP"), (load_blend_data, "blend")],
    ids=["ocp", "blend"],
)
def test_a_mat_file_with_several_candidate_variables_names_them(tmp_path, loader, kind):
    """Without ``variable_name`` the first candidate is read. Which one that is
    depends on the file's key order, so the alternatives are listed rather than
    passed over in silence. Naming one silences the warning.
    """
    path = tmp_path / f"two_{kind}.mat"
    _write_two_curve_mat(path, ("curveA", "curveB"))

    with pytest.warns(UserWarning, match=f"holds 2 variables with {kind} data"):
        loader(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, capacity = loader(path, variable_name="curveB")
    assert float(capacity.min()) == pytest.approx(1.0)
