"""Tests for ``pydma.utils.dma_config`` and ``pydma.utils.roi``.

Configuration mistakes are the class of error that otherwise runs to
completion and produces a plausible-looking wrong number: a misspelled field
name that is silently ignored, a bound pair that crosses, an ROI given in
percent. Each of those has a guard; these pin that the guard fires, and that
it fires on assignment too, not only in the constructor.
"""

import warnings

import numpy as np
import pytest

from pydma.utils.dma_config import DMAConfig
from pydma.utils.roi import build_roi_mask, normalize_roi

# ---------------------------------------------------------------------------
# The LFP preset really avoids the plateau
# ---------------------------------------------------------------------------


def test_lfp_preset_roi_selects_only_the_two_edges():
    """The preset splits the OCV ROI into 0-15 % and 85-100 % SOC.

    On ``linspace(0, 1, 1001)`` the step is 0.001, so each interval holds 151
    grid points and the mask selects 302 of 1001 -- about 30 %, comfortably
    under a third. A preset that collapsed back to a single [0, 1] interval
    would select all 1001.
    """
    config = DMAConfig.lfp_preset()
    q = np.linspace(0.0, 1.0, 1001)

    mask = build_roi_mask(q, config.roi_ocv_min, config.roi_ocv_max)

    assert int(mask.sum()) == 151 + 151
    assert mask.mean() < 0.35
    # The flat middle is what the split exists to skip.
    assert not mask[(q > 0.15) & (q < 0.85)].any()
    # Both edges of the cell are still covered.
    assert bool(mask[0]) and bool(mask[-1])


# ---------------------------------------------------------------------------
# DMAConfig re-validates on assignment
# ---------------------------------------------------------------------------


def test_typo_in_a_field_name_is_rejected_instead_of_silently_stored():
    """``config.direciton = ...`` used to attach a stray attribute that the
    rest of the code never read, leaving the run on the default direction."""
    config = DMAConfig()

    with pytest.raises(AttributeError, match="has no field 'direciton'"):
        config.direciton = "discharge"


@pytest.mark.parametrize(
    "field, value, index",
    [
        ("lower_bounds", (2.0, -1.0, 1.0, -1.0), 0),
        ("upper_bounds", (2.0, 0.0, 1.0, 0.0), 2),
    ],
    ids=["lower-meets-upper-at-0", "upper-meets-lower-at-2"],
)
def test_crossing_parameter_bounds_are_rejected_and_name_the_slot(field, value, index):
    """A bound pair where lower >= upper leaves the optimizer with an empty
    interval. The message has to name the offending slot, so both a first and
    a later slot are exercised: reporting slot 0 for every case would pass one
    of these and fail the other.
    """
    config = DMAConfig()

    with pytest.raises(
        ValueError, match=rf"lower_bounds\[{index}\] must be below upper_bounds\[{index}\]"
    ):
        setattr(config, field, value)


def test_gamma_upper_bound_above_one_is_rejected_on_assignment():
    """A blend fraction above 1 is not a fraction. The constructor validates
    it, and so must the assignment path."""
    config = DMAConfig()

    with pytest.raises(ValueError, match=r"gamma_anode_blend2_upper must be within \(0, 1\]"):
        config.gamma_anode_blend2_upper = 1.5


def test_misspelled_speed_preset_is_rejected_on_assignment():
    """'throrough' is a declared field name with an undeclared value, so only
    the value check catches it."""
    config = DMAConfig()

    with pytest.raises(ValueError, match="speed_preset must be 'fast', 'medium', or 'thorough'"):
        config.speed_preset = "throrough"


def test_a_valid_assignment_still_goes_through():
    """The re-validation must not reject the ordinary case; without this the
    tests above would also pass on a __setattr__ that rejects everything."""
    config = DMAConfig()

    config.speed_preset = "fast"
    config.gamma_anode_blend2_upper = 0.5

    assert config.speed_preset == "fast"
    assert config.gamma_anode_blend2_upper == 0.5


@pytest.mark.parametrize(
    "field, value, exc",
    [
        ("speed_preset", "throrough", ValueError),
        ("gamma_anode_blend2_upper", 1.5, ValueError),
        ("algorithm", None, AttributeError),
    ],
    ids=["value-error", "bounds-error", "attribute-error-inside-validate"],
)
def test_a_rejected_assignment_leaves_the_previous_value_in_place(field, value, exc):
    """The assignment happens before the re-validation runs, so a rejected
    value has to be rolled back or a caught exception leaves the configuration
    holding it. ``algorithm = None`` is the case that is not a ``ValueError``:
    it fails inside ``.lower()``, and a rollback that only catches ValueError
    leaves an unreadable ``algorithm`` behind.
    """
    config = DMAConfig()
    before = getattr(config, field)

    with pytest.raises(exc):
        setattr(config, field, value)

    assert getattr(config, field) == before
    # The configuration is still usable afterwards, not just equal-looking.
    config._validate()


# ---------------------------------------------------------------------------
# normalize_roi
# ---------------------------------------------------------------------------


def test_roi_given_in_percent_is_rejected():
    """ROI bounds are SOC fractions. A 10 passed for 10 % would otherwise
    widen the region to everything the grid holds."""
    with pytest.raises(ValueError, match="not percent"):
        normalize_roi(0.1, 10)


def test_overlapping_split_roi_intervals_warn():
    """roi_min carries the first interval and roi_max the second, so a swapped
    or overlapping pair quietly covers the whole range instead of two edges."""
    with pytest.warns(UserWarning, match="Split ROI intervals overlap"):
        intervals = normalize_roi((0.0, 0.5), (0.4, 1.0))

    assert intervals == ((0.0, 0.5), (0.4, 1.0))


def test_disjoint_split_roi_intervals_do_not_warn():
    """The negative control: the LFP-style split is the intended usage and
    must stay quiet, or the warning above carries no information."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intervals = normalize_roi((0.0, 0.15), (0.85, 1.0))

    assert intervals == ((0.0, 0.15), (0.85, 1.0))
    assert [str(w.message) for w in caught] == []
