"""Tests for ``pydma.analysis.degradation`` — the degradation-mode arithmetic
and the masked MSE both the objective and the post-fit metrics run on.

The expected numbers here are worked out by hand from the formula the module
documents, never by calling the function under test, so a sign flip or a
swapped pair of arguments inside ``degradation.py`` shows up as a failure
rather than being reproduced on both sides of the assertion.
"""

import math

import numpy as np
import pytest

from pydma.analysis.degradation import calculate_degradation_modes, calculate_mse

# ---------------------------------------------------------------------------
# A hand-worked non-zero degradation scenario
# ---------------------------------------------------------------------------
#
# Documented formula (degradation.py, "Notes" section of
# calculate_degradation_modes):
#
#     capa_anode     = alpha_an * capa_actual
#     capa_cathode   = alpha_ca * capa_actual
#     capa_inventory = (alpha_ca + beta_ca - beta_an) * capa_actual
#     LAM_an = (capa_anode_init     - capa_anode)     / capa_anode_init
#     LAM_ca = (capa_cathode_init   - capa_cathode)   / capa_cathode_init
#     LLI    = (capa_inventory_init - capa_inventory) / capa_inventory_init
#
# The constants below are chosen so the three modes come out at exactly
# 10 %, 5 % and 3 % — three distinct values, so a swapped return field is a
# failure and not a coincidence:
#
#     capa_anode     = 1.35   * 4.0 = 5.40   -> (6.00 - 5.40) / 6.00 = 0.10
#     capa_cathode   = 1.1875 * 4.0 = 4.75   -> (5.00 - 4.75) / 5.00 = 0.05
#     capa_inventory = (1.1875 - 0.025 + 0.05) * 4.0 = 4.85
#                                            -> (5.00 - 4.85) / 5.00 = 0.03
CAPA_ACTUAL = 4.0
ALPHA_AN = 1.35
BETA_AN = -0.05
ALPHA_CA = 1.1875
BETA_CA = -0.025
CAPA_ANODE_INIT = 6.0
CAPA_CATHODE_INIT = 5.0
CAPA_INVENTORY_INIT = 5.0


def test_degradation_modes_match_the_documented_formula():
    """Each mode equals the loss the docstring defines, recomputed here.

    Every expected value is arithmetic on the inputs, transcribed from the
    documented formula. A sign flip in ``_safe_loss`` (current - init instead
    of init - current), a swap of the anode and cathode capacities, or the
    inventory picking up ``beta_an`` with the wrong sign each break this test:
    the three modes are 10 %, 5 % and 3 %, all distinct and all non-zero.
    """
    params = np.array([ALPHA_AN, BETA_AN, ALPHA_CA, BETA_CA, 0.0, 0.0, 0.0, 0.0])

    result = calculate_degradation_modes(
        params=params,
        capa_actual=CAPA_ACTUAL,
        capa_anode_init=CAPA_ANODE_INIT,
        capa_cathode_init=CAPA_CATHODE_INIT,
        capa_inventory_init=CAPA_INVENTORY_INIT,
    )

    capa_anode = ALPHA_AN * CAPA_ACTUAL
    capa_cathode = ALPHA_CA * CAPA_ACTUAL
    capa_inventory = (ALPHA_CA + BETA_CA - BETA_AN) * CAPA_ACTUAL
    expected_lam_an = (CAPA_ANODE_INIT - capa_anode) / CAPA_ANODE_INIT
    expected_lam_ca = (CAPA_CATHODE_INIT - capa_cathode) / CAPA_CATHODE_INIT
    expected_lli = (CAPA_INVENTORY_INIT - capa_inventory) / CAPA_INVENTORY_INIT

    assert math.isclose(result.lam_anode, expected_lam_an, rel_tol=1e-12)
    assert math.isclose(result.lam_cathode, expected_lam_ca, rel_tol=1e-12)
    assert math.isclose(result.lli, expected_lli, rel_tol=1e-12)

    # The scenario was designed to land on round percentages; pin them so a
    # future edit to the constants above cannot quietly drift off the design.
    assert math.isclose(expected_lam_an, 0.10, abs_tol=1e-12)
    assert math.isclose(expected_lam_ca, 0.05, abs_tol=1e-12)
    assert math.isclose(expected_lli, 0.03, abs_tol=1e-12)


def test_inventory_loss_reads_beta_anode_with_a_minus_sign():
    """``beta_an`` enters the inventory negatively, so making it more negative
    RAISES the inventory and LOWERS the LLI.

    Read off the documented ``capa_inventory = (alpha_ca + beta_ca - beta_an)
    * capa_actual``. A ``+ beta_an`` typo passes the test above only if the
    scenario happens to be symmetric, so the direction is pinned separately.
    """
    beta_an_more_negative = BETA_AN - 0.02
    params = np.array([ALPHA_AN, beta_an_more_negative, ALPHA_CA, BETA_CA, 0.0, 0.0, 0.0, 0.0])

    result = calculate_degradation_modes(
        params=params,
        capa_actual=CAPA_ACTUAL,
        capa_anode_init=CAPA_ANODE_INIT,
        capa_cathode_init=CAPA_CATHODE_INIT,
        capa_inventory_init=CAPA_INVENTORY_INIT,
    )

    capa_inventory = (ALPHA_CA + BETA_CA - beta_an_more_negative) * CAPA_ACTUAL
    expected_lli = (CAPA_INVENTORY_INIT - capa_inventory) / CAPA_INVENTORY_INIT

    assert math.isclose(result.lli, expected_lli, rel_tol=1e-12)
    assert expected_lli < 0.03, "a more negative beta_an must reduce the LLI"


def test_single_component_electrode_puts_the_whole_loss_on_blend1():
    """With both gammas at zero the electrode is single-component: blend1
    carries the whole loss and blend2 none of it.

    ``_safe_loss`` returns 0 for a zero initial sub-capacity, so blend2 is 0
    and blend1 has to reproduce the whole-electrode number exactly.
    """
    params = np.array([ALPHA_AN, BETA_AN, ALPHA_CA, BETA_CA, 0.0, 0.0, 0.0, 0.0])

    result = calculate_degradation_modes(
        params=params,
        capa_actual=CAPA_ACTUAL,
        capa_anode_init=CAPA_ANODE_INIT,
        capa_cathode_init=CAPA_CATHODE_INIT,
        capa_inventory_init=CAPA_INVENTORY_INIT,
        gamma_an_blend2_init=0.0,
        gamma_ca_blend2_init=0.0,
    )

    assert result.lam_anode_blend2 == 0.0
    assert result.lam_cathode_blend2 == 0.0
    assert math.isclose(result.lam_anode_blend1, result.lam_anode, rel_tol=1e-12)
    assert math.isclose(result.lam_cathode_blend1, result.lam_cathode, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# calculate_mse
# ---------------------------------------------------------------------------


def test_masked_mse_averages_over_the_masked_points_only():
    """The divisor is the number of masked points, not the array length.

    measured  = [1, 2, 3, 4], calculated = [1, 4, 3, 8], mask = [T, T, F, F]
    squared errors inside the mask: 0 and 4, so MSE = (0 + 4) / 2 = 2.0.
    Dividing by 4 instead would give 1.0.
    """
    measured = np.array([1.0, 2.0, 3.0, 4.0])
    calculated = np.array([1.0, 4.0, 3.0, 8.0])
    mask = np.array([True, True, False, False])

    assert calculate_mse(measured, calculated, mask) == pytest.approx(2.0, rel=1e-12)


def test_an_integer_mask_selects_the_same_points_as_a_boolean_one():
    """``~mask`` on an integer array is a bitwise complement: 1 becomes -2 and
    0 becomes -1, so the exclusion lands on the last two positions instead of
    on the unmasked ones. Here the single unmasked point is the first, and the
    two readings differ: 4.0 against the correct 1.0.

    measured = 0, calculated = [4, 1, 1, 1, 1], mask keeps points 1..4, so the
    squared errors inside the mask are four times 1 and MSE = 4 / 4 = 1.0.
    """
    measured = np.zeros(5)
    calculated = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
    int_mask = np.array([0, 1, 1, 1, 1])

    assert calculate_mse(measured, calculated, int_mask) == pytest.approx(1.0, rel=1e-12)
    assert calculate_mse(measured, calculated, int_mask.astype(bool)) == pytest.approx(
        1.0, rel=1e-12
    )


def test_empty_mask_is_infinite_not_zero():
    """A fit evaluated over zero points is never a perfect fit."""
    measured = np.array([1.0, 2.0, 3.0])
    calculated = np.array([1.5, 2.5, 3.5])
    mask = np.zeros(3, dtype=bool)

    assert calculate_mse(measured, calculated, mask) == float("inf")


def test_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="must have the same shape"):
        calculate_mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0]))
