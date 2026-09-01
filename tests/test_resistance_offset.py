"""Tests for the fitted series-resistance correction (parameter slot 9).

The correction adds one constant, ``sign(direction) * R * |I|``, to the
reconstructed full-cell OCV. Three things about it can go wrong quietly: the
sign can be taken from nothing, the term can be added where it does not belong
(the derivative curves), and the slot can shift the meaning of a stored
parameter vector. Each of those has a test here, and the last test fits a
synthetic cell whose measurement carries a known offset, so a term that is
declared but not wired through does not pass.
"""

import warnings

import numpy as np
import pytest

from pydma.analysis.dva import precompute_dva
from pydma.analysis.ica import precompute_ica
from pydma.core.analyzer import DMAAnalyzer
from pydma.core.objectives import electrode_potential_on_q, fit_dva, fit_ica, fit_ocv
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.utils.dma_config import DMAConfig
from pydma.utils.results import FittedParams
from pydma.utils.roi import build_roi_mask

# Hand constants for the sign checks: 0.05 Ohm at 0.12 A is 6.0 mV.
R_OHM = 0.05
CURRENT_A = 0.12
OFFSET_V = 0.006

# The synthetic cell the end-to-end tests fit.
TRUE_PARAMS = FittedParams(alpha_an=1.05, beta_an=-0.02, alpha_ca=1.10, beta_ca=-0.05)
CELL_CAPACITY_AH = 2.0
FIT_CURRENT_A = 0.1
FIT_R_OHM = 0.05
FIT_OFFSET_V = 0.005


def _anode() -> ElectrodeOCP:
    """Graphite-like anode: steep at low lithiation, flat in the middle."""
    soc = np.linspace(0.0, 1.0, 401)
    return ElectrodeOCP(
        soc=soc,
        voltage=0.40 * np.exp(-9.0 * soc) + 0.09 + 0.05 * (1.0 - soc),
        name="synthetic anode",
        electrode_type="anode",
    )


def _cathode() -> ElectrodeOCP:
    """Layered-oxide-like cathode, monotonically falling with delithiation."""
    soc = np.linspace(0.0, 1.0, 401)
    return ElectrodeOCP(
        soc=soc,
        voltage=4.25 - 0.85 * soc - 0.25 * soc**3,
        name="synthetic cathode",
        electrode_type="cathode",
    )


def _model_ocv(params: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Full-cell OCV of the synthetic pair without any resistance term."""
    anode_pot = electrode_potential_on_q(
        _anode(),
        q,
        alpha=params[0],
        beta=params[1],
        gamma_blend2=0.0,
        inhom=0.0,
        inhom_offset=0.0,
        is_blend=False,
    )
    cathode_pot = electrode_potential_on_q(
        _cathode(),
        q,
        alpha=params[2],
        beta=params[3],
        gamma_blend2=0.0,
        inhom=0.0,
        inhom_offset=0.0,
        is_blend=False,
    )
    return np.asarray(cathode_pot - anode_pot)


def _params(r_offset: float) -> np.ndarray:
    return np.array([1.05, -0.02, 1.10, -0.05, 0.0, 0.0, 0.0, 0.0, r_offset])


def _fit_config(**overrides) -> DMAConfig:
    """A one-run fit of the synthetic cell.

    ``smoothing_points=1`` makes both smoothing passes the identity, so the
    synthetic measurement reaches the objective as it was built and the exact
    solution is inside the model. The DVA and ICA weights are zero because a
    constant voltage term does not appear in either, so they would only add a
    residual that is blind to the parameter under test.
    """
    settings = dict(
        direction="charge",
        data_length=150,
        smoothing_points=1,
        speed_preset="medium",
        weight_ocv=100.0,
        weight_dva=0.0,
        weight_ica=0.0,
        req_accepted=1,
        max_tries_overall=1,
        rmse_threshold=0.01,
        random_seed=11,
        print_progress=False,
        pocv_current_a=FIT_CURRENT_A,
    )
    settings.update(overrides)
    return DMAConfig(**settings)


def _synthetic_measurement(config: DMAConfig, offset_v: float) -> tuple[np.ndarray, np.ndarray]:
    """Measured (capacity, voltage) of the synthetic cell, lifted by ``offset_v``."""
    generator = DMAAnalyzer(config, anode=_anode(), cathode=_cathode())
    curves = generator.compute_simulated_curves(TRUE_PARAMS, n_points=config.data_length)
    return CELL_CAPACITY_AH * curves["capacity"], curves["voltage"] + offset_v


# ---------------------------------------------------------------------------
# 1) Parameter layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_slots", [4, 5, 6, 7, 8])
def test_a_vector_shorter_than_the_resistance_slot_reads_as_unset(n_slots):
    """A vector that never reaches the slot carries no resistance to report, so
    it reads as 'not fitted' rather than as a fitted zero. ``to_array`` still
    writes 0.0 there, which is what leaves the reconstruction untouched."""
    stored = np.array([1.05, -0.02, 1.10, -0.05, 0.2, 0.3, 0.04, 0.05])[:n_slots]

    params = FittedParams.from_array(stored)

    assert params.r_offset_ohm is None
    np.testing.assert_array_equal(params.to_array()[:n_slots], stored)
    np.testing.assert_array_equal(params.to_array()[n_slots:], np.zeros(9 - n_slots))


def test_a_nine_element_vector_reads_the_resistance_slot():
    stored = np.array([1.05, -0.02, 1.10, -0.05, 0.2, 0.3, 0.04, 0.05, 0.047])

    params = FittedParams.from_array(stored)

    assert params.r_offset_ohm == 0.047
    np.testing.assert_array_equal(params.to_array(), stored)


def test_a_tenth_slot_is_rejected_instead_of_truncated():
    """A longer vector carries something this layout has no name for."""
    with pytest.raises(ValueError, match="at most 9 elements"):
        FittedParams.from_array(np.zeros(10))


def test_the_dict_round_trip_carries_the_resistance():
    params = FittedParams(1.05, -0.02, 1.10, -0.05, r_offset_ohm=0.031)

    restored = FittedParams.from_dict(params.to_dict())

    assert restored.r_offset_ohm == 0.031
    assert FittedParams.from_dict(FittedParams(1.0, 0.0, 1.0, 0.0).to_dict()).r_offset_ohm is None


# ---------------------------------------------------------------------------
# 2) Sign and magnitude
# ---------------------------------------------------------------------------


def test_the_ocv_residual_carries_the_signed_offset():
    """0.05 Ohm at 0.12 A is 6.0 mV, upwards on charge and downwards on discharge.

    The measurement here is the model curve itself, so the whole residual is
    the resistance term and its RMS reads off as a voltage.
    """
    q = np.linspace(0.0, 1.0, 401)
    unshifted = _model_ocv(_params(0.0), q)
    shifted = unshifted + OFFSET_V

    def rms(meas, sign):
        mse = fit_ocv(
            _params(R_OHM),
            anode=_anode(),
            cathode=_cathode(),
            meas_voltage=meas,
            q=q,
            roi_ocv_min=0.0,
            roi_ocv_max=1.0,
            r_offset_sign=sign,
            pocv_current_a=CURRENT_A,
        )
        return float(np.sqrt(mse))

    # Against the unshifted measurement the model is off by exactly the term.
    assert abs(rms(unshifted, 1.0) - OFFSET_V) < 1e-12
    # Against a measurement lifted by the same 6.0 mV, charge cancels it...
    assert rms(shifted, 1.0) < 1e-12
    # ... and discharge doubles it, which is what a flipped sign looks like.
    assert abs(rms(shifted, -1.0) - 2 * OFFSET_V) < 1e-12


@pytest.mark.parametrize(
    "direction, expected",
    [("charge", OFFSET_V), ("discharge", -OFFSET_V)],
)
def test_the_direction_decides_which_way_the_reconstruction_moves(direction, expected):
    """A charge pOCV lies above the OCV and a discharge pOCV below it."""
    config = DMAConfig(direction=direction, pocv_current_a=CURRENT_A, print_progress=False)
    analyzer = DMAAnalyzer(config, anode=_anode(), cathode=_cathode())

    without = analyzer.compute_simulated_curves(TRUE_PARAMS, n_points=201)["voltage"]
    with_offset = analyzer.compute_simulated_curves(
        FittedParams(
            alpha_an=TRUE_PARAMS.alpha_an,
            beta_an=TRUE_PARAMS.beta_an,
            alpha_ca=TRUE_PARAMS.alpha_ca,
            beta_ca=TRUE_PARAMS.beta_ca,
            r_offset_ohm=R_OHM,
        ),
        n_points=201,
    )["voltage"]

    assert np.max(np.abs((with_offset - without) - expected)) < 1e-12


# ---------------------------------------------------------------------------
# 3) The limit is capacity-normalized
# ---------------------------------------------------------------------------


def test_one_limit_gives_every_cell_the_same_voltage_headroom():
    """0.25 Ohm*Ah at C/20 is 12.5 mV of headroom, whatever the cell holds.

    r_max = 0.25 / capa and I = capa / 20, so r_max * I = 0.25 / 20 = 0.0125 V
    and the capacity cancels. A limit stated in plain Ohm would give the small
    cell several times the headroom of the large one.
    """
    for capacity_ah in (2.939, 1.868, 1.215):
        config = DMAConfig(
            allow_resistance_offset=True,
            resistance_offset_limit_ohm_ah=0.25,
            pocv_current_a=capacity_ah / 20.0,
            print_progress=False,
        )

        lb, ub = config.get_full_bounds(capa_actual=capacity_ah)

        assert lb[8] == 0.0
        assert ub[8] == pytest.approx(0.25 / capacity_ah, rel=1e-12)
        assert ub[8] * config.pocv_current_a == pytest.approx(0.0125, abs=1e-12)


def test_the_negative_flag_opens_the_bounds_symmetrically():
    config = DMAConfig(
        allow_resistance_offset=True,
        allow_negative_resistance_offset=True,
        resistance_offset_limit_ohm_ah=0.25,
        pocv_current_a=0.1,
        print_progress=False,
    )

    lb, ub = config.get_full_bounds(capa_actual=2.0)
    init = config.get_initial_guess(capa_actual=2.0)

    assert (lb[8], ub[8]) == (-0.125, 0.125)
    assert init[8] == 0.0
    assert len(init) == 9
    assert config.get_active_param_mask()[8] is True


def test_the_bound_needs_a_capacity():
    config = DMAConfig(allow_resistance_offset=True, pocv_current_a=0.1, print_progress=False)

    with pytest.raises(ValueError, match="capa_actual is required"):
        config.get_full_bounds()


# ---------------------------------------------------------------------------
# 4) The derivative terms do not see the constant
# ---------------------------------------------------------------------------


def test_dva_and_ica_are_bit_identical_with_and_without_the_offset():
    """A constant drops out of a derivative, so both terms must return the
    very same float, not merely a close one."""
    q = np.linspace(0.0, 1.0, 401)
    ocv = _model_ocv(_params(0.0), q)
    meas_dva = precompute_dva(q, ocv, q0=1.0)
    meas_ica = precompute_ica(q, ocv, q0=1.0)
    mask = build_roi_mask(q, 0.1, 0.9)

    common = dict(
        anode=_anode(),
        cathode=_cathode(),
        q=q,
        roi_mask=mask,
        q0=1.0,
    )
    dva_off = fit_dva(_params(0.0), meas_dva=meas_dva, **common)
    dva_on = fit_dva(_params(R_OHM), meas_dva=meas_dva, **common)
    ica_off = fit_ica(_params(0.0), meas_ica=meas_ica, **common)
    ica_on = fit_ica(_params(R_OHM), meas_ica=meas_ica, **common)

    assert dva_on == dva_off
    assert ica_on == ica_off


# ---------------------------------------------------------------------------
# 5) Off by default
# ---------------------------------------------------------------------------


def test_a_default_fit_reports_no_resistance():
    config = _fit_config(speed_preset="fast", pocv_current_a=0.0)
    meas_cap, meas_volt = _synthetic_measurement(config, 0.0)
    analyzer = DMAAnalyzer(config, anode=_anode(), cathode=_cathode())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyzer.analyze(measured_capacity=meas_cap, measured_voltage=meas_volt)

    assert config.allow_resistance_offset is False
    assert result.fitted_params.r_offset_ohm is None
    assert result.config_snapshot["allow_resistance_offset"] is False


def test_an_eight_element_vector_reconstructs_bit_identically_to_a_zeroed_nine():
    """The slot only means something when it is non-zero, so a stored
    8-parameter fit has to give back the very same curve."""
    config = DMAConfig(direction="charge", pocv_current_a=CURRENT_A, print_progress=False)
    analyzer = DMAAnalyzer(config, anode=_anode(), cathode=_cathode())
    eight = np.array([1.05, -0.02, 1.10, -0.05, 0.0, 0.0, 0.0, 0.0])

    from_eight = analyzer.compute_simulated_curves(FittedParams.from_array(eight), n_points=257)
    from_nine = analyzer.compute_simulated_curves(
        FittedParams.from_array(np.append(eight, 0.0)), n_points=257
    )
    unset = FittedParams.from_array(eight)
    unset.r_offset_ohm = None
    from_unset = analyzer.compute_simulated_curves(unset, n_points=257)

    np.testing.assert_array_equal(from_eight["voltage"], from_nine["voltage"])
    np.testing.assert_array_equal(from_eight["voltage"], from_unset["voltage"])
    np.testing.assert_array_equal(from_eight["dva"], from_nine["dva"])


# ---------------------------------------------------------------------------
# 6) The flag needs a current
# ---------------------------------------------------------------------------


def test_the_flag_without_a_current_is_rejected():
    """R and I only ever appear as their product, so a zero current leaves the
    resistance free to be anything at all."""
    with pytest.raises(ValueError, match="not identifiable"):
        DMAConfig(allow_resistance_offset=True, pocv_current_a=0.0)

    config = DMAConfig(pocv_current_a=0.0, print_progress=False)
    with pytest.raises(ValueError, match="not identifiable"):
        config.allow_resistance_offset = True


def test_a_negative_current_is_rejected():
    """The magnitude carries the current and the direction carries the sign,
    so a signed current would apply the direction twice."""
    with pytest.raises(ValueError, match="magnitude in A"):
        DMAConfig(pocv_current_a=-0.1)


# ---------------------------------------------------------------------------
# 7) A resistance on its bound is reported as such
# ---------------------------------------------------------------------------


def test_a_resistance_pinned_to_its_bound_warns():
    """A limit of 0.02 Ohm*Ah on a 2 Ah cell leaves 1.0 mV of headroom against
    a 5 mV offset, so the fit can only run into the bound. A value sitting
    there is a level correction the model cannot otherwise make, not a
    measured resistance, and it has to say so."""
    config = _fit_config(
        allow_resistance_offset=True,
        resistance_offset_limit_ohm_ah=0.02,
    )
    meas_cap, meas_volt = _synthetic_measurement(config, FIT_OFFSET_V)
    analyzer = DMAAnalyzer(config, anode=_anode(), cathode=_cathode())

    with pytest.warns(UserWarning, match="sits on its bound"):
        result = analyzer.analyze(measured_capacity=meas_cap, measured_voltage=meas_volt)

    r_max = 0.02 / CELL_CAPACITY_AH
    assert result.fitted_params.r_offset_ohm == pytest.approx(r_max, rel=1e-3)


# ---------------------------------------------------------------------------
# 8) Acceptance: a known offset comes back out of a fit
# ---------------------------------------------------------------------------


def test_a_known_offset_is_recovered_from_a_synthetic_cell():
    """End-to-end: build a cell from two known half-cells, lift the measurement
    by 5.0 mV, and fit with the correction enabled.

    0.05 Ohm at 0.1 A is that 5.0 mV, so the fit has to come back with the
    resistance. It fails on a flipped sign, which lands near -0.05 Ohm, and on
    a term that is declared but never reaches the objective, which leaves the
    slot free to stop anywhere in its bounds.
    """
    config = _fit_config(
        allow_resistance_offset=True,
        allow_negative_resistance_offset=True,
    )
    meas_cap, meas_volt = _synthetic_measurement(config, FIT_OFFSET_V)
    analyzer = DMAAnalyzer(config, anode=_anode(), cathode=_cathode())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyzer.analyze(measured_capacity=meas_cap, measured_voltage=meas_volt)

    recovered = result.fitted_params.r_offset_ohm
    assert recovered == pytest.approx(FIT_R_OHM, rel=0.10)
    # The shape parameters come back too, so the level did not eat the fit.
    assert result.fitted_params.alpha_an == pytest.approx(TRUE_PARAMS.alpha_an, abs=5e-3)
    assert result.fitted_params.alpha_ca == pytest.approx(TRUE_PARAMS.alpha_ca, abs=5e-3)
