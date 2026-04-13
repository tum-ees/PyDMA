from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydma.analysis.degradation import calculate_degradation_modes
from pydma.core.analyzer import DMAAnalyzer
from pydma.core.objectives import apply_params_to_electrode
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.electrodes.inhomogeneity import calculate_inhomogeneity, get_inhomogeneity_distribution
from pydma.utils.results import FittedParams, ReferenceData


def test_compare_with_reference_uses_shared_degradation_calculation():
    analyzer = DMAAnalyzer()

    reference_params = FittedParams(
        alpha_an=1.2,
        beta_an=-0.1,
        alpha_ca=1.1,
        beta_ca=-0.05,
        gamma_blend2_an=0.25,
        gamma_blend2_ca=0.4,
    )
    current_params = FittedParams(
        alpha_an=1.1,
        beta_an=-0.08,
        alpha_ca=1.0,
        beta_ca=-0.04,
        gamma_blend2_an=0.2,
        gamma_blend2_ca=0.35,
    )

    analyzer.reference_data = ReferenceData(
        capa_anode_init=reference_params.alpha_an * 5.0,
        capa_cathode_init=reference_params.alpha_ca * 5.0,
        capa_inventory_init=(
            reference_params.alpha_ca + reference_params.beta_ca - reference_params.beta_an
        )
        * 5.0,
        gamma_an_blend2_init=reference_params.gamma_blend2_an,
        gamma_ca_blend2_init=reference_params.gamma_blend2_ca,
        reference_capacity=5.0,
    )

    result = analyzer.compare_with_reference(
        reference_params=reference_params,
        current_params=current_params,
        current_capacity=4.5,
    )

    expected = calculate_degradation_modes(
        params=np.array([1.1, -0.08, 1.0, -0.04, 0.2, 0.35, 0.0, 0.0], dtype=float),
        capa_actual=4.5,
        capa_anode_init=reference_params.alpha_an * 5.0,
        capa_cathode_init=reference_params.alpha_ca * 5.0,
        capa_inventory_init=(
            reference_params.alpha_ca + reference_params.beta_ca - reference_params.beta_an
        )
        * 5.0,
        gamma_an_blend2_init=reference_params.gamma_blend2_an,
        gamma_ca_blend2_init=reference_params.gamma_blend2_ca,
    )

    np.testing.assert_allclose(
        [
            result.lli,
            result.lam_anode,
            result.lam_cathode,
            result.lam_anode_blend1,
            result.lam_anode_blend2,
            result.lam_cathode_blend1,
            result.lam_cathode_blend2,
            result.capacity_loss,
        ],
        [
            expected.lli,
            expected.lam_anode,
            expected.lam_cathode,
            expected.lam_anode_blend1,
            expected.lam_anode_blend2,
            expected.lam_cathode_blend1,
            expected.lam_cathode_blend2,
            0.1,
        ],
    )


def test_reset_state_clears_capacity_warning_state():
    analyzer = DMAAnalyzer()
    analyzer._capacity_history = [1.0, 1.0]
    analyzer._normalized_soc_warning_issued = True

    analyzer.reset_state()

    assert analyzer._capacity_history == []
    assert analyzer._normalized_soc_warning_issued is False


def test_fitted_params_to_array_converts_optional_none_values_to_zero():
    params = FittedParams(
        alpha_an=1.0,
        beta_an=0.0,
        alpha_ca=1.1,
        beta_ca=-0.1,
        gamma_blend2_an=None,
        gamma_blend2_ca=None,
        inhom_an=None,
        inhom_ca=None,
    )

    arr = params.to_array()

    assert arr.dtype.kind == "f"
    np.testing.assert_allclose(arr, np.array([1.0, 0.0, 1.1, -0.1, 0.0, 0.0, 0.0, 0.0]))


def test_prepare_measured_data_keeps_matlab_q0_on_normalized_soc_axis():
    analyzer = DMAAnalyzer()
    cap = np.array([0.0, 2.1, 4.2])
    volt = np.array([3.0, 3.5, 4.0])

    _, _, q0, cap_span = analyzer._prepare_measured_data(cap, volt)

    assert np.isclose(q0, 1.0)
    assert np.isclose(cap_span, 4.2)


def test_calculate_inhomogeneity_matches_matlab_offset_formula():
    soc = np.linspace(0.0, 1.0, 11)
    voltage = soc**2
    sigma = 0.12
    offset = 0.35

    x, weights = get_inhomogeneity_distribution(sigma)
    x_dev = x - 1.0
    alpha_eff = offset + (1.0 - offset) * soc
    x_query = soc[:, None] + alpha_eff[:, None] * x_dev[None, :]
    expected = np.column_stack(
        [
            np.interp(x_query[:, j], soc, voltage, left=voltage[0], right=voltage[-1])
            for j in range(len(x))
        ]
    ) @ weights

    actual = calculate_inhomogeneity(
        soc,
        voltage,
        sigma,
        inhom_offset_fraction=offset,
    )

    np.testing.assert_allclose(actual, expected)


def test_apply_params_to_electrode_passes_inhomogeneity_offset():
    electrode = ElectrodeOCP(
        soc=np.linspace(0.0, 1.0, 21),
        voltage=np.linspace(0.1, 0.2, 21) ** 2,
        electrode_type="anode",
    )

    soc_out, voltage_out = apply_params_to_electrode(
        electrode,
        alpha=1.0,
        beta=0.0,
        inhom=0.1,
        inhom_offset=0.4,
    )
    expected_voltage = calculate_inhomogeneity(
        electrode.soc,
        electrode.voltage,
        0.1,
        inhom_offset_fraction=0.4,
    )

    np.testing.assert_allclose(soc_out, electrode.soc)
    np.testing.assert_allclose(voltage_out, expected_voltage)
