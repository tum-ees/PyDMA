"""Tests for ``pydma.utils.balancing`` — the simulator-agnostic
``c_max`` / ``c_init`` derivation from a PyDMA fit + cell geometry."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydma.utils.balancing import (
    F_FARADAY,
    CellGeometry,
    ElectrodeBalancing,
    derive_balancing,
    derive_balancing_from_result,
)


def _geom():
    """Stable illustrative geometry — same numbers used in the upstream tutorial."""
    return CellGeometry(
        eps_s_neg=0.775,
        eps_s_pos=0.719,
        L_neg=50.0e-6,
        L_pos=42.0e-6,
        A=1.294 * 2 * 0.063,
        Q_BoL_Ah=4.4707,
    )


def _expected_c_max(Q_BoL_Ah, F, eps, L, A, util):
    return Q_BoL_Ah * 3600.0 / (F * eps * L * A * util)


def test_derive_balancing_matches_closed_form():
    """c_max and c_init match the closed-form formula by hand."""
    geom = _geom()
    sto_neg = (0.05, 0.95)        # util_neg = 0.90
    sto_pos = (0.85, 0.10)        # util_pos = 0.75
    bal = derive_balancing(
        sto_window_neg=sto_neg, sto_window_pos=sto_pos, geometry=geom,
    )
    util_neg = sto_neg[1] - sto_neg[0]
    util_pos = sto_pos[0] - sto_pos[1]
    expected_c_max_neg = _expected_c_max(
        geom.Q_BoL_Ah, F_FARADAY, geom.eps_s_neg, geom.L_neg, geom.A, util_neg,
    )
    expected_c_max_pos = _expected_c_max(
        geom.Q_BoL_Ah, F_FARADAY, geom.eps_s_pos, geom.L_pos, geom.A, util_pos,
    )
    assert math.isclose(bal.c_max_neg, expected_c_max_neg, rel_tol=1e-12)
    assert math.isclose(bal.c_max_pos, expected_c_max_pos, rel_tol=1e-12)
    assert math.isclose(bal.util_neg, util_neg, rel_tol=1e-12)
    assert math.isclose(bal.util_pos, util_pos, rel_tol=1e-12)
    # c_init at SoC=0 sits at the V_min endpoint of each window
    assert math.isclose(bal.c_init_neg(0.0), sto_neg[0] * bal.c_max_neg, rel_tol=1e-12)
    assert math.isclose(bal.c_init_pos(0.0), sto_pos[0] * bal.c_max_pos, rel_tol=1e-12)
    # c_init at SoC=1 sits at the V_max endpoint
    assert math.isclose(bal.c_init_neg(1.0), sto_neg[1] * bal.c_max_neg, rel_tol=1e-12)
    assert math.isclose(bal.c_init_pos(1.0), sto_pos[1] * bal.c_max_pos, rel_tol=1e-12)


def test_pybamm_overrides_returns_canonical_keys():
    bal = derive_balancing(
        sto_window_neg=(0.05, 0.95), sto_window_pos=(0.85, 0.10), geometry=_geom(),
    )
    overrides = bal.pybamm_overrides(soc=0.5)
    assert set(overrides) == {
        "Maximum concentration in negative electrode [mol.m-3]",
        "Maximum concentration in positive electrode [mol.m-3]",
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
    }
    assert overrides["Maximum concentration in negative electrode [mol.m-3]"] == bal.c_max_neg
    assert overrides["Maximum concentration in positive electrode [mol.m-3]"] == bal.c_max_pos
    assert overrides["Initial concentration in negative electrode [mol.m-3]"] == bal.c_init_neg(0.5)
    assert overrides["Initial concentration in positive electrode [mol.m-3]"] == bal.c_init_pos(0.5)


def test_negative_window_must_be_increasing():
    with pytest.raises(ValueError, match="sto_window_neg must be increasing"):
        derive_balancing(
            sto_window_neg=(0.95, 0.05),         # decreasing -> util_neg <= 0
            sto_window_pos=(0.85, 0.10),
            geometry=_geom(),
        )


def test_positive_window_must_be_decreasing_in_x_p_convention():
    with pytest.raises(ValueError, match="sto_window_pos must be decreasing"):
        derive_balancing(
            sto_window_neg=(0.05, 0.95),
            sto_window_pos=(0.10, 0.85),         # increasing -> util_pos <= 0 in x_p
            geometry=_geom(),
        )


def test_window_length_must_be_two():
    with pytest.raises(ValueError, match="must each have exactly 2 entries"):
        derive_balancing(
            sto_window_neg=(0.05, 0.50, 0.95),   # three entries
            sto_window_pos=(0.85, 0.10),
            geometry=_geom(),
        )


def test_generator_input_works():
    """Iterable typing + tuple coercion should accept generators."""
    geom = _geom()
    bal_seq = derive_balancing(
        sto_window_neg=(0.05, 0.95), sto_window_pos=(0.85, 0.10), geometry=geom,
    )
    bal_gen = derive_balancing(
        sto_window_neg=(x for x in (0.05, 0.95)),
        sto_window_pos=(x for x in (0.85, 0.10)),
        geometry=geom,
    )
    assert math.isclose(bal_gen.c_max_neg, bal_seq.c_max_neg, rel_tol=1e-12)
    assert math.isclose(bal_gen.c_max_pos, bal_seq.c_max_pos, rel_tol=1e-12)


class _FakeResult:
    """Minimal stand-in for DMAResult — only `voltage_anchored_windows` is needed."""

    def __init__(self, sto_window_an, sto_window_ca):
        self._an = tuple(sto_window_an)
        self._ca = tuple(sto_window_ca)
        self.calls = []

    def voltage_anchored_windows(self, v_min=None, v_max=None, on_out_of_range="raise"):
        self.calls.append({"v_min": v_min, "v_max": v_max, "on_out_of_range": on_out_of_range})
        return {"sto_window_an": self._an, "sto_window_ca": self._ca}


def test_derive_balancing_from_result_full_range():
    fake = _FakeResult((0.05, 0.95), (0.85, 0.10))
    bal = derive_balancing_from_result(fake, _geom())
    # Full-range path: voltage_anchored_windows called without v_min/v_max
    assert fake.calls == [{"v_min": None, "v_max": None, "on_out_of_range": "clip"}]
    assert math.isclose(bal.util_neg, 0.90, rel_tol=1e-12)


def test_derive_balancing_from_result_anchored():
    fake = _FakeResult((0.05, 0.95), (0.85, 0.10))
    bal = derive_balancing_from_result(fake, _geom(), v_min=2.5, v_max=4.2)
    assert fake.calls == [{"v_min": 2.5, "v_max": 4.2, "on_out_of_range": "clip"}]
    assert isinstance(bal, ElectrodeBalancing)


@pytest.mark.parametrize(
    "v_min, v_max",
    [(2.5, None), (None, 4.2)],
)
def test_partial_voltage_cutoff_raises(v_min, v_max):
    fake = _FakeResult((0.05, 0.95), (0.85, 0.10))
    with pytest.raises(ValueError, match="Pass both v_min and v_max"):
        derive_balancing_from_result(fake, _geom(), v_min=v_min, v_max=v_max)


def test_q_li_consistency_with_x_100_y_100():
    """Q_Li = x_100 * Q_n + y_100 * Q_p (ESOH identity)."""
    geom = _geom()
    sto_neg = (0.05, 0.95)
    sto_pos = (0.85, 0.10)
    bal = derive_balancing(
        sto_window_neg=sto_neg, sto_window_pos=sto_pos, geometry=geom,
    )
    expected_Q_Li = sto_neg[1] * bal.Q_n + sto_pos[1] * bal.Q_p
    assert math.isclose(bal.Q_Li, expected_Q_Li, rel_tol=1e-12)
