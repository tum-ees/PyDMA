"""Tests for ``pydma.utils.balancing`` — the simulator-agnostic
``c_max`` / ``c_init`` derivation from a PyDMA fit + cell geometry."""

import math

import pytest

from pydma.utils.balancing import (
    F_FARADAY,
    CellGeometry,
    ElectrodeBalancing,
    apply_aging,
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


def _expected_q_electrode(Q_BoL_Ah, F, eps, L, A, util):
    """Theoretical electrode capacity [A.h] = F.eps.L.A.c_max / 3600."""
    return _expected_c_max(Q_BoL_Ah, F, eps, L, A, util) * F * eps * L * A / 3600.0


def test_derive_balancing_matches_closed_form():
    """c_max and c_init match the closed-form formula by hand."""
    geom = _geom()
    sto_neg = (0.05, 0.95)  # util_neg = 0.90
    sto_pos = (0.85, 0.10)  # util_pos = 0.75
    bal = derive_balancing(
        sto_window_neg=sto_neg,
        sto_window_pos=sto_pos,
        geometry=geom,
    )
    util_neg = sto_neg[1] - sto_neg[0]
    util_pos = sto_pos[0] - sto_pos[1]
    expected_c_max_neg = _expected_c_max(
        geom.Q_BoL_Ah,
        F_FARADAY,
        geom.eps_s_neg,
        geom.L_neg,
        geom.A,
        util_neg,
    )
    expected_c_max_pos = _expected_c_max(
        geom.Q_BoL_Ah,
        F_FARADAY,
        geom.eps_s_pos,
        geom.L_pos,
        geom.A,
        util_pos,
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
        sto_window_neg=(0.05, 0.95),
        sto_window_pos=(0.85, 0.10),
        geometry=_geom(),
    )
    overrides = bal.pybamm_overrides(soc=0.5, include_geometry=False)
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


def test_pybamm_overrides_ships_the_geometry_c_max_was_solved_from():
    """c_max only holds together with the eps_s / L it was derived from, so the
    default carries them along."""
    geom = _geom()
    bal = derive_balancing(
        sto_window_neg=(0.05, 0.95),
        sto_window_pos=(0.85, 0.10),
        geometry=geom,
    )
    overrides = bal.pybamm_overrides(soc=0.5)
    assert overrides["Negative electrode active material volume fraction"] == geom.eps_s_neg
    assert overrides["Positive electrode active material volume fraction"] == geom.eps_s_pos
    assert overrides["Negative electrode thickness [m]"] == geom.L_neg
    assert overrides["Positive electrode thickness [m]"] == geom.L_pos


def test_negative_window_must_be_increasing():
    with pytest.raises(ValueError, match="sto_window_neg must be increasing"):
        derive_balancing(
            sto_window_neg=(0.95, 0.05),  # decreasing -> util_neg <= 0
            sto_window_pos=(0.85, 0.10),
            geometry=_geom(),
        )


def test_positive_window_must_be_decreasing_in_x_p_convention():
    with pytest.raises(ValueError, match="sto_window_pos must be decreasing"):
        derive_balancing(
            sto_window_neg=(0.05, 0.95),
            sto_window_pos=(0.10, 0.85),  # increasing -> util_pos <= 0 in x_p
            geometry=_geom(),
        )


def test_window_length_must_be_two():
    with pytest.raises(ValueError, match="must each have exactly 2 entries"):
        derive_balancing(
            sto_window_neg=(0.05, 0.50, 0.95),  # three entries
            sto_window_pos=(0.85, 0.10),
            geometry=_geom(),
        )


def test_generator_input_works():
    """Iterable typing + tuple coercion should accept generators."""
    geom = _geom()
    bal_seq = derive_balancing(
        sto_window_neg=(0.05, 0.95),
        sto_window_pos=(0.85, 0.10),
        geometry=geom,
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
    assert fake.calls == [{"v_min": None, "v_max": None, "on_out_of_range": "raise"}]
    assert math.isclose(bal.util_neg, 0.90, rel_tol=1e-12)


def test_derive_balancing_from_result_anchored():
    fake = _FakeResult((0.05, 0.95), (0.85, 0.10))
    bal = derive_balancing_from_result(fake, _geom(), v_min=2.5, v_max=4.2)
    assert fake.calls == [{"v_min": 2.5, "v_max": 4.2, "on_out_of_range": "raise"}]
    assert isinstance(bal, ElectrodeBalancing)


def test_derive_balancing_from_result_passes_clip_policy_through():
    fake = _FakeResult((0.05, 0.95), (0.85, 0.10))
    derive_balancing_from_result(fake, _geom(), v_min=2.5, v_max=4.2, on_out_of_range="clip")
    assert fake.calls == [{"v_min": 2.5, "v_max": 4.2, "on_out_of_range": "clip"}]


@pytest.mark.parametrize(
    "v_min, v_max",
    [(2.5, None), (None, 4.2)],
)
def test_partial_voltage_cutoff_raises(v_min, v_max):
    fake = _FakeResult((0.05, 0.95), (0.85, 0.10))
    with pytest.raises(ValueError, match="Pass both v_min and v_max"):
        derive_balancing_from_result(fake, _geom(), v_min=v_min, v_max=v_max)


def test_q_li_consistency_with_x_100_y_100():
    """Q_Li = x_100 * Q_n + y_100 * Q_p (ESOH identity).

    Both electrode capacities are rebuilt here from the geometry and the sto
    windows through the same closed form ``_expected_c_max`` uses, so the
    identity is checked against the cell the balancing was derived from rather
    than against the object's own ``Q_n`` / ``Q_p``. Reading those back would
    make the assertion true whatever ``Q_Li`` was assembled from.
    """
    geom = _geom()
    sto_neg = (0.05, 0.95)
    sto_pos = (0.85, 0.10)
    util_neg = sto_neg[1] - sto_neg[0]
    util_pos = sto_pos[0] - sto_pos[1]
    bal = derive_balancing(
        sto_window_neg=sto_neg,
        sto_window_pos=sto_pos,
        geometry=geom,
    )

    expected_Q_n = _expected_q_electrode(
        geom.Q_BoL_Ah, F_FARADAY, geom.eps_s_neg, geom.L_neg, geom.A, util_neg
    )
    expected_Q_p = _expected_q_electrode(
        geom.Q_BoL_Ah, F_FARADAY, geom.eps_s_pos, geom.L_pos, geom.A, util_pos
    )
    # F.eps.L.A cancels against the c_max denominator, so the theoretical
    # electrode capacity is the BoL capacity spread over the window it uses.
    assert math.isclose(expected_Q_n, geom.Q_BoL_Ah / util_neg, rel_tol=1e-12)
    assert math.isclose(expected_Q_p, geom.Q_BoL_Ah / util_pos, rel_tol=1e-12)
    assert math.isclose(bal.Q_n, expected_Q_n, rel_tol=1e-12)
    assert math.isclose(bal.Q_p, expected_Q_p, rel_tol=1e-12)

    expected_Q_Li = sto_neg[1] * expected_Q_n + sto_pos[1] * expected_Q_p
    assert math.isclose(bal.Q_Li, expected_Q_Li, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# apply_aging: BoL c_max stays, eps_s carries the LAM
# ---------------------------------------------------------------------------

BOL_STO_NEG = (0.05, 0.95)  # util_neg = 0.90
BOL_STO_POS = (0.85, 0.10)  # util_pos = 0.75
AGED_STO_NEG = (0.08, 0.90)  # util_neg = 0.82
AGED_STO_POS = (0.80, 0.15)  # util_pos = 0.65
LAM_NEG = 0.2
LAM_POS = 0.05


def _bol_balancing():
    return derive_balancing(
        sto_window_neg=BOL_STO_NEG,
        sto_window_pos=BOL_STO_POS,
        geometry=_geom(),
    )


def test_apply_aging_keeps_c_max_and_moves_the_lam_into_eps_s():
    """c_max is a material property, so ageing must leave it untouched.

    That is the whole point of the function: re-deriving c_max from an aged
    eps_s cancels the (1 - lam) factor and hands back the BoL electrode
    capacity. Holding c_max fixed is what makes the LAM visible in Q_n / Q_p.
    The two LAM values differ, so swapping lam_neg and lam_pos fails here.
    """
    geom = _geom()
    bol = _bol_balancing()

    aged = apply_aging(
        bol,
        lam_neg=LAM_NEG,
        lam_pos=LAM_POS,
        sto_window_neg=AGED_STO_NEG,
        sto_window_pos=AGED_STO_POS,
    )

    # (i) the core property: c_max unchanged bit for bit, not merely close.
    assert aged.c_max_neg == bol.c_max_neg
    assert aged.c_max_pos == bol.c_max_pos

    # (ii) the active-material fractions carry the loss; the rest of the
    # geometry is untouched.
    assert math.isclose(aged.eps_s_neg, geom.eps_s_neg * (1.0 - LAM_NEG), rel_tol=1e-12)
    assert math.isclose(aged.eps_s_pos, geom.eps_s_pos * (1.0 - LAM_POS), rel_tol=1e-12)
    assert aged.L_neg == geom.L_neg
    assert aged.L_pos == geom.L_pos
    assert aged.A == geom.A


def test_apply_aging_recomputes_q_from_the_aged_windows():
    """Q_n / Q_p follow from the aged eps_s and the BoL c_max, and Q_Li from
    the aged windows. Every expected value is rebuilt from the geometry."""
    geom = _geom()
    bol = _bol_balancing()

    aged = apply_aging(
        bol,
        lam_neg=LAM_NEG,
        lam_pos=LAM_POS,
        sto_window_neg=AGED_STO_NEG,
        sto_window_pos=AGED_STO_POS,
    )

    util_neg_bol = BOL_STO_NEG[1] - BOL_STO_NEG[0]
    util_pos_bol = BOL_STO_POS[0] - BOL_STO_POS[1]
    c_max_neg = _expected_c_max(
        geom.Q_BoL_Ah, F_FARADAY, geom.eps_s_neg, geom.L_neg, geom.A, util_neg_bol
    )
    c_max_pos = _expected_c_max(
        geom.Q_BoL_Ah, F_FARADAY, geom.eps_s_pos, geom.L_pos, geom.A, util_pos_bol
    )
    eps_s_neg_aged = geom.eps_s_neg * (1.0 - LAM_NEG)
    eps_s_pos_aged = geom.eps_s_pos * (1.0 - LAM_POS)
    expected_Q_n = F_FARADAY * eps_s_neg_aged * geom.L_neg * geom.A * c_max_neg / 3600.0
    expected_Q_p = F_FARADAY * eps_s_pos_aged * geom.L_pos * geom.A * c_max_pos / 3600.0

    # The same number from the other direction: F.eps.L.A cancels, leaving the
    # BoL capacity over its own window, reduced by the LAM.
    assert math.isclose(expected_Q_n, geom.Q_BoL_Ah * (1.0 - LAM_NEG) / util_neg_bol, rel_tol=1e-12)
    assert math.isclose(expected_Q_p, geom.Q_BoL_Ah * (1.0 - LAM_POS) / util_pos_bol, rel_tol=1e-12)

    assert math.isclose(aged.Q_n, expected_Q_n, rel_tol=1e-12)
    assert math.isclose(aged.Q_p, expected_Q_p, rel_tol=1e-12)
    # The LAM really is visible, and it is the electrode's own LAM: a swap
    # would put 0.95 on the negative electrode instead of 0.80.
    assert math.isclose(aged.Q_n, bol.Q_n * (1.0 - LAM_NEG), rel_tol=1e-12)
    assert not math.isclose(aged.Q_n, bol.Q_n * (1.0 - LAM_POS), rel_tol=1e-3)

    # The aged windows are adopted, and Q_Li follows from them.
    assert aged.sto_window_neg == AGED_STO_NEG
    assert aged.sto_window_pos == AGED_STO_POS
    assert math.isclose(aged.util_neg, AGED_STO_NEG[1] - AGED_STO_NEG[0], rel_tol=1e-12)
    assert math.isclose(aged.util_pos, AGED_STO_POS[0] - AGED_STO_POS[1], rel_tol=1e-12)
    assert math.isclose(
        aged.Q_Li,
        AGED_STO_NEG[1] * expected_Q_n + AGED_STO_POS[1] * expected_Q_p,
        rel_tol=1e-12,
    )


@pytest.mark.parametrize(
    "lam_neg, lam_pos",
    [(-0.1, 0.0), (1.0, 0.0), (0.0, 1.5)],
    ids=["negative", "total-loss", "above-one"],
)
def test_apply_aging_rejects_lam_outside_the_unit_interval(lam_neg, lam_pos):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        apply_aging(
            _bol_balancing(),
            lam_neg=lam_neg,
            lam_pos=lam_pos,
            sto_window_neg=AGED_STO_NEG,
            sto_window_pos=AGED_STO_POS,
        )
