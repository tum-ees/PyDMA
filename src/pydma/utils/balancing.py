"""
Electrode balancing export — PyDMA fit + cell geometry → c_max / c_init.

This module is **simulator-agnostic**. The math is universal full-cell
electrochemistry: given a PyDMA voltage-anchored stoichiometry window pair
plus the cell's electrode geometry (active-material volume fraction
``eps_s``, electrode thickness ``L``, electrode area ``A``) and the BoL
capacity ``Q_BoL``, it returns ``c_max`` and ``c_init(SoC)`` for both
electrodes. The same numbers feed any DFN / SPM / SPMe simulator
(PyBaMM, COMSOL Battery Module, hand-rolled Newton solvers, ...).

A small PyBaMM-specific convenience method
(:meth:`ElectrodeBalancing.pybamm_overrides`) wraps the four scalars
in a dict keyed by PyBaMM's exact parameter-name strings, so PyBaMM
users can do ``pv.update(bal.pybamm_overrides(soc=0.0))`` without
typing out the long parameter names. For other simulators, just read
``bal.c_max_neg`` etc. directly.

Math
----
Charge balance for one electrode (single side of the cell):

.. math::

    Q_{BoL}\\,[\\mathrm{A\\cdot h}] \\cdot 3600 / F
        = (\\varepsilon_s\\,L\\,A) \\cdot c_{\\max} \\cdot \\mathrm{util}

where ``util = sto_window[1] - sto_window[0]`` is the dimensionless
width of the lithiation window the cell uses (PyDMA's
``voltage_anchored_windows(...)`` output, anchored at the measured
V_min / V_max). Solving:

.. math::

    c_{\\max} &= \\frac{Q_{BoL} \\cdot 3600}
                       {F\\,\\varepsilon_s\\,L\\,A\\,\\mathrm{util}} \\\\
    c_{init}(\\mathrm{SoC}) &= \\bigl(\\mathrm{sto\\_window}[0]
        + \\mathrm{SoC} \\cdot \\mathrm{util}\\bigr)\\,c_{\\max}

It is the only formulation that uses *only* PyDMA outputs + the geometry
the user already has — no separate literature ``c_max`` input needed.

Cathode convention
------------------
PyDMA's ``voltage_anchored_windows`` already returns the cathode window
in PyBaMM's ``x_p`` convention (1 = fully lithiated at V_min,
0 = fully delithiated at V_max). So ``sto_window_ca[1]`` (the value at
V_max / 100% cell SOC) is the small number, ``sto_window_ca[0]`` is the
large one. We take ``util_pos = sto_window_ca[0] - sto_window_ca[1]``
(a positive number); ``c_init_pos`` correctly decreases with rising SoC.

Composite anodes
----------------
For a Si-Gr blend, the *whole-anode* sto window and util describe a
single effective electrode whose OCP is the blend curve at the fitted
``gamma_Si``. This is what a DFN with a single (averaged) anode OCP
needs. They are *not* the per-phase Gr/Si stoichiometries; for those
use :meth:`pydma.BlendElectrode.get_component_stoichiometry_window`.
"""

from dataclasses import dataclass
from collections.abc import Iterable


F_FARADAY = 96485.0  # C / mol


@dataclass(frozen=True)
class CellGeometry:
    """User-supplied cell geometry — independent of PyDMA."""

    eps_s_neg: float    # negative electrode active-material volume fraction
    eps_s_pos: float    # positive electrode active-material volume fraction
    L_neg: float        # negative electrode thickness [m]
    L_pos: float        # positive electrode thickness [m]
    A: float            # electrode area (width × height, single side) [m^2]
    Q_BoL_Ah: float     # measured BoL capacity from the pOCV PyDMA was fit on [A.h]


@dataclass(frozen=True)
class ElectrodeBalancing:
    """Balancing result: ``c_max`` / ``c_init`` for both electrodes.

    ``c_init_neg(soc)`` and ``c_init_pos(soc)`` are functions of cell SOC
    in [0, 1], so you can re-seed the simulation at any state of charge
    without re-running the derivation.

    Simulator-agnostic. Use the scalars directly for any DFN/SPM
    framework; use :meth:`pybamm_overrides` for the PyBaMM-specific
    parameter-name dict.
    """

    c_max_neg: float                   # mol / m^3
    c_max_pos: float                   # mol / m^3
    sto_window_neg: tuple[float, float]
    sto_window_pos: tuple[float, float]
    util_neg: float                    # = sto_window_neg[1] - sto_window_neg[0]
    util_pos: float                    # = sto_window_pos[0] - sto_window_pos[1] (positive)
    Q_n: float                         # theoretical electrode capacity [A.h] = F·ε·L·A·c_max/3600
    Q_p: float
    Q_Li: float                        # cyclable Li at 100 % SOC [A.h]
    eps_s_neg: float
    eps_s_pos: float
    L_neg: float
    L_pos: float
    A: float

    def c_init_neg(self, soc: float) -> float:
        """Negative-electrode initial concentration [mol/m^3] at the given cell SOC."""
        x_lo, x_hi = self.sto_window_neg
        return (x_lo + soc * (x_hi - x_lo)) * self.c_max_neg

    def c_init_pos(self, soc: float) -> float:
        """Positive-electrode initial concentration [mol/m^3] at the given cell SOC."""
        y_lo, y_hi = self.sto_window_pos     # y_hi at SoC=1 (delithiated, small value)
        return (y_lo + soc * (y_hi - y_lo)) * self.c_max_pos

    # ------------------------------------------------------------------
    # PyBaMM-specific convenience (only this method is PyBaMM-flavoured)
    # ------------------------------------------------------------------
    def pybamm_overrides(self, soc: float = 0.0) -> dict:
        """Return a dict ready for ``pybamm.ParameterValues.update(...)``.

        This is a thin convenience that wraps the four scalars
        (``c_max_neg``, ``c_max_pos``, ``c_init_neg(soc)``,
        ``c_init_pos(soc)``) under PyBaMM's exact parameter-name strings.
        For non-PyBaMM simulators, ignore this method and use the
        attributes directly.

        Does NOT touch OCPs, eps_s, L, A — caller supplies those (they are
        either already in the base parameter set or handled separately).
        """
        return {
            "Maximum concentration in negative electrode [mol.m-3]": self.c_max_neg,
            "Maximum concentration in positive electrode [mol.m-3]": self.c_max_pos,
            "Initial concentration in negative electrode [mol.m-3]": self.c_init_neg(soc),
            "Initial concentration in positive electrode [mol.m-3]": self.c_init_pos(soc),
        }


def derive_balancing(
    *,
    sto_window_neg: Iterable[float],
    sto_window_pos: Iterable[float],
    geometry: CellGeometry,
) -> ElectrodeBalancing:
    """Map a PyDMA voltage-anchored window pair onto ``c_max`` / ``c_init``.

    Parameters
    ----------
    sto_window_neg, sto_window_pos
        Two-element sequences from
        :meth:`pydma.DMAResult.voltage_anchored_windows`. Must use PyDMA's
        PyBaMM-aligned cathode convention (small at V_max, large at V_min)
        — which is what ``voltage_anchored_windows`` already returns.
    geometry
        User-supplied cell geometry + BoL capacity. None of these come
        from PyDMA.

    Returns
    -------
    ElectrodeBalancing
        Simulator-agnostic ``c_max`` / ``c_init`` values plus their
        ingredients.

    Notes
    -----
    For aged CUs, pass that CU's ``voltage_anchored_windows`` output AND
    that CU's ``geometry.eps_s_* * (1 - lam_*)`` as ``eps_s_*``. The
    window already encodes LLI; the ``eps_s`` scaling encodes LAM.
    """
    neg = tuple(sto_window_neg)
    pos = tuple(sto_window_pos)
    if len(neg) != 2 or len(pos) != 2:
        raise ValueError(
            f"sto_window_neg / sto_window_pos must each have exactly 2 entries; "
            f"got {len(neg)} and {len(pos)}"
        )
    x_lo, x_hi = float(neg[0]), float(neg[1])
    y_lo, y_hi = float(pos[0]), float(pos[1])

    util_neg = x_hi - x_lo
    util_pos = y_lo - y_hi   # PyBaMM x_p convention: y_lo (V_min) > y_hi (V_max)
    if util_neg <= 0:
        raise ValueError(
            f"sto_window_neg must be increasing (V_min -> V_max); got "
            f"[{x_lo}, {x_hi}]"
        )
    if util_pos <= 0:
        raise ValueError(
            f"sto_window_pos must be decreasing (V_min -> V_max) in PyBaMM "
            f"x_p convention; got [{y_lo}, {y_hi}]"
        )

    Q_BoL_C = geometry.Q_BoL_Ah * 3600.0
    denom_neg = F_FARADAY * geometry.eps_s_neg * geometry.L_neg * geometry.A * util_neg
    denom_pos = F_FARADAY * geometry.eps_s_pos * geometry.L_pos * geometry.A * util_pos
    c_max_neg = Q_BoL_C / denom_neg
    c_max_pos = Q_BoL_C / denom_pos

    Q_n = F_FARADAY * geometry.eps_s_neg * geometry.L_neg * geometry.A * c_max_neg / 3600.0
    Q_p = F_FARADAY * geometry.eps_s_pos * geometry.L_pos * geometry.A * c_max_pos / 3600.0
    Q_Li = x_hi * Q_n + y_hi * Q_p   # ESOH: Q_Li = x_100·Q_n + y_100·Q_p

    return ElectrodeBalancing(
        c_max_neg=c_max_neg,
        c_max_pos=c_max_pos,
        sto_window_neg=(x_lo, x_hi),
        sto_window_pos=(y_lo, y_hi),
        util_neg=util_neg,
        util_pos=util_pos,
        Q_n=Q_n,
        Q_p=Q_p,
        Q_Li=Q_Li,
        eps_s_neg=geometry.eps_s_neg,
        eps_s_pos=geometry.eps_s_pos,
        L_neg=geometry.L_neg,
        L_pos=geometry.L_pos,
        A=geometry.A,
    )


def derive_balancing_from_result(
    result,
    geometry: CellGeometry,
    v_min: float | None = None,
    v_max: float | None = None,
) -> ElectrodeBalancing:
    """Convenience wrapper: pull anchored windows from a :class:`DMAResult`.

    If both ``v_min`` and ``v_max`` are ``None``, the full reconstructed-OCV
    range is used (no anchoring shrinkage). Pass the measured pOCV voltage
    cutoffs to anchor at the cell's actual operating window. Passing only
    one of the two raises ``ValueError`` — partial cutoffs aren't supported
    because anchoring is two-sided by definition.
    """
    if (v_min is None) != (v_max is None):
        raise ValueError(
            "Pass both v_min and v_max for voltage anchoring, or neither "
            "(full reconstructed-OCV range). Got "
            f"v_min={v_min!r}, v_max={v_max!r}."
        )
    if v_min is None:
        anchored = result.voltage_anchored_windows(on_out_of_range="clip")
    else:
        # XOR check above guarantees v_max is not None when v_min is not None
        assert v_max is not None
        anchored = result.voltage_anchored_windows(
            v_min=float(v_min), v_max=float(v_max), on_out_of_range="clip"
        )
    return derive_balancing(
        sto_window_neg=anchored["sto_window_an"],
        sto_window_pos=anchored["sto_window_ca"],
        geometry=geometry,
    )


__all__ = [
    "F_FARADAY",
    "CellGeometry",
    "ElectrodeBalancing",
    "derive_balancing",
    "derive_balancing_from_result",
]
