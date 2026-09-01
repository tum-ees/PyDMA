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

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, replace

F_FARADAY = 96485.0  # C / mol

# Intercalation hosts land between roughly 1e3 and 1e6 mol/m^3.
_C_MAX_MIN = 1e3
_C_MAX_MAX = 1e6


@dataclass(frozen=True)
class CellGeometry:
    """User-supplied cell geometry — independent of PyDMA."""

    eps_s_neg: float  # negative electrode active-material volume fraction
    eps_s_pos: float  # positive electrode active-material volume fraction
    L_neg: float  # negative electrode thickness [m]
    L_pos: float  # positive electrode thickness [m]
    A: float  # electrode area (width × height, single side) [m^2]
    Q_BoL_Ah: float  # measured BoL capacity from the pOCV PyDMA was fit on [A.h]

    def __post_init__(self) -> None:
        """Reject geometry that cannot describe a cell."""
        for name in ("eps_s_neg", "eps_s_pos", "L_neg", "L_pos", "A", "Q_BoL_Ah"):
            value = float(getattr(self, name))
            if not value > 0:
                raise ValueError(f"CellGeometry.{name} must be positive, got {value!r}.")


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

    c_max_neg: float  # mol / m^3
    c_max_pos: float  # mol / m^3
    sto_window_neg: tuple[float, float]
    sto_window_pos: tuple[float, float]
    util_neg: float  # = sto_window_neg[1] - sto_window_neg[0]
    util_pos: float  # = sto_window_pos[0] - sto_window_pos[1] (positive)
    Q_n: float  # theoretical electrode capacity [A.h] = F·ε·L·A·c_max/3600
    Q_p: float
    Q_Li: float  # cyclable Li at 100 % SOC [A.h]
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
        y_lo, y_hi = self.sto_window_pos  # y_hi at SoC=1 (delithiated, small value)
        return (y_lo + soc * (y_hi - y_lo)) * self.c_max_pos

    # ------------------------------------------------------------------
    # PyBaMM-specific convenience (only this method is PyBaMM-flavoured)
    # ------------------------------------------------------------------
    def pybamm_overrides(self, soc: float = 0.0, include_geometry: bool = True) -> dict:
        """Return a dict ready for ``pybamm.ParameterValues.update(...)``.

        This is a thin convenience that wraps the four concentration scalars
        (``c_max_neg``, ``c_max_pos``, ``c_init_neg(soc)``,
        ``c_init_pos(soc)``) under PyBaMM's exact parameter-name strings.
        For non-PyBaMM simulators, ignore this method and use the
        attributes directly.

        ``c_max`` was solved from ``eps_s``, ``L`` and ``A``, so it is only
        valid together with exactly that geometry: leaving the base parameter
        set's own ``eps_s`` or ``L`` in place rescales the electrode capacity
        away from ``Q_BoL``. ``include_geometry=True`` therefore also writes
        the four ``eps_s`` / thickness values back. The electrode AREA is not
        a single PyBaMM parameter (it is width × height × electrode count), so
        the caller has to set it to the same ``A`` this balancing used.

        Parameters
        ----------
        soc : float
            Cell SOC in [0, 1] the initial concentrations are seeded at.
        include_geometry : bool
            Whether to add the volume fractions and thicknesses that ``c_max``
            is tied to. Pass False only when the base parameter set already
            carries exactly these values.

        Does NOT touch OCPs — the caller supplies those.
        """
        overrides = {
            "Maximum concentration in negative electrode [mol.m-3]": self.c_max_neg,
            "Maximum concentration in positive electrode [mol.m-3]": self.c_max_pos,
            "Initial concentration in negative electrode [mol.m-3]": self.c_init_neg(soc),
            "Initial concentration in positive electrode [mol.m-3]": self.c_init_pos(soc),
        }
        if include_geometry:
            overrides.update(
                {
                    "Negative electrode active material volume fraction": self.eps_s_neg,
                    "Positive electrode active material volume fraction": self.eps_s_pos,
                    "Negative electrode thickness [m]": self.L_neg,
                    "Positive electrode thickness [m]": self.L_pos,
                }
            )
        return overrides


def _read_windows(
    sto_window_neg: Iterable[float],
    sto_window_pos: Iterable[float],
) -> tuple[float, float, float, float, float, float]:
    """Validate a window pair and return (x_lo, x_hi, y_lo, y_hi, util_neg, util_pos)."""
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
    util_pos = y_lo - y_hi  # PyBaMM x_p convention: y_lo (V_min) > y_hi (V_max)
    if util_neg <= 0:
        raise ValueError(
            f"sto_window_neg must be increasing (V_min -> V_max); got " f"[{x_lo}, {x_hi}]"
        )
    if util_pos <= 0:
        raise ValueError(
            f"sto_window_pos must be decreasing (V_min -> V_max) in PyBaMM "
            f"x_p convention; got [{y_lo}, {y_hi}]"
        )
    return x_lo, x_hi, y_lo, y_hi, util_neg, util_pos


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
    This maps a BoL fit. Do NOT reach for an aged CU by scaling ``eps_s_*``
    with ``(1 - lam_*)`` here: this function re-solves ``c_max`` from the same
    ``eps_s``, so the factor cancels exactly in ``c_max``, in ``Q_n`` and in
    ``Q_p``, and the aged cell comes back with its BoL electrode capacities.
    LAM enters through :func:`apply_aging`, which holds the BoL ``c_max``
    fixed and scales only ``eps_s``.
    """
    x_lo, x_hi, y_lo, y_hi, util_neg, util_pos = _read_windows(sto_window_neg, sto_window_pos)

    Q_BoL_C = geometry.Q_BoL_Ah * 3600.0
    denom_neg = F_FARADAY * geometry.eps_s_neg * geometry.L_neg * geometry.A * util_neg
    denom_pos = F_FARADAY * geometry.eps_s_pos * geometry.L_pos * geometry.A * util_pos
    c_max_neg = Q_BoL_C / denom_neg
    c_max_pos = Q_BoL_C / denom_pos

    implausible = [
        (name, value)
        for name, value in (("c_max_neg", c_max_neg), ("c_max_pos", c_max_pos))
        if not _C_MAX_MIN <= value <= _C_MAX_MAX
    ]
    if implausible:
        listed = ", ".join(f"{name}={value:.4g}" for name, value in implausible)
        warnings.warn(
            f"Derived {listed} mol/m^3, outside the {_C_MAX_MIN:g}-{_C_MAX_MAX:g} range "
            "intercalation hosts occupy. Check the geometry units and the "
            "stoichiometry windows.",
            stacklevel=2,
        )

    Q_n = F_FARADAY * geometry.eps_s_neg * geometry.L_neg * geometry.A * c_max_neg / 3600.0
    Q_p = F_FARADAY * geometry.eps_s_pos * geometry.L_pos * geometry.A * c_max_pos / 3600.0
    Q_Li = x_hi * Q_n + y_hi * Q_p  # ESOH: Q_Li = x_100·Q_n + y_100·Q_p

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


def apply_aging(
    bal_bol: ElectrodeBalancing,
    *,
    lam_neg: float,
    lam_pos: float,
    sto_window_neg: Iterable[float],
    sto_window_pos: Iterable[float],
) -> ElectrodeBalancing:
    """Carry a BoL balancing to an aged check-up.

    ``c_max`` is a material property: it does not change as the cell ages, only
    how much of the material is still connected does. So this keeps
    ``c_max_neg`` / ``c_max_pos`` from ``bal_bol`` FIXED, scales the active
    material fractions with ``(1 - lam)``, and adopts the aged stoichiometry
    windows.

    That fixed ``c_max`` is the whole point. :func:`derive_balancing` solves
    ``c_max`` from ``eps_s``, so feeding it an aged ``eps_s`` cancels the
    ``(1 - lam)`` factor: it appears once in the denominator of ``c_max`` and
    once in ``Q_n = F eps_s L A c_max``, and the aged electrode comes back at
    its BoL capacity. Holding ``c_max`` fixed is what makes the LAM visible in
    ``Q_n`` / ``Q_p`` / ``Q_Li``.

    Parameters
    ----------
    bal_bol
        Balancing derived at BoL, the source of the fixed ``c_max`` values and
        of the geometry.
    lam_neg, lam_pos
        Loss of active material per electrode, in [0, 1). Determine these on
        the same anchored window basis as the windows passed below: LAM is a
        ratio of electrode capacities, and each window basis (raw fit,
        voltage-anchored, OCP-model-anchored) sizes those capacities
        differently, so mixing two bases scales the aged electrode by a factor
        that belongs to neither.
    sto_window_neg, sto_window_pos
        The aged CU's stoichiometry windows, same convention as
        :func:`derive_balancing`.

    Returns
    -------
    ElectrodeBalancing
        Aged balancing: BoL ``c_max``, scaled ``eps_s``, aged windows, and
        ``Q_n`` / ``Q_p`` / ``Q_Li`` recomputed from them.
    """
    for name, lam in (("lam_neg", lam_neg), ("lam_pos", lam_pos)):
        if not 0.0 <= float(lam) < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {lam!r}.")

    x_lo, x_hi, y_lo, y_hi, util_neg, util_pos = _read_windows(sto_window_neg, sto_window_pos)

    eps_s_neg = bal_bol.eps_s_neg * (1.0 - float(lam_neg))
    eps_s_pos = bal_bol.eps_s_pos * (1.0 - float(lam_pos))

    Q_n = F_FARADAY * eps_s_neg * bal_bol.L_neg * bal_bol.A * bal_bol.c_max_neg / 3600.0
    Q_p = F_FARADAY * eps_s_pos * bal_bol.L_pos * bal_bol.A * bal_bol.c_max_pos / 3600.0
    Q_Li = x_hi * Q_n + y_hi * Q_p  # ESOH: Q_Li = x_100·Q_n + y_100·Q_p

    return replace(
        bal_bol,
        sto_window_neg=(x_lo, x_hi),
        sto_window_pos=(y_lo, y_hi),
        util_neg=util_neg,
        util_pos=util_pos,
        Q_n=Q_n,
        Q_p=Q_p,
        Q_Li=Q_Li,
        eps_s_neg=eps_s_neg,
        eps_s_pos=eps_s_pos,
    )


def derive_balancing_from_result(
    result,
    geometry: CellGeometry,
    v_min: float | None = None,
    v_max: float | None = None,
    on_out_of_range: str = "raise",
) -> ElectrodeBalancing:
    """Convenience wrapper: pull anchored windows from a :class:`DMAResult`.

    If both ``v_min`` and ``v_max`` are ``None``, the anchoring falls back to
    the range the reconstruction covers, which is what
    :meth:`DMAResult.voltage_anchored_windows` defaults to. Pass the measured
    pOCV voltage cutoffs to anchor at the cell's actual operating window.
    Passing only one of the two raises ``ValueError`` — partial cutoffs aren't
    supported because anchoring is two-sided by definition.

    ``on_out_of_range`` is handed straight to
    :meth:`DMAResult.voltage_anchored_windows`. It defaults to ``"raise"``
    there and here: a cutoff outside the reconstruction silently becomes a
    narrower window under ``"clip"``, and a narrower window inflates ``c_max``.
    Where clipping is intended, ``"clip"`` reports what it did as a warning.
    """
    if (v_min is None) != (v_max is None):
        raise ValueError(
            "Pass both v_min and v_max for voltage anchoring, or neither "
            "(full reconstructed-OCV range). Got "
            f"v_min={v_min!r}, v_max={v_max!r}."
        )
    if v_min is None:
        anchored = result.voltage_anchored_windows(on_out_of_range=on_out_of_range)
    else:
        # XOR check above guarantees v_max is not None when v_min is not None
        if v_max is None:
            raise RuntimeError(
                "v_max is None although v_min is not; the XOR check above missed it."
            )
        anchored = result.voltage_anchored_windows(
            v_min=float(v_min), v_max=float(v_max), on_out_of_range=on_out_of_range
        )

    if anchored.get("clipped_low") or anchored.get("clipped_high"):
        warnings.warn(
            f"Voltage anchoring clipped the requested cutoffs "
            f"[{anchored.get('requested_v_min')!r}, {anchored.get('requested_v_max')!r}] V "
            f"to [{anchored.get('v_min')!r}, {anchored.get('v_max')!r}] V, the range the "
            "reconstruction covers. The narrower window raises c_max.",
            stacklevel=2,
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
    "apply_aging",
    "derive_balancing",
    "derive_balancing_from_result",
]
