"""Monotonicity transforms for post-fit silicon-curve sto axes.

PyDMA's silicon-curve generator builds a (V, sto) pair where sto is the
normalised lithiation capacity. Downstream consumers may need that sto
axis strictly monotone — sometimes for an algorithm invariant (PAV's
input/output contract), sometimes for a CSV-backed interpolant in a
composite-electrode solver such as PyBaMM. This module gathers three
related helpers; see each function's docstring for details.

================ =================================================================
Transform        Role
================ =================================================================
_pav_isotonic    Pool-Adjacent-Violators isotonic regression. Front-line
                 monotone filter. Used internally by ``generate_si_curve``.

_collapse_plateaus
                 Drop PAV plateau interiors and separate the endpoints by a
                 bounded shift to get strict sto. Internal helper, opt-in via
                 ``generate_si_curve(collapse_plateaus=True)``.

pchip_resample_for_pybamm
                 PCHIP shape-preserving resample of the raw PAV output onto
                 a uniform sto grid (default 1001 points) with an optional
                 endpoint-V snap. Public — produces a CSV that's
                 well-behaved under CasADi/IDAS interpolation in PyBaMM.
                 Can also be used as a DMAAnalyzer refit input when
                 ``DMAConfig(smoothing_points=1)`` is set (the default 30
                 over-smooths the already-PCHIP-smoothed curve).
================ =================================================================
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

__all__ = ["pchip_resample_for_pybamm"]


def _pav_isotonic(
    q_raw: NDArray[np.floating],
    direction: str = "nondecreasing",
) -> NDArray[np.floating]:
    """Isotonic regression via Pool Adjacent Violators (PAV).

    Enforces monotonicity with minimal L2 change.

    Parameters
    ----------
    q_raw : NDArray
        Input capacity values.
    direction : str
        'nondecreasing' or 'nonincreasing'.

    Returns
    -------
    NDArray
        Monotone capacity values.
    """
    q = q_raw.copy()
    if direction == "nonincreasing":
        q = -q
    n = len(q)
    val = np.zeros(n)
    sz = np.zeros(n, dtype=int)
    nb = 0
    for i in range(n):
        nb += 1
        val[nb - 1] = q[i]
        sz[nb - 1] = 1
        while nb > 1 and val[nb - 2] > val[nb - 1]:
            total = sz[nb - 2] + sz[nb - 1]
            val[nb - 2] = (val[nb - 2] * sz[nb - 2] + val[nb - 1] * sz[nb - 1]) / total
            sz[nb - 2] = total
            nb -= 1
    result = np.empty(n)
    idx = 0
    for k in range(nb):
        result[idx : idx + sz[k]] = val[k]
        idx += sz[k]
    if direction == "nonincreasing":
        result = -result
    return result


def _ulp(value: float) -> float:
    """One unit in the last place of ``value``'s magnitude, always positive.

    ``np.spacing`` returns a NEGATIVE step for a negative argument. The
    internal sign flip in :func:`_collapse_plateaus` that lets a falling
    curve share the rising code path makes every working value negative
    there, so measuring the spacing of the raw value would point the ulp
    comparisons and the repair sweep in the wrong direction. Measuring on
    the magnitude keeps the step positive in both directions and matches
    MATLAB's ``eps(x)``, which is likewise defined on ``abs(x)``.
    """
    return float(np.spacing(abs(float(value))))


def _collapse_plateaus(
    voltage: NDArray[np.floating],
    capacity: NDArray[np.floating],
    eps: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Replace plateaus in ``capacity`` (runs of equal values) with their two
    voltage endpoints, separated by a bounded shift so the result is strictly
    monotonic.

    PAV (isotonic regression) pools violating points to a common mean, which
    yields runs of identical capacity. The resulting (capacity, voltage)
    relation is a function of voltage but not of capacity — multiple voltages
    map to the same SOC. Downstream code that interpolates SOC -> voltage
    needs a strictly monotone capacity axis.

    For each run of length L >= 2 at level q:

    - keep only the first and last index of the run (drop the interior),
    - move the two kept samples to ``q - s`` and ``q + s`` (reversed for a
      non-increasing curve).

    Samples that are not part of a plateau keep their exact capacity, so linear
    interpolation across the original plateau range is preserved to within
    ``s`` and voltage -> capacity consumers see essentially the same curve.

    The shift ``s`` is bounded twice, and both bounds are what keeps the output
    strictly monotone:

    - ``s <= eps``, the global tie-breaking scale, and
    - ``s <=`` a quarter of the distance to the neighbouring plateau levels, so
      the shifted endpoints of two adjacent runs can never meet, however close
      the two levels are.

    Levels closer than eight floating-point ulps of their magnitude are pooled
    into a single run before the shift is computed. At that distance a quarter
    of the gap is no longer representable, so the shifted endpoint would round
    straight back onto its own level and the pair would tie again. Levels that
    close are the same level at machine precision, so the merged run keeps the
    first and the last sample of the whole group.

    A run that sits exactly on the boundary of the capacity range is shifted
    INWARD only: the boundary sample keeps its exact value and its partner
    moves into the range. The output therefore stays inside the input range by
    construction and is never clamped back. Clamping was the origin of a silent
    data loss: it pushed shifted samples back onto the boundary value,
    re-created exact ties there, and let a downstream deduplication drop the
    tied samples together with their voltage support.

    Parameters
    ----------
    voltage : NDArray
        Voltage values, sorted with ``capacity``.
    capacity : NDArray
        Monotone (post-PAV) capacity values, possibly with plateaus.
    eps : float, optional
        Upper bound on the tie-breaking shift. If ``None``, chosen
        automatically as 1e-5 of the capacity range. Robust under cubic-spline
        interpolation downstream while still being numerically tiny. The
        per-run quarter-gap bound applies on top of it.

    Returns
    -------
    tuple[NDArray, NDArray]
        Strictly monotone (voltage, capacity) with plateaus collapsed. Free of
        ties by construction, so a consumer never has to deduplicate.

    Raises
    ------
    ValueError
        If ``voltage`` and ``capacity`` do not hold the same number of
        samples, or if the snapped capacity curve is constant so there is no
        non-degenerate range to collapse onto.
    RuntimeError
        If the constructed output is not strictly monotone, or if it left the
        range of the snapped input curve. Both properties hold by construction,
        so either exception signals a defect in this function rather than an
        unusable input. Raised rather than asserted so the guarantees survive
        ``python -O``.
    """
    v_in: NDArray[np.float64] = np.asarray(voltage, dtype=np.float64).ravel()
    q_in: NDArray[np.float64] = np.asarray(capacity, dtype=np.float64).ravel()
    if v_in.shape != q_in.shape:
        raise ValueError(
            f"voltage and capacity must hold the same number of samples; got "
            f"{v_in.shape} vs {q_in.shape}."
        )
    n = int(q_in.size)
    if n < 2:
        return voltage, capacity

    direction = 1 if q_in[-1] >= q_in[0] else -1

    # PAV-pooled means can drift in floating point so adjacent buckets that
    # should have merged end up slightly violating monotonicity. The cumulative
    # min/max snap forces (non-)monotonicity exactly, so the plateau detection
    # below works on a clean signal. Flipping the sign for a falling curve lets
    # both directions share one code path: q_work is non-decreasing, and the
    # mirrored result is flipped back at the end.
    q_work: NDArray[np.float64] = (
        np.maximum.accumulate(q_in) if direction > 0 else -np.minimum.accumulate(q_in)
    )
    q_work_min = float(q_work[0])
    q_work_max = float(q_work[-1])
    if q_work_max <= q_work_min:
        raise ValueError(
            f"Collapsing plateaus requires a non-degenerate capacity range, but the "
            f"snapped curve is constant at q = {direction * q_work_min!r}."
        )

    q_range = float(q_in.max() - q_in.min())
    eps_shift = q_range * 1e-5 if eps is None else float(eps)
    eps_shift = max(eps_shift, float(np.finfo(np.float64).eps))

    # Runs of equal capacity. Their levels are strictly increasing.
    starts: NDArray[np.intp] = np.concatenate(
        (np.zeros(1, dtype=np.intp), np.flatnonzero(np.diff(q_work) > 0) + 1)
    )
    ends: NDArray[np.intp] = np.concatenate((starts[1:] - 1, np.array([n - 1], dtype=np.intp)))
    levels: NDArray[np.float64] = q_work[starts].astype(np.float64, copy=True)

    # Pool levels that are numerically indistinguishable, so every surviving
    # gap is wide enough for a representable quarter-gap shift. Requiring eight
    # ulps keeps each shifted endpoint at least two ulps away from its own
    # level under round-to-nearest, on both sides of the run.
    keep_run: NDArray[np.bool_] = np.ones(levels.size, dtype=bool)
    reference = float(levels[0])
    for r in range(1, int(levels.size)):
        level_r = float(levels[r])
        if level_r - reference <= 8.0 * max(_ulp(level_r), _ulp(reference)):
            keep_run[r] = False
        else:
            reference = level_r
    if not bool(keep_run.all()):
        kept: NDArray[np.intp] = np.flatnonzero(keep_run)
        merged_ends: NDArray[np.intp] = np.empty(kept.size, dtype=np.intp)
        merged_ends[:-1] = ends[kept[1:] - 1]
        merged_ends[-1] = ends[-1]
        merged_levels: NDArray[np.float64] = levels[kept].astype(np.float64, copy=True)
        # The first group starts at the lower end of the range anyway. The last
        # group is represented by the upper end instead of by its own first
        # level, so both boundary values stay exact and the inward-only shift
        # below still recognises them.
        merged_levels[-1] = levels[-1]
        starts = starts[kept]
        ends = merged_ends
        levels = merged_levels
    n_runs = int(levels.size)

    # Shift budget per run: the global scale, capped at a quarter of the
    # distance to either neighbouring level.
    shift: NDArray[np.float64] = np.full(n_runs, eps_shift, dtype=np.float64)
    if n_runs > 1:
        gaps: NDArray[np.float64] = np.diff(levels)
        shift[:-1] = np.minimum(shift[:-1], 0.25 * gaps)
        shift[1:] = np.minimum(shift[1:], 0.25 * gaps)

    # Keep the first and the last sample of every run and separate them.
    v_out: NDArray[np.float64] = np.empty(2 * n_runs, dtype=np.float64)
    q_out: NDArray[np.float64] = np.empty(2 * n_runs, dtype=np.float64)
    p = 0
    for r in range(n_runs):
        level = float(levels[r])
        s = float(shift[r])
        v_out[p] = v_in[starts[r]]
        if ends[r] == starts[r]:
            q_out[p] = level  # isolated sample: keep it untouched
            p += 1
            continue
        q_out[p] = level if level == q_work_min else level - s
        p += 1
        v_out[p] = v_in[ends[r]]
        q_out[p] = level if level == q_work_max else level + s
        p += 1
    v_out = v_out[:p]
    q_out = q_out[:p]

    # Safety net. After the pooling above every shift is representable and
    # neighbouring output values differ by at least two ulps, so this sweep is
    # not expected to change anything. It steps by a single ulp and the result
    # is not clamped back into the range: a clamp would re-create exactly the
    # ties this function exists to prevent.
    for k in range(1, p):
        if q_out[k] <= q_out[k - 1]:
            q_out[k] = q_out[k - 1] + _ulp(float(q_out[k - 1]))

    # Contract of this function, guaranteed by construction rather than
    # repaired.
    violations: NDArray[np.intp] = np.flatnonzero(np.diff(q_out) <= 0)
    if violations.size:
        i = int(violations[0])
        raise RuntimeError(
            f"Collapsed capacity is not strictly monotone: samples {i} and {i + 1} sit "
            f"at q = {direction * float(q_out[i])!r} and "
            f"{direction * float(q_out[i + 1])!r} (voltages {float(v_out[i])!r} and "
            f"{float(v_out[i + 1])!r})."
        )
    if float(q_out[0]) < q_work_min or float(q_out[-1]) > q_work_max:
        raise RuntimeError(
            f"Collapsed capacity left the range of the input curve: endpoints "
            f"{direction * float(q_out[0])!r} and {direction * float(q_out[-1])!r} are "
            f"outside the snapped range bounded by {direction * q_work_min!r} and "
            f"{direction * q_work_max!r}."
        )

    q_signed: NDArray[np.float64] = q_out if direction > 0 else -q_out
    return v_out, q_signed


def pchip_resample_for_pybamm(
    voltage: NDArray[np.floating],
    sto: NDArray[np.floating],
    *,
    n_points: int = 1001,
    snap_endpoint: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """PCHIP shape-preserving resample of a (V, sto) silicon curve onto a
    uniform sto grid, with optional endpoint snap to the raw minimum V.

    .. warning::

       **This function produces a curve that is pre-smoothed by PCHIP.**

       The output is intended primarily for PyBaMM (or any CasADi/IDAS
       solver) reading a CSV-backed OCP interpolant — see "Why PyBaMM
       needs this" below for the motivation.

       The output can ALSO be used as a :class:`DMAAnalyzer` refit input,
       but only if you set :class:`DMAConfig.smoothing_points = 1` (so
       PyDMA's own LOWESS becomes effective identity). At the documented
       default ``smoothing_points = 30`` the discharge refit basin-escapes
       (RMSE 3.3 -> 21 mV, γ_Si 0.293 -> 0.18) because the PCHIP smoothing
       already removed the V-axis density inside the PAV plateau that the
       30-point LOWESS expects.

    Why PyBaMM needs this
    ---------------------
    PyBaMM's CSV-backed OCP interpolant trips over near-vertical features
    in the OCP table. The M35A silicon delithiation curve carries one such
    feature at sto ≈ 1: the **silicon saturation knee**, where V drops
    from ~0.18 V to ~0.086 V across just the last ~0.1 % of the sto axis
    (~535 raw PAV samples in a tied-sto plateau). With those raw samples
    preserved at native density, the SPM/DFN composite-electrode run
    shows a ~50 mV upward jump in ``V_cell`` as the cell traverses the
    knee during a C/25 discharge — a numerical artefact, not a physical
    feature of the cell. Resampling onto a uniform 1001-point sto grid
    via PCHIP smooths the step over a handful of grid cells, turning the
    visible artefact into a continuous ramp that CasADi/IDAS integrates
    through cleanly (0 upward dV > 5 mV in the SPM trace). PyDMA itself
    never sees the knee as a problem because LOWESS in V-space averages
    the plateau samples to V_min ≈ 0.086 V before they reach the
    optimizer — only the downstream PyBaMM interpolant cares.

    Duplicate sto handling
    ----------------------
    The expected input is the RAW PAV output, i.e.
    ``generate_si_curve(monotone_filter=True, collapse_plateaus=False)``.
    That curve carries genuine plateaus — several per cent of the samples
    share an sto value — and the dedup in step 1 below is the intended
    handling for them. The PCHIP interpolant needs a strictly increasing
    abscissa, and the dropped samples sit inside a plateau whose sto extent
    the retained points still span, while ``snap_endpoint`` restores the
    lowest-V sample explicitly.

    Output of :func:`_collapse_plateaus` is a different case: since PyDMA
    1.1.2 that function is guaranteed tie-free, so passing its result here
    leaves the dedup a no-op. A tie reaching this function from a collapsed
    curve would therefore indicate a defect in :func:`_collapse_plateaus`
    rather than a plateau to be squashed.

    Recipe
    ------
    1. Sort the input (V, sto) pairs by sto, drop duplicate-sto rows
       (keep first occurrence per group).
    2. Fit a :class:`scipy.interpolate.PchipInterpolator` through the
       deduplicated points.
    3. Sample on ``np.linspace(sto.min(), sto.max(), n_points)``.
    4. If ``snap_endpoint=True``, overwrite the final ``v_grid`` value
       with ``min(voltage)`` — the physical silicon-saturation V — so
       the curve preserves the lowest-V endpoint that the dedup step
       and PCHIP smoothing would otherwise round off.

    Parameters
    ----------
    voltage : NDArray
        Silicon OCP voltage samples ``V`` from
        :func:`pydma.silicon.generator.generate_si_curve` with
        ``monotone_filter=True, collapse_plateaus=False``.
    sto : NDArray
        Matching normalised-capacity (stoichiometry) samples. May contain
        duplicates (plateaus) — the dedup step drops them.
    n_points : int, keyword-only, default 1001
        Number of points in the uniform sto grid. 1001 has been the
        operational choice since 2026-04-28; densities up to 20000 give
        identical SPM behaviour (the PCHIP smoothing is the active
        mechanism, not the grid density).
    snap_endpoint : bool, keyword-only, default True
        If True, force ``v_grid[-1] = min(voltage)`` so the
        silicon-saturation V is preserved as the table's last point.

    Returns
    -------
    sto_out : NDArray
        Uniform stoichiometry grid, ``n_points`` long, strictly
        monotone-increasing in ``[sto.min(), sto.max()]``.
    voltage_out : NDArray
        PCHIP-resampled voltage samples matching ``sto_out``. The last
        point is overwritten by ``min(voltage)`` if ``snap_endpoint``.

    Notes
    -----
    The returned arrays serialise cleanly with any writer
    (``np.savetxt``, ``pandas.DataFrame.to_csv``, ...). When writing the
    CSV, document the source and the PyDMA-refit caveat in the comment
    header so downstream users don't accidentally feed it back.

    See also
    --------
    _pav_isotonic, _collapse_plateaus,
    :func:`pydma.silicon.generator.generate_si_curve`
    """
    voltage = np.asarray(voltage, dtype=np.float64)
    sto = np.asarray(sto, dtype=np.float64)
    if voltage.shape != sto.shape:
        raise ValueError(
            f"voltage and sto must have identical shape; got {voltage.shape} " f"vs {sto.shape}."
        )
    if voltage.size < 2:
        raise ValueError(f"Need at least 2 samples; got {voltage.size}.")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError(f"n_points must be an integer >= 2; got {n_points!r}.")

    raw_min_V = float(voltage.min())

    order = np.argsort(sto, kind="stable")
    sto_sorted = sto[order]
    v_sorted = voltage[order]

    # Deduplicate sto: keep first occurrence per duplicate-sto group.
    keep = np.concatenate(([True], np.diff(sto_sorted) > 0))
    sto_dedup = sto_sorted[keep]
    v_dedup = v_sorted[keep]
    if sto_dedup.size < 2:
        raise ValueError(
            "After dedup, fewer than 2 unique sto values remain. " "Input is degenerate."
        )

    pchip = PchipInterpolator(sto_dedup, v_dedup)
    sto_grid = np.linspace(sto_dedup[0], sto_dedup[-1], n_points)
    v_grid = pchip(sto_grid)

    if snap_endpoint:
        v_grid[-1] = raw_min_V

    return sto_grid, v_grid
