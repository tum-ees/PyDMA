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
                 Drop PAV plateau interiors and eps-shift the endpoints to
                 get strict sto. Internal helper, opt-in via
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
    direction: str = 'nondecreasing',
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
    if direction == 'nonincreasing':
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
        result[idx:idx + sz[k]] = val[k]
        idx += sz[k]
    if direction == 'nonincreasing':
        result = -result
    return result


def _collapse_plateaus(
    voltage: NDArray[np.floating],
    capacity: NDArray[np.floating],
    eps: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Replace plateaus in ``capacity`` (runs of equal values) with their two
    voltage endpoints, shifted by ±eps so the result is strictly monotonic.

    PAV (isotonic regression) pools violating points to a common mean, which
    yields runs of identical capacity. The resulting (capacity, voltage)
    relation is a function of voltage but not of capacity — multiple voltages
    map to the same SOC. Downstream code that interpolates SOC -> voltage
    needs a strictly monotone capacity axis.

    For each run of length L >= 2 at value q:
      - keep only the first and last index of the run (drop interior),
      - set their capacity values to ``q - eps`` and ``q + eps`` (or reversed
        for a non-increasing curve), clamped to the original [min, max] so
        the output never extends past the input range.

    Linear interpolation across the original plateau range is preserved within
    eps, so voltage -> capacity consumers see essentially the same curve.

    Parameters
    ----------
    voltage : NDArray
        Voltage values, sorted with ``capacity``.
    capacity : NDArray
        Monotone (post-PAV) capacity values, possibly with plateaus.
    eps : float, optional
        Tie-breaking shift. If ``None``, chosen automatically as 1e-5 of the
        capacity range. Robust under cubic-spline interpolation downstream
        while still being numerically tiny.

    Returns
    -------
    tuple[NDArray, NDArray]
        Strictly monotone (voltage, capacity) with plateaus collapsed.
    """
    n = len(capacity)
    if n < 2:
        return voltage, capacity

    direction = 1 if capacity[-1] >= capacity[0] else -1

    q_min = float(capacity.min())
    q_max = float(capacity.max())
    q_range = q_max - q_min

    if eps is None:
        eps = q_range * 1e-5 if q_range > 0 else 1e-9
    eps = max(float(eps), np.finfo(np.float64).eps)

    # PAV-pooled means can drift in floating point so adjacent buckets that
    # should have merged end up slightly violating monotonicity. Cumulative
    # min/max snap forces (non-)monotonicity exactly so plateau detection
    # below works on a clean signal.
    q_arr = capacity.astype(np.float64)
    q_clean = np.maximum.accumulate(q_arr) if direction > 0 else np.minimum.accumulate(q_arr)

    # Mark interior of plateaus for removal
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and q_clean[j + 1] == q_clean[i]:
            j += 1
        if j > i + 1:
            keep[i + 1:j] = False
        i = j + 1

    v_out = voltage[keep].copy()
    q_out = q_clean[keep].astype(np.float64, copy=True)

    # Break remaining adjacent ties by shifting endpoints in the monotone
    # direction; clamp to the original range so the output never extends
    # past [q_min, q_max].
    m = len(q_out)
    k = 0
    while k < m - 1:
        if q_out[k] == q_out[k + 1]:
            q_out[k] = max(q_min, min(q_max, q_out[k] - direction * eps))
            q_out[k + 1] = max(q_min, min(q_max, q_out[k + 1] + direction * eps))
            k += 2
        else:
            k += 1

    # Final enforcement: forward sweep adds a tiny correction wherever the
    # shift step (combined with clamping at the boundary) left adjacent
    # values weakly ordered. Guarantees strict monotonicity for downstream
    # cubic-spline interpolators without re-arranging the curve shape.
    min_step = max(eps * 0.1, np.finfo(np.float64).eps)
    if direction > 0:
        for k in range(1, m):
            if q_out[k] <= q_out[k - 1]:
                q_out[k] = q_out[k - 1] + min_step
    else:
        for k in range(1, m):
            if q_out[k] >= q_out[k - 1]:
                q_out[k] = q_out[k - 1] - min_step
    q_out = np.clip(q_out, q_min, q_max)

    return v_out, q_out


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
            f"voltage and sto must have identical shape; got {voltage.shape} "
            f"vs {sto.shape}."
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
            "After dedup, fewer than 2 unique sto values remain. "
            "Input is degenerate."
        )

    pchip = PchipInterpolator(sto_dedup, v_dedup)
    sto_grid = np.linspace(sto_dedup[0], sto_dedup[-1], n_points)
    v_grid = pchip(sto_grid)

    if snap_endpoint:
        v_grid[-1] = raw_min_V

    return sto_grid, v_grid
