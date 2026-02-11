"""Plotting functions for DMA results.

This module provides visualization functions for:
- OCV, DVA, and ICA comparisons
- Degradation mode tracking over aging
- Parameter evolution plots
- Multi-panel result summaries
- MATLAB-style OCV+DVA composite figures with alpha/beta annotations
  drawn as arrows on electrode reconstruction curves

Conventions used in this module:
- For the OCV+DVA view, alpha and beta labels are shown on
  the reconstructed electrode curves and always refer to full-cell SOC in
  the current aging state (current CU).
- Figure layout, limits, label positioning, and colors are handled here so
  notebook and script users get consistent publication-style output.
"""

from typing import Sequence
import re
import warnings
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pydma.analysis.dva import calculate_dva
from pydma.utils.results import DMAResult, DegradationModes, AgingStudyResults


# TUM colors (from TUM_colors.m)
TUM_COLORS = {
    'blue': np.array([48, 112, 179]) / 255,      # #3070B3 - TUM main blue (tumBlue)
    'blue_dark': np.array([7, 33, 64]) / 255,    # #072140 - tumBlueDark
    'dark_blue': np.array([0, 82, 147]) / 255,   # #005293
    'light_blue': np.array([100, 160, 200]) / 255,  # #64A0C8
    'lighter_blue': np.array([152, 198, 234]) / 255,  # #98C6EA
    'orange': np.array([227, 114, 34]) / 255,    # #E37222
    'green': np.array([162, 173, 0]) / 255,      # #A2AD00
    'black': np.array([0, 0, 0]) / 255,
    'gray': np.array([153, 153, 153]) / 255,     # #999999
    'medium_gray': np.array([106, 117, 126]) / 255,  # #6A757E
    'dark_gray': np.array([71, 80, 88]) / 255,   # #475058
    'light_gray': np.array([218, 215, 203]) / 255,   # #DAD7CB
}


def _setup_style(latex_fonts: bool = False, use_tex: bool = False) -> None:
    """Set up matplotlib style for DMA plots.

    Parameters
    ----------
    latex_fonts : bool, optional
        If True, use a LaTeX-like serif font stack for all labels and text
        without requiring a full LaTeX installation.
    use_tex : bool, optional
        If True, enable full LaTeX rendering via ``text.usetex``.
        Requires a working TeX installation in the runtime environment.
    """
    style = {
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    }
    if latex_fonts:
        style.update(
            {
                'font.family': 'serif',
                'font.serif': [
                    'Computer Modern Roman',
                    'CMU Serif',
                    'Times New Roman',
                    'DejaVu Serif',
                ],
                'mathtext.fontset': 'cm',
                # Keep math symbols italic (U, Q, C, etc.); use \mathrm{...}
                # in labels for upright text/units where needed.
                'mathtext.default': 'it',
            }
        )
    style['text.usetex'] = bool(use_tex)
    plt.rcParams.update(style)


def _first_nonempty_array(
    *arrays: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Return the first non-empty array from a candidate list."""
    for arr in arrays:
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=np.float64).flatten()
        if arr_np.size > 0:
            return arr_np
    return np.array([], dtype=np.float64)


def _concat_finite_arrays(
    *arrays: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Concatenate all finite values from multiple arrays."""
    finite_parts: list[NDArray[np.floating]] = []
    for arr in arrays:
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=np.float64).flatten()
        arr_finite = arr_np[np.isfinite(arr_np)]
        if arr_finite.size > 0:
            finite_parts.append(arr_finite)
    if not finite_parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(finite_parts)


def _detect_cu_index(cu_name: str) -> int | None:
    """Extract CU index from names like 'CU1', 'entry03', etc."""
    if not cu_name:
        return None
    match = re.search(r"(\d+)", cu_name)
    if match is None:
        return None
    return int(match.group(1))


def plot_ocv_model_param_show(
    result: DMAResult,
    simulated_curves: dict[str, NDArray[np.floating]] | None = None,
    *,
    cell_name: str | None = None,
    cu_index: int | None = None,
    label_cfg: dict[str, str] | None = None,
    figsize_cm: tuple[float, float] = (20.0, 16.0),
    y_max_dva: float = 2.5,
    legend_ncols: int = 2,
    latex_fonts: bool = False,
    use_tex: bool = False,
) -> Figure:
    """Plot MATLAB-style OCV + DVA overview with alpha/beta annotations.

    This replicates the layout of MATLAB's ``plot_OCV_model_param_show.m``:
    OCV on top (2/3 height), DVA below (1/3 height), with parameter arrows
    and labels for ``alpha`` and ``beta`` on the electrode reconstructions.

    Parameters
    ----------
    result : DMAResult
        Single-CU DMA result containing measured/reconstructed curves and
        fitted parameters.
    simulated_curves : dict, optional
        Optional simulated curve dictionary (e.g., from
        ``DMAAnalyzer.compute_simulated_curves``). Used as fallback when
        reconstructed arrays are not embedded in ``result``.
    cell_name : str, optional
        Kept for API compatibility. Title is intentionally omitted.
    cu_index : int, optional
        Kept for API compatibility. Title is intentionally omitted.
    label_cfg : dict, optional
        Label overrides. Supported keys:
        - ``label_cathode`` (default: ``"Cathode"``)
        - ``label_anode`` (default: ``"Anode"``)
    figsize_cm : tuple, optional
        Figure size in centimeters (width, height), default ``(20, 16)``.
    y_max_dva : float, optional
        Default upper y-limit cap for DVA panel (default ``2.5``).
        If full-cell DVA data within 10-90% SOC exceeds this value, the
        upper limit is automatically increased by 5%.
    legend_ncols : int, optional
        Number of columns in the legend (default ``2``).
    latex_fonts : bool, optional
        If True, use LaTeX-like serif fonts for all plot text.
    use_tex : bool, optional
        If True, enable full LaTeX rendering (requires TeX installed).

    Returns
    -------
    Figure
        Matplotlib figure object.

    Raises
    ------
    ValueError
        If measured/reconstructed OCV/DVA or electrode reconstruction data
        required for the MATLAB-style overlay is missing.
    """
    _setup_style(latex_fonts=latex_fonts, use_tex=use_tex)
    if legend_ncols < 1:
        raise ValueError(f"legend_ncols must be >= 1, got {legend_ncols}")

    labels = {
        "label_cathode": "Cathode",
        "label_anode": "Anode",
    }
    if label_cfg:
        normalized = dict(label_cfg)
        if "labelCathode" in normalized and "label_cathode" not in normalized:
            normalized["label_cathode"] = str(normalized["labelCathode"])
        if "labelAnode" in normalized and "label_anode" not in normalized:
            normalized["label_anode"] = str(normalized["labelAnode"])
        labels.update(
            {
                "label_cathode": str(normalized.get("label_cathode", labels["label_cathode"])),
                "label_anode": str(normalized.get("label_anode", labels["label_anode"])),
            }
        )

    measured_q = _first_nonempty_array(result.measured_capacity, result.soc_measured)
    measured_u = _first_nonempty_array(result.measured_voltage, result.ocv_measured)

    reconstructed_q = _first_nonempty_array(
        result.soc_reconstructed,
        result.dva_q_reconstructed,
        None if simulated_curves is None else simulated_curves.get("capacity"),
    )
    reconstructed_u = _first_nonempty_array(
        result.ocv_reconstructed,
        None if simulated_curves is None else simulated_curves.get("voltage"),
    )

    measured_q_dva = _first_nonempty_array(
        result.dva_q_measured,
        measured_q,
    )
    measured_dva = _first_nonempty_array(result.dva_measured, result.measured_dva)
    reconstructed_q_dva = _first_nonempty_array(
        result.dva_q_reconstructed,
        reconstructed_q,
        None if simulated_curves is None else simulated_curves.get("capacity"),
    )
    reconstructed_dva = _first_nonempty_array(
        result.dva_reconstructed,
        None if simulated_curves is None else simulated_curves.get("dva"),
    )

    cathode_soc = _first_nonempty_array(
        result.cathode_soc,
        None if simulated_curves is None else simulated_curves.get("cathode_soc"),
    )
    cathode_u = _first_nonempty_array(
        result.cathode_potential,
        None if simulated_curves is None else simulated_curves.get("cathode_voltage"),
    )
    anode_soc = _first_nonempty_array(
        result.anode_soc,
        None if simulated_curves is None else simulated_curves.get("anode_soc"),
    )
    anode_u = _first_nonempty_array(
        result.anode_potential,
        None if simulated_curves is None else simulated_curves.get("anode_voltage"),
    )

    if measured_q.size == 0 or measured_u.size == 0:
        raise ValueError(
            "Measured OCV arrays are missing. "
            "Expected capacity/SOC and voltage in DMAResult."
        )
    if reconstructed_q.size == 0 or reconstructed_u.size == 0:
        raise ValueError(
            "Reconstructed OCV arrays are missing. Provide result.soc_reconstructed/"
            "result.ocv_reconstructed or simulated_curves with 'capacity' and 'voltage'."
        )
    if measured_q_dva.size == 0 or measured_dva.size == 0:
        raise ValueError(
            "Measured DVA arrays are missing. "
            "Expected in result.dva_q_measured/dva_measured."
        )
    if reconstructed_q_dva.size == 0 or reconstructed_dva.size == 0:
        raise ValueError(
            "Reconstructed DVA arrays are missing. Provide result.dva_q_reconstructed/"
            "result.dva_reconstructed or simulated_curves with 'capacity' and 'dva'."
        )
    if cathode_soc.size == 0 or cathode_u.size == 0 or anode_soc.size == 0 or anode_u.size == 0:
        raise ValueError(
            "Electrode reconstruction arrays are missing. "
            "Expected anode/cathode SOC and potential in DMAResult or simulated_curves."
        )

    alpha_an = result.fitted_params.alpha_an
    beta_an = result.fitted_params.beta_an
    alpha_ca = result.fitted_params.alpha_ca
    beta_ca = result.fitted_params.beta_ca

    _, _, dva_cath = calculate_dva(cathode_soc, cathode_u, smooth=True, smooth_window=30)
    q_dva_cath, _, _ = calculate_dva(cathode_soc, cathode_u, smooth=False)
    q_dva_an, _, dva_an = calculate_dva(anode_soc, anode_u, smooth=True, smooth_window=30)

    # MATLAB-aligned plotting constants
    title_font = 12
    base_font = 10
    axis_tick_font = 9
    legend_font = 9
    lw_main = 2
    lw_dva = 2
    lw_guide = 1
    axes_box_line_width = 1.5
    arrow_line_width = 2

    x_pad_left_frac = 0.05
    x_pad_right_frac = 0.08
    x_pad_zero_left_frac = 0.02

    y_pad_lower_ocv = 0.30
    y_pad_upper_ocv_frac = 0.04
    y_pad_lower_dva_frac = 0.05

    alpha_cat_text_down = 0.15
    alpha_an_text_up = 0.30
    beta_cat_text_up = 0.50
    beta_an_text_up = -0.04

    alpha_cat_arrow_y_offset = 0.04
    alpha_an_arrow_y_offset = 0.05
    beta_cat_arrow_y_offset = -0.02
    beta_an_arrow_y_offset = 0.09
    beta_cat_x_shift_to_zero = 0.01

    col_meas = 0.5 * np.array([1.0, 1.0, 1.0])
    col_model = TUM_COLORS["black"]
    col_cathode = TUM_COLORS["green"]
    col_anode = TUM_COLORS["blue"]

    x_all = _concat_finite_arrays(
        measured_q,
        reconstructed_q,
        cathode_soc,
        anode_soc,
        measured_q_dva,
        reconstructed_q_dva,
        q_dva_cath,
        q_dva_an,
    )
    if x_all.size == 0:
        x_min_data = 0.0
        x_max_data = 1.0
    else:
        x_min_data = float(np.min(x_all))
        x_max_data = float(np.max(x_all))
    range_x = max(x_max_data - x_min_data, 1.0)
    x_min_need = min(x_min_data, 0.0) - (x_pad_left_frac + x_pad_zero_left_frac) * range_x
    x_max_need = x_max_data + x_pad_right_frac * range_x
    x_lim_dyn = (x_min_need, x_max_need)

    y_all_ocv = _concat_finite_arrays(measured_u, reconstructed_u, cathode_u, anode_u)
    if y_all_ocv.size == 0:
        y_min_data = 0.0
        y_max_data = 4.4
    else:
        y_min_data = float(np.min(y_all_ocv))
        y_max_data = float(np.max(y_all_ocv))
    range_y = max(y_max_data - y_min_data, 1.0)
    y_need_top = max(
        y_max_data,
        float(np.max(cathode_u) + alpha_cat_arrow_y_offset),
        float(np.max(anode_u) + alpha_an_arrow_y_offset),
        float(np.max(anode_u) + alpha_an_text_up),
    )
    y_need_bot = min(
        y_min_data,
        float(np.min(cathode_u) + beta_cat_arrow_y_offset),
        float(np.min(anode_u) + beta_an_arrow_y_offset),
    )
    y_lim_ocv = (
        y_need_bot - y_pad_lower_ocv,
        y_need_top + y_pad_upper_ocv_frac * range_y,
    )

    y_all_dva = _concat_finite_arrays(
        measured_dva,
        reconstructed_dva,
        dva_cath,
        np.abs(dva_an),
    )
    if y_all_dva.size == 0:
        y_min_d = 0.0
        y_max_d = y_max_dva
    else:
        y_min_d = float(np.min(y_all_dva))
        y_max_d = float(np.max(y_all_dva))
    range_yd = max(y_max_d - y_min_d, 1.0)
    # MATLAB-consistent lower bound:
    # keep some fractional padding below the DVA minimum, clamped to 0.
    y_lim_dva_min = max(0.0, y_min_d - y_pad_lower_dva_frac * range_yd)
    y_lim_dva_max = float(y_max_dva)

    # check measured/reconstructed full-cell curves in the 10-90% SOC window.
    # If the data peak there exceeds the configured y-limit max, expand by 5%.
    fullcell_roi_min = 0.10
    fullcell_roi_max = 0.90
    roi_peaks: list[float] = []
    for q_arr, d_arr in (
        (measured_q_dva, measured_dva),
        (reconstructed_q_dva, reconstructed_dva),
    ):
        n = min(q_arr.size, d_arr.size)
        if n == 0:
            continue
        q_roi = q_arr[:n]
        d_roi = d_arr[:n]
        mask_roi = (
            np.isfinite(q_roi)
            & np.isfinite(d_roi)
            & (q_roi >= fullcell_roi_min)
            & (q_roi <= fullcell_roi_max)
        )
        if np.any(mask_roi):
            roi_peaks.append(float(np.max(d_roi[mask_roi])))
    if roi_peaks:
        max_roi_peak = max(roi_peaks)
        if max_roi_peak > y_lim_dva_max:
            y_lim_dva_max = 1.05 * max_roi_peak

    y_lim_dva = (y_lim_dva_min, y_lim_dva_max)

    fig_w_cm, fig_h_cm = figsize_cm
    compact_height = fig_h_cm <= 12.0
    fig = plt.figure(
        figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
        facecolor="white",
    )
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.06)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Title intentionally omitted to keep the plot area clean and maximize
    # consistency across CUs in overview workflows.
    _ = cell_name
    _ = cu_index
    _ = title_font
    # Compromise spacing: closer than original, but not as tight as the last revision.
    legend_y = 0.05 if compact_height else 0.045
    axes_top = 0.96

    # OCV panel (2x height)
    h_meas, = ax1.plot(
        measured_q,
        measured_u,
        color=col_meas,
        linewidth=lw_main,
        linestyle="-",
        label="FC measured",
    )
    h_recon, = ax1.plot(
        reconstructed_q,
        reconstructed_u,
        color=col_model,
        linewidth=lw_main,
        linestyle="-.",
        label="FC reconstructed",
    )
    h_cath, = ax1.plot(
        cathode_soc,
        cathode_u,
        color=col_cathode,
        linewidth=lw_main,
        linestyle="-.",
        label=f"{labels['label_cathode']} reconstructed",
    )
    h_an, = ax1.plot(
        anode_soc,
        anode_u,
        color=col_anode,
        linewidth=lw_main,
        linestyle="-.",
        label=f"{labels['label_anode']} reconstructed",
    )

    ax1.set_xlim(*x_lim_dyn)
    ax1.set_ylim(*y_lim_ocv)
    ax1.set_yticks(np.arange(0.0, 5.0, 1.0))
    ax1.set_ylabel(r"$U$ / $\mathrm{V}$", fontsize=base_font)
    ax1.tick_params(axis="both", labelsize=axis_tick_font)
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_box_line_width)

    x_left, x_right = ax1.get_xlim()
    if x_left <= 0.0 <= x_right:
        ax1.plot([0.0, 0.0], [y_lim_ocv[0], y_lim_ocv[1]], "--k", linewidth=lw_guide)
    if x_left <= 1.0 <= x_right:
        ax1.plot([1.0, 1.0], [y_lim_ocv[0], y_lim_ocv[1]], "--k", linewidth=lw_guide)

    fig.legend(
        handles=[h_meas, h_recon, h_cath, h_an],
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_ncols,
        fontsize=legend_font,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    def draw_double_arrow(
        ax: Axes,
        x_start: float,
        x_end: float,
        y_level: float,
        color: NDArray[np.floating],
    ) -> None:
        ax.annotate(
            "",
            xy=(x_end, y_level),
            xytext=(x_start, y_level),
            arrowprops={
                "arrowstyle": "<->",
                "color": color,
                "linewidth": arrow_line_width,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "mutation_scale": 12.0,
            },
        )

    draw_double_arrow(
        ax1,
        float(np.min(cathode_soc)),
        float(np.max(cathode_soc)),
        float(np.max(cathode_u) + alpha_cat_arrow_y_offset),
        col_cathode,
    )
    ax1.text(
        float(0.5 * (np.min(cathode_soc) + np.max(cathode_soc))),
        float(np.max(cathode_u) - alpha_cat_text_down),
        rf"$\alpha_{{\mathrm{{cat}}}} = {alpha_ca:.2f}$",
        color=col_cathode,
        fontsize=base_font,
        ha="center",
        va="center",
    )

    draw_double_arrow(
        ax1,
        float(np.min(anode_soc)),
        float(np.max(anode_soc)),
        float(np.max(anode_u) + alpha_an_arrow_y_offset),
        col_anode,
    )
    ax1.text(
        float(0.5 * (np.min(anode_soc) + np.max(anode_soc))),
        float(np.max(anode_u) + alpha_an_text_up),
        rf"$\alpha_{{\mathrm{{an}}}} = {alpha_an:.2f}$",
        color=col_anode,
        fontsize=base_font,
        ha="center",
        va="center",
    )

    draw_double_arrow(
        ax1,
        # Keep cathode beta arrow visible even when cathode SOC extends below 0.
        # Using min(0, ...) avoids collapsing the arrow to zero length.
        float(min(0.0, np.min(cathode_soc) - beta_cat_x_shift_to_zero)),
        0.0,
        float(np.min(cathode_u) + beta_cat_arrow_y_offset),
        col_cathode,
    )
    draw_double_arrow(
        ax1,
        float(np.min(anode_soc)),
        0.0,
        float(np.min(anode_u) + beta_an_arrow_y_offset),
        col_anode,
    )
    beta_text_x = 0.03 if x_lim_dyn[0] <= 0.03 <= x_lim_dyn[1] else x_lim_dyn[0] + 0.03 * (
        x_lim_dyn[1] - x_lim_dyn[0]
    )
    ax1.text(
        beta_text_x,
        float(np.min(cathode_u) + beta_cat_text_up),
        rf"$\beta_{{\mathrm{{cat}}}} = {beta_ca:.2f}$",
        color=col_cathode,
        fontsize=base_font,
        ha="left",
        va="center",
    )
    ax1.text(
        beta_text_x,
        float(np.min(anode_u) + beta_an_text_up),
        rf"$\beta_{{\mathrm{{an}}}} = {beta_an:.2f}$",
        color=col_anode,
        fontsize=base_font,
        ha="left",
        va="center",
    )

    # DVA panel (1x height)
    ax2.plot(
        measured_q_dva,
        measured_dva,
        color=col_meas,
        linewidth=lw_dva,
        linestyle="-",
    )
    ax2.plot(
        reconstructed_q_dva,
        reconstructed_dva,
        color=col_model,
        linewidth=lw_dva,
        linestyle="-.",
    )
    ax2.plot(
        q_dva_cath,
        dva_cath,
        color=col_cathode,
        linewidth=lw_dva,
        linestyle="-.",
    )
    ax2.plot(
        q_dva_an,
        np.abs(dva_an),
        color=col_anode,
        linewidth=lw_dva,
        linestyle="-.",
    )

    ax2.set_xlim(*x_lim_dyn)
    ax2.set_ylim(*y_lim_dva)
    y_ticks = np.arange(0.0, np.floor(y_lim_dva[1]) + 1.0, 1.0)
    if y_ticks.size == 0:
        y_ticks = np.array([0.0])
    ax2.set_yticks(y_ticks)
    # Re-apply limits because setting ticks can auto-expand axes.
    ax2.set_ylim(*y_lim_dva)
    ax2.set_xlabel("SOC / -", fontsize=base_font)
    ax2.set_ylabel(r"$dU(dQ)^{-1}\cdot C_{\mathrm{act}}$ / $\mathrm{V}$", fontsize=base_font)
    ax2.tick_params(axis="both", labelsize=axis_tick_font)
    ax2.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_box_line_width)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.24 if compact_height else 0.21,
        top=axes_top,
    )
    return fig


def plot_ocv_comparison(
    measured_capacity: NDArray[np.floating],
    measured_voltage: NDArray[np.floating],
    simulated_capacity: NDArray[np.floating] | None = None,
    simulated_voltage: NDArray[np.floating] | None = None,
    ax: Axes | None = None,
    title: str = "OCV Comparison",
) -> Axes:
    """Plot measured vs simulated OCV curves.

    Parameters
    ----------
    measured_capacity : NDArray
        Measured capacity values
    measured_voltage : NDArray
        Measured voltage values
    simulated_capacity : NDArray, optional
        Simulated capacity values
    simulated_voltage : NDArray, optional
        Simulated voltage values
    ax : Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title

    Returns
    -------
    Axes
        The matplotlib axes object
    """
    _setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Plot measured
    ax.plot(
        measured_capacity, measured_voltage,
        '-', color=TUM_COLORS['blue'], linewidth=2,
        label='Measured'
    )

    # Plot simulated if provided
    if simulated_capacity is not None and simulated_voltage is not None:
        ax.plot(
            simulated_capacity, simulated_voltage,
            '--', color=TUM_COLORS['orange'], linewidth=2,
            label='Simulated'
        )

    def _is_normalized_capacity(values: NDArray[np.floating]) -> bool:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return False
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        return (vmin >= -1e-6) and (vmax <= 1.0 + 1e-6)

    is_normalized = _is_normalized_capacity(measured_capacity)
    if simulated_capacity is not None:
        is_normalized = is_normalized and _is_normalized_capacity(simulated_capacity)

    # MATLAB-style limits (plot_OCV_model_param_show.m)
    x_vals = [np.asarray(measured_capacity)]
    y_vals = [np.asarray(measured_voltage)]
    if simulated_capacity is not None and simulated_voltage is not None:
        x_vals.append(np.asarray(simulated_capacity))
        y_vals.append(np.asarray(simulated_voltage))

    x_all = (
        np.concatenate([xv[np.isfinite(xv)] for xv in x_vals if xv.size])
        if x_vals
        else np.array([])
    )
    y_all = (
        np.concatenate([yv[np.isfinite(yv)] for yv in y_vals if yv.size])
        if y_vals
        else np.array([])
    )

    if x_all.size:
        x_min = float(np.min(x_all))
        x_max = float(np.max(x_all))
        range_x = max(x_max - x_min, 1.0)
        x_min_need = min(x_min, 0.0) - (0.05 + 0.02) * range_x
        x_max_need = x_max + 0.08 * range_x
        ax.set_xlim(x_min_need, x_max_need)

    if y_all.size:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        range_y = max(y_max - y_min, 1.0)
        y_lim = (y_min - 0.30, y_max + 0.04 * range_y)
        ax.set_ylim(*y_lim)

    ax.set_xlabel(r'SOC / -' if is_normalized else r'$Q$ / Ah')
    ax.set_ylabel(r'$U$ / V')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def plot_dva_comparison(
    measured_soc: NDArray[np.floating],
    measured_dva: NDArray[np.floating],
    simulated_soc: NDArray[np.floating] | None = None,
    simulated_dva: NDArray[np.floating] | None = None,
    soc_min: float | None = None,
    soc_max: float | None = None,
    ax: Axes | None = None,
    title: str = "DVA Comparison",
) -> Axes:
    """Plot measured vs simulated DVA curves.

    Parameters
    ----------
    measured_soc : NDArray
        Measured SOC values (0-1)
    measured_dva : NDArray
        Measured DVA values (dV/dQ)
    simulated_soc : NDArray, optional
        Simulated SOC values (0-1)
    simulated_dva : NDArray, optional
        Simulated DVA values
    soc_min : float, optional
        Minimum SOC for ROI shading
    soc_max : float, optional
        Maximum SOC for ROI shading
    ax : Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title

    Returns
    -------
    Axes
        The matplotlib axes object
    """
    _setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Plot ROI region if specified
    if soc_min is not None and soc_max is not None:
        ax.axvspan(soc_min, soc_max, alpha=0.1, color='green', label='ROI')

    # Plot measured
    ax.plot(
        measured_soc, measured_dva,
        '-', color=TUM_COLORS['blue'], linewidth=1.5,
        label='Measured'
    )

    # Plot simulated if provided
    if simulated_soc is not None and simulated_dva is not None:
        ax.plot(
            simulated_soc, simulated_dva,
            '--', color=TUM_COLORS['orange'], linewidth=1.5,
            label='Simulated'
        )

    # MATLAB-style limits (plot_OCV_model_param_show.m)
    x_vals = [np.asarray(measured_soc)]
    y_vals = [np.asarray(measured_dva)]
    if simulated_soc is not None and simulated_dva is not None:
        x_vals.append(np.asarray(simulated_soc))
        y_vals.append(np.asarray(simulated_dva))

    x_all = (
        np.concatenate([xv[np.isfinite(xv)] for xv in x_vals if xv.size])
        if x_vals
        else np.array([])
    )
    y_all = (
        np.concatenate([yv[np.isfinite(yv)] for yv in y_vals if yv.size])
        if y_vals
        else np.array([])
    )

    if x_all.size:
        x_min = float(np.min(x_all))
        x_max = float(np.max(x_all))
        range_x = max(x_max - x_min, 1.0)
        x_min_need = min(x_min, 0.0) - (0.05 + 0.02) * range_x
        x_max_need = x_max + 0.08 * range_x
        ax.set_xlim(x_min_need, x_max_need)

    if y_all.size:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        range_y = max(y_max - y_min, 1.0)
        y_lim = (max(0.0, y_min - 0.05 * range_y), 3.2)
        ax.set_ylim(*y_lim)

    ax.set_xlabel(r'SOC / -')
    ax.set_ylabel(r'$dU/dQ \cdot C_{\mathrm{act}}$ / V')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def plot_ica_comparison(
    measured_voltage: NDArray[np.floating],
    measured_ica: NDArray[np.floating],
    simulated_voltage: NDArray[np.floating] | None = None,
    simulated_ica: NDArray[np.floating] | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    ax: Axes | None = None,
    title: str = "ICA Comparison",
) -> Axes:
    """Plot measured vs simulated ICA curves.

    Parameters
    ----------
    measured_voltage : NDArray
        Measured voltage values
    measured_ica : NDArray
        Measured ICA values (dQ/dV)
    simulated_voltage : NDArray, optional
        Simulated voltage values
    simulated_ica : NDArray, optional
        Simulated ICA values
    v_min : float, optional
        Minimum voltage for ROI shading
    v_max : float, optional
        Maximum voltage for ROI shading
    ax : Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title

    Returns
    -------
    Axes
        The matplotlib axes object
    """
    _setup_style()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Plot ROI region if specified
    if v_min is not None and v_max is not None:
        ax.axvspan(v_min, v_max, alpha=0.1, color='green', label='ROI')

    # Plot measured
    ax.plot(
        measured_voltage, measured_ica,
        '-', color=TUM_COLORS['blue'], linewidth=1.5,
        label='Measured'
    )

    # Plot simulated if provided
    if simulated_voltage is not None and simulated_ica is not None:
        ax.plot(
            simulated_voltage, simulated_ica,
            '--', color=TUM_COLORS['orange'], linewidth=1.5,
            label='Simulated'
        )

    ax.set_xlabel(r'$U$ / V')
    ax.set_ylabel(r'$dQ/dU$ / Ah V$^{-1}$')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def plot_degradation_modes(
    degradation_modes: DegradationModes,
    ax: Axes | None = None,
    title: str = "Degradation Modes",
    show_values: bool = True,
    show_anode_blend: bool = False,
    show_cathode_blend: bool = False,
) -> Axes:
    """Plot degradation modes as a bar chart.

    Parameters
    ----------
    degradation_modes : DegradationModes
        Degradation modes to plot
    ax : Axes, optional
        Matplotlib axes to plot on
    title : str, optional
        Plot title
    show_values : bool, optional
        Whether to show numeric values on bars
    show_anode_blend : bool, optional
        Show anode blend1 and blend2 bars (default: False)
    show_cathode_blend : bool, optional
        Show cathode blend1 and blend2 bars (default: False)

    Returns
    -------
    Axes
        The matplotlib axes object
    """
    _setup_style()

    if ax is None:
        # Wider figure if showing blend components
        n_extra = (2 if show_anode_blend else 0) + (2 if show_cathode_blend else 0)
        fig_width = 6 + n_extra * 0.8
        _, ax = plt.subplots(figsize=(fig_width, 4))

    # Build dynamic bar data
    modes = []
    values = []
    colors = []

    # LLI
    modes.append('LLI')
    values.append(degradation_modes.lli * 100)
    colors.append(TUM_COLORS['blue'])

    # Anode LAM
    modes.append('LAM_an')
    values.append(degradation_modes.lam_an * 100)
    colors.append(TUM_COLORS['orange'])

    # Anode blend components (if enabled)
    if show_anode_blend:
        modes.append('An-blend1')
        values.append(degradation_modes.lam_anode_blend1 * 100)
        colors.append(TUM_COLORS['light_blue'])

        modes.append('An-blend2')
        values.append(degradation_modes.lam_anode_blend2 * 100)
        colors.append(TUM_COLORS['dark_blue'])

    # Cathode LAM
    modes.append('LAM_ca')
    values.append(degradation_modes.lam_ca * 100)
    colors.append(TUM_COLORS['green'])

    # Cathode blend components (if enabled)
    if show_cathode_blend:
        modes.append('Ca-blend1')
        values.append(degradation_modes.lam_cathode_blend1 * 100)
        colors.append(TUM_COLORS['lighter_blue'])

        modes.append('Ca-blend2')
        values.append(degradation_modes.lam_cathode_blend2 * 100)
        colors.append(TUM_COLORS['medium_gray'])

    # Capacity loss (always last)
    modes.append('Capacity\nLoss')
    values.append(degradation_modes.capacity_loss * 100)
    colors.append(TUM_COLORS['gray'])

    # Create bar chart
    bars = ax.bar(modes, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    if show_values:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{value:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9,
            )

    ax.set_ylabel('Degradation / %')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate labels if many bars
    if len(modes) > 5:
        ax.tick_params(axis='x', rotation=30)

    return ax


def plot_dma_result(
    result: DMAResult,
    simulated_curves: dict[str, NDArray[np.floating]] | None = None,
    figsize: tuple[float, float] = (12, 8),
    show_anode_blend: bool = False,
    show_cathode_blend: bool = False,
) -> Figure:
    """Create a multi-panel summary of DMA results.

    Creates a 2x2 subplot with:
    - OCV comparison
    - DVA comparison
    - ICA comparison
    - Degradation modes bar chart

    Parameters
    ----------
    result : DMAResult
        DMA result to visualize
    simulated_curves : dict, optional
        Simulated curves from analyzer.compute_simulated_curves()
    figsize : tuple, optional
        Figure size (width, height)
    show_anode_blend : bool, optional
        Show anode blend1 and blend2 in degradation modes bar chart (default: False)
    show_cathode_blend : bool, optional
        Show cathode blend1 and blend2 in degradation modes bar chart (default: False)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Degradation Mode Analysis Results', fontsize=14, fontweight='bold')

    # Prepare simulated data if available
    sim_cap = sim_volt = sim_dva = sim_ica = None
    if simulated_curves is not None:
        sim_cap = simulated_curves.get('capacity')
        sim_volt = simulated_curves.get('voltage')
        sim_dva = simulated_curves.get('dva')
        sim_ica = simulated_curves.get('ica')

    # OCV comparison
    plot_ocv_comparison(
        result.measured_capacity,
        result.measured_voltage,
        sim_cap, sim_volt,
        ax=axes[0, 0],
        title='OCV Comparison',
    )

    # DVA comparison
    plot_dva_comparison(
        result.measured_capacity,
        result.measured_dva,
        sim_cap, sim_dva,
        ax=axes[0, 1],
        title='DVA Comparison',
    )

    # ICA comparison
    plot_ica_comparison(
        result.measured_voltage,
        result.measured_ica,
        sim_volt, sim_ica,
        ax=axes[1, 0],
        title='ICA Comparison',
    )

    # Degradation modes
    plot_degradation_modes(
        result.degradation_modes,
        ax=axes[1, 1],
        title=f'Degradation Modes (RMSE: {result.rmse * 1000:.1f} mV)',
        show_anode_blend=show_anode_blend,
        show_cathode_blend=show_cathode_blend,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    return fig


def plot_aging_study(
    results: AgingStudyResults | Sequence[DMAResult],
    x_values: Sequence[float] | None = None,
    x_label: str = "EFC",
    figsize: tuple[float, float] | None = None,
    plot_cathode: bool = True,
    plot_anode: bool = True,
    plot_lli: bool = True,
    plot_rmse: bool = True,
    use_anode_blend: bool = False,
    use_cathode_blend: bool = False,
    calendar_aging: bool = False,
    labels: dict[str, str] | None = None,
) -> Figure:
    """Plot degradation modes evolution over aging (MATLAB-style).

    Creates separate panels for each degradation mode, all in TUM Blue.
    Matches the MATLAB plot_DMA.m style with tiledlayout.

    Parameters
    ----------
    results : AgingStudyResults or Sequence[DMAResult]
        Aging study results or list of DMA results
    x_values : Sequence[float], optional
        X-axis values (e.g., EFC). If not provided, uses efc_values or indices.
    x_label : str, optional
        Label for x-axis (default: "EFC")
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    plot_cathode : bool, optional
        Show cathode LAM panel (default: True)
    plot_anode : bool, optional
        Show anode LAM panel (default: True)
    plot_lli : bool, optional
        Show LLI panel (default: True)
    plot_rmse : bool, optional
        Show RMSE panel (default: True)
    use_anode_blend : bool, optional
        Show separate anode blend1/blend2 panels (default: False)
    use_cathode_blend : bool, optional
        Show separate cathode blend1/blend2 panels (default: False)
    calendar_aging : bool, optional
        If True, x-label shows "RPT Number", else "EFC" (default: False)
    labels : dict, optional
        Custom labels for panels. Keys: 'cathode', 'anode', 'lli', 'rmse',
        'anode_blend1', 'anode_blend2', 'cathode_blend1', 'cathode_blend2'

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    _setup_style()

    # Default labels
    default_labels = {
        'cathode': 'Cathode',
        'anode': 'Anode',
        'lli': 'Charge-carrier-inv',
        'rmse': 'RMSE',
        'anode_blend1': 'An-blend1',
        'anode_blend2': 'An-blend2',
        'cathode_blend1': 'Ca-blend1',
        'cathode_blend2': 'Ca-blend2',
    }
    if labels:
        default_labels.update(labels)
    labels = default_labels

    # Handle different input types
    if isinstance(results, AgingStudyResults):
        if x_values is None:
            if results.efc_values:
                x_values = results.efc_values
            else:
                x_values = list(range(1, len(results.results) + 1))
        result_list = (
            [results.results[name] for name in results.cu_labels]
            if results.cu_labels
            else list(results.results.values())
        )
    else:
        if x_values is None:
            x_values = list(range(1, len(results) + 1))
        result_list = list(results)

    # Build list of panels to plot
    panels = []  # List of (label, data_array)

    # Cathode
    if plot_cathode:
        lam_ca = [r.degradation_modes.lam_ca * 100 for r in result_list]
        panels.append((labels['cathode'], lam_ca))
        if use_cathode_blend:
            lam_ca_b1 = [r.degradation_modes.lam_cathode_blend1 * 100 for r in result_list]
            lam_ca_b2 = [r.degradation_modes.lam_cathode_blend2 * 100 for r in result_list]
            panels.append((labels['cathode_blend1'], lam_ca_b1))
            panels.append((labels['cathode_blend2'], lam_ca_b2))

    # Anode
    if plot_anode:
        lam_an = [r.degradation_modes.lam_an * 100 for r in result_list]
        panels.append((labels['anode'], lam_an))
        if use_anode_blend:
            lam_an_b1 = [r.degradation_modes.lam_anode_blend1 * 100 for r in result_list]
            lam_an_b2 = [r.degradation_modes.lam_anode_blend2 * 100 for r in result_list]
            panels.append((labels['anode_blend1'], lam_an_b1))
            panels.append((labels['anode_blend2'], lam_an_b2))

    # LLI (Charge-carrier inventory)
    if plot_lli:
        lli = [r.degradation_modes.lli * 100 for r in result_list]
        panels.append((labels['lli'], lli))

    # RMSE
    rmse_data = None
    if plot_rmse:
        rmse_data = [r.rmse * 1000 for r in result_list]  # V -> mV

    n_panels = len(panels) + (1 if plot_rmse else 0)
    if n_panels == 0:
        raise ValueError("At least one panel must be enabled")

    # Auto-calculate figure size (similar to MATLAB: 20cm wide, 6cm tall)
    if figsize is None:
        figsize = (20 * 0.393701, 6 * 0.393701 * 1.2)  # cm to inches, slightly taller

    # Style constants (matching MATLAB)
    tum_blue = TUM_COLORS['blue']
    markers = ['-o', '-s', '-d', '-^', '-v', '-*']
    font_sz = 10
    line_width = 2
    marker_size = 5

    # Calculate common y-limits for LAM panels
    all_lam_values = []
    for _, data in panels:
        all_lam_values.extend(data)
    if all_lam_values:
        y_min = min(all_lam_values)
        y_max = max(all_lam_values)
        y_range = y_max - y_min if y_max > y_min else 1.0
        y_min_padded = y_min - 0.05 * y_range
        y_max_padded = y_max + 0.05 * y_range
    else:
        y_min_padded, y_max_padded = 0, 1

    # Create figure with custom spacing:
    # - tighter spacing between DM panels
    # - larger gap before RMSE panel
    if plot_rmse and panels:
        fig = plt.figure(figsize=figsize)
        width_ratios = [1.0] * len(panels) + [0.35, 1.0]  # spacer, then RMSE
        gs = fig.add_gridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.07)
        lam_axes = [fig.add_subplot(gs[0, i]) for i in range(len(panels))]
        ax_rmse = fig.add_subplot(gs[0, -1])
    elif plot_rmse:
        fig, single_ax = plt.subplots(1, 1, figsize=figsize, sharey=False)
        lam_axes = []
        ax_rmse = single_ax
    else:
        fig, raw_axes = plt.subplots(1, len(panels), figsize=figsize, sharey=False)
        lam_axes = [raw_axes] if len(panels) == 1 else list(raw_axes)
        ax_rmse = None

    # X-axis label based on aging type
    if calendar_aging:
        actual_x_label = "RPT Number / -"
    else:
        actual_x_label = x_label if x_label else "EFC"

    # Plot LAM panels
    for idx, (label, data) in enumerate(panels):
        ax = lam_axes[idx]
        marker = markers[idx % len(markers)]

        ax.plot(x_values, data, marker, color=tum_blue,
                linewidth=line_width, markersize=marker_size)

        ax.set_title(label, fontsize=1.2 * font_sz)
        ax.set_xlabel(actual_x_label, fontsize=font_sz)
        ax.set_xlim(auto=True)
        ax.set_ylim(y_min_padded, y_max_padded)
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction='out', length=0)

        # Y-label only on first panel
        if idx == 0:
            ax.set_ylabel('Capacity Loss / %', fontsize=font_sz)
        else:
            ax.set_yticklabels([])

    # RMSE panel (separate y-axis)
    if plot_rmse and rmse_data is not None and ax_rmse is not None:
        ax_rmse.plot(x_values, rmse_data, '-o', color=tum_blue,
                     linewidth=line_width, markersize=marker_size)

        ax_rmse.set_title(labels['rmse'], fontsize=1.2 * font_sz)
        ax_rmse.set_xlabel(actual_x_label, fontsize=font_sz)
        ax_rmse.set_ylabel('RMSE / mV', fontsize=font_sz)
        ax_rmse.set_xlim(auto=True)
        ax_rmse.set_ylim(0, max(rmse_data) * 1.1 if max(rmse_data) > 0 else 1)
        ax_rmse.grid(True, alpha=0.3)
        ax_rmse.tick_params(direction='out', length=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    return fig


class DMAPlotter:
    """Convenience class for creating DMA visualizations.

    Provides methods for common visualization tasks with
    consistent styling.

    Parameters
    ----------
    style : str, optional
        Matplotlib style to use

    Examples
    --------
    >>> plotter = DMAPlotter()
    >>> plotter.plot_result(result)
    >>> plotter.show()
    """

    def __init__(self, style: str | None = None):
        if style is not None:
            plt.style.use(style)
        _setup_style()
        self._figures: list[Figure] = []

    def plot_result(
        self,
        result: DMAResult,
        simulated_curves: dict[str, NDArray[np.floating]] | None = None,
        show_anode_blend: bool = False,
        show_cathode_blend: bool = False,
    ) -> Figure:
        """Plot DMA result summary."""
        fig = plot_dma_result(
            result,
            simulated_curves,
            show_anode_blend=show_anode_blend,
            show_cathode_blend=show_cathode_blend,
        )
        self._figures.append(fig)
        return fig

    def plot_ocv_dva_parameters(
        self,
        result: DMAResult,
        simulated_curves: dict[str, NDArray[np.floating]] | None = None,
        *,
        cell_name: str | None = None,
        cu_index: int | None = None,
        label_cfg: dict[str, str] | None = None,
        figsize_cm: tuple[float, float] = (20.0, 16.0),
        legend_ncols: int = 2,
        latex_fonts: bool = False,
        use_tex: bool = False,
    ) -> Figure:
        """Plot MATLAB-style OCV+DVA panel with alpha/beta annotations."""
        fig = plot_ocv_model_param_show(
            result=result,
            simulated_curves=simulated_curves,
            cell_name=cell_name,
            cu_index=cu_index,
            label_cfg=label_cfg,
            figsize_cm=figsize_cm,
            legend_ncols=legend_ncols,
            latex_fonts=latex_fonts,
            use_tex=use_tex,
        )
        self._figures.append(fig)
        return fig

    def plot_aging(
        self,
        results: AgingStudyResults | Sequence[DMAResult],
        x_values: Sequence[float] | None = None,
        x_label: str = "Cycle Number",
    ) -> Figure:
        """Plot aging study results."""
        fig = plot_aging_study(results, x_values, x_label)
        self._figures.append(fig)
        return fig

    def plot_comparison(
        self,
        result1: DMAResult,
        result2: DMAResult,
        labels: tuple[str, str] = ("Before", "After"),
        figsize: tuple[float, float] = (10, 6),
    ) -> Figure:
        """Compare two DMA results side by side."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Comparison: {labels[0]} vs {labels[1]}', fontsize=14, fontweight='bold')

        # Plot degradation modes for each
        plot_degradation_modes(
            result1.degradation_modes,
            ax=axes[0],
            title=labels[0],
        )
        plot_degradation_modes(
            result2.degradation_modes,
            ax=axes[1],
            title=labels[1],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        self._figures.append(fig)
        return fig

    def show(self) -> None:
        """Show all created figures."""
        plt.show()

    def save_all(
        self,
        prefix: str = "dma_plot",
        format: str = "png",
        dpi: int = 300,
    ) -> list[str]:
        """Save all created figures.

        Parameters
        ----------
        prefix : str
            Filename prefix
        format : str
            File format (png, pdf, svg, etc.)
        dpi : int
            Resolution for raster formats

        Returns
        -------
        list[str]
            List of saved file paths
        """
        paths = []
        for i, fig in enumerate(self._figures):
            path = f"{prefix}_{i+1}.{format}"
            fig.savefig(path, format=format, dpi=dpi, bbox_inches='tight')
            paths.append(path)
        return paths

    def close_all(self) -> None:
        """Close all created figures."""
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
