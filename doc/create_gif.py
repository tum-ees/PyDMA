"""Create an animated GIF of OCV electrode balancing shift over aging.

Run from PyDMA root:
    python doc/create_gif.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

# Use a non-interactive backend for script execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pydma import (  # noqa: E402
    BlendElectrode,
    DMAAnalyzer,
    DMAConfig,
    ElectrodeOCP,
    load_ocp,
    plot_ocv_model_param_show,
)


FIGSIZE_CM = (13.0, 12.0)  # Match getting_started OCV+DVA figure geometry
FRAME_DPI = 200
FRAME_DURATION_MS = 800
LAST_FRAME_DURATION_MS = 2000
TARGET_GIF_WIDTH_PX = 900


def _find_testdata_dir(root: Path) -> Path:
    candidates = [
        root / "TestData",
        root / "notebooks" / "TestData",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find TestData directory. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _entry_number(path: Path) -> int:
    match = re.search(r"entry(\d+)", path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse entry number from {path.name}")
    return int(match.group(1))


def _load_blend_electrodes(input_data_dir: Path) -> tuple[BlendElectrode, ElectrodeOCP]:
    graphite_df = pd.read_csv(input_data_dir / "Graphite" / "Gr_Lithiation_Kuecher.csv")
    silicon_df = pd.read_csv(
        input_data_dir / "Silicon" / "SiReconstr_Lithiation_Kuecher_P45B_Anode_0C03.csv"
    )
    cathode = load_ocp(
        input_data_dir / "NCA" / "GITT_P45b_Cat_NCA_JN_VS_Coin_1_GITT__Extracted_Continuous_pOCP.csv",
        electrode_type="cathode",
        smooth=False,
    )

    graphite = ElectrodeOCP(
        soc=graphite_df["normalizedCapacity"].to_numpy(),
        voltage=graphite_df["voltage"].to_numpy(),
        name="Graphite (Kuecher)",
    )
    silicon = ElectrodeOCP(
        soc=silicon_df["normalizedCapacity"].to_numpy(),
        voltage=silicon_df["voltage"].to_numpy(),
        name="Silicon (Reconstructed)",
    )
    si_gr_anode = BlendElectrode(
        blend1=graphite,
        blend2=silicon,
        name="Si-Gr Anode (P45B)",
    )
    return si_gr_anode, cathode


def _load_all_pocv_entries(testdata_dir: Path) -> list[tuple[int, Path, np.ndarray, np.ndarray, float | None]]:
    expected = [testdata_dir / f"FR23_pOCV_CH_entry{i:02d}.csv" for i in range(1, 10)]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected pOCV files: {missing}")

    fit_data: list[tuple[int, Path, np.ndarray, np.ndarray, float | None]] = []
    for path in expected:
        df = pd.read_csv(path)
        cap = df["Ah_Step"].to_numpy()
        volt = df["U"].to_numpy()
        mask = ~(np.isnan(cap) | np.isnan(volt))
        cap = cap[mask]
        volt = volt[mask]
        efc = float(df["EFC"].iloc[0]) if "EFC" in df.columns else None
        fit_data.append((_entry_number(path), path, cap, volt, efc))

    fit_data.sort(key=lambda x: x[0])
    return fit_data


def _build_blend_config() -> DMAConfig:
    return DMAConfig(
        speed_preset="medium",
        workers=-1,
        direction="charge",
        data_length=1000,
        smoothing_points=30,
        roi_ocv_min=0.0,
        roi_ocv_max=1.0,
        roi_dva_min=0.10,
        roi_dva_max=0.90,
        roi_ica_min=0.13,
        roi_ica_max=0.90,
        weight_ocv=100.0,
        weight_dva=1.0,
        weight_ica=0.0,
        use_anode_blend=True,
        use_cathode_blend=False,
        gamma_anode_blend2_init=0.25,
        gamma_anode_blend2_upper=0.30,
        req_accepted=2,
        max_tries_overall=10,
        rmse_threshold=0.01,
        allow_anode_inhomogeneity=True,
        allow_cathode_inhomogeneity=True,
        allow_first_cycle_inhomogeneity=True,
        max_inhomogeneity=0.3,
        max_inhomogeneity_delta=0.1,
        max_anode_gain=0.01,
        max_cathode_gain=0.01,
        max_anode_blend1_gain=0.005,
        max_anode_blend2_gain=0.01,
        max_anode_loss=1.0,
        max_cathode_loss=1.0,
        max_anode_blend1_loss=1.0,
        max_anode_blend2_loss=1.0,
    )


def _progress_callback(accepted: int, rejected: int, total: int) -> None:
    print(
        f"    run {total:02d}: accepted={accepted}, rejected={rejected}",
        flush=True,
    )


def _compute_shared_limits(results: list, label_cfg: dict[str, str]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_mins: list[float] = []
    x_maxs: list[float] = []
    ocv_y_mins: list[float] = []
    ocv_y_maxs: list[float] = []
    dva_y_mins: list[float] = []
    dva_y_maxs: list[float] = []

    for idx, result in enumerate(results, start=1):
        fig = plot_ocv_model_param_show(
            result,
            cell_name="P45B",
            cu_index=idx,
            label_cfg=label_cfg,
            figsize_cm=FIGSIZE_CM,
            legend_ncols=2,
            latex_fonts=True,
        )
        ax_ocv, ax_dva = fig.axes[:2]
        x0, x1 = ax_ocv.get_xlim()
        y0, y1 = ax_ocv.get_ylim()
        d0, d1 = ax_dva.get_ylim()
        x_mins.append(x0)
        x_maxs.append(x1)
        ocv_y_mins.append(y0)
        ocv_y_maxs.append(y1)
        dva_y_mins.append(d0)
        dva_y_maxs.append(d1)
        plt.close(fig)

    shared_xlim = (min(x_mins), max(x_maxs))
    shared_ocv_ylim = (min(ocv_y_mins), max(ocv_y_maxs))
    shared_dva_ylim = (min(dva_y_mins), max(dva_y_maxs))
    return shared_xlim, shared_ocv_ylim, shared_dva_ylim


def _update_soc_guides(ax, y_limits: tuple[float, float]) -> None:
    y0, y1 = y_limits
    for line in ax.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        if x_data.size == 2 and y_data.size == 2 and np.isfinite(x_data).all():
            if np.isclose(x_data[0], x_data[1]) and (
                np.isclose(x_data[0], 0.0) or np.isclose(x_data[0], 1.0)
            ):
                line.set_ydata([y0, y1])


def _save_frames(
    results: list,
    frame_dir: Path,
    label_cfg: dict[str, str],
    shared_xlim: tuple[float, float],
    shared_ocv_ylim: tuple[float, float],
    shared_dva_ylim: tuple[float, float],
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    dva_ticks = np.arange(0.0, np.floor(shared_dva_ylim[1]) + 1.0, 1.0)
    if dva_ticks.size == 0:
        dva_ticks = np.array([0.0])

    for idx, result in enumerate(results, start=1):
        fig = plot_ocv_model_param_show(
            result,
            cell_name="P45B",
            cu_index=idx,
            label_cfg=label_cfg,
            figsize_cm=FIGSIZE_CM,
            legend_ncols=2,
            latex_fonts=True,
        )
        ax_ocv, ax_dva = fig.axes[:2]
        ax_ocv.set_xlim(*shared_xlim)
        ax_dva.set_xlim(*shared_xlim)
        ax_ocv.set_ylim(*shared_ocv_ylim)
        ax_dva.set_ylim(*shared_dva_ylim)
        ax_dva.set_yticks(dva_ticks)
        ax_dva.set_ylim(*shared_dva_ylim)
        _update_soc_guides(ax_ocv, shared_ocv_ylim)

        frame_path = frame_dir / f"frame_{idx:02d}.png"
        fig.savefig(frame_path, dpi=FRAME_DPI, facecolor="white")
        frame_paths.append(frame_path)
        plt.close(fig)
        print(f"  Saved frame {idx:02d}/09 -> {frame_path.name}", flush=True)

    return frame_paths


def _create_gif(frame_paths: list[Path], output_gif: Path) -> None:
    if not frame_paths:
        raise ValueError("No frames available for GIF creation.")

    resample = getattr(Image, "Resampling", Image).LANCZOS
    adaptive_palette = getattr(Image, "ADAPTIVE", None)
    if adaptive_palette is None:
        adaptive_palette = Image.Palette.ADAPTIVE
    images: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            frame = img.convert("RGB")
            if frame.width != TARGET_GIF_WIDTH_PX:
                target_height = int(round(frame.height * TARGET_GIF_WIDTH_PX / frame.width))
                frame = frame.resize((TARGET_GIF_WIDTH_PX, target_height), resample=resample)
            images.append(frame.convert("P", palette=adaptive_palette))

    durations = [FRAME_DURATION_MS] * len(images)
    durations[-1] = LAST_FRAME_DURATION_MS

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate PNG frames in doc/_tmp_ocp_shift_frames instead of deleting them.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional custom directory for intermediate frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("Starting GIF creation workflow...", flush=True)

    testdata_dir = _find_testdata_dir(ROOT)
    input_data_dir = testdata_dir / "InputData"
    output_gif = ROOT / "doc" / "OCP_shift_over_SOC.gif"
    frame_dir = (
        args.frames_dir.resolve()
        if args.frames_dir is not None
        else ROOT / "doc" / "_tmp_ocp_shift_frames"
    )

    print(f"Using TestData directory: {testdata_dir}", flush=True)
    print("Loading electrode data (Si-Gr blend anode + NCA cathode)...", flush=True)
    si_gr_anode, cathode = _load_blend_electrodes(input_data_dir)
    fit_data = _load_all_pocv_entries(testdata_dir)
    print(f"Loaded {len(fit_data)} pOCV entries (entry01..entry09).", flush=True)

    reference_capacity = float(fit_data[0][2].max() - fit_data[0][2].min())
    print(f"Reference capacity from entry01: {reference_capacity:.4f} Ah", flush=True)

    config = _build_blend_config()
    analyzer = DMAAnalyzer(config)
    analyzer.set_anode(si_gr_anode)
    analyzer.set_cathode(cathode)
    analyzer.set_reference_capacity(reference_capacity)

    results = []
    print("Running blend DMA fits...", flush=True)
    for idx, (entry_no, file_path, cap, volt, _efc) in enumerate(fit_data, start=1):
        cu_name = f"CU{entry_no}"
        print(f"[{idx}/9] Fitting {cu_name} from {file_path.name}", flush=True)
        t_start = time.perf_counter()
        result = analyzer.analyze(
            measured_capacity=cap,
            measured_voltage=volt,
            progress_callback=_progress_callback,
        )
        elapsed = time.perf_counter() - t_start
        result.cu_name = cu_name
        results.append(result)
        print(
            f"    done in {elapsed:.2f} s | RMSE={result.rmse * 1000:.2f} mV",
            flush=True,
        )

    label_cfg = {
        "label_cathode": "Cathode",
        "label_anode": "Anode",
    }

    print("Computing shared axis limits across all CUs...", flush=True)
    shared_xlim, shared_ocv_ylim, shared_dva_ylim = _compute_shared_limits(results, label_cfg)
    print(
        "  Shared limits "
        f"x={shared_xlim}, ocv_y={shared_ocv_ylim}, dva_y={shared_dva_ylim}",
        flush=True,
    )

    print("Rendering PNG frames...", flush=True)
    frame_paths: list[Path] = []
    try:
        frame_paths = _save_frames(
            results=results,
            frame_dir=frame_dir,
            label_cfg=label_cfg,
            shared_xlim=shared_xlim,
            shared_ocv_ylim=shared_ocv_ylim,
            shared_dva_ylim=shared_dva_ylim,
        )

        print("Creating animated GIF...", flush=True)
        _create_gif(frame_paths, output_gif)
        print(f"GIF written to: {output_gif}", flush=True)
    finally:
        if not args.keep_frames and frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)
            print(f"Cleaned up intermediate frames: {frame_dir}", flush=True)
        elif args.keep_frames:
            print(f"Kept intermediate frames: {frame_dir}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
