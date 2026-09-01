"""Scientific regression tests.

These exercise the full DMA pipeline against saved golden numbers, using a
fixed ``DMAConfig.random_seed`` so the optimizer is deterministic. They
catch silent behavior shifts that the unit tests miss (e.g. a wrong
default propagating through the optimizer).

Run only these:

    pytest -m scientific

Run everything except these (fast suite):

    pytest -m "not scientific"

To refresh the golden numbers after an intentional, validated change,
re-run ``tests/_golden_probe.py`` and replace the JSON.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pydma import DMAAnalyzer, DMAConfig, ElectrodeOCP, load_ocp

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = REPO_ROOT / "notebooks" / "TestData" / "InputData"
POCV_DIR = REPO_ROOT / "notebooks" / "TestData"
GOLDEN_DIR = Path(__file__).parent / "golden"

_REQUIRED_FILES = [
    INPUT_DIR / "SiGr_blend_anode" / "P45B_Anode_Lithiation_0C03.csv",
    INPUT_DIR / "NCA" / "GITT_P45b_Cat_NCA_JN_VS_Coin_1_GITT__Extracted_Continuous_pOCP.csv",
    POCV_DIR / "FR23_pOCV_CH_entry01.csv",
    GOLDEN_DIR / "p45b_serial23_entry01_cu1_nonblend.json",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED_FILES),
    reason=(
        "scientific regression test data not available "
        "(typical for sdist installs); run from a git checkout"
    ),
)

# Tolerances chosen to (a) survive numpy/scipy patch-level drift on the
# supported floors and (b) still catch real shifts. RMSE moves the least
# (~1e-7 across reruns); fitted params can move at the 5th-6th decimal.
ABS_TOL_RMSE = 5e-5  # 0.05 mV
ABS_TOL_PARAM = 5e-4
ABS_TOL_DEGMODE = 1e-3  # 0.1 % LLI/LAM
REL_TOL = 2e-3  # 0.2 %


def _assert_close(actual: float, expected: float, *, abs_tol: float, name: str) -> None:
    assert math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=abs_tol), (
        f"{name}: actual={actual!r}, expected={expected!r}, "
        f"diff={actual - expected!r} (abs_tol={abs_tol}, rel_tol={REL_TOL})"
    )


@pytest.mark.scientific
def test_p45b_serial23_entry01_cu1_nonblend() -> None:
    """Lock down the non-blend P45B/NCA fit on FR23_pOCV_CH_entry01.

    Smallest possible end-to-end scientific case:
      - single CU, single optimizer run, fast preset
      - no inhomogeneity, no blend
      - deterministic seed
    """
    golden_path = GOLDEN_DIR / "p45b_serial23_entry01_cu1_nonblend.json"
    with open(golden_path) as f:
        golden = json.load(f)
    cfg_in = golden["config"]

    anode_df = pd.read_csv(INPUT_DIR / "SiGr_blend_anode" / "P45B_Anode_Lithiation_0C03.csv")
    anode = ElectrodeOCP(
        soc=anode_df["normalizedCapacity"].values,
        voltage=anode_df["voltage"].values,
        name="Si-Gr Anode (P45B)",
    )
    cathode = load_ocp(
        INPUT_DIR / "NCA" / "GITT_P45b_Cat_NCA_JN_VS_Coin_1_GITT__Extracted_Continuous_pOCP.csv",
        electrode_type="cathode",
        smooth=False,
    )

    pocv = pd.read_csv(POCV_DIR / "FR23_pOCV_CH_entry01.csv")
    mask = ~(pocv["Ah_Step"].isna() | pocv["U"].isna())
    meas_cap = pocv["Ah_Step"][mask].to_numpy()
    meas_volt = pocv["U"][mask].to_numpy()
    ref_cap = float(meas_cap.max() - meas_cap.min())

    config = DMAConfig(
        speed_preset=cfg_in["speed_preset"],
        direction=cfg_in["direction"],
        data_length=cfg_in["data_length"],
        smoothing_points=cfg_in["smoothing_points"],
        weight_ocv=cfg_in["weight_ocv"],
        weight_dva=cfg_in["weight_dva"],
        weight_ica=cfg_in["weight_ica"],
        req_accepted=cfg_in["req_accepted"],
        max_tries_overall=cfg_in["max_tries_overall"],
        rmse_threshold=cfg_in["rmse_threshold"],
        allow_anode_inhomogeneity=cfg_in["allow_anode_inhomogeneity"],
        allow_cathode_inhomogeneity=cfg_in["allow_cathode_inhomogeneity"],
        workers=cfg_in["workers"],
        random_seed=cfg_in["random_seed"],
        print_progress=False,
    )

    analyzer = DMAAnalyzer(config)
    analyzer.set_anode(anode)
    analyzer.set_cathode(cathode)
    analyzer.set_reference_capacity(ref_cap)

    result = analyzer.analyze(measured_capacity=meas_cap, measured_voltage=meas_volt)

    _assert_close(result.rmse, golden["rmse"], abs_tol=ABS_TOL_RMSE, name="rmse")
    _assert_close(
        result.rmse_fit_region,
        golden["rmse_fit_region"],
        abs_tol=ABS_TOL_RMSE,
        name="rmse_fit_region",
    )
    _assert_close(
        result.rmse_full_range,
        golden["rmse_full_range"],
        abs_tol=ABS_TOL_RMSE,
        name="rmse_full_range",
    )

    fp = result.fitted_params
    g_fp = golden["fitted_params"]
    for key in (
        "alpha_an",
        "beta_an",
        "alpha_ca",
        "beta_ca",
        "utilization_an",
        "utilization_ca",
        "sto_init_an",
        "sto_init_ca",
    ):
        _assert_close(float(getattr(fp, key)), g_fp[key], abs_tol=ABS_TOL_PARAM, name=key)

    dm = result.degradation_modes
    g_dm = golden["degradation_modes"]
    for key in ("lli", "lam_an", "lam_ca"):
        _assert_close(float(getattr(dm, key)), g_dm[key], abs_tol=ABS_TOL_DEGMODE, name=key)


@pytest.mark.scientific
def test_random_seed_is_deterministic() -> None:
    """Two analyzer instances with the same seed must produce bit-identical fits.

    Cheap sanity check that protects the seeding wiring itself: if a future
    refactor breaks ``DMAConfig.random_seed`` propagation into
    ``differential_evolution``, the parameters will diverge here long before
    the heavier golden test starts flapping.
    """
    anode_df = pd.read_csv(INPUT_DIR / "SiGr_blend_anode" / "P45B_Anode_Lithiation_0C03.csv")
    anode = ElectrodeOCP(
        soc=anode_df["normalizedCapacity"].values,
        voltage=anode_df["voltage"].values,
        name="Si-Gr Anode (P45B)",
    )
    cathode = load_ocp(
        INPUT_DIR / "NCA" / "GITT_P45b_Cat_NCA_JN_VS_Coin_1_GITT__Extracted_Continuous_pOCP.csv",
        electrode_type="cathode",
        smooth=False,
    )
    pocv = pd.read_csv(POCV_DIR / "FR23_pOCV_CH_entry01.csv")
    mask = ~(pocv["Ah_Step"].isna() | pocv["U"].isna())
    meas_cap = pocv["Ah_Step"][mask].to_numpy()
    meas_volt = pocv["U"][mask].to_numpy()
    ref_cap = float(meas_cap.max() - meas_cap.min())

    def _fit() -> tuple[float, float, float, float, float]:
        cfg = DMAConfig(
            speed_preset="fast",
            direction="charge",
            req_accepted=1,
            max_tries_overall=1,
            allow_anode_inhomogeneity=False,
            allow_cathode_inhomogeneity=False,
            print_progress=False,
            random_seed=4321,
        )
        a = DMAAnalyzer(cfg)
        a.set_anode(anode)
        a.set_cathode(cathode)
        a.set_reference_capacity(ref_cap)
        r = a.analyze(measured_capacity=meas_cap, measured_voltage=meas_volt)
        fp = r.fitted_params
        return (
            float(r.rmse),
            float(fp.alpha_an),
            float(fp.beta_an),
            float(fp.alpha_ca),
            float(fp.beta_ca),
        )

    first = _fit()
    second = _fit()
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
