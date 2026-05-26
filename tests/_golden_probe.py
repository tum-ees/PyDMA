"""Probe: produce the deterministic P45B golden numbers used in the regression test.

This script is one-shot. It is NOT a test, just a helper to generate the
golden JSON. Run it whenever you intentionally want to refresh the
reference values, then commit the updated tests/golden/p45b_*.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from pydma import DMAAnalyzer, DMAConfig, ElectrodeOCP, load_ocp  # noqa: E402

INPUT_DIR = repo_root / "notebooks" / "TestData" / "InputData"
POCV_FILE = repo_root / "notebooks" / "TestData" / "FR23_pOCV_CH_entry01.csv"

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

pocv = pd.read_csv(POCV_FILE)
mask = ~(pocv["Ah_Step"].isna() | pocv["U"].isna())
meas_cap = pocv["Ah_Step"][mask].to_numpy()
meas_volt = pocv["U"][mask].to_numpy()
ref_cap = float(meas_cap.max() - meas_cap.min())

config = DMAConfig(
    speed_preset="fast",
    direction="charge",
    data_length=1000,
    smoothing_points=30,
    weight_ocv=100.0,
    weight_dva=1.0,
    weight_ica=0.0,
    req_accepted=1,
    max_tries_overall=1,
    rmse_threshold=0.01,
    allow_anode_inhomogeneity=False,
    allow_cathode_inhomogeneity=False,
    print_progress=False,
    workers=1,
    random_seed=1234,
)

analyzer = DMAAnalyzer(config)
analyzer.set_anode(anode)
analyzer.set_cathode(cathode)
analyzer.set_reference_capacity(ref_cap)

result = analyzer.analyze(measured_capacity=meas_cap, measured_voltage=meas_volt)

fp = result.fitted_params
dm = result.degradation_modes
golden = {
    "case": "p45b_serial23_entry01_cu1_nonblend",
    "config": {
        "speed_preset": config.speed_preset,
        "direction": config.direction,
        "smoothing_points": config.smoothing_points,
        "data_length": config.data_length,
        "weight_ocv": config.weight_ocv,
        "weight_dva": config.weight_dva,
        "weight_ica": config.weight_ica,
        "req_accepted": config.req_accepted,
        "max_tries_overall": config.max_tries_overall,
        "rmse_threshold": config.rmse_threshold,
        "allow_anode_inhomogeneity": config.allow_anode_inhomogeneity,
        "allow_cathode_inhomogeneity": config.allow_cathode_inhomogeneity,
        "workers": config.workers,
        "random_seed": config.random_seed,
    },
    "rmse": float(result.rmse),
    "rmse_fit_region": float(result.rmse_fit_region),
    "rmse_full_range": float(result.rmse_full_range),
    "fitted_params": {
        "alpha_an": float(fp.alpha_an),
        "beta_an": float(fp.beta_an),
        "alpha_ca": float(fp.alpha_ca),
        "beta_ca": float(fp.beta_ca),
        "utilization_an": float(fp.utilization_an),
        "utilization_ca": float(fp.utilization_ca),
        "sto_init_an": float(fp.sto_init_an),
        "sto_init_ca": float(fp.sto_init_ca),
    },
    "degradation_modes": {
        "lli": float(dm.lli),
        "lam_an": float(dm.lam_an),
        "lam_ca": float(dm.lam_ca),
    },
}
print(json.dumps(golden, indent=2))
