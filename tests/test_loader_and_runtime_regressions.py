from pathlib import Path
import sys
import tempfile
import itertools
from unittest.mock import patch
import warnings

import numpy as np
from scipy.io import savemat


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydma.core.analyzer import DMAAnalyzer
from pydma.core.optimizer import MultiRunResult, OptimizationRun
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.preprocessing.loader import load_aging_study, load_ocp, load_pocv
from pydma.utils.dma_config import DMAConfig
from pydma.utils.results import DMAResult


def test_load_pocv_detects_fullcell_columns_and_capacity():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cu1.csv"
        path.write_text(
            "Time,U,Ah_Step,pOCV_CH\n"
            "0,3.00,0.00,4.20\n"
            "1,3.50,2.10,4.20\n"
            "2,4.00,4.20,4.20\n",
            encoding="utf-8",
        )

        capacity_axis, voltage, capacity = load_pocv(path)

    np.testing.assert_allclose(capacity_axis, np.array([0.0, 2.1, 4.2]))
    np.testing.assert_allclose(voltage, np.array([3.0, 3.5, 4.0]))
    assert capacity == 4.2


def test_load_pocv_returns_none_for_normalized_soc_without_capacity_column():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cu1.csv"
        path.write_text(
            "SOC,U\n"
            "0.0,3.0\n"
            "0.5,3.5\n"
            "1.0,4.0\n",
            encoding="utf-8",
        )

        soc, voltage, capacity = load_pocv(path)

    np.testing.assert_allclose(soc, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(voltage, np.array([3.0, 3.5, 4.0]))
    assert capacity is None


def test_load_ocp_handles_nested_mat_struct():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "electrode.mat"
        savemat(
            path,
            {
                "electrode": {
                    "SOC": np.array([0.0, 0.5, 1.0]),
                    "UCa": np.array([3.0, 3.5, 4.0]),
                }
            },
        )

        ocp = load_ocp(path, electrode_type="cathode")

    np.testing.assert_allclose(ocp.soc, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(ocp.voltage, np.array([3.0, 3.5, 4.0]))


def test_load_aging_study_selects_requested_direction():
    with tempfile.TemporaryDirectory() as td:
        cu_dir = Path(td) / "CU1"
        cu_dir.mkdir()
        (cu_dir / "entry_charge.csv").write_text(
            "Ah_Step,U,pOCV_CH\n0.0,3.0,4.2\n1.0,3.5,4.2\n2.0,4.0,4.2\n",
            encoding="utf-8",
        )
        (cu_dir / "entry_discharge.csv").write_text(
            "Ah_Step,U,pOCV_DCH\n0.0,4.0,4.1\n1.0,3.6,4.1\n2.0,3.2,4.1\n",
            encoding="utf-8",
        )

        charge_data = load_aging_study(td, direction="charge")
        discharge_data = load_aging_study(td, direction="discharge")

    assert charge_data["CU1"][1][0] == 3.0
    assert discharge_data["CU1"][1][0] == 4.0


def test_load_aging_study_reads_single_mat_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "study.mat"
        savemat(
            path,
            {
                "CU1": {
                    "SOC": np.array([0.0, 0.5, 1.0]),
                    "U": np.array([3.0, 3.5, 4.0]),
                    "Capacity": np.array([5.0]),
                },
                "CU2": {
                    "SOC": np.array([0.0, 0.5, 1.0]),
                    "U": np.array([3.1, 3.6, 4.1]),
                    "Capacity": np.array([4.8]),
                },
            },
        )

        data = load_aging_study(path, direction="charge")

    assert list(data.keys()) == ["CU1", "CU2"]
    assert data["CU1"][2] == 5.0
    assert data["CU2"][2] == 4.8


def test_load_aging_study_reads_table_like_single_mat_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "study_table.mat"
        savemat(
            path,
            {
                "study": {
                    "CU": np.array(["CU1", "CU2"], dtype=object),
                    "Testdata_pOCV_CH": np.array(
                        [
                            {"SOC": np.array([0.0, 0.5, 1.0]), "U": np.array([3.0, 3.5, 4.0])},
                            {"SOC": np.array([0.0, 0.5, 1.0]), "U": np.array([3.1, 3.6, 4.1])},
                        ],
                        dtype=object,
                    ),
                    "pOCV_CH": np.array([5.0, 4.8]),
                }
            },
        )

        data = load_aging_study(path, direction="charge")

    assert list(data.keys()) == ["CU1", "CU2"]
    np.testing.assert_allclose(data["CU1"][1], np.array([3.0, 3.5, 4.0]))
    np.testing.assert_allclose(data["CU2"][1], np.array([3.1, 3.6, 4.1]))
    assert data["CU1"][2] == 5.0
    assert data["CU2"][2] == 4.8


def test_analyze_aging_study_requires_actual_capacity_when_loader_returns_none():
    analyzer = DMAAnalyzer()

    try:
        analyzer.analyze_aging_study(
            {"CU1": (np.array([0.0, 0.5, 1.0]), np.array([3.0, 3.5, 4.0]), None)}
        )
    except ValueError as exc:
        assert "actual capacity value" in str(exc)
    else:
        raise AssertionError("Expected ValueError when actual capacity is missing.")


def test_analyze_aging_study_passes_actual_capacity_to_analyze():
    class RecordingAnalyzer(DMAAnalyzer):
        def __init__(self):
            super().__init__()
            self.actual_capacities = []

        def analyze(self, *args, actual_capacity=None, **kwargs):
            self.actual_capacities.append(actual_capacity)
            return DMAResult(capacity=float(actual_capacity or 0.0))

    analyzer = RecordingAnalyzer()
    analyzer.analyze_aging_study(
        {"CU1": (np.array([0.0, 0.5, 1.0]), np.array([3.0, 3.5, 4.0]), 4.2)}
    )

    assert analyzer.actual_capacities == [4.2]


def test_analyze_aging_study_loads_path_using_config_direction():
    class RecordingAnalyzer(DMAAnalyzer):
        def __init__(self, config):
            super().__init__(config=config)
            self.actual_capacities = []
            self.first_voltages = []

        def analyze(self, measured_capacity, measured_voltage, actual_capacity=None, **kwargs):
            self.actual_capacities.append(actual_capacity)
            self.first_voltages.append(float(np.asarray(measured_voltage)[0]))
            return DMAResult(capacity=float(actual_capacity or 0.0))

    config = DMAConfig(direction="discharge")
    analyzer = RecordingAnalyzer(config=config)

    with tempfile.TemporaryDirectory() as td:
        cu_dir = Path(td) / "CU1"
        cu_dir.mkdir()
        (cu_dir / "entry_charge.csv").write_text(
            "Ah_Step,U,pOCV_CH\n0.0,3.0,4.2\n1.0,3.5,4.2\n2.0,4.0,4.2\n",
            encoding="utf-8",
        )
        (cu_dir / "entry_discharge.csv").write_text(
            "Ah_Step,U,pOCV_DCH\n0.0,4.0,4.1\n1.0,3.6,4.1\n2.0,3.2,4.1\n",
            encoding="utf-8",
        )

        analyzer.analyze_aging_study(td)

    assert analyzer.actual_capacities == [4.1]
    assert analyzer.first_voltages == [4.0]


def test_analyze_aging_study_rejects_normalized_two_tuple_study():
    analyzer = DMAAnalyzer()
    pocv_data = {
        "CU1": (np.array([0.0, 0.5, 1.0]), np.array([3.0, 3.5, 4.0])),
        "CU2": (np.array([0.0, 0.5, 1.0]), np.array([3.1, 3.6, 4.1])),
    }

    try:
        analyzer.analyze_aging_study(pocv_data)
    except ValueError as exc:
        assert "normalized 0..1 capacity axes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous normalized 2-tuple study.")


def test_analyze_aging_study_warns_for_two_tuple_input():
    class RecordingAnalyzer(DMAAnalyzer):
        def analyze(self, *args, actual_capacity=None, **kwargs):
            return DMAResult(capacity=float(actual_capacity or 0.0))

    analyzer = RecordingAnalyzer()
    pocv_data = {
        "CU1": (np.array([0.0, 0.4, 0.8]), np.array([3.0, 3.4, 3.8])),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer.analyze_aging_study(pocv_data)

    assert any("2-tuple aging-study input assumes" in str(w.message) for w in caught)


def test_config_algorithm_and_initial_guess_are_active():
    config = DMAConfig(
        algorithm="DE",
        use_anode_blend=True,
        use_cathode_blend=True,
        gamma_anode_blend2_init=0.22,
        gamma_cathode_blend2_init=0.44,
    )

    init = config.get_initial_guess()

    assert config.algorithm == "differential_evolution"
    assert init[4] == 0.22
    assert init[5] == 0.44


def test_get_initial_guess_stays_within_bounds_for_all_flag_combos():
    for use_ab, use_cb, allow_inhom_an, allow_inhom_ca in itertools.product(
        [False, True],
        repeat=4,
    ):
        cfg = DMAConfig(
            use_anode_blend=use_ab,
            use_cathode_blend=use_cb,
            allow_anode_inhomogeneity=allow_inhom_an,
            allow_cathode_inhomogeneity=allow_inhom_ca,
        )
        init = cfg.get_initial_guess()
        lb, ub = cfg.get_full_bounds()
        assert np.all(init >= lb - 1e-12)
        assert np.all(init <= ub + 1e-12)


def test_get_initial_guess_stays_within_bounds_with_previous_inhom_constraints():
    cfg = DMAConfig(
        allow_anode_inhomogeneity=True,
        allow_cathode_inhomogeneity=True,
        max_inhomogeneity=0.3,
        max_inhomogeneity_delta=0.05,
    )

    init = cfg.get_initial_guess(inhom_an_prev=0.12, inhom_ca_prev=0.18)
    lb, ub = cfg.get_full_bounds(inhom_an_prev=0.12, inhom_ca_prev=0.18)

    assert np.all(init >= lb - 1e-12)
    assert np.all(init <= ub + 1e-12)


def test_analyze_sets_real_fit_metrics_and_result_metadata():
    q = np.linspace(0.0, 1.0, 100)
    anode = ElectrodeOCP(
        soc=q,
        voltage=0.2 - 0.1 * q,
        electrode_type="anode",
    )
    cathode = ElectrodeOCP(
        soc=q,
        voltage=3.0 + 0.5 * q,
        electrode_type="cathode",
    )
    measured_voltage = (3.0 + 0.5 * q) - (0.2 - 0.1 * q)

    config = DMAConfig(
        weight_ocv=1.0,
        weight_dva=0.0,
        weight_ica=0.0,
        req_accepted=1,
        max_tries_overall=1,
        print_progress=False,
        speed_preset="fast",
    )
    analyzer = DMAAnalyzer(config=config, anode=anode, cathode=cathode)

    dummy_result = MultiRunResult(
        best_params=np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        best_cost=123.0,
        best_rmse=0.5,
        accepted_runs=[],
        rejected_runs=[
            OptimizationRun(
                params=np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                cost=123.0,
                rmse=0.5,
                success=True,
                n_iterations=1,
                n_function_evals=1,
            )
        ],
    )

    class DummyOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            return dummy_result

    with patch("pydma.core.analyzer.DMAOptimizer", DummyOptimizer):
        result = analyzer.analyze(
            measured_capacity=q,
            measured_voltage=measured_voltage,
            actual_capacity=4.2,
        )

    assert result.is_accepted is False
    assert result.status == "rejected_above_threshold"
    assert result.algorithm == "differential_evolution"
    assert result.capacity == 4.2
    assert result.fit_ocv_mse < 1e-10
    assert result.fit_dva_mse == 0.0
    assert result.fit_ica_mse == 0.0


def test_reorientation_warning_is_once_until_reset():
    analyzer = DMAAnalyzer()
    measured_capacity = np.array([0.0, 0.5, 1.0])
    measured_voltage = np.array([4.2, 3.6, 3.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer._prepare_measured_data(measured_capacity, measured_voltage)
        analyzer._prepare_measured_data(measured_capacity, measured_voltage)

    reorientation_warnings = [
        warning for warning in caught if "Reoriented the full-cell axis" in str(warning.message)
    ]
    assert len(reorientation_warnings) == 1

    analyzer.reset_state()

    with warnings.catch_warnings(record=True) as caught_after_reset:
        warnings.simplefilter("always")
        analyzer._prepare_measured_data(measured_capacity, measured_voltage)

    reorientation_warnings_after_reset = [
        warning
        for warning in caught_after_reset
        if "Reoriented the full-cell axis" in str(warning.message)
    ]
    assert len(reorientation_warnings_after_reset) == 1


def test_load_aging_study_is_exported_top_level():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import pydma

    assert callable(pydma.load_aging_study)
