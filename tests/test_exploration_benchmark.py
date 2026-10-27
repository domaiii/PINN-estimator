import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from evaluation.benchmark_exploration import BenchmarkConfig, run_benchmark
from source_estimation_pinn.environment import OccupancyGrid
from source_estimation_pinn.estimation import GasSourceConfig


@pytest.fixture
def synthetic_data(tmp_path):
    data_dir = tmp_path / "data"
    scenario = data_dir / "synthetic_room" / "scenario_test"
    scenario.mkdir(parents=True)
    Image.fromarray(np.full((6, 6), 255, dtype=np.uint8)).save(scenario / "map.png")
    occupancy_path = scenario / "occupancy.yaml"
    occupancy_path.write_text(
        "image: map.png\nresolution: 0.2\norigin: [0, 0, 0]\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    points = OccupancyGrid.from_occupancy_file(occupancy_path).free_points
    concentration = np.exp(-np.sum((points - [0.5, 0.5]) ** 2, axis=1) / 0.1)
    pd.DataFrame({"Points:0": points[:, 0], "Points:1": points[:, 1],
                  "C": concentration}).to_csv(scenario / "gas.csv", index=False)
    pd.DataFrame({"Points:0": [0, 1.2, 0, 1.2],
                  "Points:1": [0, 0, 1.2, 1.2],
                  "U:0": [0.1] * 4, "U:1": [0.0] * 4}).to_csv(
                      scenario / "wind.csv", index=False)
    (scenario / "metadata.json").write_text(json.dumps({
        "occupancy": "occupancy.yaml", "wind": "wind.csv",
        "files": {"gas": "gas.csv"},
        "source": {"position": [0.5, 0.5], "sigma": 0.1, "rate": 1.0},
        "diffusivity": 0.02, "diffusion_speed_std": 0.0, "dt": 0.1,
    }))
    return data_dir


def test_parallel_trials_match_sequential_trials(tmp_path, synthetic_data):
    config = BenchmarkConfig(
        data_dir=synthetic_data, scenarios=("synthetic_room/scenario_test",),
        seeds=(0, 1), n_initial_measurements=2, n_acquisitions=1,
        initial_training_steps=2, update_training_steps=2,
        n_collocation_points=8, cpu_threads=1, parallel_runs=1,
        gas_absolute_noise_std=0.001,
        pinn=GasSourceConfig(concentration_hidden_layers=1,
                            concentration_hidden_dim=4, source_hidden_layers=1,
                            source_hidden_dim=4, device="cpu"),
        output_dir=tmp_path / "sequential",
    )
    sequential = run_benchmark(config)
    parallel = run_benchmark(replace(config, parallel_runs=2, output_dir=tmp_path / "parallel"))
    columns = ["selection_seconds", "training_seconds"]
    left = pd.read_csv(sequential / "metrics.csv").drop(columns=columns)
    right = pd.read_csv(parallel / "metrics.csv").drop(columns=columns)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert len(left) == 8
    for archive in (sequential / "runs").rglob("*.npz"):
        other = parallel / archive.relative_to(sequential)
        with np.load(archive) as a, np.load(other) as b:
            assert a.files == b.files
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])
    for name in ("summary.png", "position_by_scenario.png", "source_field_by_scenario.png"):
        assert (parallel / name).is_file()


def test_six_variant_study_is_paired_and_parallel_reproducible(tmp_path, synthetic_data):
    from evaluation.benchmark_exploration import run_exploration_benchmark

    config = BenchmarkConfig(
        data_dir=synthetic_data, scenarios=("synthetic_room/scenario_test",),
        seeds=(0, 1), n_initial_measurements=2, n_acquisitions=2,
        initial_training_steps=3, update_training_steps=2,
        n_collocation_points=8, cpu_threads=1, parallel_runs=1,
        gas_peak_noise_alpha=0.05,
        pinn=GasSourceConfig(concentration_hidden_layers=1, concentration_hidden_dim=4,
                            source_hidden_layers=1, source_hidden_dim=4),
        output_dir=tmp_path / "study_sequential",
    )
    sequential = run_exploration_benchmark(config)
    parallel = run_exploration_benchmark(
        replace(config, parallel_runs=2, output_dir=tmp_path / "study_parallel"))
    left = pd.read_csv(sequential / "metrics.csv").drop(columns=["selection_seconds", "training_seconds"])
    right = pd.read_csv(parallel / "metrics.csv").drop(columns=["selection_seconds", "training_seconds"])
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    expected_labels = {f"{method}_{mode}" for method in ("Random", "CLLLA", "QLLLA")
                       for mode in ("static", "adaptive")}
    assert set(left.strategy) == expected_labels
    assert len(left) == 36
    for seed_dir in (sequential / "runs").glob("*/*/seed_*"):
        archives = sorted(seed_dir.glob("*.npz"))
        assert len(archives) == 6
        hashes, starts, first_observations = [], [], []
        initial_fields = {}
        random_paths = []
        for archive in archives:
            with np.load(archive) as saved, np.load(parallel / archive.relative_to(sequential)) as other:
                for key in saved.files:
                    np.testing.assert_array_equal(saved[key], other[key])
                hashes.append(saved["initial_parameter_sha256"].item())
                starts.append(saved["measurement_indices"][:2].copy())
                first_observations.append(saved["observed_concentrations"][:2].copy())
                mode = saved["dynamic_weights"].item()
                initial_fields.setdefault(mode, []).append(saved["predicted_source"][0].copy())
                assert saved["training_losses"].shape[0] == 7
                np.testing.assert_array_equal(saved["training_round"], [0, 0, 0, 1, 1, 2, 2])
                weight_names = saved["training_weight_names"].tolist()
                data_weights = saved["training_weights"][:, weight_names.index("gas_data")]
                assert (not np.allclose(data_weights, data_weights[0])) == mode
                if "Random_" in archive.name:
                    random_paths.append(saved["measurement_indices"].copy())
        assert len(set(hashes)) == 1
        for start, observed in zip(starts, first_observations):
            np.testing.assert_array_equal(start, starts[0])
            np.testing.assert_array_equal(observed, first_observations[0])
        for fields in initial_fields.values():
            for values in fields:
                np.testing.assert_array_equal(values, fields[0])
        np.testing.assert_array_equal(*random_paths)

    metadata = json.loads((sequential / "config.json").read_text())
    scenario = metadata["scenarios"][0]
    assert scenario["effective_gas_absolute_noise_std"] == pytest.approx(
        0.05 * scenario["peak_concentration"])
    effective = scenario["effective_pinn_by_strategy"]
    assert [c["dynamic_weights"] for c in effective] == [False] * 3 + [True] * 3
    for name in ("summary.csv", "summary.png", "position_by_scenario.png",
                 "source_field_by_scenario.png", "input_manifest.json",
                 "provenance.json", "source_snapshot/evaluation/benchmark_exploration.py"):
        assert (sequential / name).is_file()
    manifest = json.loads((sequential / "input_manifest.json").read_text())
    assert set(manifest[0]["files"]) == {"metadata", "gas", "wind", "occupancy", "occupancy_image"}
    assert all(len(info["sha256"]) == 64 for info in manifest[0]["files"].values())
    with pytest.raises(FileExistsError):
        run_exploration_benchmark(config)


def test_cllla_selects_largest_available_concentration_variance():
    from types import SimpleNamespace
    from evaluation.benchmark_exploration import ConcentrationUncertaintySampling
    from evaluation.benchmark_exploration import SamplingContext
    import torch

    gradients = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    laplace = SimpleNamespace(
        covariance=torch.eye(2),
        split_covariance=lambda covariance: (torch.diag(torch.tensor([1., 4.])), None, None),
        concentration_gradient_wrt_parameters=lambda xy: gradients,
    )
    estimator = SimpleNamespace(laplace=laplace, _tensor=torch.as_tensor)
    context = SamplingContext(SimpleNamespace(estimator=estimator), np.zeros((3, 2)),
                              np.array([True, True, False]), np.random.default_rng(0))
    # Variances are 1, 4, 5; point 2 is already measured.
    assert ConcentrationUncertaintySampling().select(context) == 1


def test_peak_noise_is_constant_and_scales_with_scenario_peak():
    from evaluation.benchmark_exploration import noisy_measurements

    config = BenchmarkConfig(gas_peak_noise_alpha=0.05)
    concentrations = np.array([0.0, 0.001, 0.2, 2.0])
    observed = noisy_measurements(concentrations, config, seed=3)
    draws = np.random.default_rng(np.random.SeedSequence([3, 2])).standard_normal(4)
    np.testing.assert_allclose(observed - concentrations, draws * 0.1)
    np.testing.assert_allclose(
        noisy_measurements(10 * concentrations, config, seed=3), 10 * observed)
    with pytest.raises(ValueError, match="cannot be combined"):
        BenchmarkConfig(gas_peak_noise_alpha=0.05, gas_relative_noise_std=0.05)


def test_exploration_does_not_import_legacy_sampling():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import importlib.abc
import sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'evaluation.benchmark_sampling':
            raise ImportError('Legacy benchmark is unavailable')
sys.meta_path.insert(0, BlockLegacy())
from evaluation.benchmark_exploration import CONFIG, study_strategies
assert len(study_strategies()) == 6
assert CONFIG.gas_peak_noise_alpha == 0.05
"""], check=True)
