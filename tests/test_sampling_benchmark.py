import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from evaluation.benchmark_sampling import BenchmarkConfig, run_benchmark
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
