import numpy as np
import pytest
import torch

from source_estimation_pinn.environment import CellType, OccupancyGrid, WindField
from source_estimation_pinn.estimation import (
    GasSourceConfig,
    UncertaintyBasedGasSourceEstimator,
)
from source_estimation_pinn.measurements import GasSampleSet


@pytest.fixture
def explorer():
    occupancy = OccupancyGrid(
        occupancy=np.full((2, 2), CellType.FREE, dtype=np.uint8),
        resolution=1.0,
        origin=np.array([0.0, 0.0]),
    )
    wind = WindField(
        positions=np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]),
        velocities=np.zeros((4, 2)),
    )
    torch.manual_seed(0)
    return UncertaintyBasedGasSourceEstimator(
        occupancy,
        wind,
        GasSourceConfig(
            concentration_hidden_layers=1,
            concentration_hidden_dim=2,
            source_hidden_layers=1,
            source_hidden_dim=2,
        ),
        initial_training_steps=1,
        update_training_steps=1,
        n_collocation_points=4,
    )

def test_suggestion_masks_previously_measured_points(explorer, monkeypatch):
    explorer.fit(GasSampleSet(np.array([[0.5, 0.5]]), np.array([0.1])))
    monkeypatch.setattr(
        explorer.estimator,
        "source_uncertainty_reduction",
        lambda *args, **kwargs: torch.tensor([10.0, 1.0, 2.0, 3.0]),
    )

    np.testing.assert_array_equal(explorer.suggest_next_measurement(), [1.5, 1.5])


def test_new_measurement_is_stored_retrained_and_excluded(explorer, monkeypatch):
    explorer.fit(GasSampleSet(np.array([[0.5, 0.5]]), np.array([0.1])))
    next_point = np.array([1.5, 0.5])
    history = explorer.add_measurement(next_point, 0.2)

    assert len(history.total) == 1
    assert len(explorer.samples.positions) == 2
    with pytest.raises(ValueError, match="already"):
        explorer.add_measurement(next_point, 0.3)

    monkeypatch.setattr(
        explorer.estimator,
        "source_uncertainty_reduction",
        lambda *args, **kwargs: torch.tensor([10.0, 9.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(explorer.suggest_next_measurement(), [1.5, 1.5])