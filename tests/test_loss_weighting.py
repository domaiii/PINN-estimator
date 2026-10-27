import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

root = Path.cwd()
if not (root / "source_estimation_pinn").exists():
    root = root.parent
sys.path.insert(0, str(root))

from source_estimation_pinn.environment import CellType, OccupancyGrid, WindField
from source_estimation_pinn.estimation import (
    GasSourceConfig,
    GasSourceEstimator,
)
from source_estimation_pinn.measurements import GasSampleSet
from source_estimation_pinn.estimation.loss_weighting import LearnedWeights, StaticWeights

REFERENCE = Path(__file__).parent / "test_data" / "loss_weighting_reference_static.csv"


@pytest.fixture
def estimator() -> GasSourceEstimator:
    torch.manual_seed(0)
    cells = np.full((4, 4), CellType.FREE, dtype=np.uint8)
    cells[1, 1] = CellType.OCCUPIED
    cells[2, -1] = CellType.OPEN_BOUNDARY
    occupancy = OccupancyGrid(
        occupancy=cells,
        resolution=1.0,
        origin=np.array([0.0, 0.0]),
    )

    corners = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])
    wind = WindField(positions=corners, velocities=np.tile([-0.3, 0.0], (4, 1)))

    config = GasSourceConfig(
        concentration_hidden_layers=2,
        concentration_hidden_dim=4,
        source_hidden_layers=2,
        source_hidden_dim=4,
    )

    return GasSourceEstimator(
        occupancy=occupancy,
        wind=wind,
        config=config,
    )


@pytest.fixture
def samples() -> GasSampleSet:
    positions = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 2.5], [0.5, 3.5]])
    concentrations = np.array([1.0, 2.0, 3.0, 0.5])
    return GasSampleSet(positions, concentrations)


def test_fit_and_predict(estimator, samples):
    estimator.fit(samples, steps=10, n_collocation_points=32)
    test_positions = estimator.occupancy.free_points
    concentration = estimator.predict_concentration(test_positions)
    source = estimator.predict_source(test_positions)

    assert concentration.shape == source.shape == (len(test_positions),)
    assert np.isfinite(concentration).all() and np.isfinite(source).all()


def test_loss_reference(estimator, samples):
    estimator.fit(samples, steps=10, n_collocation_points=32)
    losses = estimator.history.losses
    df = pd.DataFrame(losses)

    reference = pd.read_csv(REFERENCE)
    pd.testing.assert_frame_equal(df, reference, rtol=1e-3, atol=0.0)


def test_learned_weights_start_at_configured_values():
    config = GasSourceConfig(lambda_gas_data=2.0, lambda_ad=3.0,
                             lambda_inflow=4.0, lambda_gas_wall=5.0,
                             lambda_nonfree=0.7, lambda_sparse=0.02)
    learned = LearnedWeights(config)
    for name, expected in StaticWeights(config)().items():
        torch.testing.assert_close(learned()[name], expected)
    assert len(list(learned.parameters())) == 5
    torch.testing.assert_close(
        learned.log_term({"gas_wall", "nonfree"}),
        learned.log_eps_gas_wall + learned.log_eps_nonfree,
    )


def test_optimizer_updates_all_active_learned_weights(estimator, samples):
    dynamic = GasSourceEstimator(estimator.occupancy, estimator.wind,
                                 replace(estimator.config, dynamic_weights=True))
    before = {name: parameter.detach().clone()
              for name, parameter in dynamic.loss_weights.named_parameters()}
    history = dynamic.fit(samples, steps=3, n_collocation_points=8)
    for name, parameter in dynamic.loss_weights.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert not torch.equal(parameter.detach(), before[name])
    for name in ("gas_data", "ad", "inflow", "gas_wall", "nonfree"):
        assert len(history.weights[name]) == 3
        assert np.isfinite(history.weights[name]).all()
    assert np.isfinite(history.total).all()


@pytest.mark.parametrize("open_edges", [False, True])
def test_empty_groups_do_not_optimize_their_log_scales(open_edges):
    cells = np.full((4, 4), CellType.FREE, dtype=np.uint8)
    if open_edges:
        cells[[0, -1], :] = CellType.OPEN_BOUNDARY
        cells[:, [0, -1]] = CellType.OPEN_BOUNDARY
    grid = OccupancyGrid(cells, 1.0, np.zeros(2))
    wind = WindField(np.array([[0, 0], [4, 0], [0, 4], [4, 4]]), np.zeros((4, 2)))
    config = GasSourceConfig(dynamic_weights=True, concentration_hidden_layers=1,
                             concentration_hidden_dim=4, source_hidden_layers=1,
                             source_hidden_dim=4)
    dynamic = GasSourceEstimator(grid, wind, config)
    inactive_names = ["log_eps_inflow", "log_eps_gas_wall" if open_edges else "log_eps_nonfree"]
    before = {name: getattr(dynamic.loss_weights, name).detach().clone()
              for name in inactive_names}
    samples = GasSampleSet(grid.free_points[:2], np.array([0.1, 0.2]))
    dynamic.fit(samples, steps=3, n_collocation_points=4)
    for name in inactive_names:
        torch.testing.assert_close(getattr(dynamic.loss_weights, name), before[name],
                                   rtol=0, atol=0)


@pytest.mark.parametrize("name", ["lambda_inflow", "lambda_gas_wall", "lambda_nonfree"])
def test_learned_boundary_weights_require_positive_initial_values(name):
    with pytest.raises(ValueError, match=name):
        LearnedWeights(replace(GasSourceConfig(), **{name: 0.0}))
