import numpy as np
import pytest
import torch

from source_estimation_pinn.environment import (
    CellType,
    OccupancyGrid,
    WindField,
)
from source_estimation_pinn.estimation import (
    GasSourceConfig,
    GasSourceEstimator,
)
from source_estimation_pinn.estimation.laplace import JointSourceLaplace
from source_estimation_pinn.measurements import GasSampleSet


@pytest.mark.parametrize("small_eigenvalue", [-1e-5, 0.0, 1e-9, 0.5])
def test_covariance_from_loss_clips_hessian_eigenvalues(small_eigenvalue):
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    eigenvalues = torch.tensor([small_eigenvalue, 2.0], dtype=torch.float64)
    hessian = rotation @ torch.diag(eigenvalues) @ rotation.T

    def loss(parameters):
        return 0.5 * parameters @ hessian @ parameters

    covariance = JointSourceLaplace.covariance_from_loss(
        loss, torch.zeros(2, dtype=torch.float64)
    )
    expected = rotation @ torch.diag(eigenvalues.clamp_min(1e-7).reciprocal()) @ rotation.T
    torch.testing.assert_close(covariance, expected)
    torch.testing.assert_close(covariance, covariance.T)
    assert torch.isfinite(covariance).all()
    assert torch.linalg.cholesky_ex(covariance).info.item() == 0


def test_covariance_from_ill_conditioned_float32_hessian_is_positive_definite():
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float32)
    hessian = rotation @ torch.diag(torch.tensor([-1e-8, 0.04])) @ rotation.T

    def loss(parameters):
        return 0.5 * parameters @ hessian @ parameters

    covariance = JointSourceLaplace.covariance_from_loss(
        loss, torch.zeros(2, dtype=torch.float32)
    )
    assert covariance.dtype == torch.float64
    assert torch.linalg.cholesky_ex(covariance).info.item() == 0


@pytest.fixture
def estimator() -> GasSourceEstimator:
    torch.manual_seed(0)

    occupancy = OccupancyGrid(
        occupancy=np.full(
            (2, 2),
            CellType.FREE,
            dtype=np.uint8,
        ),
        resolution=1.0,
        origin=np.array([0.0, 0.0]),
    )

    wind_positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 2.0],
    ])
    wind_velocities = np.zeros((4, 2))

    wind = WindField(
        positions=wind_positions,
        velocities=wind_velocities,
    )

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
def fitted_estimator(estimator):
    samples = GasSampleSet(
        positions=np.array([[0.5, 0.5], [1.5, 1.5]]),
        concentrations=np.array([0.2, 0.1]),
    )
    estimator.fit(samples, steps=1, n_collocation_points=4)
    return estimator


def test_concentration_gradient_wrt_ll_params(fitted_estimator):
    estimator = fitted_estimator
    xy = estimator._tensor([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    gradient = estimator.laplace.concentration_gradient_wrt_parameters(xy)

    # Independently differentiate each scalar output of the actual network.
    layer = estimator.model.c_net.net[-1]
    expected_rows = []
    for point in xy:
        concentration = estimator.model.c_net(point[None, :]).sum()
        weight_gradient, bias_gradient = torch.autograd.grad(
            concentration, (layer.weight, layer.bias)
        )
        expected_rows.append(torch.cat([
            weight_gradient.reshape(-1), bias_gradient.reshape(-1)
        ]))
    expected = torch.stack(expected_rows)

    assert gradient.shape == (len(xy), layer.in_features + 1)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected)


def test_source_gradient_wrt_ll_params(fitted_estimator):
    estimator = fitted_estimator
    xy = estimator._tensor([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    gradient = estimator.laplace.source_gradient_wrt_parameters(xy)

    layer = estimator.model.q_net.net[-1]
    expected_rows = []
    for point in xy:
        source = estimator.model.q_net(point[None, :]).sum()
        weight_gradient, bias_gradient = torch.autograd.grad(
            source, (layer.weight, layer.bias)
        )
        expected_rows.append(torch.cat([
            weight_gradient.reshape(-1), bias_gradient.reshape(-1)
        ]))

    assert gradient.shape == (len(xy), layer.in_features + 1)
    torch.testing.assert_close(gradient, torch.stack(expected_rows))

def test_uncertainty_reduction_batch_matches_single_points(fitted_estimator):
    xy = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    batch_scores = fitted_estimator.source_uncertainty_reduction(xy)
    single_scores = torch.cat([
        fitted_estimator.source_uncertainty_reduction(point[None, :])
        for point in xy
    ])
    torch.testing.assert_close(batch_scores, single_scores)


@pytest.mark.parametrize("coupled", [True, False], ids=["known_scores", "no_coupling"])
def test_uncertainty_reduction_known_covariance(estimator, monkeypatch, coupled):
    # Pad the small numerical example to the actual model dimensions.
    n_c = estimator.model.c_net.net[-1].in_features + 1
    n_q = estimator.model.q_net.net[-1].in_features + 1
    covariance = torch.eye(n_c + n_q, device=estimator.device)
    covariance[n_c:, n_c:] *= 2.0
    if coupled:
        covariance[n_c, 0] = 0.5
        covariance[0, n_c] = 0.5
    estimator.laplace.covariance = covariance

    gradients = covariance.new_zeros((3, n_c))
    gradients[0, 0] = 1.0
    gradients[1, 1] = 1.0
    gradients[2, :2] = 1.0
    monkeypatch.setattr(
        estimator.laplace, "concentration_gradient_wrt_parameters", lambda xy: gradients
    )
    def source_gradients(points):
        result = covariance.new_zeros((len(points), n_q))
        result[:, 0] = 1.0
        return result

    monkeypatch.setattr(
        estimator.laplace, "source_gradient_wrt_parameters", source_gradients
    )

    xy = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    scores = estimator.source_uncertainty_reduction(xy)
    expected = covariance.new_tensor([0.25, 0.0, 0.125] if coupled else [0.0] * 3)
    assert scores.shape == (3,)
    torch.testing.assert_close(scores, expected)


def test_training_invalidates_laplace_cache(fitted_estimator):
    fitted_estimator.laplace.compute_covariance()
    assert fitted_estimator.laplace.covariance is not None
    fitted_estimator.training_step(n_collocation_points=4)
    assert fitted_estimator.laplace.covariance is None
