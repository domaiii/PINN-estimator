import numpy as np
import pytest
import torch

from source_estimation_pinn.estimation.laplace import JointSourceLaplace
from source_estimation_pinn.environment import (
    CellType,
    OccupancyGrid,
    WindField,
)
from source_estimation_pinn.measurements import GasSampleSet, WindSampleSet
from source_estimation_pinn.estimation import (
    GasSourceConfig,
    GasSourceEstimator,
    WindGasSourceConfig,
    WindGasSourceEstimator,
)


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


def test_last_layer_reconstruction(estimator):
    xy = estimator._tensor(np.array([
        [1.0, 1.0],
        [4.0, 3.0],
        [8.0, 5.0],
    ]))

    # Concentration network
    c_net = estimator.model.c_net
    h_c = c_net.last_hidden_features(xy)
    z_c_from_features = c_net.eval_last_layer(h_c)
    z_c = c_net.raw_output(xy)
    c_reconstructed = torch.nn.functional.softplus(z_c)

    torch.testing.assert_close(z_c_from_features, z_c)
    torch.testing.assert_close(c_reconstructed, c_net(xy))

    # Source network
    q_net = estimator.model.q_net
    h_q = q_net.last_hidden_features(xy)
    z_q_from_features = q_net.eval_last_layer(h_q)
    z_q = q_net.raw_output(xy)
    q_reconstructed = torch.nn.functional.softplus(z_q)

    torch.testing.assert_close(z_q_from_features, z_q)
    torch.testing.assert_close(q_reconstructed, q_net(xy))

def test_last_layer_loss(fitted_estimator):
    estimator = fitted_estimator
    data = estimator._require_training_data()
    indices = torch.arange(len(data.xy_free), device=estimator.device)

    normal_components = estimator._loss_components(
        data,
        indices,
        estimator.model.c_net,
        estimator.model.q_net,
        compute_wind_loss=False,
        detach_wind=True,
    )

    params = estimator.model.last_layer_params
    prior = (
        0.5
        * estimator.config.last_layer_prior_precision
        * torch.sum(params**2)
    )
    normal_loss = sum(normal_components.values()) + prior

    external_params = params.detach().clone().requires_grad_(True)
    external_loss = estimator.loss_for_last_layer(external_params)

    torch.testing.assert_close(external_loss, normal_loss)


def test_joint_last_layer_loss_is_differentiable(fitted_estimator):
    estimator = fitted_estimator

    params = (
        estimator.model.last_layer_params
        .detach()
        .clone()
        .requires_grad_(True)
    )
    loss = estimator.loss_for_last_layer(params)

    gradient = torch.autograd.grad(loss, params)[0]
    gradient_c, gradient_q = estimator.model.split_ll_params(gradient)

    assert torch.isfinite(gradient).all()
    assert torch.any(gradient_c != 0)
    assert torch.any(gradient_q != 0)


def test_joint_last_layer_covariance(fitted_estimator):
    estimator = fitted_estimator

    params = estimator.model.last_layer_params
    covariance = estimator.laplace.compute_covariance()

    expected_shape = (len(params), len(params))
    assert covariance.shape == expected_shape
    assert torch.isfinite(covariance).all()

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

def test_uncertainty_reduction_score(fitted_estimator):
    xy = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    score = fitted_estimator.source_uncertainty_reduction(xy)
    assert score.shape == (len(xy),)
    assert torch.isfinite(score).all()
    assert torch.all(score >= 0)


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
        covariance[n_c, 0] = 1.0
        covariance[0, n_c] = 1.0
    estimator.laplace.covariance = covariance

    gradients = covariance.new_zeros((3, n_c))
    gradients[0, 0] = 1.0
    gradients[1, 1] = 1.0
    gradients[2, :2] = 1.0
    monkeypatch.setattr(
        estimator.laplace, "concentration_gradient_wrt_parameters", lambda xy: gradients
    )

    xy = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    scores = estimator.source_uncertainty_reduction(xy)
    expected = covariance.new_tensor([1.0, 0.0, 0.5] if coupled else [0.0] * 3)
    assert scores.shape == (3,)
    torch.testing.assert_close(scores, expected)


def test_laplace_is_only_available_with_known_wind(estimator):
    joint = WindGasSourceEstimator(
        occupancy=estimator.occupancy,
        config=WindGasSourceConfig(),
    )
    for name in (
        "loss_for_last_layer",
        "source_uncertainty_reduction",
        "laplace",
    ):
        assert hasattr(estimator, name)
        assert not hasattr(joint, name)

    # Joint training must still work without any Laplace state.
    positions = np.array([[0.5, 0.5], [1.5, 1.5]])
    joint.fit(
        GasSampleSet(positions=positions, concentrations=np.array([0.2, 0.1])),
        WindSampleSet(positions=positions, wind_vectors=np.zeros((2, 2))),
        steps=1,
        n_collocation_points=4,
    )
    assert not hasattr(joint, "laplace")


def test_training_invalidates_laplace_cache(fitted_estimator):
    fitted_estimator.laplace.compute_covariance()
    assert fitted_estimator.laplace.covariance is not None
    fitted_estimator.training_step(n_collocation_points=4)
    assert fitted_estimator.laplace.covariance is None
