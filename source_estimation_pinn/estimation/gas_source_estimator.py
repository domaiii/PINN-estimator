from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..environment import OccupancyGrid, WindField
from ..measurements import GasSampleSet, WindSampleSet
from .configs import (
    EstimationHistory,
    GasSourceConfig,
    StreamFunctionWindConfig,
    WindGasSourceConfig,
)
from .laplace import JointSourceLaplace
from .losses import AdvectionDiffusionState, GasDistributionLoss, GasSourceLoss
from .networks import PositiveFieldNet
from .wind_estimator import (
    _build_wind_loss,
    _build_wind_network,
    _wind_loss_components,
)


@dataclass
class _TrainingData:
    xy_free: torch.Tensor
    xy_nonfree: torch.Tensor
    xy_measurements: torch.Tensor
    c_measurements: torch.Tensor
    xy_walls: torch.Tensor
    wall_normals: torch.Tensor
    xy_open: torch.Tensor
    open_normals: torch.Tensor


class _GasSourceModel(nn.Module):
    def __init__(self, bounds: tuple[np.ndarray, np.ndarray], config: GasSourceConfig):
        super().__init__()
        lower_bound, upper_bound = bounds
        self.c_net = PositiveFieldNet(
            lower_bound,
            upper_bound,
            hidden_layers=config.concentration_hidden_layers,
            hidden_dim=config.concentration_hidden_dim,
            initial_value=config.concentration_initial_value,
        )
        self.q_net = PositiveFieldNet(
            lower_bound,
            upper_bound,
            hidden_layers=config.source_hidden_layers,
            hidden_dim=config.source_hidden_dim,
            initial_value=config.source_initial_value,
        )

    @property
    def last_layer_params(self) -> torch.Tensor:
        """Return the parameters of the last layers of both networks."""
        return torch.cat([
            JointSourceLaplace.linear_layer_parameters(self.c_net.net[-1]),
            JointSourceLaplace.linear_layer_parameters(self.q_net.net[-1]),
        ])

    def split_ll_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a concatenated last-layer parameter vector into two parts."""
        c_last_layer_size = self.c_net.net[-1].in_features + 1
        q_last_layer_size = self.q_net.net[-1].in_features + 1
        
        if len(params) != c_last_layer_size + q_last_layer_size:
            raise ValueError(
                f"params must have length {c_last_layer_size + q_last_layer_size}"
            )
        return params[:c_last_layer_size], params[c_last_layer_size:]


class _WindGasSourceModel(_GasSourceModel):
    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        config: WindGasSourceConfig,
    ):
        super().__init__(bounds, config)
        self.wind_net = _build_wind_network(bounds, config.wind)


class _GasSourceEstimatorBase(ABC):
    """Shared training machinery for known- and estimated-wind variants."""

    def __init__(
        self,
        occupancy: OccupancyGrid,
        config: GasSourceConfig,
        model: _GasSourceModel,
        optimizer: torch.optim.Optimizer,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.occupancy = occupancy
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.gas_distribution_loss = GasDistributionLoss(
            lambda_data=config.lambda_gas_data,
            lambda_AD=config.lambda_ad,
            lambda_wall=config.lambda_gas_wall,
            lambda_inflow=config.lambda_inflow,
            diffusion_constant=config.diffusion_constant,
        ).to(self.device)
        self.gas_source_loss = GasSourceLoss(
            lambda_sparse=config.lambda_sparse,
            lambda_nonfree=config.lambda_nonfree,
        ).to(self.device)

        self.history: EstimationHistory | None = None
        self._training_data: _TrainingData | None = None

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    def _prepare_common(self, gas_samples: GasSampleSet) -> None:
        if len(gas_samples.positions) == 0:
            raise ValueError("gas_samples must not be empty")

        free_points = self.occupancy.free_points
        if len(free_points) == 0:
            raise ValueError("occupancy grid contains no free points")

        xx, yy = np.meshgrid(
            self.occupancy.x_centers,
            self.occupancy.y_centers,
        )
        nonfree_mask = ~self.occupancy.free_mask
        nonfree_points = np.column_stack(
            (xx[nonfree_mask], yy[nonfree_mask])
        )
        wall_points, wall_normals = self.occupancy.wall_boundary_samples()
        open_points, open_normals = self.occupancy.open_boundary_samples()

        self._training_data = _TrainingData(
            xy_free=self._tensor(free_points),
            xy_nonfree=self._tensor(nonfree_points),
            xy_measurements=self._tensor(gas_samples.positions),
            c_measurements=self._tensor(gas_samples.concentrations),
            xy_walls=self._tensor(wall_points),
            wall_normals=self._tensor(wall_normals),
            xy_open=self._tensor(open_points),
            open_normals=self._tensor(open_normals),
        )

    def _require_training_data(self) -> _TrainingData:
        if self._training_data is None:
            raise RuntimeError("Call fit(...) first")
        return self._training_data

    @abstractmethod
    def _wind_components(
        self,
        data: _TrainingData,
        indices: torch.Tensor,
        xy_collocation: torch.Tensor,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return collocation velocity and optional wind-loss components."""

    @abstractmethod
    def _inflow_concentration(
        self,
        data: _TrainingData,
        concentration: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor | None:
        """Return predicted concentration at the current inflow points."""

    def _gas_distribution_loss_components(
        self,
        data: _TrainingData,
        concentration: Callable[[torch.Tensor], torch.Tensor],
        ad_state: AdvectionDiffusionState,
    ) -> dict[str, torch.Tensor]:
        """Return data, AD, wall, and inflow loss components."""
        if len(data.xy_walls) == 0:
            dc_dn_walls = None
        else:
            xy_walls = data.xy_walls.detach().requires_grad_(True)
            c_walls = concentration(xy_walls)
            grad_c_walls = torch.autograd.grad(
                c_walls,
                xy_walls,
                grad_outputs=torch.ones_like(c_walls),
                create_graph=True,
            )[0]
            dc_dn_walls = torch.sum(
                grad_c_walls * data.wall_normals,
                dim=1,
            )

        return self.gas_distribution_loss.components(
            data.c_measurements,
            concentration(data.xy_measurements),
            ad_state,
            dc_dn_walls,
            self._inflow_concentration(data, concentration),
        )

    def _loss_components(
        self,
        data: _TrainingData,
        colloc_indices: torch.Tensor,
        concentration: Callable[[torch.Tensor], torch.Tensor],
        source: Callable[[torch.Tensor], torch.Tensor],
        *,
        compute_wind_loss: bool = True,
        detach_wind: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Evaluate all loss components."""
        xy_collocation = data.xy_free[colloc_indices].detach().requires_grad_(True)
        c_collocation = concentration(xy_collocation)
        q_collocation = source(xy_collocation)
        uv_collocation, wind_components = self._wind_components(
            data,
            colloc_indices,
            xy_collocation,
            compute_loss=compute_wind_loss,
        )

        if detach_wind:
            uv_collocation = uv_collocation.detach()

        ad_state = AdvectionDiffusionState(
            xy=xy_collocation,
            c=c_collocation,
            q=q_collocation,
            uv=uv_collocation,
        )
        components = self._gas_distribution_loss_components(
            data,
            concentration,
            ad_state,
        )
        components.update(
            self.gas_source_loss.components(
                q_collocation, 
                source(data.xy_nonfree)
            )
        )
        components.update(wind_components)

        return components

    def loss_on_random_collocation(
        self,
        n_collocation_points: int,
    ) -> dict[str, torch.Tensor]:
        data = self._require_training_data()
        if n_collocation_points <= 0:
            raise ValueError("n_collocation_points must be positive")

        indices = torch.randint(
            len(data.xy_free),
            (n_collocation_points,),
            device=self.device,
        )
        components = self._loss_components(
            data,
            indices,
            self.model.c_net,
            self.model.q_net,
        )
        params_last_layer = self.model.last_layer_params
        components["last_layer_prior"] = (
            0.5
            * self.config.last_layer_prior_precision
            * torch.sum(params_last_layer**2)
        )
        return {"total": sum(components.values()), **components}

    def training_step(
        self,
        n_collocation_points: int,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.loss_on_random_collocation(n_collocation_points)
        losses["total"].backward()
        self.optimizer.step()
        return {
            name: value.detach().cpu().item()
            for name, value in losses.items()
        }

    def _fit_prepared(
        self,
        steps: int,
        n_collocation_points: int,
        callback: Callable[[int, dict[str, float]], None] | None,
    ) -> EstimationHistory:
        if steps <= 0:
            raise ValueError("steps must be positive")

        recorded: dict[str, list[float]] = {}
        for step in range(steps):
            losses = self.training_step(n_collocation_points)
            for name, value in losses.items():
                recorded.setdefault(name, []).append(value)
            if callback is not None:
                callback(step, losses)

        self.history = EstimationHistory(
            losses={
                name: np.asarray(values, dtype=float)
                for name, values in recorded.items()
            }
        )
        return self.history

    def predict_concentration(self, xy: np.ndarray) -> np.ndarray:
        return self._predict(self.model.c_net, xy)

    def predict_source(self, xy: np.ndarray) -> np.ndarray:
        return self._predict(self.model.q_net, xy)

    def _predict(self, network: nn.Module, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float)
        single_point = points.ndim == 1
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("xy must have shape (2,) or (n, 2)")

        self.model.eval()
        with torch.no_grad():
            values = network(self._tensor(points)).detach().cpu().numpy()
        return values[0] if single_point else values

    def get_source_estimate(self) -> np.ndarray:
        free_points = self.occupancy.free_points
        source_values = self.predict_source(free_points)
        return free_points[int(np.argmax(source_values))].copy()


class GasSourceEstimator(_GasSourceEstimatorBase):
    """Estimate concentration and source fields for a known wind field.

    Joint c-q last-layer Laplace and acquisition scores are available here
    only: the supplied wind is fixed and has no modeled uncertainty.
    """

    def __init__(
        self,
        occupancy: OccupancyGrid,
        wind: WindField,
        config: GasSourceConfig | None = None,
    ):
        config = config or GasSourceConfig()
        device = torch.device(config.device)
        bounds = (occupancy.lower_bound, occupancy.upper_bound)
        model = _GasSourceModel(bounds, config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        super().__init__(occupancy, config, model, optimizer)
        self.wind = wind
        self.laplace = JointSourceLaplace(
            self.model.c_net, self.model.q_net, self.loss_for_last_layer
        )
        self._known_uv_free: torch.Tensor | None = None
        self._known_xy_inflow: torch.Tensor | None = None

    def _wind_components(
        self,
        data: _TrainingData,
        indices: torch.Tensor,
        xy_collocation: torch.Tensor,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._known_uv_free is None:
            raise RuntimeError("Known wind has not been prepared")
        return self._known_uv_free[indices], {}

    def _inflow_concentration(
        self,
        data: _TrainingData,
        concentration: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor | None:
        if self._known_xy_inflow is None or len(self._known_xy_inflow) == 0:
            return None
        return concentration(self._known_xy_inflow)

    def fit(
        self,
        gas_samples: GasSampleSet,
        steps: int = 1000,
        n_collocation_points: int = 1000,
        callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> EstimationHistory:
        self.laplace.invalidate()
        self._prepare_common(gas_samples)
        data = self._require_training_data()
        free_points = data.xy_free.detach().cpu().numpy()
        self._known_uv_free = self._tensor(
            self.wind.velocity_at(free_points)
        )

        if len(data.xy_open) == 0:
            self._known_xy_inflow = data.xy_open
        else:
            open_points = data.xy_open.detach().cpu().numpy()
            open_normals = data.open_normals.detach().cpu().numpy()
            normal_velocity = np.sum(
                self.wind.velocity_at(open_points) * open_normals,
                axis=1,
            )
            self._known_xy_inflow = self._tensor(
                open_points[normal_velocity < 0.0]
            )

        return self._fit_prepared(steps, n_collocation_points, callback)

    def training_step(self, n_collocation_points: int) -> dict[str, float]:
        """Invalidate the known-wind Laplace approximation before training."""
        self.laplace.invalidate()
        return super().training_step(n_collocation_points)

    def loss_for_last_layer(
        self,
        params_last_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Return the joint concentration-source loss for last-layer parameters."""
        data = self._require_training_data()
        expected_parameters = (self.model.c_net.net[-1].in_features + 1
                               + self.model.q_net.net[-1].in_features + 1
        )   
        if (
            params_last_layer.ndim != 1
            or len(params_last_layer) != expected_parameters
        ):
            raise ValueError(
                f"params_last_layer must have shape ({expected_parameters},)"
            )

        ll_params_c, ll_params_q = self.model.split_ll_params(params_last_layer)

        def concentration(xy: torch.Tensor) -> torch.Tensor:
            last_layer_input = self.model.c_net.last_hidden_features(xy)
            weight = ll_params_c[:-1]
            bias = ll_params_c[-1]
            pre_softplus_output = last_layer_input @ weight + bias
            return torch.nn.functional.softplus(pre_softplus_output)

        def source(xy: torch.Tensor) -> torch.Tensor:
            ll_input = self.model.q_net.last_hidden_features(xy)
            ll_weights = ll_params_q[:-1]
            ll_bias = ll_params_q[-1]
            pre_softplus_output = ll_input @ ll_weights + ll_bias
            return torch.nn.functional.softplus(pre_softplus_output)

        indices = torch.arange(len(data.xy_free), device=self.device)
        components = self._loss_components(
            data,
            indices,
            concentration,
            source,
            compute_wind_loss=False,
            detach_wind=True,
        )
        prior_loss = (
            0.5
            * self.config.last_layer_prior_precision
            * torch.sum(params_last_layer**2)
        )
        return sum(components.values()) + prior_loss

    def source_uncertainty_reduction(self, xy: np.ndarray) -> torch.Tensor:
        """Score candidate concentration measurements assuming known, fixed wind."""
        return self.laplace.source_uncertainty_reduction(self._tensor(xy))


class WindGasSourceEstimator(_GasSourceEstimatorBase):
    """Jointly estimate wind, concentration, and source fields.

    Laplace uncertainty and acquisition scores are not implemented for this
    estimator, since uncertainty in the estimated wind must also be considered.
    """

    def __init__(
        self,
        occupancy: OccupancyGrid,
        config: WindGasSourceConfig | None = None,
    ):
        config = config or WindGasSourceConfig()
        device = torch.device(config.device)
        bounds = (occupancy.lower_bound, occupancy.upper_bound)
        model = _WindGasSourceModel(bounds, config).to(device)
        gas_parameters = list(model.c_net.parameters()) + list(
            model.q_net.parameters()
        )
        optimizer = torch.optim.Adam([
            {"params": gas_parameters, "lr": config.learning_rate},
            {"params": model.wind_net.parameters(), "lr": config.wind.learning_rate},
        ])
        super().__init__(occupancy, config, model, optimizer)
        self.config = config
        self.wind_distribution_loss = _build_wind_loss(config.wind).to(
            self.device
        )
        self._xy_wind_measurements: torch.Tensor | None = None
        self._uv_measurements: torch.Tensor | None = None

    @property
    def _joint_model(self) -> _WindGasSourceModel:
        return self.model  # type: ignore[return-value]

    def _wind_components(
        self,
        data: _TrainingData,
        indices: torch.Tensor,
        xy_collocation: torch.Tensor,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._xy_wind_measurements is None or self._uv_measurements is None:
            raise RuntimeError("Wind measurements have not been prepared")

        if not compute_loss:
            return self._joint_model.wind_net(xy_collocation)[:, :2], {}

        return _wind_loss_components(
            self.config.wind,
            self._joint_model.wind_net,
            self.wind_distribution_loss,
            self._xy_wind_measurements,
            self._uv_measurements,
            xy_collocation,
            data.xy_walls,
            data.wall_normals,
            data.xy_free[:1],
        )

    def _inflow_concentration(
        self,
        data: _TrainingData,
        concentration: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor | None:
        if len(data.xy_open) == 0:
            return None
        uv_open = self._joint_model.wind_net(data.xy_open)[:, :2]
        normal_velocity = torch.sum(uv_open * data.open_normals, dim=1)
        inflow_mask = normal_velocity.detach() < 0.0
        if not torch.any(inflow_mask):
            return None
        return concentration(data.xy_open[inflow_mask])

    def fit(
        self,
        gas_samples: GasSampleSet,
        wind_samples: WindSampleSet,
        steps: int = 1000,
        n_collocation_points: int = 1000,
        callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> EstimationHistory:
        if len(wind_samples.positions) == 0:
            raise ValueError("wind_samples must not be empty")
        self._prepare_common(gas_samples)
        self._xy_wind_measurements = self._tensor(wind_samples.positions)
        self._uv_measurements = self._tensor(wind_samples.wind_vectors)

        return self._fit_prepared(steps, n_collocation_points, callback)

    def predict_wind(self, xy: np.ndarray) -> np.ndarray:
        values = self._predict(self._joint_model.wind_net, xy)
        return values[..., :2]

    def predict_pressure(self, xy: np.ndarray) -> np.ndarray:
        if isinstance(self.config.wind, StreamFunctionWindConfig):
            raise RuntimeError("The streamfunction formulation has no pressure field")
        values = self._predict(self._joint_model.wind_net, xy)
        return values[..., 2]

    def predict_wind_divergence(self, xy: np.ndarray) -> np.ndarray:
        """Evaluate du/dx + dv/dy at one or multiple positions."""
        points = np.asarray(xy, dtype=float)
        single_point = points.ndim == 1
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("xy must have shape (2,) or (n, 2)")

        self.model.eval()
        xy_tensor = self._tensor(points).detach().requires_grad_(True)
        uv = self._joint_model.wind_net(xy_tensor)[:, :2]
        grad_u = torch.autograd.grad(
            uv[:, 0],
            xy_tensor,
            grad_outputs=torch.ones_like(uv[:, 0]),
            retain_graph=True,
        )[0]
        grad_v = torch.autograd.grad(
            uv[:, 1],
            xy_tensor,
            grad_outputs=torch.ones_like(uv[:, 1]),
        )[0]
        divergence = (grad_u[:, 0] + grad_v[:, 1]).detach().cpu().numpy()
        return divergence[0] if single_point else divergence
