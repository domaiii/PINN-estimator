from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..environment import OccupancyGrid, WindField
from ..measurements import GasSampleSet, WindSampleSet

from .losses import (
    AdvectionDiffusionState,
    GasDistributionLoss,
    GasSourceLoss,
    StreamFunctionWindLoss,
    WindDistributionLoss,
)
from .networks import PositiveFieldNet, StreamFunctionNet2D, WindNet2D


@dataclass(frozen=True)
class GasSourceConfig:
    concentration_hidden_layers: int = 3
    concentration_hidden_dim: int = 64
    source_hidden_layers: int = 2
    source_hidden_dim: int = 32
    concentration_initial_value: float = 1e-3
    source_initial_value: float = 1e-3
    diffusion_constant: float = 1e-3
    lambda_gas_data: float = 1.0
    lambda_ad: float = 1.0
    lambda_gas_wall: float = 1.0
    lambda_inflow: float = 1.0
    lambda_sparse: float = 1e-3
    lambda_nonfree: float = 1e-2
    learning_rate: float = 1e-3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.concentration_hidden_layers < 1:
            raise ValueError("concentration_hidden_layers must be positive")
        if self.concentration_hidden_dim < 1:
            raise ValueError("concentration_hidden_dim must be positive")
        if self.source_hidden_layers < 1:
            raise ValueError("source_hidden_layers must be positive")
        if self.source_hidden_dim < 1:
            raise ValueError("source_hidden_dim must be positive")
        if self.diffusion_constant <= 0.0:
            raise ValueError("diffusion_constant must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class NavierStokesWindConfig:
    hidden_layers: int = 3
    hidden_dim: int = 32
    lambda_data: float = 1.0
    lambda_smooth: float = 1e-3
    lambda_div: float = 1e-2
    lambda_momentum: float = 1e-2
    lambda_pressure: float = 1.0
    lambda_wall: float = 1.0
    kinematic_viscosity: float = 1e-5
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.hidden_layers < 1 or self.hidden_dim < 1:
            raise ValueError("wind network dimensions must be positive")
        if self.kinematic_viscosity < 0.0:
            raise ValueError("kinematic_viscosity must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("wind learning_rate must be positive")


@dataclass(frozen=True)
class StreamFunctionWindConfig:
    hidden_layers: int = 4
    hidden_dim: int = 64
    lambda_data: float = 1.0
    lambda_smooth: float = 1e-3
    lambda_wall: float = 10.0
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.hidden_layers < 1 or self.hidden_dim < 1:
            raise ValueError("wind network dimensions must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("wind learning_rate must be positive")


WindConfig = NavierStokesWindConfig | StreamFunctionWindConfig


@dataclass(frozen=True)
class WindGasSourceConfig(GasSourceConfig):
    wind: WindConfig = field(default_factory=NavierStokesWindConfig)



def _build_wind_network(
    bounds: tuple[np.ndarray, np.ndarray],
    config: WindConfig,
) -> nn.Module:
    lower_bound, upper_bound = bounds
    network_class = (
        StreamFunctionNet2D
        if isinstance(config, StreamFunctionWindConfig)
        else WindNet2D
    )
    return network_class(
        lower_bound,
        upper_bound,
        hidden_layers=config.hidden_layers,
        hidden_dim=config.hidden_dim,
    )


def _build_wind_loss(config: WindConfig) -> nn.Module:
    if isinstance(config, StreamFunctionWindConfig):
        return StreamFunctionWindLoss(
            lambda_data=config.lambda_data,
            lambda_smooth=config.lambda_smooth,
            lambda_wall=config.lambda_wall,
        )
    return WindDistributionLoss(
        lambda_data=config.lambda_data,
        lambda_smooth_w=config.lambda_smooth,
        lambda_div=config.lambda_div,
        lambda_mom=config.lambda_momentum,
        lambda_p=config.lambda_pressure,
        lambda_wall=config.lambda_wall,
        nu=config.kinematic_viscosity,
    )


def _wind_loss_components(
    config: WindConfig,
    network: nn.Module,
    loss: nn.Module,
    xy_data: torch.Tensor,
    uv_data: torch.Tensor,
    xy_collocation: torch.Tensor,
    xy_walls: torch.Tensor,
    wall_normals: torch.Tensor,
    pressure_reference_xy: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    wind_collocation = network(xy_collocation)
    uv_pred_data = network(xy_data)[:, :2]
    uv_pred_wall = None if len(xy_walls) == 0 else network(xy_walls)[:, :2]

    if isinstance(config, StreamFunctionWindConfig):
        components = loss.components(
            uv_pred_data,
            uv_data,
            wind_collocation,
            xy_collocation,
            uv_pred_wall,
            wall_normals,
        )
        return wind_collocation, components

    p_reference = network(pressure_reference_xy)[:, 2]
    components = loss.components(
        uv_pred_data,
        uv_data,
        wind_collocation,
        xy_collocation,
        uv_pred_wall,
        p_reference,
    )
    return wind_collocation[:, :2], components


@dataclass
class EstimationHistory:
    """Weighted loss values recorded during training."""

    losses: dict[str, np.ndarray]

    @property
    def total(self) -> np.ndarray:
        return self.losses["total"]


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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return collocation velocity and optional wind-loss components."""

    @abstractmethod
    def _inflow_concentration(
        self,
        data: _TrainingData,
    ) -> torch.Tensor | None:
        """Return predicted concentration at the current inflow points."""

    def compute_losses(
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
        xy_collocation = data.xy_free[indices].detach().requires_grad_(True)
        c_collocation = self.model.c_net(xy_collocation)
        q_collocation = self.model.q_net(xy_collocation)
        uv_collocation, wind_components = self._wind_components(
            data,
            indices,
            xy_collocation,
        )
        ad_state = AdvectionDiffusionState(
            xy=xy_collocation,
            c=c_collocation,
            q=q_collocation,
            uv=uv_collocation,
        )

        if len(data.xy_walls) == 0:
            dc_dn_walls = None
        else:
            xy_walls = data.xy_walls.detach().requires_grad_(True)
            c_walls = self.model.c_net(xy_walls)
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

        components = self.gas_distribution_loss.components(
            data.c_measurements,
            self.model.c_net(data.xy_measurements),
            ad_state,
            dc_dn_walls,
            self._inflow_concentration(data),
        )
        components.update(
            self.gas_source_loss.components(
                q_collocation,
                self.model.q_net(data.xy_nonfree),
            )
        )
        components.update(wind_components)
        return {"total": sum(components.values()), **components}

    def training_step(
        self,
        n_collocation_points: int,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_losses(n_collocation_points)
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


class WindEstimator:
    """Estimate a wind field from measurements and a selectable physical prior."""

    def __init__(
        self,
        occupancy: OccupancyGrid,
        config: WindConfig | None = None,
        device: str = "cpu",
    ):
        self.occupancy = occupancy
        self.config = config or NavierStokesWindConfig()
        self.device = torch.device(device)
        bounds = (occupancy.lower_bound, occupancy.upper_bound)
        self.network = _build_wind_network(bounds, self.config).to(self.device)
        self.loss = _build_wind_loss(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )
        self.history: EstimationHistory | None = None

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.network.parameters()
            if parameter.requires_grad
        )

    def fit(
        self,
        wind_samples: WindSampleSet,
        steps: int = 1000,
        n_collocation_points: int = 1000,
        callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> EstimationHistory:
        if len(wind_samples.positions) == 0:
            raise ValueError("wind_samples must not be empty")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if n_collocation_points <= 0:
            raise ValueError("n_collocation_points must be positive")

        xy_data = self._tensor(wind_samples.positions)
        uv_data = self._tensor(wind_samples.wind_vectors)
        xy_free = self._tensor(self.occupancy.free_points)
        if len(xy_free) == 0:
            raise ValueError("occupancy grid contains no free points")
        wall_points, wall_normals = self.occupancy.wall_boundary_samples()
        xy_walls = self._tensor(wall_points)
        wall_normals = self._tensor(wall_normals)

        recorded: dict[str, list[float]] = {}
        self.network.train()
        for step in range(steps):
            indices = torch.randint(
                len(xy_free),
                (n_collocation_points,),
                device=self.device,
            )
            xy_collocation = xy_free[indices].detach().requires_grad_(True)
            _, components = _wind_loss_components(
                self.config,
                self.network,
                self.loss,
                xy_data,
                uv_data,
                xy_collocation,
                xy_walls,
                wall_normals,
                xy_free[:1],
            )
            losses = {"total": sum(components.values()), **components}

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            self.optimizer.step()

            values = {
                name: value.detach().cpu().item()
                for name, value in losses.items()
            }
            for name, value in values.items():
                recorded.setdefault(name, []).append(value)
            if callback is not None:
                callback(step, values)

        self.history = EstimationHistory(
            losses={
                name: np.asarray(values, dtype=float)
                for name, values in recorded.items()
            }
        )
        return self.history

    def predict_wind(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float)
        single_point = points.ndim == 1
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("xy must have shape (2,) or (n, 2)")

        self.network.eval()
        values = self.network(self._tensor(points))[:, :2]
        result = values.detach().cpu().numpy()
        return result[0] if single_point else result

    def predict_pressure(self, xy: np.ndarray) -> np.ndarray:
        if isinstance(self.config, StreamFunctionWindConfig):
            raise RuntimeError("The streamfunction formulation has no pressure field")
        points = np.asarray(xy, dtype=float)
        single_point = points.ndim == 1
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("xy must have shape (2,) or (n, 2)")

        self.network.eval()
        values = self.network(self._tensor(points))[:, 2]
        result = values.detach().cpu().numpy()
        return result[0] if single_point else result


class GasSourceEstimator(_GasSourceEstimatorBase):
    """Estimate concentration and source fields for a known wind field."""

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
        self._known_uv_free: torch.Tensor | None = None
        self._known_xy_inflow: torch.Tensor | None = None

    def _wind_components(
        self,
        data: _TrainingData,
        indices: torch.Tensor,
        xy_collocation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._known_uv_free is None:
            raise RuntimeError("Known wind has not been prepared")
        return self._known_uv_free[indices], {}

    def _inflow_concentration(
        self,
        data: _TrainingData,
    ) -> torch.Tensor | None:
        if self._known_xy_inflow is None or len(self._known_xy_inflow) == 0:
            return None
        return self.model.c_net(self._known_xy_inflow)

    def fit(
        self,
        gas_samples: GasSampleSet,
        steps: int = 1000,
        n_collocation_points: int = 1000,
        callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> EstimationHistory:
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


class WindGasSourceEstimator(_GasSourceEstimatorBase):
    """Jointly estimate wind, concentration, and source fields."""

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._xy_wind_measurements is None or self._uv_measurements is None:
            raise RuntimeError("Wind measurements have not been prepared")

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
    ) -> torch.Tensor | None:
        if len(data.xy_open) == 0:
            return None
        uv_open = self._joint_model.wind_net(data.xy_open)[:, :2]
        normal_velocity = torch.sum(uv_open * data.open_normals, dim=1)
        inflow_mask = normal_velocity.detach() < 0.0
        if not torch.any(inflow_mask):
            return None
        return self.model.c_net(data.xy_open[inflow_mask])

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
