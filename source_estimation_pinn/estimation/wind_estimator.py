from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ..environment import OccupancyGrid
from ..measurements import WindSampleSet
from .configs import (
    EstimationHistory,
    NavierStokesWindConfig,
    StreamFunctionWindConfig,
    WindConfig,
)
from .losses import StreamFunctionWindLoss, WindDistributionLoss
from .networks import StreamFunctionNet2D, WindNet2D


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

