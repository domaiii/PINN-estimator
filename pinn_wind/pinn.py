from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import mean

from .io_tools import OccupancyMap


class WindNet2D(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden_units: int = 20):
        super().__init__()

        layers = [
            torch.nn.Linear(2, hidden_units),
            torch.nn.Tanh(),
        ]

        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(hidden_units, hidden_units))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(hidden_units, 3))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


class PINNLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_smooth: float = 0.001,
        lambda_div: float = 0.01,
        lambda_mom: float = 0.01,
        lambda_p: float = 1.0,
        lambda_wall: float = 1.0,
        nu: float = 1e-3,
    ):
        super().__init__()
        self.data_loss_fn = torch.nn.MSELoss(reduction="mean")
        self.lambda_data = lambda_data
        self.lambda_smooth = lambda_smooth
        self.lambda_div = lambda_div
        self.lambda_mom = lambda_mom
        self.lambda_p = lambda_p
        self.lambda_wall = lambda_wall
        self.nu = nu

        self._current_loss_data = 0.0
        self._current_loss_smooth = 0.0
        self._current_loss_div = 0.0
        self._current_loss_mom = 0.0
        self._current_loss_p = 0.0
        self._current_loss_wall = 0.0
        self._current_loss_total = 0.0

    @property
    def current_losses(self) -> dict[str, float]:
        return {
            "data": self._current_loss_data,
            "smooth": self._current_loss_smooth,
            "div": self._current_loss_div,
            "mom": self._current_loss_mom,
            "p": self._current_loss_p,
            "wall": self._current_loss_wall,
            "total": self._current_loss_total,
        }

    def components(
        self,
        uv_pred_data: torch.Tensor,
        uv_data: torch.Tensor,
        w_pred_prior: torch.Tensor,
        xy_prior: torch.Tensor,
        uv_pred_wall: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        u_pred = w_pred_prior[:, 0]
        v_pred = w_pred_prior[:, 1]
        p_pred = w_pred_prior[:, 2]
        ones = torch.ones_like(u_pred)

        grad_u_dxy = torch.autograd.grad(u_pred, xy_prior, ones, create_graph=True)[0]
        grad_v_dxy = torch.autograd.grad(v_pred, xy_prior, ones, create_graph=True)[0]
        grad_p_dxy = torch.autograd.grad(p_pred, xy_prior, ones, create_graph=True)[0]

        u_x = grad_u_dxy[:, 0]
        u_y = grad_u_dxy[:, 1]
        v_x = grad_v_dxy[:, 0]
        v_y = grad_v_dxy[:, 1]
        p_x = grad_p_dxy[:, 0]
        p_y = grad_p_dxy[:, 1]

        grad_ux_dxy = torch.autograd.grad(u_x, xy_prior, ones, create_graph=True)[0]
        grad_uy_dxy = torch.autograd.grad(u_y, xy_prior, ones, create_graph=True)[0]
        grad_vx_dxy = torch.autograd.grad(v_x, xy_prior, ones, create_graph=True)[0]
        grad_vy_dxy = torch.autograd.grad(v_y, xy_prior, ones, create_graph=True)[0]

        u_xx = grad_ux_dxy[:, 0]
        u_yy = grad_uy_dxy[:, 1]
        v_xx = grad_vx_dxy[:, 0]
        v_yy = grad_vy_dxy[:, 1]

        data_loss = self.data_loss_fn(uv_pred_data, uv_data)
        smooth_loss = mean(grad_u_dxy**2 + grad_v_dxy**2)
        div_loss = torch.mean((u_x + v_y) ** 2)

        mom_x = -self.nu * (u_xx + u_yy) + p_x + u_pred * u_x + v_pred * u_y
        mom_y = -self.nu * (v_xx + v_yy) + p_y + u_pred * v_x + v_pred * v_y
        mom_loss = mean(mom_x**2 + mom_y**2)
        p_loss = mean(p_pred) ** 2

        if uv_pred_wall is None:
            wall_loss = torch.zeros((), dtype=w_pred_prior.dtype, device=w_pred_prior.device)
        else:
            wall_loss = torch.mean(uv_pred_wall**2)

        return {
            "data": data_loss,
            "smooth": smooth_loss,
            "div": div_loss,
            "mom": mom_loss,
            "p": p_loss,
            "wall": wall_loss,
        }

    def weighted_components(
        self,
        uv_pred_data: torch.Tensor,
        uv_data: torch.Tensor,
        w_pred_prior: torch.Tensor,
        xy_prior: torch.Tensor,
        uv_pred_wall: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        components = self.components(
            uv_pred_data,
            uv_data,
            w_pred_prior,
            xy_prior,
            uv_pred_wall=uv_pred_wall,
        )
        return {
            "data": self.lambda_data * components["data"],
            "smooth": self.lambda_smooth * components["smooth"],
            "div": self.lambda_div * components["div"],
            "mom": self.lambda_mom * components["mom"],
            "p": self.lambda_p * components["p"],
            "wall": self.lambda_wall * components["wall"],
        }

    def forward(
        self,
        uv_pred_data: torch.Tensor,
        uv_data: torch.Tensor,
        w_pred_prior: torch.Tensor,
        xy_prior: torch.Tensor,
        uv_pred_wall: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weighted = self.weighted_components(
            uv_pred_data,
            uv_data,
            w_pred_prior,
            xy_prior,
            uv_pred_wall=uv_pred_wall,
        )
        total = sum(weighted.values())

        self._current_loss_data = weighted["data"].detach().cpu().item()
        self._current_loss_smooth = weighted["smooth"].detach().cpu().item()
        self._current_loss_div = weighted["div"].detach().cpu().item()
        self._current_loss_mom = weighted["mom"].detach().cpu().item()
        self._current_loss_p = weighted["p"].detach().cpu().item()
        self._current_loss_wall = weighted["wall"].detach().cpu().item()
        self._current_loss_total = total.detach().cpu().item()

        return total


@dataclass
class TrainingHistory:
    loss: np.ndarray


class PINNWindEstimator:
    def __init__(
        self,
        n_layers: int = 3,
        hidden_units: int = 20,
        learning_rate: float = 1e-2,
        lambda_data: float = 1.0,
        lambda_smooth: float = 0.001,
        lambda_div: float = 0.01,
        lambda_mom: float = 0.01,
        lambda_p: float = 1.0,
        lambda_wall: float = 1.0,
        nu: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = WindNet2D(n_layers, hidden_units).to(self.device)
        self.loss_fn = PINNLoss(
            lambda_data=lambda_data,
            lambda_smooth=lambda_smooth,
            lambda_div=lambda_div,
            lambda_mom=lambda_mom,
            lambda_p=lambda_p,
            lambda_wall=lambda_wall,
            nu=nu,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history: TrainingHistory | None = None

    def fit(
        self,
        occupancy_map: OccupancyMap,
        measurements: pd.DataFrame,
        steps: int = 500,
    ) -> TrainingHistory:
        xy_data = measurements[["Points:0", "Points:1"]].to_numpy(dtype=np.float32)
        uv_data = measurements[["U:0", "U:1"]].to_numpy(dtype=np.float32)
        return self.fit_xy(
            xy_data,
            uv_data,
            occupancy_map.free_points,
            wall_points=occupancy_map.wall_points,
            steps=steps,
        )

    def fit_xy(
        self,
        xy_data: np.ndarray,
        uv_data: np.ndarray,
        prior_points: np.ndarray,
        wall_points: np.ndarray | None = None,
        steps: int = 500,
    ) -> TrainingHistory:
        xy_data_t = torch.tensor(xy_data, dtype=torch.float32, device=self.device)
        uv_data_t = torch.tensor(uv_data, dtype=torch.float32, device=self.device)
        prior_points_t = torch.tensor(prior_points, dtype=torch.float32, device=self.device)
        xy_prior = prior_points_t.clone().detach().requires_grad_()
        if wall_points is None or len(wall_points) == 0:
            wall_points_t = None
        else:
            wall_points_t = torch.tensor(wall_points, dtype=torch.float32, device=self.device)

        loss_values = np.zeros(steps, dtype=float)

        self.model.train()
        for i in range(steps):
            w_pred_data = self.model(xy_data_t)
            w_pred_prior = self.model(xy_prior)
            uv_pred_data = w_pred_data[:, :2]
            uv_pred_wall = None if wall_points_t is None else self.model(wall_points_t)[:, :2]

            loss = self.loss_fn(
                uv_pred_data,
                uv_data_t,
                w_pred_prior,
                xy_prior,
                uv_pred_wall=uv_pred_wall,
            )
            loss_values[i] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.history = TrainingHistory(loss=loss_values)
        return self.history

    def predict_points(self, xy: np.ndarray) -> np.ndarray:
        xy_t = torch.tensor(xy, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(xy_t)[:, :2].cpu().numpy()

    def predict_map(self, occupancy_map: OccupancyMap) -> tuple[np.ndarray, np.ndarray]:
        uv_free = self.predict_points(occupancy_map.free_points)

        u = np.full(occupancy_map.occupancy.shape, np.nan, dtype=float)
        v = np.full(occupancy_map.occupancy.shape, np.nan, dtype=float)
        u[occupancy_map.free_mask] = uv_free[:, 0]
        v[occupancy_map.free_mask] = uv_free[:, 1]
        return u, v
