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

        self._normalize = False
        self._xy_lb = torch.tensor([torch.nan, torch.nan])
        self._xy_ub = torch.tensor([torch.nan, torch.nan])

        layers = [
            torch.nn.Linear(2, hidden_units),
            torch.nn.Tanh(),
        ]

        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(hidden_units, hidden_units))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(hidden_units, 3))

        self.net = torch.nn.Sequential(*layers)

    def normalize_domain(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        if self._normalize:
            raise RuntimeError("Domain is already normalized.")

        self._normalize = True
        self._xy_lb = torch.as_tensor(
            lower_bound,
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        self._xy_ub = torch.as_tensor(
            upper_bound,
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
    
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if self._normalize:
            xy = 2.0 * (xy - self._xy_lb)/(self._xy_ub - self._xy_lb) - 1.0

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

    def forward(
        self,
        # Measurements
        uv_pred_data: torch.Tensor,
        uv_data: torch.Tensor,
        # Free collocation
        w_pred_collocation: torch.Tensor,
        xy_collocation: torch.Tensor,
        #Walls
        uv_pred_wall: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u_pred = w_pred_collocation[:, 0]
        v_pred = w_pred_collocation[:, 1]
        p_pred = w_pred_collocation[:, 2]
        ones = torch.ones_like(u_pred)

        grad_u_dxy = torch.autograd.grad(u_pred, xy_collocation, ones, create_graph=True)[0]
        grad_v_dxy = torch.autograd.grad(v_pred, xy_collocation, ones, create_graph=True)[0]
        grad_p_dxy = torch.autograd.grad(p_pred, xy_collocation, ones, create_graph=True)[0]

        u_x = grad_u_dxy[:, 0]
        u_y = grad_u_dxy[:, 1]
        v_x = grad_v_dxy[:, 0]
        v_y = grad_v_dxy[:, 1]
        p_x = grad_p_dxy[:, 0]
        p_y = grad_p_dxy[:, 1]

        data_loss = self.lambda_data * self.data_loss_fn(uv_pred_data, uv_data)
        smooth_loss = self.lambda_smooth * mean(grad_u_dxy**2 + grad_v_dxy**2)
        div_loss = self.lambda_div * torch.mean((u_x + v_y) ** 2)

        if self.lambda_mom > 0.0:
            grad_ux_dxy = torch.autograd.grad(u_x, xy_collocation, ones, create_graph=True)[0]
            grad_uy_dxy = torch.autograd.grad(u_y, xy_collocation, ones, create_graph=True)[0]
            grad_vx_dxy = torch.autograd.grad(v_x, xy_collocation, ones, create_graph=True)[0]
            grad_vy_dxy = torch.autograd.grad(v_y, xy_collocation, ones, create_graph=True)[0]

            u_xx = grad_ux_dxy[:, 0]
            u_yy = grad_uy_dxy[:, 1]
            v_xx = grad_vx_dxy[:, 0]
            v_yy = grad_vy_dxy[:, 1]

            mom_x = -self.nu * (u_xx + u_yy) + p_x + u_pred * u_x + v_pred * u_y
            mom_y = -self.nu * (v_xx + v_yy) + p_y + u_pred * v_x + v_pred * v_y
            mom_loss = self.lambda_mom * mean(mom_x**2 + mom_y**2)
        else:
            mom_loss = torch.zeros((), dtype=w_pred_collocation.dtype, device=w_pred_collocation.device)

        p_loss = self.lambda_p * mean(p_pred) ** 2

        if uv_pred_wall is None:
            wall_loss = torch.zeros((), 
                                    dtype=w_pred_collocation.dtype, 
                                    device=w_pred_collocation.device)
        else:
            wall_loss = self.lambda_wall * torch.mean(uv_pred_wall**2)

        total = data_loss + smooth_loss + div_loss + mom_loss + p_loss + wall_loss

        self._current_loss_data = data_loss.detach().cpu().item()
        self._current_loss_smooth = smooth_loss.detach().cpu().item()
        self._current_loss_div = div_loss.detach().cpu().item()
        self._current_loss_mom = mom_loss.detach().cpu().item()
        self._current_loss_p = p_loss.detach().cpu().item()
        self._current_loss_wall = wall_loss.detach().cpu().item()
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
        n_collocation_points: int,
        steps: int = 500,
    ) -> TrainingHistory:
        xy_data = measurements[["Points:0", "Points:1"]].to_numpy(dtype=np.float32)
        uv_data = measurements[["U:0", "U:1"]].to_numpy(dtype=np.float32)

        return self.fit_xy(
            xy_data,
            uv_data,
            occupancy_map.free_points,
            n_collocation_points,
            wall_points=occupancy_map.wall_points,
            steps=steps
        )

    def fit_xy(
        self,
        xy_data: np.ndarray,
        uv_data: np.ndarray,
        prior_points: np.ndarray,
        n_collocation_points: int,
        wall_points: np.ndarray | None = None,
        steps: int = 500,
    ) -> TrainingHistory:
        xy_data_t = torch.tensor(xy_data, dtype=torch.float32, device=self.device)
        uv_data_t = torch.tensor(uv_data, dtype=torch.float32, device=self.device)
        colloc_points_t = torch.tensor(prior_points, dtype=torch.float32, device=self.device)

        if wall_points is None or len(wall_points) == 0:
            wall_points_t = None
        else:
            wall_points_t = torch.tensor(wall_points, dtype=torch.float32, device=self.device)

        loss_values = np.zeros(steps, dtype=float)

        self.model.train()
        for i in range(steps):
            uv_pred_data = self.model(xy_data_t)[:, :2]

            # Draw collocation points WITH replacement (slightly, but not notably faster)
            # random_ids = torch.randperm(
            #     colloc_points_t.shape[0],
            #     device=self.device
            # )[:n_collocation_points]

            # Draw collocation points WITHOUT replacement
            random_ids = torch.randint(
                colloc_points_t.shape[0],
                (n_collocation_points,),
                device=self.device,
            )

            xy_collocation = colloc_points_t[random_ids].clone().detach().requires_grad_()
            w_pred_collocation = self.model(xy_collocation)

            uv_pred_wall = None if wall_points_t is None else self.model(wall_points_t)[:, :2]

            loss = self.loss_fn(
                uv_pred_data,
                uv_data_t,
                w_pred_collocation,
                xy_collocation,
                uv_pred_wall=uv_pred_wall,
            )
            loss_values[i] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.history = TrainingHistory(loss=loss_values)
        return self.history
    
    def normalize_domain(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        self.model.normalize_domain(lower_bound, upper_bound)

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
