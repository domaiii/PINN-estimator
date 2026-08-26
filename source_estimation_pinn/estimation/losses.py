from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AdvectionDiffusionState:
    xy: torch.Tensor
    c: torch.Tensor
    q: torch.Tensor
    uv: torch.Tensor


class GasDistributionLoss(nn.Module):
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_AD: float = 1.0,
        lambda_wall: float = 1.0,
        lambda_inflow: float = 1.0,
        diffusion_constant: float = 1e-3,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_AD = lambda_AD
        self.lambda_wall = lambda_wall
        self.lambda_inflow = lambda_inflow
        self.diffusion_constant = diffusion_constant

    def components(
        self,
        c_measurements: torch.Tensor,
        c_pred_at_measurements: torch.Tensor,
        AD_collocation: AdvectionDiffusionState,
        dc_dn_walls: torch.Tensor | None = None,
        c_inflow: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        data_loss = self.lambda_data * torch.mean(
            (c_measurements - c_pred_at_measurements) ** 2
        )

        grad_c = torch.autograd.grad(
            AD_collocation.c,
            AD_collocation.xy,
            grad_outputs=torch.ones_like(AD_collocation.c),
            create_graph=True,
        )[0]
        c_x = grad_c[:, 0]
        c_y = grad_c[:, 1]
        c_xx = torch.autograd.grad(
            c_x,
            AD_collocation.xy,
            grad_outputs=torch.ones_like(c_x),
            create_graph=True,
        )[0][:, 0]
        c_yy = torch.autograd.grad(
            c_y,
            AD_collocation.xy,
            grad_outputs=torch.ones_like(c_y),
            create_graph=True,
        )[0][:, 1]

        ad_residual = (
            -self.diffusion_constant * (c_xx + c_yy)
            + AD_collocation.uv[:, 0] * c_x
            + AD_collocation.uv[:, 1] * c_y
            - AD_collocation.q
        )
        ad_loss = self.lambda_AD * torch.mean(ad_residual**2)

        wall_loss = AD_collocation.c.new_zeros(())
        if dc_dn_walls is not None and dc_dn_walls.numel() > 0:
            wall_loss = self.lambda_wall * torch.mean(dc_dn_walls**2)

        inflow_loss = AD_collocation.c.new_zeros(())
        if c_inflow is not None and c_inflow.numel() > 0:
            inflow_loss = self.lambda_inflow * torch.mean(c_inflow**2)

        return {
            "gas_data": data_loss,
            "ad": ad_loss,
            "gas_wall": wall_loss,
            "inflow": inflow_loss,
        }

    def forward(
        self,
        c_measurements: torch.Tensor,
        c_pred_at_measurements: torch.Tensor,
        AD_collocation: AdvectionDiffusionState,
        dc_dn_walls: torch.Tensor | None = None,
        c_inflow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sum(
            self.components(
                c_measurements,
                c_pred_at_measurements,
                AD_collocation,
                dc_dn_walls,
                c_inflow,
            ).values()
        )


class WindDistributionLoss(nn.Module):
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_smooth_w: float = 0.001,
        lambda_div: float = 0.01,
        lambda_mom: float = 0.01,
        lambda_p: float = 1.0,
        lambda_wall: float = 1.0,
        nu: float = 1e-5,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_smooth_w = lambda_smooth_w
        self.lambda_div = lambda_div
        self.lambda_mom = lambda_mom
        self.lambda_p = lambda_p
        self.lambda_wall = lambda_wall
        self.nu = nu

    def components(
        self,
        uv_pred_data: torch.Tensor,
        uv_data: torch.Tensor,
        uvp_collocation: torch.Tensor,
        xy_collocation: torch.Tensor,
        uv_pred_wall: torch.Tensor | None = None,
        p_reference: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        u_pred = uvp_collocation[:, 0]
        v_pred = uvp_collocation[:, 1]
        p_pred = uvp_collocation[:, 2]
        ones = torch.ones_like(u_pred)

        grad_u_dxy = torch.autograd.grad(
            u_pred, xy_collocation, ones, create_graph=True
        )[0]
        grad_v_dxy = torch.autograd.grad(
            v_pred, xy_collocation, ones, create_graph=True
        )[0]
        grad_p_dxy = torch.autograd.grad(
            p_pred, xy_collocation, ones, create_graph=True
        )[0]

        u_x, u_y = grad_u_dxy[:, 0], grad_u_dxy[:, 1]
        v_x, v_y = grad_v_dxy[:, 0], grad_v_dxy[:, 1]
        p_x, p_y = grad_p_dxy[:, 0], grad_p_dxy[:, 1]

        data_loss = self.lambda_data * torch.mean(
            (uv_pred_data - uv_data) ** 2
        )
        smooth_loss = self.lambda_smooth_w * torch.mean(
            grad_u_dxy**2 + grad_v_dxy**2
        )
        div_loss = self.lambda_div * torch.mean((u_x + v_y) ** 2)

        if self.lambda_mom > 0.0:
            grad_ux_dxy = torch.autograd.grad(
                u_x, xy_collocation, ones, create_graph=True
            )[0]
            grad_uy_dxy = torch.autograd.grad(
                u_y, xy_collocation, ones, create_graph=True
            )[0]
            grad_vx_dxy = torch.autograd.grad(
                v_x, xy_collocation, ones, create_graph=True
            )[0]
            grad_vy_dxy = torch.autograd.grad(
                v_y, xy_collocation, ones, create_graph=True
            )[0]

            u_xx = grad_ux_dxy[:, 0]
            u_yy = grad_uy_dxy[:, 1]
            v_xx = grad_vx_dxy[:, 0]
            v_yy = grad_vy_dxy[:, 1]

            mom_x = (
                -self.nu * (u_xx + u_yy)
                + p_x
                + u_pred * u_x
                + v_pred * u_y
            )
            mom_y = (
                -self.nu * (v_xx + v_yy)
                + p_y
                + u_pred * v_x
                + v_pred * v_y
            )
            momentum_loss = self.lambda_mom * torch.mean(
                mom_x**2 + mom_y**2
            )
        else:
            momentum_loss = uvp_collocation.new_zeros(())

        if p_reference is None or p_reference.numel() == 0:
            pressure_loss = uvp_collocation.new_zeros(())
        else:
            pressure_loss = self.lambda_p * torch.mean(p_reference**2)
        if uv_pred_wall is None or uv_pred_wall.numel() == 0:
            wall_loss = uvp_collocation.new_zeros(())
        else:
            wall_loss = self.lambda_wall * torch.mean(uv_pred_wall**2)

        return {
            "wind_data": data_loss,
            "wind_smooth": smooth_loss,
            "div": div_loss,
            "momentum": momentum_loss,
            "pressure": pressure_loss,
            "wind_wall": wall_loss,
        }

    def forward(
        self,
        uv_pred_data: torch.Tensor,
        uv_data: torch.Tensor,
        uvp_collocation: torch.Tensor,
        xy_collocation: torch.Tensor,
        uv_pred_wall: torch.Tensor | None = None,
        p_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sum(
            self.components(
                uv_pred_data,
                uv_data,
                uvp_collocation,
                xy_collocation,
                uv_pred_wall,
                p_reference,
            ).values()
        )


class GasSourceLoss(nn.Module):
    def __init__(
        self,
        lambda_nonfree: float = 1e-2,
        lambda_sparse: float = 1e-3,
    ):
        super().__init__()
        self.lambda_nonfree = lambda_nonfree
        self.lambda_sparse = lambda_sparse

    def components(
        self,
        q_free: torch.Tensor,
        q_nonfree: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_sparse = self.lambda_sparse * torch.mean(q_free)
        if q_nonfree.numel() == 0:
            loss_nonfree = q_free.new_zeros(())
        else:
            loss_nonfree = self.lambda_nonfree * torch.mean(q_nonfree**2)
        return {
            "sparse": loss_sparse,
            "nonfree": loss_nonfree,
        }

    def forward(
        self,
        q_free: torch.Tensor,
        q_nonfree: torch.Tensor,
    ) -> torch.Tensor:
        return sum(self.components(q_free, q_nonfree).values())
