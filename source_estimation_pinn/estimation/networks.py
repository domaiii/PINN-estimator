from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class WindNet2D(nn.Module):
    """Approximate the two-dimensional wind and pressure fields (x, y) -> (u, v, p)."""

    def __init__(
        self,
        lower_bound: tuple[float, float] | np.ndarray,
        upper_bound: tuple[float, float] | np.ndarray,
        hidden_layers: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        lower_bound_t = torch.as_tensor(lower_bound, dtype=torch.float32)
        upper_bound_t = torch.as_tensor(upper_bound, dtype=torch.float32)

        if lower_bound_t.shape != (2,) or upper_bound_t.shape != (2,):
            raise ValueError("lower_bound and upper_bound must have shape (2,)")
        if torch.any(upper_bound_t <= lower_bound_t):
            raise ValueError("upper_bound must be greater than lower_bound")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        self.register_buffer("lower_bound", lower_bound_t)
        self.register_buffer("upper_bound", upper_bound_t)

        layers: list[nn.Module] = [
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: Tensor) -> Tensor:
        xy_normalized = 2.0 * (
            (xy - self.lower_bound) / (self.upper_bound - self.lower_bound)
        ) - 1.0
        return self.net(xy_normalized)


class VelocityNet2D(nn.Module):
    """Approximate a two-dimensional velocity field (x, y) -> (u, v)."""

    def __init__(
        self,
        lower_bound: tuple[float, float] | np.ndarray,
        upper_bound: tuple[float, float] | np.ndarray,
        hidden_layers: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        lower_bound_t = torch.as_tensor(lower_bound, dtype=torch.float32)
        upper_bound_t = torch.as_tensor(upper_bound, dtype=torch.float32)

        if lower_bound_t.shape != (2,) or upper_bound_t.shape != (2,):
            raise ValueError("lower_bound and upper_bound must have shape (2,)")
        if torch.any(upper_bound_t <= lower_bound_t):
            raise ValueError("upper_bound must be greater than lower_bound")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        self.register_buffer("lower_bound", lower_bound_t)
        self.register_buffer("upper_bound", upper_bound_t)

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.Tanh()))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: Tensor) -> Tensor:
        xy_normalized = 2.0 * (
            (xy - self.lower_bound) / (self.upper_bound - self.lower_bound)
        ) - 1.0
        return self.net(xy_normalized)


class StreamFunctionNet2D(nn.Module):
    """Represent an exactly divergence-free 2D velocity through a streamfunction."""

    def __init__(
        self,
        lower_bound: tuple[float, float] | np.ndarray,
        upper_bound: tuple[float, float] | np.ndarray,
        hidden_layers: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        lower_bound_t = torch.as_tensor(lower_bound, dtype=torch.float32)
        upper_bound_t = torch.as_tensor(upper_bound, dtype=torch.float32)

        if lower_bound_t.shape != (2,) or upper_bound_t.shape != (2,):
            raise ValueError("lower_bound and upper_bound must have shape (2,)")
        if torch.any(upper_bound_t <= lower_bound_t):
            raise ValueError("upper_bound must be greater than lower_bound")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        self.register_buffer("lower_bound", lower_bound_t)
        self.register_buffer("upper_bound", upper_bound_t)

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.Tanh()))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: Tensor) -> Tensor:
        # Velocity is a spatial derivative of psi, so autograd must remain enabled
        # even while the caller evaluates the model for plotting.
        with torch.enable_grad():
            if not xy.requires_grad:
                xy = xy.detach().requires_grad_(True)
            xy_normalized = 2.0 * (
                (xy - self.lower_bound) / (self.upper_bound - self.lower_bound)
            ) - 1.0
            psi = self.net(xy_normalized)[:, 0]
            grad_psi = torch.autograd.grad(
                psi,
                xy,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
            )[0]
            return torch.stack((grad_psi[:, 1], -grad_psi[:, 0]), dim=1)


class PositiveFieldNet(nn.Module):
    """Approximate a positive scalar field over a rectangular 2D domain."""

    def __init__(
        self,
        lower_bound: tuple[float, float] | np.ndarray,
        upper_bound: tuple[float, float] | np.ndarray,
        hidden_layers: int = 2,
        hidden_dim: int = 32,
        initial_value: float = 1e-3,
    ):
        super().__init__()

        lower_bound_t = torch.as_tensor(lower_bound, dtype=torch.float32)
        upper_bound_t = torch.as_tensor(upper_bound, dtype=torch.float32)

        if lower_bound_t.shape != (2,) or upper_bound_t.shape != (2,):
            raise ValueError("lower_bound and upper_bound must have shape (2,)")
        if torch.any(upper_bound_t <= lower_bound_t):
            raise ValueError("upper_bound must be greater than lower_bound")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if initial_value <= 0.0:
            raise ValueError("initial_value must be positive")

        self.register_buffer("lower_bound", lower_bound_t)
        self.register_buffer("upper_bound", upper_bound_t)

        layers: list[nn.Module] = [
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Initialize softplus(net(x)) close to the requested positive value.
        with torch.no_grad():
            self.net[-1].bias.fill_(np.log(np.expm1(max(initial_value, 1e-8))))

    def last_hidden_features(self, xy: Tensor) -> Tensor:
        """Return the output of the final hidden layer."""
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (n, 2)")

        xy_normalized = 2.0 * (
            (xy - self.lower_bound) / (self.upper_bound - self.lower_bound)
        ) - 1.0
        return self.net[:-1](xy_normalized)

    def eval_last_layer(self, features: Tensor) -> Tensor:
        """Evaluate the final linear layer on precomputed features."""
        return self.net[-1](features)[:, 0]

    def raw_output(self, xy: Tensor) -> Tensor:
        """Return the scalar output before the softplus activation."""
        return self.eval_last_layer(self.last_hidden_features(xy))

    def forward(self, xy: Tensor) -> Tensor:
        return nn.functional.softplus(self.raw_output(xy))
