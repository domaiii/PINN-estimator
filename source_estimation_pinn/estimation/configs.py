from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    last_layer_prior_precision: float = 1e-3
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
        if self.last_layer_prior_precision < 0.0:
            raise ValueError("last_layer_prior_precision must be non-negative")
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


@dataclass
class EstimationHistory:
    """Weighted loss values recorded during training."""

    losses: dict[str, np.ndarray]

    @property
    def total(self) -> np.ndarray:
        return self.losses["total"]
