import numpy as np
import torch

from .configs import (
    GasSourceConfig,
    NavierStokesWindConfig,
    StreamFunctionWindConfig,
    WindGasSourceConfig,
)
from .networks import PositiveFieldNet, StreamFunctionNet2D, WindNet2D


class GasSourceModel(torch.nn.Module):
    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        config: GasSourceConfig,
    ):
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
        last_layer_params_c = torch.cat(
            [self.c_net.net[-1].weight.reshape(-1), self.c_net.net[-1].bias.reshape(-1)]
        )
        last_layer_params_q = torch.cat(
            [self.q_net.net[-1].weight.reshape(-1), self.q_net.net[-1].bias.reshape(-1)]
        )
        return torch.cat([last_layer_params_c, last_layer_params_q])

    def split_ll_params(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a concatenated last-layer parameter vector into two parts."""
        c_last_layer_size = self.c_net.net[-1].in_features + 1
        return params[:c_last_layer_size], params[c_last_layer_size:]


def build_wind_network(
    bounds: tuple[np.ndarray, np.ndarray],
    config: NavierStokesWindConfig | StreamFunctionWindConfig,
) -> torch.nn.Module:
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


class WindGasSourceModel(GasSourceModel):
    def __init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        config: WindGasSourceConfig,
    ):
        super().__init__(bounds, config)
        self.wind_net = build_wind_network(bounds, config.wind)
