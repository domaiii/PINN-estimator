import torch

from source_estimation_pinn.estimation.configs import GasSourceConfig


class StaticWeights(torch.nn.Module):
    def __init__(self, config: GasSourceConfig):
        super().__init__()
        self.lambda_sparse = config.lambda_sparse
        self.lambda_nonfree = config.lambda_nonfree
        self.lambda_gas_data = config.lambda_gas_data
        self.lambda_ad = config.lambda_ad
        self.lambda_gas_wall = config.lambda_gas_wall
        self.lambda_inflow = config.lambda_inflow

    def forward(self):
        return {
            "sparse": torch.as_tensor(self.lambda_sparse),
            "nonfree": torch.as_tensor(self.lambda_nonfree),
            "gas_data": torch.as_tensor(self.lambda_gas_data),
            "ad": torch.as_tensor(self.lambda_ad),
            "gas_wall": torch.as_tensor(self.lambda_gas_wall),
            "inflow": torch.as_tensor(self.lambda_inflow),
        }

    def log_term(self, active_components=None):
        return 0.0


class LearnedWeights(StaticWeights):
    def __init__(self, config: GasSourceConfig):
        super().__init__(config)

        for name in ("lambda_gas_data", "lambda_ad", "lambda_inflow",
                     "lambda_gas_wall", "lambda_nonfree"):
            value = getattr(config, name)
            if not 0.0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive for learned weights")

        self.log_eps_data = torch.nn.Parameter(
            -0.5 * torch.log(torch.tensor(2.0 * config.lambda_gas_data))
        )
        self.log_eps_ad = torch.nn.Parameter(
            -0.5 * torch.log(torch.tensor(2.0 * config.lambda_ad))
        )
        self.log_eps_inflow = torch.nn.Parameter(
            -0.5 * torch.log(torch.tensor(2.0 * config.lambda_inflow))
        )
        self.log_eps_gas_wall = torch.nn.Parameter(
            -0.5 * torch.log(torch.tensor(2.0 * config.lambda_gas_wall))
        )
        self.log_eps_nonfree = torch.nn.Parameter(
            -0.5 * torch.log(torch.tensor(2.0 * config.lambda_nonfree))
        )

    def forward(self):
        weights = super().forward()
        weights["gas_data"] = 0.5 * torch.exp(-2.0 * self.log_eps_data)
        weights["ad"] = 0.5 * torch.exp(-2.0 * self.log_eps_ad)
        weights["inflow"] = 0.5 * torch.exp(-2.0 * self.log_eps_inflow)
        weights["gas_wall"] = 0.5 * torch.exp(-2.0 * self.log_eps_gas_wall)
        weights["nonfree"] = 0.5 * torch.exp(-2.0 * self.log_eps_nonfree)
        return weights

    def log_term(self, active_components=None):
        log_eps = {
            "gas_data": self.log_eps_data,
            "ad": self.log_eps_ad,
            "inflow": self.log_eps_inflow,
            "gas_wall": self.log_eps_gas_wall,
            "nonfree": self.log_eps_nonfree,
        }
        return sum(value for name, value in log_eps.items()
                   if active_components is None or name in active_components)
