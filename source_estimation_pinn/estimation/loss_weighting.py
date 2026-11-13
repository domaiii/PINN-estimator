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
    """Learn selected residual variances, optionally above fixed variance floors."""

    PARAMETER_NAMES = {"gas_data": "log_eps_data", "ad": "log_eps_ad",
                       "inflow": "log_eps_inflow", "gas_wall": "log_eps_gas_wall",
                       "nonfree": "log_eps_nonfree"}

    def __init__(self, config: GasSourceConfig):
        super().__init__(config)
        self.terms = config.learned_weight_terms
        self.variance_floors = {"gas_data": config.gas_data_variance_floor,
                                "ad": config.ad_variance_floor}
        initial = super().forward()
        for name in self.terms:
            weight = float(initial[name])
            if not 0.0 < weight < float("inf"):
                raise ValueError(f"lambda_{name} must be finite and positive for learned weights")
            variance = 1.0 / (2.0 * weight) - self.variance_floors.get(name, 0.0)
            if variance <= 0:
                raise ValueError(f"variance floor for {name} must be below its initial variance")
            setattr(self, self.PARAMETER_NAMES[name], torch.nn.Parameter(
                0.5 * torch.log(torch.tensor(variance))))

    def _log_variance(self, name):
        log_variance = 2.0 * getattr(self, self.PARAMETER_NAMES[name])
        floor = self.variance_floors.get(name, 0.0)
        if floor:
            # log(floor + exp(2s)), evaluated stably for very small residual scales.
            log_variance = torch.logaddexp(log_variance,
                                           log_variance.new_tensor(floor).log())
        return log_variance

    def forward(self):
        weights = super().forward()
        for name in self.terms:
            weights[name] = 0.5 * torch.exp(-self._log_variance(name))
        return weights

    def log_term(self, active_components=None):
        return sum(0.5 * self._log_variance(name) for name in self.terms
                   if active_components is None or name in active_components)
