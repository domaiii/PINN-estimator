from collections.abc import Callable

import torch
from torch import Tensor, nn


class JointSourceLaplace:
    """Joint c-q last-layer Laplace with fixed wind and hidden-layer weights.

    The supplied loss must depend on the last-layer parameters ordered as
    [c weights, c bias, q weights, q bias]. Its training data and wind are fixed.
    """

    def __init__(
        self,
        c_net: nn.Module,
        q_net: nn.Module,
        loss_function: Callable[[Tensor], Tensor],
    ):
        self.c_net = c_net
        self.q_net = q_net
        self.loss_function = loss_function
        self.covariance: Tensor | None = None

    def invalidate(self) -> None:
        """Discard the approximation after training or changing observations."""
        self.covariance = None

    def compute_covariance(self) -> Tensor:
        parameters = torch.cat([
            self.linear_layer_parameters(self.c_net.net[-1]),
            self.linear_layer_parameters(self.q_net.net[-1]),
        ]).detach()
        self.covariance = self.covariance_from_loss(
            self.loss_function, parameters
        ).detach()
        return self.covariance

    def split_covariance(self, covariance: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the cc, qq and qc blocks, in that order."""
        n_c = self.c_net.net[-1].in_features + 1
        return (
            covariance[:n_c, :n_c],
            covariance[n_c:, n_c:],
            covariance[n_c:, :n_c],
        )

    def concentration_gradient_wrt_parameters(self, xy: Tensor) -> Tensor:
        """Return one concentration-parameter gradient per point: (N, n_c)."""
        last_layer_input = self.c_net.last_hidden_features(xy)
        augmented_input = torch.cat(
            (last_layer_input, torch.ones_like(last_layer_input[:, :1])), dim=1
        )
        concentration_params = self.linear_layer_parameters(self.c_net.net[-1])
        pre_softplus_output = augmented_input @ concentration_params
        return augmented_input * torch.sigmoid(pre_softplus_output)[:, None]

    def source_uncertainty_reduction(self, xy: Tensor) -> Tensor:
        """Score concentration measurements with wind treated as known and fixed."""
        if self.covariance is None:
            self.compute_covariance()

        cov_cc, _, cov_qc = self.split_covariance(self.covariance)
        grad_c = self.concentration_gradient_wrt_parameters(xy).T.to(cov_cc.dtype)
        numerator = torch.sum((cov_qc @ grad_c)**2, dim=0)
        denominator = (grad_c.T @ cov_cc @ grad_c).diagonal()
        return numerator / (denominator + 1e-8)

    @staticmethod
    def linear_layer_parameters(layer: nn.Linear) -> Tensor:
        """Flatten a linear layer weight and bias into one parameter vector."""
        if layer.bias is None:
            raise ValueError("The linear layer must have a bias")
        return torch.cat((layer.weight.reshape(-1), layer.bias.reshape(-1)))


    @staticmethod
    def covariance_from_loss(
        loss_function: Callable[[Tensor], Tensor],
        parameters: Tensor,
    ) -> Tensor:
        """Invert the symmetric Hessian after flooring its eigenvalues at 1e-7.

        This matches the original MWE and inverts the clipped Hessian,
        not necessarily the original Hessian. Use float64 for the eigendecomposition
        and covariance so a large eigenvalue ratio does not spoil definiteness.
        """
        hessian = torch.autograd.functional.hessian(loss_function, parameters)
        hessian = hessian.to(torch.float64)
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        eigenvalues = eigenvalues.clamp_min(1e-7)
        covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
        return 0.5 * (covariance + covariance.T)
