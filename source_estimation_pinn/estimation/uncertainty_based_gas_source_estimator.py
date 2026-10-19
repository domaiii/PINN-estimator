from __future__ import annotations

import numpy as np

from ..environment import OccupancyGrid, WindField
from ..measurements import GasSampleSet
from .configs import EstimationHistory, GasSourceConfig
from .gas_source_estimator import GasSourceEstimator


class UncertaintyBasedGasSourceEstimator:
    """Public interface for uncertainty-based PINN gas source localization.

    The estimator trains on accumulated concentration measurements and uses
    source-oriented LLLA to propose the next informative measurement point.
    """

    def __init__(
        self,
        occupancy: OccupancyGrid,
        wind: WindField,
        config: GasSourceConfig | None = None,
        initial_training_steps: int = 1000,
        update_training_steps: int = 150,
        n_collocation_points: int = 1000,
    ):
        if initial_training_steps < 1 or update_training_steps < 1:
            raise ValueError("training steps must be positive")
        if n_collocation_points < 1:
            raise ValueError("n_collocation_points must be positive")

        self.occupancy = occupancy
        self.candidate_points = occupancy.free_points
        if len(self.candidate_points) == 0:
            raise ValueError("occupancy grid contains no free points")

        self.estimator = GasSourceEstimator(occupancy, wind, config)
        self.initial_training_steps = initial_training_steps
        self.update_training_steps = update_training_steps
        self.n_collocation_points = min(n_collocation_points, len(self.candidate_points))
        self._available = np.ones(len(self.candidate_points), dtype=bool)
        self._samples: GasSampleSet | None = None

    @property
    def samples(self) -> GasSampleSet:
        """Return a copy of all observations used for the current model."""
        if self._samples is None:
            raise RuntimeError("Call fit(...) first")
        return GasSampleSet(
            self._samples.positions.copy(),
            self._samples.concentrations.copy(),
        )

    def fit(self, initial_samples: GasSampleSet) -> EstimationHistory:
        """Train the internal estimator on initial, distinct grid measurements."""
        indices = self._candidate_indices(initial_samples.positions)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("initial_samples must not contain duplicate points")

        self._samples = GasSampleSet(
            initial_samples.positions.copy(),
            initial_samples.concentrations.copy(),
        )
        self._available.fill(True)
        self._available[indices] = False
        return self.estimator.fit(
            self._samples,
            steps=self.initial_training_steps,
            n_collocation_points=self.n_collocation_points,
        )

    def suggest_next_measurement(self) -> np.ndarray:
        """Return the highest-scoring free point that has not been measured."""
        self._require_fitted()
        if not np.any(self._available):
            raise RuntimeError("all candidate points have been measured")

        scores = self.estimator.source_uncertainty_reduction(
            self.candidate_points,
            source_eval_points=self.candidate_points,
        ).detach().cpu().numpy()
        if not np.isfinite(scores[self._available]).all():
            raise RuntimeError("LLLA produced non-finite acquisition scores")
        index = int(np.argmax(np.where(self._available, scores, -np.inf)))
        return self.candidate_points[index].copy()

    def add_measurement(
        self,
        position: np.ndarray,
        concentration: float,
    ) -> EstimationHistory:
        """Add one new observation and retrain the internal estimator."""
        self._require_fitted()
        index = self._candidate_indices(position)[0]
        if not self._available[index]:
            raise ValueError("position has already been measured")
        if not np.isfinite(concentration):
            raise ValueError("concentration must be finite")

        self._samples.append(self.candidate_points[index], concentration)
        history = self.estimator.fit(
            self._samples,
            steps=self.update_training_steps,
            n_collocation_points=self.n_collocation_points,
        )
        self._available[index] = False
        return history

    def get_source_estimate(self) -> np.ndarray:
        """Return the current source estimate from the internal estimator."""
        self._require_fitted()
        return self.estimator.get_source_estimate()

    def _candidate_indices(self, positions: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(positions, dtype=float))
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("positions must have shape (2,) or (n, 2)")

        distances = np.linalg.norm(
            points[:, None, :] - self.candidate_points[None, :, :],
            axis=2,
        )
        indices = np.argmin(distances, axis=1)
        tolerance = self.occupancy.resolution * 1e-6
        if np.any(distances[np.arange(len(points)), indices] > tolerance):
            raise ValueError("positions must be free occupancy grid points")
        return indices

    def _require_fitted(self) -> None:
        if self._samples is None:
            raise RuntimeError("Call fit(...) first")