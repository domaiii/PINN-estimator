import numpy as np
from dataclasses import dataclass

from source_estimation_pinn.environment import CellType, OccupancyGrid, WindField

@dataclass
class GasSource:
    """Physical parameters of a Gaussian gas source."""

    position: np.ndarray
    rate: float
    sigma: float

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (2,):
            raise ValueError("source position must have shape (2,)")
        if self.rate < 0 or self.sigma <= 0:
            raise ValueError("Invalid gas source parameter")


class GasDispersion:
    """Small 2-D Gaussian-filament dispersion simulator."""

    def __init__(
        self,
        wind_field: WindField,
        occupancy: OccupancyGrid,
        source: GasSource,
        filaments_per_sec: float = 20.0,
        diffusion_speed_std: float = 0.5,
        initial_sigma: float = 0.05,
        diffusivity: float = 1e-2,
        max_age: float = 60.0,
    ):
        if (
            filaments_per_sec <= 0
            or initial_sigma <= 0
            or diffusivity < 0
            or max_age <= 0
        ):
            raise ValueError("Invalid filament simulation parameter")

        self.wind_field = wind_field
        self.occupancy = occupancy
        self.source = source
        self.filaments_per_sec = float(filaments_per_sec)
        self.diffusion_speed_std = float(diffusion_speed_std)
        self.initial_sigma = float(initial_sigma)
        self.diffusivity = float(diffusivity)
        self.max_age = float(max_age)

        if not self.occupancy.is_strictly_free(self.source.position):
            source_state = self.occupancy.cell_type_at(self.source.position)
            raise ValueError(
                f"Source position {self.source.position} is not strictly free "
                f"(containing cell: {source_state.name})"
            )

        self.time = 0.0
        self.positions = np.empty((0, 2), dtype=float)
        self.ages = np.empty(0, dtype=float)
        self.masses = np.empty(0, dtype=float)
        self._release_remainder = 0.0

    def step(self, dt: float) -> None:
        """Advance the complete simulation by one time step."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.spawn_filaments(dt)
        self.update_filaments(dt)
        self.kill_filaments()
        self.time += dt

    def spawn_filaments(self, dt: float) -> None:
        self._release_remainder += self.filaments_per_sec * dt
        count = int(np.floor(self._release_remainder + 1e-12))
        self._release_remainder = max(0.0, self._release_remainder - count)
        if count == 0:
            return

        spawn = self._sample_source_positions(count)
        masses = np.full(count, self.source.rate / self.filaments_per_sec)
        self.positions = np.vstack([self.positions, spawn])
        self.ages = np.concatenate([self.ages, np.zeros(count)])
        self.masses = np.concatenate([self.masses, masses])

    def _sample_source_positions(self, count: int) -> np.ndarray:
        accepted = []
        missing = count

        for _ in range(100):
            candidates = np.random.normal(
                loc=self.source.position,
                scale=self.source.sigma,
                size=(max(2 * missing, 10), 2),
            )
            states = self.occupancy.cell_type_at(candidates)
            selected = candidates[states == CellType.FREE][:missing]
            accepted.append(selected)
            missing -= len(selected)

            if missing == 0:
                return np.vstack(accepted)

        raise RuntimeError(
            "Could not spawn enough filaments in free cells. "
            "The source may be too close to an obstacle."
        )

    def update_filaments(self, dt: float) -> None:
        if len(self.positions) == 0:
            return

        velocity = self.wind_field.velocity_at(self.positions)
        velocity += np.random.normal(scale=self.diffusion_speed_std, size=velocity.shape)
        proposed_position_changes = velocity * dt

        new_states = self.occupancy.cell_type_at(
            self.positions + proposed_position_changes
        )

        free_mask = new_states == CellType.FREE
        occupied_mask = new_states == CellType.OCCUPIED
        open_boundary_mask = new_states == CellType.OPEN_BOUNDARY

        self.positions[free_mask] += proposed_position_changes[free_mask]
        self.positions[occupied_mask] = self.adjust_wall_filaments(
            self.positions[occupied_mask], proposed_position_changes[occupied_mask]
        )

        self.positions = self.positions[np.logical_not(open_boundary_mask)]
        self.ages = self.ages[np.logical_not(open_boundary_mask)]
        self.masses = self.masses[np.logical_not(open_boundary_mask)]
        
        self.ages += dt

    def kill_filaments(self) -> None:
        keep = self.ages <= self.max_age
        self.positions = self.positions[keep]
        self.ages = self.ages[keep]
        self.masses = self.masses[keep]

    def concentration_at(self, position: np.ndarray):
        """Evaluate concentration at one point or at an array of points."""
        query = np.asarray(position, dtype=float)
        single_point = query.ndim == 1
        query = np.atleast_2d(query)

        if query.shape[1] != 2:
            raise ValueError("position must have shape (2,) or (n, 2)")
        if len(self.positions) == 0:
            result = np.zeros(len(query))
            return float(result[0]) if single_point else result

        sigma_squared = self.initial_sigma**2 + 2.0 * self.diffusivity * self.ages
        distance_squared = np.sum(
            (query[:, None, :] - self.positions[None, :, :]) ** 2,
            axis=2,
        )
        contributions = (
            self.masses[None, :]
            / (2.0 * np.pi * sigma_squared[None, :])
            * np.exp(-distance_squared / (2.0 * sigma_squared[None, :]))
        )
        result = contributions.sum(axis=1)
        return float(result[0]) if single_point else result

    def adjust_wall_filaments(
        self,
        positions: np.ndarray,
        proposed_changes: np.ndarray,
    ) -> np.ndarray:
        """Slide wall-colliding filaments along an axis-aligned wall."""

        if len(positions) == 0:
            return positions

        states = self.occupancy.cell_type_at(positions + proposed_changes)

        if np.any(states != CellType.OCCUPIED):
            raise ValueError("'adjust_wall_filaments' can only be called for filaments "
                             "in occupied cells.")

        corrected_positions = positions.copy()

        for pos, change, corrected in zip(
            positions,
            proposed_changes,
            corrected_positions,
        ):
            x_candidate = pos + np.array([change[0], 0.0])
            y_candidate = pos + np.array([0.0, change[1]])

            if self.occupancy.cell_type_at(x_candidate) == CellType.FREE:
                corrected[0] = x_candidate[0]

            elif self.occupancy.cell_type_at(y_candidate) == CellType.FREE:
                corrected[1] = y_candidate[1]

        return corrected_positions

    @property
    def number_of_filaments(self) -> int:
        return len(self.positions)
