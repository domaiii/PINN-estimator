#%%
import numpy as np

try:
    from environment import WindField, OccupancyGrid, CellType
except ImportError:
    from dispersion.environment import WindField, OccupancyGrid, CellType


class GasDispersion:
    """Small 2-D Gaussian-filament dispersion simulator."""

    def __init__(
        self,
        wind_field: WindField,
        source_x: float = 2.0,
        source_y: float = 2.0,
        source_sigma: float = 0.1,
        source_rate: float = 1.0,
        release_rate: float = 20.0,
        diffusion_speed_std: float = 0.5,
        initial_sigma: float = 0.05,
        diffusivity: float = 1e-2,
        max_age: float = 60.0,
    ):
        if release_rate <= 0 or initial_sigma <= 0 or diffusivity < 0 or max_age <= 0:
            raise ValueError("Invalid filament simulation parameter")

        self.wind_field = wind_field
        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.source_sigma = float(source_sigma)
        self.source_rate = float(source_rate)
        self.release_rate = float(release_rate)
        self.diffusion_speed_std = float(diffusion_speed_std)
        self.initial_sigma = float(initial_sigma)
        self.diffusivity = float(diffusivity)
        self.max_age = float(max_age)

        self.occupancy = None

        self.time = 0.0
        self.positions = np.empty((0, 2), dtype=float)
        self.ages = np.empty(0, dtype=float)
        self.masses = np.empty(0, dtype=float)
        self._release_remainder = 0.0

    def create_occupancy_grid(self, mshfile: str, resolution: float = 0.1) -> None:
        self.occupancy = OccupancyGrid.from_msh_file(mshfile, resolution)

    def step(self, dt: float) -> None:
        """Advance the complete simulation by one time step."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.spawn_filaments(dt)
        self.update_filaments(dt)
        self.kill_filaments()
        self.time += dt

    def spawn_filaments(self, dt: float) -> None:
        self._release_remainder += self.release_rate * dt
        count = int(self._release_remainder)
        self._release_remainder -= count
        if count == 0:
            return

        spawn = np.random.normal(
            loc=[self.source_x, self.source_y],
            scale=self.source_sigma,
            size=(count, 2),
        )
        self.positions = np.vstack([self.positions, spawn])
        self.ages = np.concatenate([self.ages, np.zeros(count)])
        mass_per_filament = self.source_rate / self.release_rate
        self.masses = np.concatenate(
            [self.masses, np.full(count, mass_per_filament)]
        )

    def update_filaments(self, dt: float) -> None:
        if len(self.positions) == 0:
            return

        velocity = self.wind_field.velocity_at(self.positions)
        velocity += np.random.normal(scale=self.diffusion_speed_std, size=velocity.shape)
        proposed_position_changes = velocity * dt

        if self.occupancy is None:
            raise RuntimeError("Create an occupancy grid before updating filaments")

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


# %% Minimal live example
if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from IPython.display import display

    wind = WindField.from_csv("/app/data/example_labyrinth/wind_gt.csv")
    gas = GasDispersion(
        wind_field=wind,
        source_x = 1.0,
        source_y = 2.0,
        source_rate = 1.0,
        release_rate = 20.0,
        diffusion_speed_std = 0.7,
        initial_sigma=0.05,
        diffusivity=1e-2,
        max_age=30.0,
    )
    gas.create_occupancy_grid(
        "/app/data/example_labyrinth/labyrinth_2d_fine.msh",
        resolution=0.1,
    )

    height, width = gas.occupancy.occupancy.shape
    x_min, y_min = gas.occupancy.origin
    extent = [
        x_min,
        x_min + width * gas.occupancy.resolution,
        y_min,
        y_min + height * gas.occupancy.resolution,
    ]
    occupancy_cmap = ListedColormap(["#303030", "#f2f2f2", "#00a6ff"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(
        gas.occupancy.occupancy,
        origin="lower",
        extent=extent,
        cmap=occupancy_cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
        zorder=0,
    )
    particles = ax.scatter(
        [], [], s=10, color="tab:orange", label="filaments", zorder=3
    )
    source_marker = ax.scatter(
        gas.source_x, gas.source_y,
        marker="x",
        s=80,
        color="red",
        label="source",
        zorder=4,
    )
    ax.set(
        xlim=(extent[0], extent[1]),
        ylim=(extent[2], extent[3]),
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    ax.legend(handles=[
        particles,
        source_marker,
        Patch(color=occupancy_cmap(CellType.FREE), label="free"),
        Patch(color=occupancy_cmap(CellType.OCCUPIED), label="occupied"),
        Patch(
            color=occupancy_cmap(CellType.OPEN_BOUNDARY),
            label="open boundary",
        ),
    ])

    dt = 0.05
    display_handle = display(fig, display_id=True)

    for step in range(600):
        gas.step(dt)
        particles.set_offsets(gas.positions)
        ax.set_title(
            f"Gas filaments: t={gas.time:.1f} s, n={gas.number_of_filaments}"
        )

        # Updating every fifth step keeps inline notebooks responsive.
        if step % 5 == 0:
            display_handle.update(fig)

    display_handle.update(fig)
    plt.close(fig)

# %%
