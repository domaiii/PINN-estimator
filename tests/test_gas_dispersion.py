from pathlib import Path

import numpy as np
import pytest

from dispersion_physics.gas import GasDispersion, GasSource
from source_estimation_pinn.environment import CellType, OccupancyGrid


class ZeroWind:
    def velocity_at(self, positions: np.ndarray) -> np.ndarray:
        return np.zeros_like(positions, dtype=float)


def make_occupancy(with_wall: bool = False) -> OccupancyGrid:
    cells = np.full((5, 5), CellType.FREE, dtype=np.uint8)
    if with_wall:
        cells[:, 2] = CellType.OCCUPIED
    return OccupancyGrid(
        occupancy=cells,
        resolution=1.0,
        origin=np.array([0.0, 0.0]),
    )


def make_dispersion(occupancy: OccupancyGrid | None = None) -> GasDispersion:
    return GasDispersion(
        wind_field=ZeroWind(),
        occupancy=occupancy or make_occupancy(),
        source=GasSource(position=np.array([0.5, 0.5]), rate=2.0, sigma=0.1),
        filaments_per_sec=4.0,
        diffusion_speed_std=0.0,
        initial_sigma=0.2,
        diffusivity=0.0,
    )


def test_concentration_handles_empty_and_batched_queries():
    dispersion = make_dispersion()
    np.testing.assert_array_equal(
        dispersion.concentration_at(np.array([[0.5, 0.5], [1.5, 0.5]])),
        [0.0, 0.0],
    )

    dispersion.positions = np.array([[0.5, 0.5]])
    dispersion.ages = np.array([0.0])
    dispersion.masses = np.array([2.0])
    expected = 2.0 / (2.0 * np.pi * 0.2**2)

    assert dispersion.concentration_at(np.array([0.5, 0.5])) == pytest.approx(expected)
    assert dispersion.concentration_at(np.array([[0.5, 0.5]]))[0] == pytest.approx(
        expected
    )


def test_wall_blocks_concentration_from_other_side():
    dispersion = make_dispersion(make_occupancy(with_wall=True))
    dispersion.positions = np.array([[0.5, 1.5]])
    dispersion.ages = np.array([0.0])
    dispersion.masses = np.array([1.0])

    assert dispersion.concentration_at(np.array([1.5, 1.5])) > 0.0
    assert dispersion.concentration_at(np.array([3.5, 1.5])) == 0.0


def test_spawn_filaments_preserves_source_rate():
    dispersion = make_dispersion()
    dispersion._sample_source_positions = lambda count: np.tile(
        np.array([[0.5, 0.5]]), (count, 1)
    )

    dispersion.spawn_filaments(0.5)

    assert dispersion.number_of_filaments == 2
    np.testing.assert_array_equal(dispersion.masses, [0.5, 0.5])


def test_update_filaments_does_not_enter_occupied_cell():
    dispersion = make_dispersion(make_occupancy(with_wall=True))
    dispersion.positions = np.array([[1.5, 1.5]])
    dispersion.ages = np.array([0.0])
    dispersion.masses = np.array([1.0])
    dispersion.wind_field = type(
        "WindTowardsWall",
        (),
        {"velocity_at": staticmethod(lambda positions: np.tile([1.0, 0.0], (len(positions), 1)))},
    )()

    dispersion.update_filaments(1.0)

    np.testing.assert_array_equal(dispersion.positions, [[1.5, 1.5]])

def test_occupancy_map_free_edge_cells_are_open_boundaries():
    root = Path(__file__).resolve().parents[1]
    occupancy = OccupancyGrid.from_occupancy_file(
        root / "data/example_labyrinth/occupancy.yaml"
    )

    edge = np.zeros_like(occupancy.occupancy, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    free_edge = edge & (occupancy.occupancy != CellType.OCCUPIED)

    assert np.any(free_edge)
    assert np.all(occupancy.occupancy[free_edge] == CellType.OPEN_BOUNDARY)


def test_yaml_and_msh_maps_have_matching_free_space_after_alignment():
    root = Path(__file__).resolve().parents[1]
    yaml_map = OccupancyGrid.from_occupancy_file(
        root / "data/example_labyrinth/occupancy.yaml"
    )
    msh_map = OccupancyGrid.from_msh_file(
        root / "data/example_labyrinth/labyrinth_2d_fine.msh",
        resolution=0.1,
    )

    # The YAML image has a one-cell padding ring around the MSH domain.
    assert yaml_map.occupancy.shape == (
        msh_map.occupancy.shape[0] + 2,
        msh_map.occupancy.shape[1] + 2,
    )
    np.testing.assert_allclose(
        yaml_map.origin + yaml_map.resolution,
        msh_map.origin,
    )
    yaml_interior = yaml_map.occupancy[1:-1, 1:-1]
    np.testing.assert_array_equal(
        yaml_interior != CellType.OCCUPIED,
        msh_map.occupancy != CellType.OCCUPIED,
    )


def test_filament_leaves_through_open_boundary():
    occupancy = make_occupancy()
    occupancy.occupancy[2, -1] = CellType.OPEN_BOUNDARY
    dispersion = make_dispersion(occupancy)
    dispersion.positions = np.array([[3.5, 2.5]])
    dispersion.ages = np.array([0.0])
    dispersion.masses = np.array([1.0])
    dispersion.wind_field = type(
        "WindToOpenBoundary",
        (),
        {"velocity_at": staticmethod(lambda positions: np.tile([1.0, 0.0], (len(positions), 1)))},
    )()

    dispersion.update_filaments(1.0)

    assert dispersion.number_of_filaments == 0


def test_filament_cannot_skip_open_boundary_in_long_step():
    occupancy = make_occupancy()
    occupancy.occupancy[2, -1] = CellType.OPEN_BOUNDARY
    dispersion = make_dispersion(occupancy)
    dispersion.positions = np.array([[2.5, 2.5]])
    dispersion.ages = np.array([0.0])
    dispersion.masses = np.array([1.0])
    dispersion.wind_field = type(
        "FastWindToOpenBoundary",
        (),
        {"velocity_at": staticmethod(lambda positions: np.tile([3.0, 0.0], (len(positions), 1)))},
    )()

    dispersion.update_filaments(1.0)

    assert dispersion.number_of_filaments == 0
