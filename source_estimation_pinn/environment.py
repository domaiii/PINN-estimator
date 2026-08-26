from pathlib import Path
from matplotlib import pyplot, colors
from matplotlib.patches import Patch
from meshio import Mesh, read
import numpy as np
import pandas as pd
import re
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from enum import IntEnum

class CellType(IntEnum):
    """Cell types for occupancy grid."""
    OCCUPIED      = 0       # Cell is occupied (obstacle)
    FREE          = 1       # Cell is free (open space)
    OPEN_BOUNDARY = 2       # Cell is an open boundary to outside the environment


class WindField:
    """Static 2-D wind field loaded from scattered CSV samples."""

    def __init__(self, positions: np.ndarray, velocities: np.ndarray):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)

        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n, 2)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must have shape (n, 2)")

        self._linear = LinearNDInterpolator(self.positions, self.velocities)
        self._nearest = NearestNDInterpolator(self.positions, self.velocities)

    @classmethod
    def from_csv(cls, path: str | Path) -> "WindField":
        data = pd.read_csv(path)
        position_columns = ["Points:0", "Points:1"]
        velocity_columns = ["U:0", "U:1"]

        required = position_columns + velocity_columns
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError("Missing wind CSV columns: " + ", ".join(missing))

        return cls(
            data[position_columns].to_numpy(float),
            data[velocity_columns].to_numpy(float),
        )

    def velocity_at(self, position: np.ndarray) -> np.ndarray:
        """Return wind at one position ``(2,)`` or many positions ``(n, 2)``."""
        points = np.asarray(position, dtype=float)
        single_point = points.ndim == 1
        points = np.atleast_2d(points)

        if points.shape[1] != 2:
            raise ValueError("position must have shape (2,) or (n, 2)")

        velocity = np.asarray(self._linear(points), dtype=float)
        outside = np.isnan(velocity).any(axis=1)
        if np.any(outside):
            velocity[outside] = self._nearest(points[outside])

        return velocity[0] if single_point else velocity


class OccupancyGrid:
    """Occupancy grid for a 2-D environment."""

    def __init__(self, occupancy: np.ndarray, resolution: float, origin: np.ndarray):
        self.occupancy = np.asarray(occupancy, dtype=np.uint8)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)

        if self.occupancy.ndim != 2:
            raise ValueError("occupancy must be a 2-D array")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.origin.shape != (2,):
            raise ValueError("origin must have shape (2,)")

    @property
    def lower_bound(self) -> np.ndarray:
        return self.origin.copy()

    @property
    def upper_bound(self) -> np.ndarray:
        height, width = self.occupancy.shape
        return self.origin + np.array([width, height]) * self.resolution

    @property
    def x_centers(self) -> np.ndarray:
        width = self.occupancy.shape[1]
        return self.origin[0] + (np.arange(width) + 0.5) * self.resolution

    @property
    def y_centers(self) -> np.ndarray:
        height = self.occupancy.shape[0]
        return self.origin[1] + (np.arange(height) + 0.5) * self.resolution

    @property
    def xx(self) -> np.ndarray:
        return np.meshgrid(self.x_centers, self.y_centers)[0]

    @property
    def yy(self) -> np.ndarray:
        return np.meshgrid(self.x_centers, self.y_centers)[1]

    @property
    def points(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x_centers, self.y_centers)
        return np.column_stack((xx.ravel(), yy.ravel()))

    @property
    def free_mask(self) -> np.ndarray:
        return self.occupancy == CellType.FREE

    @property
    def occupied_mask(self) -> np.ndarray:
        return self.occupancy == CellType.OCCUPIED

    @property
    def open_boundary_mask(self) -> np.ndarray:
        return self.occupancy == CellType.OPEN_BOUNDARY

    @property
    def free_points(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x_centers, self.y_centers)
        return np.column_stack((xx[self.free_mask], yy[self.free_mask]))

    @property
    def occupied_points(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x_centers, self.y_centers)
        return np.column_stack((xx[self.occupied_mask], yy[self.occupied_mask]))

    @property
    def wall_mask(self) -> np.ndarray:
        occupied_neighbor = np.zeros_like(self.free_mask)
        occupied_neighbor[1:, :] |= self.occupied_mask[:-1, :]
        occupied_neighbor[:-1, :] |= self.occupied_mask[1:, :]
        occupied_neighbor[:, 1:] |= self.occupied_mask[:, :-1]
        occupied_neighbor[:, :-1] |= self.occupied_mask[:, 1:]

        wall_mask = self.free_mask & occupied_neighbor
        wall_mask[0, :] |= self.free_mask[0, :]
        wall_mask[-1, :] |= self.free_mask[-1, :]
        wall_mask[:, 0] |= self.free_mask[:, 0]
        wall_mask[:, -1] |= self.free_mask[:, -1]
        return wall_mask

    @property
    def wall_points(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x_centers, self.y_centers)
        return np.column_stack((xx[self.wall_mask], yy[self.wall_mask]))

    def wall_boundary_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return closed-wall face midpoints and normals pointing out of free space."""
        free = self.free_mask
        occupied = self.occupied_mask
        xx, yy = np.meshgrid(self.x_centers, self.y_centers)
        centers = np.stack((xx, yy), axis=-1)
        half_cell = 0.5 * self.resolution

        points = []
        normals = []

        def add_faces(mask, offset, normal):
            count = np.count_nonzero(mask)
            if count == 0:
                return
            points.append(centers[mask] + half_cell * np.asarray(offset))
            normals.append(np.tile(normal, (count, 1)))

        left = np.zeros_like(free)
        left[:, 1:] = free[:, 1:] & occupied[:, :-1]
        left[:, 0] = free[:, 0]
        add_faces(left, (-1.0, 0.0), (-1.0, 0.0))

        right = np.zeros_like(free)
        right[:, :-1] = free[:, :-1] & occupied[:, 1:]
        right[:, -1] = free[:, -1]
        add_faces(right, (1.0, 0.0), (1.0, 0.0))

        bottom = np.zeros_like(free)
        bottom[1:, :] = free[1:, :] & occupied[:-1, :]
        bottom[0, :] = free[0, :]
        add_faces(bottom, (0.0, -1.0), (0.0, -1.0))

        top = np.zeros_like(free)
        top[:-1, :] = free[:-1, :] & occupied[1:, :]
        top[-1, :] = free[-1, :]
        add_faces(top, (0.0, 1.0), (0.0, 1.0))

        if not points:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.vstack(points), np.vstack(normals)

    def open_boundary_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return outer open-boundary points and outward unit normals."""
        open_mask = self.open_boundary_mask
        covered = np.zeros_like(open_mask)
        points = []
        normals = []

        def add(points_for_side, normal):
            if len(points_for_side) == 0:
                return
            points.append(points_for_side)
            normals.append(np.tile(normal, (len(points_for_side), 1)))

        left = open_mask[:, 0]
        covered[:, 0] |= left
        add(
            np.column_stack((
                np.full(np.count_nonzero(left), self.lower_bound[0]),
                self.y_centers[left],
            )),
            (-1.0, 0.0),
        )

        right = open_mask[:, -1]
        covered[:, -1] |= right
        add(
            np.column_stack((
                np.full(np.count_nonzero(right), self.upper_bound[0]),
                self.y_centers[right],
            )),
            (1.0, 0.0),
        )

        bottom = open_mask[0, :]
        covered[0, :] |= bottom
        add(
            np.column_stack((
                self.x_centers[bottom],
                np.full(np.count_nonzero(bottom), self.lower_bound[1]),
            )),
            (0.0, -1.0),
        )

        top = open_mask[-1, :]
        covered[-1, :] |= top
        add(
            np.column_stack((
                self.x_centers[top],
                np.full(np.count_nonzero(top), self.upper_bound[1]),
            )),
            (0.0, 1.0),
        )

        if np.any(open_mask & ~covered):
            raise ValueError(
                "Cannot infer normals for OPEN_BOUNDARY cells away from the "
                "rectangular grid boundary."
            )
        if not points:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.vstack(points), np.vstack(normals)

    @classmethod
    def from_msh_file(cls, path: str | Path, resolution: float) -> "OccupancyGrid":
        """Rasterize all mesh triangles into a typed occupancy grid."""
        mesh = read(path)
        points = np.asarray(mesh.points[:, :2], dtype=float)
        triangles = cls.extract_triangles_msh(mesh)

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        width = int(np.ceil((x_max - x_min) / resolution))
        height = int(np.ceil((y_max - y_min) / resolution))

        x_centers = x_min + (np.arange(width) + 0.5) * resolution
        y_centers = y_min + (np.arange(height) + 0.5) * resolution
        occupancy_grid = np.full((height, width), CellType.OCCUPIED, dtype=np.uint8)

        for triangle in triangles:
            vertices = points[triangle]

            # All columns and rows within the bounding box of the triangle
            columns = np.flatnonzero(
                (x_centers >= vertices[:, 0].min())
                & (x_centers <= vertices[:, 0].max())
            )
            rows = np.flatnonzero(
                (y_centers >= vertices[:, 1].min())
                & (y_centers <= vertices[:, 1].max())
            )
            if len(columns) == 0 or len(rows) == 0:
                continue

            inside = cls.points_in_triangle(x_centers[columns], y_centers[rows], vertices)

            # Update the respective part of the occupancy grid with the triangle mask
            indices_bounding_box = np.ix_(rows, columns)
            occupancy_grid[indices_bounding_box] = np.where(inside, 
                                                            CellType.FREE, 
                                                            occupancy_grid[indices_bounding_box])

        # Assign OPEN_BOUNDARY to in- and outflows of the domain
        for block_index, cell_block in enumerate(mesh.cells):
            if cell_block.type != "line":
                continue

            line_segments_block = cell_block.data
            physical_IDs_block = mesh.cell_data["gmsh:physical"][block_index]

            # get phyiscal wall ID

            for segment, physical_ID in zip(line_segments_block, physical_IDs_block):
                physical_name = OccupancyGrid.physical_ID_to_name(mesh, physical_ID, 1)
                pattern = re.compile(r"(wall|obstacle|solid|no_?slip)", re.I)

                if pattern.match(physical_name):
                    continue
                
                # calculate XY coordinates of the sample points along the line segment
                points_segment = mesh.points[segment, :2]
                start_segment, end_segment = points_segment
                len_segment = np.linalg.norm(end_segment - start_segment)
                n_points = int(np.ceil(len_segment / (resolution / 2.0)))
                t = (np.arange(n_points) + 0.5) / n_points
                sample_points = start_segment + t[:, None] * (end_segment - start_segment)

                # Find the corresponding grid cells for each sample point
                cols_open = np.floor((sample_points[:, 0] - x_min) / resolution).astype(np.int32)
                rows_open = np.floor((sample_points[:, 1] - y_min) / resolution).astype(np.int32)

                cols_open = np.clip(cols_open, 0, width - 1)
                rows_open = np.clip(rows_open, 0, height - 1)

                occupancy_grid[rows_open, cols_open] = CellType.OPEN_BOUNDARY
                    

        return cls(
            occupancy=occupancy_grid,
            resolution=resolution,
            origin=np.array([x_min, y_min]),
        )

    def cell_type_at(self, position: np.ndarray) -> CellType | np.ndarray:
        """Return the cell type at one position (2,) or many positions (n, 2)."""
        position = np.asarray(position, dtype=float)
        single_position = position.ndim == 1
        position = np.atleast_2d(position)

        if position.shape[1] != 2:
            raise ValueError("position must have shape (2,) or (n, 2)")

        cols = np.floor(
            (position[:, 0] - self.origin[0]) / self.resolution
        ).astype(int)
        rows = np.floor(
            (position[:, 1] - self.origin[1]) / self.resolution
        ).astype(int)

        height, width = self.occupancy.shape
        in_bounds = (
            (rows >= 0)
            & (rows < height)
            & (cols >= 0)
            & (cols < width)
        )

        states = np.full(
            len(position), CellType.OCCUPIED, dtype=np.uint8
        )
        states[in_bounds] = self.occupancy[
            rows[in_bounds], cols[in_bounds]
        ]

        return CellType(states[0]) if single_position else states

    def is_strictly_free(self, position: np.ndarray) -> bool:
        """Return whether a point and every cell touching it are free."""
        position = np.asarray(position, dtype=float)
        if position.shape != (2,):
            raise ValueError("position must have shape (2,)")

        epsilon = self.resolution * 1e-6
        offsets = epsilon * np.array(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]
        )
        states = self.cell_type_at(position + offsets)
        return bool(np.all(states == CellType.FREE))

    def plot(
        self,
        ax=None,
        title: str | None = None,
        output_path: str | Path | None = None,
        show: bool = False,
    ):
        if ax is None:
            _, ax = pyplot.subplots()

        cmap = colors.ListedColormap(["#303030", "#f2f2f2", "#00a6ff"])
        extent = [
            self.lower_bound[0],
            self.upper_bound[0],
            self.lower_bound[1],
            self.upper_bound[1],
        ]

        ax.imshow(
            self.occupancy,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if title is not None:
            ax.set_title(title)
        ax.legend(
            handles=[
                Patch(color=cmap(CellType.FREE), label="Free"),
                Patch(color=cmap(CellType.OCCUPIED), label="Occupied"),
                Patch(color=cmap(CellType.OPEN_BOUNDARY), label="Open Boundary"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            ncol=3,
        )

        if output_path is not None:
            ax.figure.savefig(output_path, bbox_inches="tight")
        if show:
            pyplot.show()
        return ax

    def plot_occupancy(self):
        return self.plot(title="Occupancy Grid", show=True)

    @staticmethod
    def points_in_triangle(
        x: np.ndarray, y: np.ndarray, triangle: np.ndarray
    ) -> np.ndarray:
        """Return a boolean mask for points inside one triangle."""
        (x1, y1), (x2, y2), (x3, y3) = triangle
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        xx, yy = np.meshgrid(x, y)

        if np.isclose(denominator, 0.0):
            return np.zeros_like(xx, dtype=bool)

        a = ((y2 - y3) * (xx - x3) + (x3 - x2) * (yy - y3)) / denominator
        b = ((y3 - y1) * (xx - x3) + (x1 - x3) * (yy - y3)) / denominator
        c = 1.0 - a - b
        return (a >= 0.0) & (b >= 0.0) & (c >= 0.0)

    @staticmethod
    def extract_triangles_msh(mesh: Mesh) -> np.ndarray:
        """Extract an (n, 3) array of triangle vertex indices."""
        triangles = [
            np.asarray(cell_block.data)
            for cell_block in mesh.cells
            if cell_block.type == "triangle"
        ]
        if not triangles:
            raise ValueError("No triangle cells found in mesh.")
        return np.vstack(triangles)

    @staticmethod
    def extract_lines_msh(mesh: Mesh) -> np.ndarray:
        """Extract an (n, 2) array of line vertex indices."""
        lines = [
            np.asarray(cell_block.data)
            for cell_block in mesh.cells
            if cell_block.type == "line"
        ]
        if not lines:
            raise ValueError("No line cells found in mesh.")
        return np.vstack(lines)

    @staticmethod
    def physical_ID_to_name(mesh, physical_id, dimension):
        """Return the name of a physical entity given its ID and dimension."""
        for name, (ID, dim) in mesh.field_data.items():
            if ID == physical_id and dim == dimension:
                return name
        return None

