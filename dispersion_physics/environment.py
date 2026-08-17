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

    def plot_occupancy(self):
        cmap = colors.ListedColormap(["#303030", "#f2f2f2", "#00a6ff"])
        height, width = self.occupancy.shape
        x0, y0 = self.origin
        extent = [
            x0, x0 + width * self.resolution,
            y0, y0 + height * self.resolution,
        ]
        _, ax = pyplot.subplots()
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
        ax.set_title("Occupancy Grid")


        # Put a legend below current axis
        ax.legend(handles=[
            Patch(color=cmap(CellType.FREE), label="Free"),
            Patch(color=cmap(CellType.OCCUPIED), label="Occupied"),
            Patch(color=cmap(CellType.OPEN_BOUNDARY), label="Open Boundary"),
        ],
        loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, ncol=3)
        pyplot.show()

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

