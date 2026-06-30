from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


@dataclass
class OccupancyMap:
    occupancy: np.ndarray
    free_mask: np.ndarray
    solid_mask: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    resolution: float
    origin: tuple[float, float]

    @property
    def points(self) -> np.ndarray:
        return np.column_stack([self.xx.ravel(), self.yy.ravel()])

    @property
    def free_points(self) -> np.ndarray:
        return np.column_stack([self.xx[self.free_mask], self.yy[self.free_mask]])

    @property
    def solid_points(self) -> np.ndarray:
        return np.column_stack([self.xx[self.solid_mask], self.yy[self.solid_mask]])

    @property
    def wall_mask(self) -> np.ndarray:
        solid_neighbor = np.zeros_like(self.free_mask, dtype=bool)
        solid_neighbor[1:, :] |= self.solid_mask[:-1, :]
        solid_neighbor[:-1, :] |= self.solid_mask[1:, :]
        solid_neighbor[:, 1:] |= self.solid_mask[:, :-1]
        solid_neighbor[:, :-1] |= self.solid_mask[:, 1:]
        return self.free_mask & solid_neighbor

    @property
    def wall_points(self) -> np.ndarray:
        return np.column_stack([self.xx[self.wall_mask], self.yy[self.wall_mask]])

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "OccupancyMap":
        yaml_path = Path(yaml_path).resolve()
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        image_path = yaml_path.parent / data["image"]
        image = plt.imread(image_path)
        if image.ndim == 3:
            image = image[..., 0]
        image = image.astype(float)
        if image.max() <= 1.0:
            image *= 255.0
        image = np.flipud(image)

        resolution = float(data["resolution"])
        origin = data.get("origin", [0.0, 0.0, 0.0])
        origin_xy = (float(origin[0]), float(origin[1]))
        free_thresh = float(data.get("free_thresh", 0.196))
        occupied_thresh = float(data.get("occupied_thresh", 0.65))
        negate = int(data.get("negate", 0))

        if negate:
            occupancy = image / 255.0
        else:
            occupancy = (255.0 - image) / 255.0

        free_mask = occupancy < free_thresh
        solid_mask = occupancy > occupied_thresh
        unknown_mask = ~(free_mask | solid_mask)
        unknown_count = int(np.count_nonzero(unknown_mask))
        if unknown_count:
            raise ValueError(
                f"Found {unknown_count} unknown occupancy cells between "
                f"free_thresh={free_thresh} and occupied_thresh={occupied_thresh}."
            )

        n_rows, n_cols = image.shape
        x = origin_xy[0] + (np.arange(n_cols) + 0.5) * resolution
        y = origin_xy[1] + (np.arange(n_rows) + 0.5) * resolution
        xx, yy = np.meshgrid(x, y)

        return cls(
            occupancy=occupancy,
            free_mask=free_mask,
            solid_mask=solid_mask,
            xx=xx,
            yy=yy,
            resolution=resolution,
            origin=origin_xy,
        )

    def plot(
        self,
        ax: plt.Axes | None = None,
        title: str | None = None,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        values = np.zeros_like(self.occupancy, dtype=float)
        values[self.solid_mask] = 0.4

        half_cell = 0.5 * self.resolution
        extent = (
            self.xx.min() - half_cell,
            self.xx.max() + half_cell,
            self.yy.min() - half_cell,
            self.yy.max() + half_cell,
        )

        ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        if title is not None:
            ax.set_title(title)

        if output_path is not None:
            ax.figure.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()

        return ax


def draw_random_samples_csv(
    csv_path: str | Path,
    n_samples: int,
    random_seed: int = 0,
    add_noise_std: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ["Points:0", "Points:1", "U:0", "U:1"]

    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Required columns cannot be found in {csv_path}.")

    rng = np.random.default_rng(random_seed)
    sample_ids = rng.integers(0, len(df), n_samples)
    df_rows = df.iloc[sample_ids].reset_index(drop=True)
    df_rows = df_rows.drop(columns=["Points:2", "U:2"], errors="ignore")
    if add_noise_std:
        df_rows[["U:0", "U:1"]] += rng.normal(0.0, add_noise_std, (n_samples, 2))
    return df_rows
