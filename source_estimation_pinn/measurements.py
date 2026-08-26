import numpy as np
import pandas as pd
import random
from pathlib import Path
from dataclasses import dataclass

@dataclass
class GasSampleSet:
    """Container for 2D gas concentration samples."""
    positions: np.ndarray       # (n, 2)
    concentrations: np.ndarray  # (n,)
    timestamps: np.ndarray | None = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.concentrations = np.asarray(
            self.concentrations,
            dtype=float,
        )

        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n, 2)")

        if self.concentrations.shape != (len(self.positions),):
            raise ValueError("concentrations must have shape (n,)")

    def append(
        self,
        positions: np.ndarray,
        concentrations: float | np.ndarray,
    ) -> None:
        """Append one or multiple gas concentration samples."""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        concentrations = np.atleast_1d(
            np.asarray(concentrations, dtype=float)
        )

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (2,) or (n, 2)")
        if concentrations.shape != (len(positions),):
            raise ValueError(
                "concentrations must be a scalar or have shape (n,)"
            )

        self.positions = np.vstack([self.positions, positions])
        self.concentrations = np.concatenate(
            [self.concentrations, concentrations]
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        n: int | None = None,
        random_seed: int | None = None,
    ) -> "GasSampleSet":
        """Draw `n` random samples from a CSV file containing 2D gas concentration data."""
        df = pd.read_csv(csv_path)
        required_columns = {"Points:0", "Points:1", "C"}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV file must contain columns 'Points:0', 'Points:1', and 'C'")

        positions = df[["Points:0", "Points:1"]].to_numpy(float)
        concentrations = df["C"].to_numpy(float)

        if n is not None:
            if not 0 <= n <= len(positions):
                raise ValueError(
                    f"Requested {n} samples, "
                    f"but only {len(positions)} are available."
                )

            rng = np.random.default_rng(random_seed)
            indices = rng.choice(len(positions), size=n, replace=False)

            positions = positions[indices]
            concentrations = concentrations[indices]

        return cls(positions=positions, concentrations=concentrations)


@dataclass
class WindSampleSet:
    """Container for 2D wind field samples."""
    positions: np.ndarray       # (n, 2)
    wind_vectors: np.ndarray    # (n, 2)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.wind_vectors = np.asarray(
            self.wind_vectors,
            dtype=float,
        )

        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n, 2)")

        if self.wind_vectors.shape != (len(self.positions), 2):
            raise ValueError("wind_vectors must have shape (n, 2)")

    def append(
        self,
        positions: np.ndarray,
        wind_vectors: np.ndarray,
    ) -> None:
        """Append one or multiple wind vector samples."""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        wind_vectors = np.atleast_2d(
            np.asarray(wind_vectors, dtype=float)
        )

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (2,) or (n, 2)")
        if wind_vectors.shape != (len(positions), 2):
            raise ValueError(
                "wind_vectors must have shape (2,) or (n, 2)"
            )

        self.positions = np.vstack([self.positions, positions])
        self.wind_vectors = np.vstack([self.wind_vectors, wind_vectors])

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        n: int | None = None,
        random_seed: int | None = None,
        noise_std: float | None = None,
    ) -> "WindSampleSet":
        """Load wind samples, optionally subsample them and add Gaussian noise."""
        df = pd.read_csv(csv_path)

        required_columns = {"Points:0", "Points:1", "U:0", "U:1"}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV file must contain columns 'Points:0', 'Points:1', 'U:0', and 'U:1'")

        positions = df[["Points:0", "Points:1"]].to_numpy(float)
        wind_vectors = df[["U:0", "U:1"]].to_numpy(float)

        if noise_std is not None and noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")

        rng = np.random.default_rng(random_seed)
        if n is not None:
            if not 0 <= n <= len(positions):
                raise ValueError(
                    f"Requested {n} samples, "
                    f"but only {len(positions)} are available."
                )

            indices = rng.choice(len(positions), size=n, replace=False)

            positions = positions[indices]
            wind_vectors = wind_vectors[indices]

        if noise_std is not None:
            wind_vectors += rng.normal(0.0, noise_std, wind_vectors.shape)

        return cls(positions=positions, wind_vectors=wind_vectors)



if __name__ == "__main__":
    path = "/app/data/example_labyrinth/gas_gt.csv"
    random.seed(42)
    n = 10
    sample_set = GasSampleSet.from_csv(Path(path), n, random_seed=42)
    print(f"Found {len(sample_set.positions)}/{n} samples in {path}.")

    path_wind = "/app/data/example_labyrinth/wind_gt.csv"
    sample_set_wind = WindSampleSet.from_csv(Path(path_wind), n, random_seed=42)
    print(f"Found {len(sample_set_wind.positions)}/{n} samples in {path_wind}.")