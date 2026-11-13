"""Compare Random, CLLLA and QLLLA with static/adaptive loss weights.

Edit PINN_CONFIG and CONFIG below, then run python evaluation/benchmark_exploration.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import pickle
import re
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import cache
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source_estimation_pinn.environment import OccupancyGrid, WindField
from source_estimation_pinn.estimation import (
    GasSourceConfig,
    UncertaintyBasedGasSourceEstimator,
)
from source_estimation_pinn.measurements import GasSampleSet
from evaluation.experiment_logging import write_json as _json_write, save_provenance

@dataclass(frozen=True)
class BenchmarkConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    scenarios: tuple[str, ...] | None = None  # Default: all scenarios in data_dir.
    output_dir: Path | None = None  # Default: a new evaluation/results/<timestamp> directory.
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    n_initial_measurements: int = 5
    n_acquisitions: int = 50
    initial_training_steps: int = 1000
    update_training_steps: int = 200
    n_collocation_points: int = 1000
    gas_absolute_noise_std: float = 0.0  # Same concentration units as gas.csv.
    gas_relative_noise_std: float = 0.0  # E.g. 0.05 means 5% of the local true value.
    gas_peak_noise_alpha: float | None = None  # Constant sigma = alpha * scenario max(C).
    clip_negative_measurements: bool = False  # Clipping introduces bias near zero.
    source_field_metric: str = "normalized_l2"  # Alternatives: "normalized_l2", "point_w1".
    use_metadata_diffusivity: bool = True
    cpu_threads: int = 1
    parallel_runs: int = 6  # Independent scenario/seed pairs; 1 runs sequentially.
    pinn: GasSourceConfig = field(default_factory=GasSourceConfig)

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("seeds must be distinct, nonnegative and nonempty")
        for name in ("n_initial_measurements", "initial_training_steps",
                     "update_training_steps", "n_collocation_points", "cpu_threads", "parallel_runs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_acquisitions < 0:
            raise ValueError("n_acquisitions must be nonnegative")
        for std in (self.gas_absolute_noise_std, self.gas_relative_noise_std):
            if not np.isfinite(std) or std < 0:
                raise ValueError("measurement noise standard deviations must be finite and nonnegative")
        if self.gas_peak_noise_alpha is not None:
            if not np.isfinite(self.gas_peak_noise_alpha) or self.gas_peak_noise_alpha < 0:
                raise ValueError("gas_peak_noise_alpha must be finite and nonnegative")
            if self.gas_absolute_noise_std != 0 or self.gas_relative_noise_std != 0:
                raise ValueError("peak-scaled noise cannot be combined with absolute/relative noise")
        if self.source_field_metric not in FIELD_METRICS:
            raise ValueError(f"source_field_metric must be one of {tuple(FIELD_METRICS)}")


FIELD_METRICS = {
    "l2": ("source_l2", "Source-field L2 (shape and strength)"),
    "normalized_l2": ("source_normalized_l2", "Normalized source-field L2 (1/m)"),
    "point_w1": ("point_source_w1_m", "Wasserstein-1 to point source (m)"),
}


# All six variants use these settings; only dynamic_weights differs.
PINN_CONFIG = GasSourceConfig(
    concentration_hidden_layers=3, concentration_hidden_dim=32,
    source_hidden_layers=3, source_hidden_dim=16,
    concentration_initial_value=1e-3, source_initial_value=1e-3,
    diffusion_constant=0.02525,
    lambda_gas_data=1.0, lambda_ad=1.0,
    lambda_gas_wall=1.0, lambda_inflow=1.0,
    lambda_sparse=1e-3, lambda_nonfree=1e-2,
    last_layer_prior_precision=1e-6, learning_rate=2e-3,
    device="cpu",
)
CONFIG = BenchmarkConfig(
    data_dir=PROJECT_ROOT / "data",
    scenarios=None,  # All; or e.g. ("example_labyrinth/scenario_source_1.5_2_seed_7",).
    output_dir=None,  # New timestamped directory under evaluation/results/.
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    n_initial_measurements=5,
    n_acquisitions=50,
    initial_training_steps=1000,
    update_training_steps=200,
    n_collocation_points=1000,
    gas_absolute_noise_std=0.0,
    gas_relative_noise_std=0.0,
    gas_peak_noise_alpha=0.05,  # Constant sigma = 5% of each scenario's peak concentration.
    source_field_metric="point_w1",
    use_metadata_diffusivity=True,  # False for recordings without filament metadata.
    cpu_threads=1,
    parallel_runs=4,
    pinn=PINN_CONFIG,
)


@dataclass(frozen=True)
class SamplingContext:
    localizer: UncertaintyBasedGasSourceEstimator
    candidate_points: np.ndarray
    available_mask: np.ndarray
    rng: np.random.Generator

    @property
    def estimator(self):
        return self.localizer.estimator

    @property
    def measurements(self):
        return self.localizer.samples


class SourceUncertaintySampling:
    """Adapter to the project's current method, without duplicating its ranking."""

    def select(self, context: SamplingContext) -> int:
        position = context.localizer.suggest_next_measurement()
        distances = np.linalg.norm(context.candidate_points - position, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > context.localizer.occupancy.resolution * 1e-6:
            raise ValueError("method suggested a position outside the candidate grid")
        return index


class RandomSampling:
    def select(self, context: SamplingContext) -> int:
        return int(context.rng.choice(np.flatnonzero(context.available_mask)))


@dataclass(frozen=True)
class StrategySpec:
    factory: Callable
    label: str | None = None  # Optional display label, never used to choose behavior.
    dynamic_weights: bool | None = None  # None: use config.pinn.dynamic_weights.
    pinn_overrides: dict = field(default_factory=dict)
    data_variance_from_noise: bool = False


class ConcentrationUncertaintySampling:
    """CLLLA: maximize linearized concentration variance g_c Sigma_cc g_c^T."""

    def select(self, context: SamplingContext) -> int:
        laplace = context.estimator.laplace
        if laplace.covariance is None:
            laplace.compute_covariance()
        covariance_cc, _, _ = laplace.split_covariance(laplace.covariance)
        with torch.no_grad():
            gradients = laplace.concentration_gradient_wrt_parameters(
                context.estimator._tensor(context.candidate_points)
            ).to(covariance_cc.dtype)
            variance = torch.sum((gradients @ covariance_cc) * gradients, dim=1)
        scores = variance.detach().cpu().numpy()
        if not context.available_mask.any():
            raise RuntimeError("all candidate points have been measured")
        if (not np.isfinite(scores[context.available_mask]).all()
                or np.any(scores[context.available_mask] < -1e-10)):
            raise RuntimeError("CLLLA produced invalid concentration variances")
        return int(np.argmax(np.where(context.available_mask, np.maximum(scores, 0), -np.inf)))


def study_strategies():
    """Labels identify results; behavior comes from the actual strategy classes."""
    methods = ((RandomSampling, "Random"),
               (ConcentrationUncertaintySampling, "CLLLA"),
               (SourceUncertaintySampling, "QLLLA"))
    return tuple(
        StrategySpec(factory, f"{label}_{mode}", dynamic_weights=dynamic)
        for mode, dynamic in (("static", False), ("adaptive", True))
        for factory, label in methods
    )


@dataclass
class Scenario:
    directory: Path
    metadata: dict
    occupancy: OccupancyGrid
    wind: WindField
    concentrations: np.ndarray
    source_position: np.ndarray
    source_values: np.ndarray

    @property
    def points(self):
        return self.occupancy.free_points

    @property
    def name(self):
        return f"{self.directory.parent.name}/{self.directory.name}"


@cache
def _load_occupancy(path: Path, resolution: float | None):
    if path.suffix == ".msh":
        return OccupancyGrid.from_msh_file(path, resolution=resolution)
    return OccupancyGrid.from_occupancy_file(path)


@cache
def _load_wind(path: Path):
    return WindField.from_csv(path)


def ground_truth_source(points, position, sigma, rate, cell_area):
    """Discrete density of the simulator's Gaussian release, truncated to free space."""
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(rate) or rate <= 0:
        raise ValueError("source sigma and rate must be finite and positive")
    log_values = -np.sum((points - position) ** 2, axis=1) / (2 * sigma**2)
    values = np.exp(log_values - log_values.max())
    return rate * values / (cell_area * values.sum())


def load_scenario(directory: Path) -> Scenario:
    directory = directory.resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    if "occupancy" in metadata:
        occupancy = _load_occupancy((directory / metadata["occupancy"]).resolve(), None)
    else:
        occupancy = _load_occupancy(
            (directory / metadata["mesh"]).resolve(), float(metadata["resolution"])
        )
    wind = _load_wind((directory / metadata["wind"]).resolve())
    reference = GasSampleSet.from_csv(directory / metadata["files"]["gas"])
    points = occupancy.free_points
    if not len(points) or not np.isfinite(reference.positions).all():
        raise ValueError(f"{directory}: invalid gas coordinates or empty free grid")
    if not np.isfinite(reference.concentrations).all() or np.any(reference.concentrations < 0):
        raise ValueError(f"{directory}: invalid reference concentrations")
    distances, indices = cKDTree(reference.positions).query(points)
    if np.any(distances > occupancy.resolution * 1e-5) or len(np.unique(indices)) != len(points):
        raise ValueError(f"{directory}: gas.csv must cover every free grid point exactly")
    source = metadata["source"]
    position = np.asarray(source["position"], dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all() or not occupancy.is_strictly_free(position):
        raise ValueError(f"{directory}: invalid source position")
    values = ground_truth_source(
        points, position, float(source["sigma"]), float(source["rate"]), occupancy.resolution**2
    )
    return Scenario(directory, metadata, occupancy, wind,
                    reference.concentrations[indices], position, values)


def noisy_measurements(concentrations, config, seed):
    """Shared per-location noise for a paired trial; no clipping by default."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    sigma = (config.gas_peak_noise_alpha * float(np.max(concentrations))
             if config.gas_peak_noise_alpha is not None else
             np.hypot(config.gas_absolute_noise_std,
                      config.gas_relative_noise_std * concentrations))
    observed = concentrations + rng.standard_normal(len(concentrations)) * sigma
    return np.maximum(observed, 0) if config.clip_negative_measurements else observed


def source_metrics(predicted, points, true_values, true_position, cell_area):
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(predicted) != len(points) or not np.isfinite(predicted).all() or np.any(predicted < 0):
        raise ValueError("predicted source field must be finite and nonnegative")
    mass = float(cell_area * predicted.sum())
    if mass <= 0:
        raise ValueError("predicted source field has zero mass")
    estimate = points[int(np.argmax(predicted))]
    normalized = predicted / mass
    normalized_truth = true_values / (cell_area * true_values.sum())
    return {
        "source_error_m": float(np.linalg.norm(estimate - true_position)),
        "source_l2": float(np.sqrt(cell_area * np.sum((predicted - true_values)**2))),
        "source_normalized_l2": float(np.sqrt(
            cell_area * np.sum((normalized - normalized_truth)**2))),
        # This evaluates a POINT target, not EMD against the finite Gaussian.
        "point_source_w1_m": float(cell_area * np.sum(
            normalized * np.linalg.norm(points - true_position, axis=1))),
        "estimated_source_mass": mass,
        "estimated_x": float(estimate[0]),
        "estimated_y": float(estimate[1]),
    }


def _scenario_pinn_config(config, metadata):
    if not config.use_metadata_diffusivity:
        return config.pinn
    diffusivity = (float(metadata["diffusivity"])
                   + 0.5 * float(metadata["diffusion_speed_std"])**2 * float(metadata["dt"]))
    return replace(config.pinn, diffusion_constant=diffusivity)


def _strategy_pinn_config(config, case, spec):
    overrides = dict(spec.pinn_overrides)
    if spec.dynamic_weights is not None:
        overrides["dynamic_weights"] = spec.dynamic_weights
    if spec.data_variance_from_noise:
        if config.gas_peak_noise_alpha is None:
            raise ValueError("noise-based variance floor requires peak-scaled noise")
        overrides["gas_data_variance_floor"] = (
            config.gas_peak_noise_alpha * float(np.max(case.concentrations))) ** 2
    return replace(_scenario_pinn_config(config, case.metadata), **overrides)


def _strategy_label(strategy, spec):
    return spec.label or getattr(strategy, "__name__", type(strategy).__name__)


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def plot_results(output_dir: Path, field_metric: str = "l2"):
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / "metrics.csv")
    field_column, field_label = FIELD_METRICS[field_metric]
    columns = ("source_error_m", field_column)
    labels = ("Source-maximum position error (m)", field_label)

    # Each case has equal weight. Bands are SD of the per-seed case averages,
    # not confidence intervals or a claim of generalization to new environments.
    per_seed = frame.groupby(
        ["strategy", "seed", "additional_measurements"], sort=False
    )[list(columns)].mean().reset_index()
    summary = per_seed.groupby(
        ["strategy", "additional_measurements"], sort=False
    )[list(columns)].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, column, label in zip(axes, columns, labels):
        upper = 0.0
        for name, group in per_seed.groupby("strategy", sort=False):
            curve = group.groupby("additional_measurements")[column].agg(["mean", "std"]).fillna(0)
            x = curve.index.to_numpy()
            y, sd = curve["mean"].to_numpy(), curve["std"].to_numpy()
            upper = max(upper, float(np.max(y + sd)))
            line, = axis.plot(x, y, marker="o", label=name)
            axis.fill_between(x, np.maximum(0, y - sd), y + sd, alpha=0.15,
                              color=line.get_color())
        axis.set(xlabel="Additional measurements (0 = initial model per weighting mode)",
                 ylabel=label, ylim=(0, upper * 1.1 if upper > 0 else 1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"{frame['scenario'].nunique()} scenarios; "
                 f"{frame['seed'].nunique()} seeds; mean ± seed SD; lower is better")
    fig.savefig(output_dir / "summary.png", dpi=180)
    plt.close(fig)

    scenarios = frame["scenario"].unique()
    for column, label, filename in zip(
        columns, labels, ("position_by_scenario.png", "source_field_by_scenario.png")
    ):
        ncols = min(3, len(scenarios))
        nrows = (len(scenarios) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.4*nrows),
                                 squeeze=False, constrained_layout=True)
        for axis, name in zip(axes.flat, scenarios):
            case = frame[frame.scenario == name]
            upper = 0.0
            for strategy, group in case.groupby("strategy", sort=False):
                curve = group.groupby("additional_measurements")[column].agg(["mean", "std"]).fillna(0)
                x = curve.index.to_numpy()
                y, sd = curve["mean"].to_numpy(), curve["std"].to_numpy()
                upper = max(upper, float(np.max(y + sd)))
                line, = axis.plot(x, y, label=strategy)
                axis.fill_between(x, np.maximum(0, y - sd), y + sd,
                                  color=line.get_color(), alpha=0.15)
            first = case.iloc[0]
            axis.set(title=f"{first['environment']} | source ({first['true_x']:g}, {first['true_y']:g})",
                     xlabel="Additional measurements", ylabel=label,
                     ylim=(0, upper * 1.1 if upper > 0 else 1))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(scenarios):]:
            axis.set_visible(False)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def _write_trial_metrics(rows, trial_dir):
    temporary = trial_dir / "metrics.tmp.csv"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(trial_dir / "metrics.csv")


def _run_trial(config, scenario_dir, seed, specs, output):
    """Train paired strategies for one scenario/seed in an isolated process."""
    trial_dir = output / "runs" / scenario_dir.parent.name / scenario_dir.name / f"seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        case = load_scenario(scenario_dir)
        points = case.points
        if config.n_initial_measurements + config.n_acquisitions > len(points):
            raise ValueError(f"{case.name}: measurement budget exceeds available points")
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        initial_indices = initial_rng.choice(len(points), config.n_initial_measurements, replace=False)
        observed = noisy_measurements(case.concentrations, config, seed)
        initial_samples = GasSampleSet(points[initial_indices].copy(), observed[initial_indices].copy())
        seen_labels = set()
        effective_configs = [_strategy_pinn_config(config, case, spec) for spec in specs]
        for effective_pinn in dict.fromkeys(effective_configs):
            dynamic = effective_pinn.dynamic_weights
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]))
                initial = UncertaintyBasedGasSourceEstimator(
                    case.occupancy, case.wind, effective_pinn,
                    initial_training_steps=config.initial_training_steps,
                    update_training_steps=config.update_training_steps,
                    n_collocation_points=config.n_collocation_points,
                )
                print(f"{case.name}, seed {seed}: initial training", flush=True)
                start = perf_counter()
                initial_parameter_sha256 = hashlib.sha256(b"".join(
                    value.detach().cpu().numpy().tobytes()
                    for value in initial.estimator.model.state_dict().values()
                )).hexdigest()
                initial.fit(initial_samples)
                initial_seconds = perf_counter() - start
                train_state = torch.get_rng_state().clone()
                cuda_state = torch.cuda.get_rng_state_all() if devices else None
                for strategy_index, spec in enumerate(specs):
                    if effective_configs[strategy_index] != effective_pinn:
                        continue
                    torch.set_rng_state(train_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state_all(cuda_state)
                    with torch.random.fork_rng(devices=devices):
                        strategy = spec.factory()
                    name = _strategy_label(strategy, spec)
                    if name in seen_labels:
                        raise ValueError("duplicate strategy labels; provide distinct StrategySpec labels")
                    seen_labels.add(name)
                    localizer = copy.deepcopy(initial, {
                        id(case.occupancy): case.occupancy, id(case.wind): case.wind
                    })
                    available = np.ones(len(points), dtype=bool)
                    available[initial_indices] = False
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
                    measured_indices = list(initial_indices)
                    fields = []
                    histories = [localizer.estimator.history]
                    selection_seconds, training_seconds = 0.0, initial_seconds
                    try:
                        for round_number in range(config.n_acquisitions + 1):
                            if round_number:
                                context = SamplingContext(localizer, points, available.copy(), rng)
                                start = perf_counter()
                                with torch.random.fork_rng(devices=devices):
                                    select = getattr(strategy, "select", strategy)
                                    index = select(context)
                                selection_seconds = perf_counter() - start
                                if (not isinstance(index, (int, np.integer))
                                        or not 0 <= index < len(points) or not available[index]):
                                    raise ValueError("strategy must return an available candidate index")
                                index = int(index)
                                start = perf_counter()
                                localizer.add_measurement(points[index], float(observed[index]))
                                histories.append(localizer.estimator.history)
                                training_seconds = perf_counter() - start
                                available[index] = False
                                measured_indices.append(index)
                            predicted = np.asarray(localizer.estimator.predict_source(points)).reshape(-1)
                            fields.append(predicted.astype(np.float32))
                            metrics = source_metrics(predicted, points, case.source_values,
                                                     case.source_position, case.occupancy.resolution**2)
                            rows.append({
                                "scenario": case.name, "environment": case.directory.parent.name,
                                "seed": seed, "strategy": name, "additional_measurements": round_number,
                                "dynamic_weights": dynamic,
                                "total_measurements": config.n_initial_measurements + round_number,
                                "true_x": case.source_position[0], "true_y": case.source_position[1],
                                "selection_seconds": selection_seconds, "training_seconds": training_seconds,
                                **metrics,
                            })
                            print(f"  {case.name}, seed {seed}, {name}: +{round_number} | maximum error "
                                  f"{metrics['source_error_m']:.3f} m | L2 {metrics['source_l2']:.4g}",
                                  flush=True)
                        directory = output / "runs" / case.directory.parent.name / case.directory.name / f"seed_{seed}"
                        directory.mkdir(parents=True, exist_ok=True)
                        loss_names = sorted(histories[0].losses)
                        weight_names = sorted(histories[0].weights)
                        np.savez_compressed(
                            directory / f"{strategy_index}_{_slug(name)}.npz", candidate_points=points,
                            true_source=case.source_values, true_position=case.source_position,
                            predicted_source=np.asarray(fields), measurement_indices=measured_indices,
                            observed_concentrations=observed[measured_indices],
                            true_concentrations=case.concentrations[measured_indices],
                            dynamic_weights=dynamic,
                            initial_parameter_sha256=initial_parameter_sha256,
                            training_loss_names=np.asarray(loss_names),
                            training_weight_names=np.asarray(weight_names),
                            training_losses=np.concatenate([
                                np.column_stack([h.losses[k] for k in loss_names]) for h in histories
                            ]),
                            training_weights=np.concatenate([
                                np.column_stack([h.weights[k] for k in weight_names]) for h in histories
                            ]),
                            training_round=np.concatenate([
                                np.full(len(h.total), number, dtype=int)
                                for number, h in enumerate(histories)
                            ]),
                        )
                        _write_trial_metrics(rows, trial_dir)
                    except Exception as error:
                        _write_trial_metrics(rows, trial_dir)
                        _json_write(trial_dir / "failure.json", {
                            "scenario": case.name, "seed": seed, "strategy": name,
                            "error": repr(error),
                        })
                        raise RuntimeError(f"benchmark failed; partial results saved in {output}") from error
        return rows
    except Exception as error:
        if rows:
            _write_trial_metrics(rows, trial_dir)
        if not (trial_dir / "failure.json").exists():
            _json_write(trial_dir / "failure.json", {
                "scenario": f"{scenario_dir.parent.name}/{scenario_dir.name}",
                "seed": seed, "error": repr(error),
            })
        raise
    finally:
        torch.set_num_threads(previous_threads)


def _collect_metrics(output):
    """Only the parent writes the aggregate CSV, including partial trials."""
    paths = sorted((output / "runs").glob("*/*/seed_*/metrics.csv"))
    if paths:
        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        frame.sort_values(["scenario", "seed"], kind="stable", inplace=True)
        frame.to_csv(output / "metrics.csv", index=False)


def run_benchmark(
    config: BenchmarkConfig,
    strategy_factories: Sequence[Callable | StrategySpec] = (SourceUncertaintySampling, RandomSampling),
    scenario_dirs: Sequence[Path] | None = None,
) -> Path:
    """Run any strategies implementing select(context), or callable strategies.

    Parallel factories must be defined at module level.
    For lambdas/local factories use parallel_runs=1.
    Use StrategySpec(factory, label, dynamic_weights=...) to compare weighting modes.
    Each mode shares one trained initial model across its strategies; all modes
    use the same initial network seed, initial measurements and training RNG.
    A new strategy instance is created for every scenario and seed.
    """
    specs = [s if isinstance(s, StrategySpec) else StrategySpec(s) for s in strategy_factories]
    if not specs:
        raise ValueError("at least one strategy factory is required")
    if scenario_dirs is not None:
        directories = list(scenario_dirs)
    elif config.scenarios is not None:
        directories = [config.data_dir / name for name in config.scenarios]
    else:
        directories = [p.parent for p in sorted(config.data_dir.glob("*/scenario_*/metadata.json"))]
    if not directories:
        raise ValueError("no scenario metadata files found")
    directories = [Path(p).resolve() for p in directories]
    if len({(p.parent.name, p.name) for p in directories}) != len(directories):
        raise ValueError("scenario directories must be distinct")
    if config.parallel_runs > 1:
        try:
            pickle.dumps(specs)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            raise ValueError("Parallel strategy factories must be defined at module level; "
                             "use parallel_runs=1 for lambdas/local functions") from error
    scenarios = []
    for directory in directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        case = load_scenario(directory)
        concentrations = case.concentrations if config.gas_peak_noise_alpha is not None else None
        scenarios.append({
            **({"peak_concentration": float(np.max(concentrations)),
                "effective_gas_absolute_noise_std": config.gas_peak_noise_alpha * float(np.max(concentrations))}
               if concentrations is not None else {}),
            "directory": str(directory), "metadata": metadata,
            "effective_pinn": asdict(_scenario_pinn_config(config, metadata)),
            "effective_pinn_by_strategy": [asdict(_strategy_pinn_config(config, case, spec))
                                           for spec in specs],
            "gas_sha256": hashlib.sha256(
                (directory / metadata["files"]["gas"]).read_bytes()).hexdigest(),
        })

    output = config.output_dir or PROJECT_ROOT / "evaluation/results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing benchmark: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_files = {}
    for directory in directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        paths = {"metadata": directory / "metadata.json",
                 "gas": directory / metadata["files"]["gas"],
                 "wind": directory / metadata["wind"]}
        if "occupancy" in metadata:
            paths["occupancy"] = (directory / metadata["occupancy"]).resolve()
            occupancy = yaml.safe_load(paths["occupancy"].read_text())
            paths["occupancy_image"] = paths["occupancy"].parent / occupancy["image"]
        else:
            paths["mesh"] = directory / metadata["mesh"]
        input_files[str(directory)] = paths
    save_provenance(
        output, PROJECT_ROOT, input_files,
        [*sorted((PROJECT_ROOT / "source_estimation_pinn").rglob("*.py")),
         Path(__file__),
         PROJECT_ROOT / "evaluation/experiment_logging.py"],
    )
    _json_write(output / "config.json", {
        **asdict(config),
        "output_dir": str(output),
        "strategies": [{"factory": getattr(s.factory, "__qualname__", repr(s.factory)),
                        "module": getattr(s.factory, "__module__", None), "label": s.label,
                        "dynamic_weights": config.pinn.dynamic_weights
                        if s.dynamic_weights is None else s.dynamic_weights,
                        "pinn_overrides": s.pinn_overrides,
                        "data_variance_from_noise": s.data_variance_from_noise}
                       for s in specs],
        "scenarios": scenarios,
        "noise_model": ("Gaussian: sigma=gas_peak_noise_alpha*max(C_true) over scenario free grid; shared by location"
                        if config.gas_peak_noise_alpha is not None else
                        "Gaussian: sigma=sqrt(absolute_std^2+(relative_std*C_true)^2); shared by location"),
        "reference_source": "Gaussian density on free grid, normalized to metadata source.rate",
    })
    trials = [(directory, seed) for directory in directories for seed in config.seeds]
    directory, seed = trials[0]
    try:
        if config.parallel_runs == 1:
            for directory, seed in trials:
                _run_trial(config, directory, seed, specs, output)
                _collect_metrics(output)
        else:
            # Spawn avoids inheriting PyTorch/CUDA state from the parent.
            with ProcessPoolExecutor(max_workers=min(config.parallel_runs, len(trials)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_run_trial, config, directory, seed, specs, output):
                           (directory, seed) for directory, seed in trials}
                try:
                    for future in as_completed(futures):
                        directory, seed = futures[future]
                        future.result()
                        _collect_metrics(output)
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
    except Exception as error:
        _collect_metrics(output)
        _json_write(output / "failure.json", {
            "scenario": f"{directory.parent.name}/{directory.name}",
            "seed": seed, "error": repr(error),
        })
        raise RuntimeError(f"benchmark failed; partial results saved in {output}") from error
    plot_results(output, config.source_field_metric)
    print(f"Results: {output}", flush=True)
    return output


def run_exploration_benchmark(config: BenchmarkConfig = CONFIG) -> Path:
    output = config.output_dir or PROJECT_ROOT / "evaluation/results" / datetime.now().strftime(
        "exploration_%Y%m%d_%H%M%S_%f")
    return run_benchmark(replace(config, output_dir=output),
                         strategy_factories=study_strategies())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--weighting-diagnostic", action="store_true",
                       help="Paired Random comparison: static, data/PDE adaptive, data/PDE with floors")
    modes.add_argument("--data-weighting-diagnostic", action="store_true",
                       help="Paired Random comparison: static versus adaptive data weight with noise floor")
    args = parser.parse_args()
    if args.weighting_diagnostic or args.data_weighting_diagnostic:
        prefix = "data_weighting_diagnostic" if args.data_weighting_diagnostic else "weighting_diagnostic"
        diagnostic = replace(CONFIG,
            scenarios=("example_apartment/scenario_source_1.5_1_seed_7",),
            seeds=(0, 1, 2, 3), parallel_runs=2,
            output_dir=PROJECT_ROOT / "evaluation/results" / datetime.now().strftime(
                f"{prefix}_%Y%m%d_%H%M%S_%f"))
        terms = {"learned_weight_terms": ("gas_data", "ad")}
        strategies = (
            StrategySpec(RandomSampling, "Random_static", dynamic_weights=False),
            StrategySpec(RandomSampling, "Random_data_pde", dynamic_weights=True,
                         pinn_overrides=terms),
            StrategySpec(RandomSampling, "Random_data_pde_floored", dynamic_weights=True,
                         pinn_overrides={**terms, "ad_variance_floor": 1e-3},
                         data_variance_from_noise=True),
        )
        if args.data_weighting_diagnostic:
            strategies = (
                strategies[0],
                StrategySpec(RandomSampling, "Random_data_only_floored", dynamic_weights=True,
                             pinn_overrides={"learned_weight_terms": ("gas_data",)},
                             data_variance_from_noise=True),
            )
        run_benchmark(diagnostic, strategy_factories=strategies)
    else:
        run_exploration_benchmark()
