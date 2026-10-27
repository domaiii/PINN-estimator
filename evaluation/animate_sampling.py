"""Run one QLLLA example and animate it: python evaluation/animate_sampling.py.

Edit the settings below. No archived benchmark or exact replay is required.
Motion is an obstacle-aware point-robot visualization, starting at the last
initial measurement. Measurement selection does not consider travel distance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demos.exploration_helpers import AnyAngleGridPlanner
from evaluation.benchmark_exploration import load_scenario
from source_estimation_pinn.estimation import GasSourceConfig, UncertaintyBasedGasSourceEstimator
from source_estimation_pinn.measurements import GasSampleSet

# All simulation settings: scenario data, random start measurements and training.
SCENARIO_DIR = PROJECT_ROOT / "data/example_apartment/scenario_source_1.5_1_seed_7"
SEED = 1
N_INITIAL_MEASUREMENTS = 5
N_ACQUISITIONS =20  # Additional measurements; independent of update training steps.
INITIAL_TRAINING_STEPS = 1000
UPDATE_TRAINING_STEPS = 100
N_COLLOCATION_POINTS = 1000
CPU_THREADS = 1
GAS_ABSOLUTE_NOISE_STD = 0.0
GAS_RELATIVE_NOISE_STD = 0.0
PINN_CONFIG = GasSourceConfig(
    concentration_hidden_layers=3,
    concentration_hidden_dim=32,
    source_hidden_layers=3,
    source_hidden_dim=16,
    concentration_initial_value=1e-3,
    source_initial_value=1e-3,
    diffusion_constant=0.02525,  # Explicit: adjust when using other gas/wind data.
    dynamic_weights=False,
    lambda_gas_data=1.0,
    lambda_ad=1.0,
    lambda_gas_wall=1.0,
    lambda_inflow=1.0,
    lambda_sparse=1e-3,
    lambda_nonfree=1e-2,
    last_layer_prior_precision=1e-6,
    learning_rate=2e-3,
    device="cpu",
)
OUTPUT_ROOT = PROJECT_ROOT / "evaluation/results/animations"

# Video settings (playback time, not physical time).
FPS = 15
SECONDS_PER_MOVE = 1.8  # Playback duration, independent of physical travel time.
SECONDS_AT_MEASUREMENT = 0.6
DPI = 110


def simulate(case, output):
    """Train, select the actual score maximum, measure and retrain."""
    points = case.points
    if N_INITIAL_MEASUREMENTS < 1 or N_ACQUISITIONS < 0:
        raise ValueError("Need at least one initial measurement and nonnegative acquisitions")
    if N_INITIAL_MEASUREMENTS + N_ACQUISITIONS > len(points):
        raise ValueError("Measurement budget exceeds the number of free cells")
    if CPU_THREADS < 1 or SEED < 0:
        raise ValueError("CPU_THREADS must be positive and SEED nonnegative")
    for std in (GAS_ABSOLUTE_NOISE_STD, GAS_RELATIVE_NOISE_STD):
        if not np.isfinite(std) or std < 0:
            raise ValueError("Noise standard deviations must be finite and nonnegative")
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
    indices = list(rng.choice(len(points), N_INITIAL_MEASUREMENTS, replace=False))
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    sigma = np.hypot(GAS_ABSOLUTE_NOISE_STD, GAS_RELATIVE_NOISE_STD * case.concentrations)
    observed = case.concentrations + noise_rng.standard_normal(len(points)) * sigma
    localizer = UncertaintyBasedGasSourceEstimator(
        case.occupancy, case.wind, PINN_CONFIG,
        initial_training_steps=INITIAL_TRAINING_STEPS,
        update_training_steps=UPDATE_TRAINING_STEPS,
        n_collocation_points=N_COLLOCATION_POINTS,
    )
    print(f"New run: {case.name}, seed {SEED}; {N_INITIAL_MEASUREMENTS} initial + "
          f"{N_ACQUISITIONS} additional measurements", flush=True)
    print(f"Initial training: {INITIAL_TRAINING_STEPS} steps", flush=True)
    localizer.fit(GasSampleSet(points[indices], observed[indices]))
    available = np.ones(len(points), dtype=bool)
    available[indices] = False
    fields = [localizer.estimator.predict_source(points).reshape(-1)]
    scores = []
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for step in range(N_ACQUISITIONS):
        # Selection must not alter the RNG stream used for collocation sampling.
        with torch.random.fork_rng(devices=devices):
            values = localizer.estimator.source_uncertainty_reduction(points).detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite acquisition map")
        values = np.where(available, values, np.nan)
        chosen = int(np.nanargmax(values))
        scores.append(values)
        localizer.add_measurement(points[chosen], float(observed[chosen]))
        available[chosen] = False
        indices.append(chosen)
        fields.append(localizer.estimator.predict_source(points).reshape(-1))
        error = np.linalg.norm(points[int(np.argmax(fields[-1]))] - case.source_position)
        print(f"  measurement {step + 1}/{N_ACQUISITIONS}: {points[chosen]} | "
              f"{UPDATE_TRAINING_STEPS} update steps | source error {error:.3f} m", flush=True)
    archive = dict(candidate_points=points, true_position=case.source_position,
                   true_source=case.source_values, predicted_source=np.asarray(fields),
                   measurement_indices=np.asarray(indices), observed_concentrations=observed[indices],
                   true_concentrations=case.concentrations[indices])
    scores = np.asarray(scores).reshape(N_ACQUISITIONS, len(points))
    np.savez_compressed(output / "simulation.npz", **archive, acquisition_scores=scores)
    return archive, scores


def planned_routes(occupancy, positions):
    """8-neighbour shortest paths, without corner cutting, then LOS shortcuts."""
    planner = AnyAngleGridPlanner(occupancy)
    nodes = planner.nearest_nodes(positions)
    routes = []
    for start, target in zip(nodes[:-1], nodes[1:]):
        # Reuse the helper's collision-safe grid graph. Dijkstra is much faster
        # than exploring every any-angle shortcut when rendering a whole run.
        _, parents = dijkstra(planner._graph, directed=False, indices=int(start),
                              return_predecessors=True)
        chain = [int(target)]
        while chain[-1] != start:
            previous = int(parents[chain[-1]])
            if previous < 0:
                raise RuntimeError("Selected measurement is unreachable in the free grid.")
            chain.append(previous)
        chain.reverse()
        simplified, anchor = [chain[0]], 0
        while anchor < len(chain) - 1:
            farthest = anchor + 1
            for candidate in range(anchor + 2, len(chain)):
                if not planner._line_of_sight(chain[anchor], chain[candidate]):
                    break
                farthest = candidate
            simplified.append(chain[farthest])
            anchor = farthest
        routes.append(planner.points[simplified])
    return routes


def partial_route(route, fraction):
    """Interpolate at constant distance per frame along a polyline."""
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    distance = fraction * distances[-1]
    position = np.array([np.interp(distance, distances, route[:, dim]) for dim in (0, 1)])
    return np.vstack((route[distances < distance], position))


def render_video(case, archive, scores, output, *, seed, n_initial):
    """Render the states produced by simulate; no training happens here."""
    points = archive["candidate_points"]
    positions = points[archive["measurement_indices"]]
    fields = archive["predicted_source"]
    routes = planned_routes(case.occupancy, positions[n_initial - 1:])
    np.savez_compressed(output / "planned_route.npz", vertices=np.concatenate(routes) if routes else np.empty((0, 2)),
                        offsets=np.r_[0, np.cumsum([len(route) for route in routes])])

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.23, top=0.81, wspace=0.20)
    source = archive["true_position"]
    for ax, title in zip(axes, ("LLLA Uncertainty Map", "Robot Position and Trajectory", "Current Source Estimation")):
        case.occupancy.plot(ax=ax, title=title)
        ax.get_legend().remove()
        ax.scatter(*source, marker="*", c="#48ee26", edgecolors="black", s=170, zorder=8)
    extent = [case.occupancy.lower_bound[0], case.occupancy.upper_bound[0],
              case.occupancy.lower_bound[1], case.occupancy.upper_bound[1]]

    def grid(values):
        result = np.full(case.occupancy.free_mask.shape, np.nan)
        result[case.occupancy.free_mask] = values
        return np.ma.masked_invalid(result)

    first_score = scores[0] if len(scores) else np.full(len(points), np.nan)
    score_image = axes[0].imshow(grid(first_score), origin="lower", extent=extent,
                                 cmap="viridis", interpolation="nearest", vmin=0)
    source_image = axes[2].imshow(grid(fields[0]), origin="lower", extent=extent,
                                  cmap="inferno", interpolation="nearest", vmin=0,
                                  vmax=max(float(fields[0].max()), 1e-12))
    colorbar_axes = [fig.add_axes([ax.get_position().x0, 0.17, ax.get_position().width, 0.027])
                    for ax in (axes[0], axes[2])]
    score_bar = fig.colorbar(score_image, cax=colorbar_axes[0], orientation="horizontal")
    score_bar.set_label("Expected decrease in source-field variance")
    source_bar = fig.colorbar(source_image, cax=colorbar_axes[1], orientation="horizontal")
    source_bar.set_label("Source map q")
    for bar in (score_bar, source_bar):
        bar.ax.tick_params(labelsize=8)
    axes[1].scatter(*positions[:n_initial].T, s=43, facecolors="white", edgecolors="#182d46", zorder=6)
    for i, position in enumerate(positions[:n_initial]):
        axes[1].annotate(f"S{i + 1}", position, xytext=(5, 5), textcoords="offset points", fontsize=8)
    measured = axes[1].scatter([], [], c="#00bcd4", edgecolors="black", s=26, zorder=6)
    robot, = axes[1].plot([], [], "o", color="#147df5", markeredgecolor="white", markersize=11, zorder=9)
    route_line, = axes[1].plot([], [], color="#e02525", linewidth=2, zorder=5)
    targets = [ax.plot([], [], marker="x", color="#ff3864", markersize=9,
                       markeredgewidth=2, linestyle="none", zorder=8)[0] for ax in axes[:2]]
    estimated, = axes[2].plot([], [], "x", color="#00dcff", markersize=9, markeredgewidth=2)
    status = fig.suptitle("", fontsize=15, y=0.965)
    detail = fig.text(0.5, 0.89, "", ha="center", fontsize=10)
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#182d46", label="Initial Random Measurements"),
               Line2D([], [], marker="o", color="none", markerfacecolor="#147df5", label="Robot"),
               Line2D([], [], color="#e02525", label="Trajectory"),
               Line2D([], [], marker="*", color="none", markerfacecolor="#48ee26", markeredgecolor="black", markersize=12, label="GT Source"),
               Line2D([], [], marker="x", color="#ff3864", linestyle="none", label="Next Measurement"),
               Line2D([], [], marker="x", color="#00bdd8", linestyle="none", label="Estimated Source")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=6, fontsize=9, frameon=False)
    fig.text(0.5, 0.018, f"Start at S{n_initial}",
             ha="center", fontsize=9, color="#555555")
    movie_path = output / "sampling.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-threads", "1"])
    complete_route = positions[n_initial - 1:n_initial].copy()

    def draw(step, visible_route, moving):
        robot.set_data([visible_route[-1, 0]], [visible_route[-1, 1]])
        route_line.set_data(*visible_route.T)
        measured.set_offsets(positions[n_initial:n_initial + step])
        source_image.set_data(grid(fields[step]))
        source_image.set_clim(0, max(float(fields[step].max()), 1e-12))
        estimate = points[int(np.argmax(fields[step]))]
        estimated.set_data([estimate[0]], [estimate[1]])
        if step < len(scores):
            score_image.set_data(grid(scores[step]))
            score_image.set_clim(0, max(float(np.nanmax(scores[step])), 1e-12))
            target = positions[n_initial + step]
            for artist in targets:
                artist.set_data([target[0]], [target[1]])
        else:
            score_image.set_data(grid(np.full(len(points), np.nan)))
            #axes[0].set_title("Abgeschlossen – kein weiterer Messort")
            for artist in targets:
                artist.set_data([], [])
        error = np.linalg.norm(estimate - source)
        status.set_text(f"Source-LLLA · {case.directory.parent.name} · Seed {seed}")
        detail.set_text(f"{n_initial} initial random meausurements + {step}/{len(routes)} additionally acquired measurements  |  Position error: {error:.2f} m")

    print(f"Rendering {len(routes)} movements to {movie_path}", flush=True)
    with writer.saving(fig, str(movie_path), DPI):
        draw(0, complete_route, False)
        fig.savefig(output / "preview_start.png", dpi=DPI)
        for _ in range(FPS * 2):
            writer.grab_frame()
        for step, route in enumerate(routes):
            for fraction in np.linspace(0, 1, max(2, round(FPS * SECONDS_PER_MOVE))):
                visible = np.vstack((complete_route, partial_route(route, fraction)))
                draw(step, visible, True)
                writer.grab_frame()
            complete_route = np.vstack((complete_route, route[1:]))
            draw(step + 1, complete_route, False)
            for _ in range(max(1, round(FPS * SECONDS_AT_MEASUREMENT))):
                writer.grab_frame()
            if step == min(9, len(routes) - 1):
                fig.savefig(output / "preview_route.png", dpi=DPI)
            if (step + 1) % 10 == 0:
                print(f"  rendered {step + 1}/{len(routes)} movements", flush=True)
        for _ in range(FPS * 2):
            writer.grab_frame()
        fig.savefig(output / "preview_final.png", dpi=DPI)
    plt.close(fig)
    (output / "animation_metadata.json").write_text(json.dumps({
        "scenario": str(case.directory), "seed": seed, "n_initial": n_initial, "n_acquisitions": len(routes),
        "fps": FPS, "seconds_per_move": SECONDS_PER_MOVE,
        "seconds_at_measurement": SECONDS_AT_MEASUREMENT,
        "data_file": "simulation.npz", "mode": "new simulation, no archived replay",
        "left_panel": "Actual QLLLA acquisition score: mean source-field variance reduction; measured cells masked; color scale per round",
        "right_panel": "Newly computed source fields; linear color scale from zero to current field maximum, per round",
        "robot_start": "Last initial measurement; visualized motion starts after initial training",
        "route": "Visualized point-robot path: 8-neighbour Dijkstra without corner cutting, collision-checked line-of-sight shortcuts; no robot footprint",
        "motion": "Constant speed within each leg; equal playback duration per leg; not physical time; no samples collected in transit",
    }, indent=2) + "\n")
    print(f"Saved {movie_path}", flush=True)
    return movie_path


def main():
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("FFmpeg fehlt. Zuerst installieren: apt-get install --no-install-recommends ffmpeg")
    case = load_scenario(SCENARIO_DIR)
    output = OUTPUT_ROOT / case.directory.parent.name / case.directory.name / (
        f"seed_{SEED}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps({
        "scenario": str(case.directory), "scenario_metadata": case.metadata,
        "seed": SEED, "n_initial_measurements": N_INITIAL_MEASUREMENTS,
        "n_acquisitions": N_ACQUISITIONS, "initial_training_steps": INITIAL_TRAINING_STEPS,
        "update_training_steps": UPDATE_TRAINING_STEPS, "n_collocation_points": N_COLLOCATION_POINTS,
        "cpu_threads": CPU_THREADS, "gas_absolute_noise_std": GAS_ABSOLUTE_NOISE_STD,
        "gas_relative_noise_std": GAS_RELATIVE_NOISE_STD, "pinn": asdict(PINN_CONFIG),
    }, indent=2) + "\n")
    archive, scores = simulate(case, output)
    return render_video(case, archive, scores, output, seed=SEED, n_initial=N_INITIAL_MEASUREMENTS)


if __name__ == "__main__":
    main()
