#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch

from pinn_wind.io_tools import OccupancyMap, draw_random_samples_csv
from pinn_wind import pinn

random_seed = 1
measurement_noise_std = 0.1 # m/s
n_measurements = 50

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


occ = OccupancyMap.from_yaml("data/example_labyrinth/occupancy.yaml")
measurements = draw_random_samples_csv(
    "/app/data/example_labyrinth/wind_gt.csv",
    n_samples=n_measurements,
    random_seed=random_seed,
    add_noise_std=measurement_noise_std,
)

torch.manual_seed(random_seed)

estimator = pinn.PINNWindEstimator(n_layers=3,
                                   hidden_units=64,
                                   lambda_wall=0.1, 
                                   lambda_data=1.0,
                                   lambda_smooth=1e-3,
                                   lambda_div=1.0,
                                   lambda_mom=1.0,
                                   lambda_p=1e-3,
                                   learning_rate=5e-3, 
                                   nu=1e-4)

lb = occ.points.min(axis=0)
ub = occ.points.max(axis=0)
estimator.normalize_domain(lb, ub)

t_start = time.time()
history = estimator.fit(occ, measurements, n_collocation_points=500, steps=1000)
u_pred, v_pred = estimator.predict_map(occ)
t_end = time.time()
elapsed_s = t_end - t_start

# Plotting and reporting
print("-----")
print(f"Estimation based on {n_measurements} samples with noise level sigma = {measurement_noise_std} m/s")
print("-----")
print("Weighted loss components:")
print(estimator.loss_fn.current_losses)
print(f"Elapsed time for training and prediction: {elapsed_s:.2f} s.")

speed = np.sqrt(u_pred**2 + v_pred**2)

ground_truth = pd.read_csv("/app/data/example_labyrinth/wind_gt.csv")
xy_gt = ground_truth[["Points:0", "Points:1"]].to_numpy(dtype=np.float32)
uv_gt = ground_truth[["U:0", "U:1"]].to_numpy(dtype=np.float32)
uv_eval = estimator.predict_points(xy_gt)
vector_rmse = np.sqrt(np.mean(np.sum((uv_eval - uv_gt) ** 2, axis=1)))
print(f"Vector RMSE to ground truth = {vector_rmse:.4f}")

point_error = np.sqrt(np.sum((uv_eval - uv_gt) ** 2, axis=1))
rmse_map = np.full(occ.occupancy.shape, np.nan, dtype=float)
gt_cols = np.rint((xy_gt[:, 0] - occ.origin[0]) / occ.resolution - 0.5).astype(int)
gt_rows = np.rint((xy_gt[:, 1] - occ.origin[1]) / occ.resolution - 0.5).astype(int)
valid_bounds = (
    (gt_rows >= 0)
    & (gt_rows < occ.occupancy.shape[0])
    & (gt_cols >= 0)
    & (gt_cols < occ.occupancy.shape[1])
)
valid_gt = np.zeros_like(valid_bounds, dtype=bool)
valid_gt[valid_bounds] = occ.free_mask[gt_rows[valid_bounds], gt_cols[valid_bounds]]
rmse_map[gt_rows[valid_gt], gt_cols[valid_gt]] = point_error[valid_gt]

half_cell = 0.5 * occ.resolution
map_extent = (
    occ.xx.min() - half_cell,
    occ.xx.max() + half_cell,
    occ.yy.min() - half_cell,
    occ.yy.max() + half_cell,
)

fig = plt.figure(figsize=(14, 7), constrained_layout=True)
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=(1.35, 1.0),
    height_ratios=(0.65, 1.0),
    wspace=0.28,
    hspace=0.35,
)
ax_field = fig.add_subplot(gs[:, 0])
ax_loss = fig.add_subplot(gs[0, 1])
ax_rmse = fig.add_subplot(gs[1, 1])

occ.plot(ax=ax_field, title=f"PINN wind estimate ({n_measurements} samples)")

stream = ax_field.streamplot(
    occ.xx[0, :],
    occ.yy[:, 0],
    u_pred,
    v_pred,
    color=speed,
    cmap="coolwarm",
    density=1.4,
    linewidth=1.2,
    arrowsize=1.1,
)
fig.colorbar(stream.lines, ax=ax_field, label="wind speed")
ax_field.scatter(
    measurements["Points:0"],
    measurements["Points:1"],
    c="black",
    s=18,
    marker="x",
    label="Measurements",
)
ax_field.legend(loc="upper right")

ax_loss.plot(history.loss, color="orange", lw=2)
ax_loss.set_title("Training loss")
ax_loss.set_xlabel("step")
ax_loss.set_ylabel("loss")
ax_loss.grid(True, linestyle="--", alpha=0.6)

occ.plot(ax=ax_rmse, title="Local vector error")
error_cmap = plt.cm.magma.copy()
error_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
rmse_image = ax_rmse.imshow(
    rmse_map,
    origin="lower",
    extent=map_extent,
    cmap=error_cmap,
)
fig.colorbar(rmse_image, ax=ax_rmse, label="error magnitude (m/s)")

plt.show()
# %%
