#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch

from pinn_wind.io_tools import OccupancyMap, draw_random_samples_csv
from pinn_wind import pinn

random_seed = 0
measurement_noise_std = 0.1 # m/s
n_measurements = 50



occ = OccupancyMap.from_yaml("data/example_labyrinth/occupancy.yaml")
measurements = draw_random_samples_csv(
    "/app/data/example_labyrinth/wind_gt.csv",
    n_samples=n_measurements,
    random_seed=0,
    add_noise_std=measurement_noise_std,
)

torch.manual_seed(random_seed)

estimator = pinn.PINNWindEstimator(lambda_wall=0.1, 
                                   lambda_data=1.0,
                                   lambda_smooth=1e-3,
                                   lambda_div=0.1,
                                   lambda_mom=0.1,
                                   lambda_p=0.1,
                                   learning_rate=1e-2)

lb = occ.points.min(axis=0)
ub = occ.points.max(axis=0)
estimator.normalize_domain(lb, ub)

t_start = time.time()
history = estimator.fit(occ, measurements, n_collocation_points=1000, steps=500)
u_pred, v_pred = estimator.predict_map(occ)
t_end = time.time()
elapsed_s = t_end - t_start

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

fig, (ax_field, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))

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

plt.tight_layout()
plt.show()
# %%
