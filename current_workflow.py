#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinn_wind.io_tools import OccupancyMap, draw_random_samples_csv
from pinn_wind import pinn

occ = OccupancyMap.from_yaml("data/example_labyrinth/occupancy.yaml")
measurements = draw_random_samples_csv(
    "/app/data/example_labyrinth/wind_gt.csv",
    n_samples=50,
    random_seed=0,
    add_noise_std=0.0,
)

estimator = pinn.PINNWindEstimator(lambda_wall=0.1, 
                                   lambda_data=1.0,
                                   lambda_smooth=1e-3,
                                   lambda_div=0.1,
                                   lambda_mom=0.1,
                                   lambda_p=0.0,
                                   learning_rate=1e-2)

lb = occ.points.min(axis=0)
ub = occ.points.max(axis=0)

estimator.normalize_domain(lb, ub)
history = estimator.fit(occ, measurements, steps=1000)

print("Weighted loss components:")
print(estimator.loss_fn.current_losses)

u_pred, v_pred = estimator.predict_map(occ)
speed = np.sqrt(u_pred**2 + v_pred**2)

ground_truth = pd.read_csv("/app/data/example_labyrinth/wind_gt.csv")
xy_gt = ground_truth[["Points:0", "Points:1"]].to_numpy(dtype=np.float32)
uv_gt = ground_truth[["U:0", "U:1"]].to_numpy(dtype=np.float32)
uv_eval = estimator.predict_points(xy_gt)
vector_rmse = np.sqrt(np.mean(np.sum((uv_eval - uv_gt) ** 2, axis=1)))
print(f"Vector RMSE to ground truth = {vector_rmse:.4f}")

fig, (ax_field, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))

occ.plot(ax=ax_field, title=f"PINN wind estimate, RMSE={vector_rmse:.3f}")

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
    label="measurements",
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
