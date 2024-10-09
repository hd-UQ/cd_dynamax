# Imports
import numpy as np
from matplotlib import pyplot as plt


# Plot Lorenz data: states and observations
def plot_lorenz(time_grid, x_tr, x_obs, x_est=None, est_type="", t_start=None, t_end=None):

    # Squeeze time_grid
    time_grid = np.squeeze(time_grid)

    # Determine indices based on t_start and t_end
    if t_start is not None:
        start_idx = np.searchsorted(time_grid, t_start)
    else:
        start_idx = 0

    if t_end is not None:
        end_idx = np.searchsorted(time_grid, t_end, side="right")
    else:
        end_idx = len(time_grid)

    time_grid = time_grid[start_idx:end_idx]
    x_tr = x_tr[start_idx:end_idx, :]
    x_obs = x_obs[start_idx:end_idx, :]
    if x_est is not None:
        x_est = x_est[start_idx:end_idx, :]

    plt.figure()

    # Latent states
    n_states = x_tr.shape[1]

    for i in range(n_states):
        plt.plot(time_grid, x_tr[:, i], color=f"C{i}", alpha=0.5, linewidth=4, label=f"True State {i}")

    if x_est is not None:
        for i in range(n_states):
            plt.plot(
                time_grid, x_est[:, i], "--", color=f"C{i}", linewidth=1.5, label=f"{est_type} Estimated State {i}"
            )

    # Observations
    n_obs = x_obs.shape[1]
    for i in range(n_obs):
        plt.plot(
            time_grid, x_obs[:, i], "ok", color=f"C{i}", fillstyle="none", ms=1.5, label=f"Measurement {i}"
        )

    # Title, labels and legend
    plt.title("Lorenz Dynamics")
    plt.xlabel("Time $t$")
    plt.legend(loc=1, borderpad=0.5, handlelength=4, fancybox=False, edgecolor="k")

    # Show plot
    plt.show()


# Plot Lorenz estimates
def plot_lorenz_estimates(time_grid, x_tr, x_obs, x_est=None, x_unc=None, est_type="", t_start=None, t_end=None):

    # Squeeze time_grid
    time_grid = np.squeeze(time_grid)

    # Determine indices based on t_start and t_end
    if t_start is not None:
        start_idx = np.searchsorted(time_grid, t_start)
    else:
        start_idx = 0

    if t_end is not None:
        end_idx = np.searchsorted(time_grid, t_end, side="right")
    else:
        end_idx = len(time_grid)

    time_grid = time_grid[start_idx:end_idx]
    x_tr = x_tr[start_idx:end_idx, :]
    x_obs = x_obs[start_idx:end_idx, :]
    if x_est is not None:
        x_est = x_est[start_idx:end_idx, :]
    if x_unc is not None:
        x_unc = x_unc[start_idx:end_idx, :, :]

    plt.figure()

    # Latent states
    n_states = x_tr.shape[1]

    # True states
    for i in range(n_states):
        plt.plot(time_grid, x_tr[:, i], "--", color=f"C{i}", alpha=0.5, linewidth=2, label=f"True State {i}")

    # True Measurements
    n_obs = x_obs.shape[1]
    for i in range(n_obs):
        plt.plot(
            time_grid,
            x_obs[:, i],
            marker="X",
            color=f"C{i}",
            alpha=0.5,
            linestyle="None",
            label=f"Measurement {i}",
        )

    # Estimated states, with uncertainty
    if x_est is not None:
        for i in range(n_states):
            plt.plot(
                time_grid, x_est[:, i], color=f"C{i}", alpha=0.5, linewidth=2, label=f"{est_type} Estimated State {i}"
            )
        if x_unc is not None:
            for i in range(n_states):
                plt.fill_between(
                    time_grid,
                    x_est[:, i] - np.sqrt(x_unc[:, i, i]),
                    x_est[:, i] + np.sqrt(x_unc[:, i, i]),
                    color=f"C{i}",
                    alpha=0.5,
                )

    # Title, labels and legend
    plt.title("Lorenz Dynamics: True States, Observations and Estimates")
    plt.xlabel("Time $t$")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), borderpad=0.5, handlelength=4, fancybox=False, edgecolor="k")

    # Show plot
    plt.show()
