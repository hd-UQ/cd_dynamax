import sys
sys.path.append("../")
sys.path.append("../..")

import itertools
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import jax
import numpy as np
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import matplotlib.colors as mcolors
import seaborn as sns
from continuous_discrete_nonlinear_gaussian_ssm import (
    EnKFHyperParams, EKFHyperParams, UKFHyperParams
)

# ------------------------
# Plot helpers
# ------------------------
def plot_coeff_heatmaps(W_true, W_learned, exponents):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    vlim = jnp.max(jnp.abs(jnp.concatenate([W_true, W_learned])))
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    im0 = axes[0].imshow(W_true, aspect="auto", cmap="seismic", norm=norm)
    axes[0].set_title("True weights"); axes[0].set_xlabel("Term index"); axes[0].set_ylabel("State index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(W_learned, aspect="auto", cmap="seismic", norm=norm)
    axes[1].set_title("Learned weights (SVI median)"); axes[1].set_xlabel("Term index"); axes[1].set_ylabel("State index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    abs_err = jnp.abs(W_learned - W_true) #/ jnp.maximum(jnp.abs(W_true), 1e-8)
    im2 = axes[2].imshow(abs_err, aspect="auto", cmap="viridis")
    axes[2].set_title("Absolute error"); axes[2].set_xlabel("Term index"); axes[2].set_ylabel("State index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    return fig


def plot_drift_field(
    f_true,
    f_learned,
    f_learned_sd=None,     # callable or None
    x1_range=(-3.0, 3.0),
    x2_range=(-3.2, 3.2),
    num_points=50,
    return_rmse=False,
    relative_error=False,
):
    """
    Plot true vs learned drift fields (2D state space).
    Optionally include learned uncertainty (stddev).

    Args:
        f_true: callable f(x) -> (2,) array, true drift function
        f_learned: callable f(x) -> (2,) array, learned drift function
        f_learned_sd: optional callable f(x) -> (2,) array (stddev per output dim)
        x1_range: tuple (low, high) for x1 axis
        x2_range: tuple (low, high) for x2 axis
        num_points: number of grid points per axis
    """
    x1 = jnp.linspace(*x1_range, num_points)
    x2 = jnp.linspace(*x2_range, num_points)
    X1, X2 = jnp.meshgrid(x1, x2, indexing="ij")
    grid_points = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)

    # Evaluate true, learned drift and optional stddev
    f_true_vals = jax.vmap(f_true)(grid_points)        # (N, 2)
    f_learned_vals = jax.vmap(f_learned)(grid_points)  # (N, 2)

    if f_learned_sd is not None:
        f_learned_sd_vals = jax.vmap(f_learned_sd)(grid_points).squeeze(1)  # (N, 2)
        f1_sd = f_learned_sd_vals[:, 0].reshape(num_points, num_points)
        f2_sd = f_learned_sd_vals[:, 1].reshape(num_points, num_points)
    else:
        f1_sd = f2_sd = None

    # Split into components and reshape
    f1_true = f_true_vals[:, 0].reshape(num_points, num_points)
    f2_true = f_true_vals[:, 1].reshape(num_points, num_points)
    f1_learned = f_learned_vals[:, 0].reshape(num_points, num_points)
    f2_learned = f_learned_vals[:, 1].reshape(num_points, num_points)

    if relative_error:
        f1_err = (f1_learned - f1_true) / (jnp.abs(f1_true) + 1e-6)
        f2_err = (f2_learned - f2_true) / (jnp.abs(f2_true) + 1e-6)
    else:
        f1_err = f1_learned - f1_true
        f2_err = f2_learned - f2_true

    # Color normalization
    vlim1 = jnp.max(jnp.abs(jnp.concatenate([f1_true.ravel(), f1_learned.ravel()])))
    vlim2 = jnp.max(jnp.abs(jnp.concatenate([f2_true.ravel(), f2_learned.ravel()])))

    # Subplot grid: add uncertainty column if available
    ncols = 4 if f_learned_sd is not None else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5*ncols, 8), constrained_layout=True)

    # f1 row
    im0 = axes[0, 0].imshow(f1_true.T, origin="lower",
                            extent=(*x1_range, *x2_range),
                            cmap="seismic", vmin=-vlim1, vmax=vlim1, aspect="auto")
    axes[0, 0].set_title("f1 true"); fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(f1_learned.T, origin="lower",
                            extent=(*x1_range, *x2_range),
                            cmap="seismic", vmin=-vlim1, vmax=vlim1, aspect="auto")
    axes[0, 1].set_title("f1 learned"); fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(f1_err.T, origin="lower",
                            extent=(*x1_range, *x2_range),
                            cmap="viridis", aspect="auto")
    axes[0, 2].set_title("f1 error"); fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    if f1_sd is not None:
        im3 = axes[0, 3].imshow(f1_sd.T, origin="lower",
                                extent=(*x1_range, *x2_range),
                                cmap="magma", aspect="auto")
        axes[0, 3].set_title("f1 stddev"); fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # f2 row
    im4 = axes[1, 0].imshow(f2_true.T, origin="lower",
                            extent=(*x1_range, *x2_range),
                            cmap="seismic", vmin=-vlim2, vmax=vlim2, aspect="auto")
    axes[1, 0].set_title("f2 true"); fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(f2_learned.T, origin="lower",
                            extent=(*x1_range, *x2_range),
                            cmap="seismic", vmin=-vlim2, vmax=vlim2, aspect="auto")
    axes[1, 1].set_title("f2 learned"); fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(f2_err.T, origin="lower",
                            extent=(*x1_range, *x2_range),
                            cmap="viridis", aspect="auto")
    axes[1, 2].set_title("f2 error"); fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    if f2_sd is not None:
        im7 = axes[1, 3].imshow(f2_sd.T, origin="lower",
                                extent=(*x1_range, *x2_range),
                                cmap="magma", aspect="auto")
        axes[1, 3].set_title("f2 stddev"); fig.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(False)

    if return_rmse:
        rmse = jnp.sqrt(jnp.mean((f_learned_vals - f_true_vals)**2))
        return fig, rmse
    else:
        return fig


def plot_particle_diagnostics(
    t_emissions,       # shape (T,) time vector
    x_ens_filtered,    # shape (T, N, D) ensemble after update
    x_ens_predicted,   # shape (T, N, D) ensemble before update (forecast)
    observations,      # shape (T, D_obs) -- assume D_obs == D for now
    start_idx=0,
    stop_idx=None,
    figsize=(12, 6)
):
    """
    Plot particle forecasts and updates over time, along with observations.
    One subplot per state dimension.

    Args:
        t_emissions: (T,) time vector
        x_ens_filtered: (T, N, D) ensemble after update
        x_ens_predicted: (T, N, D) ensemble before update (forecast)
        observations: (T, D) true observations
        start_idx: int, start timestep
        stop_idx: int, stop timestep (exclusive)
        figsize: tuple, figure size
    """
    T, N, D = x_ens_filtered.shape
    if stop_idx is None:
        stop_idx = T

    t_range = t_emissions[start_idx:stop_idx]

    fig, axes = plt.subplots(D, 1, figsize=figsize, sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]

        # Plot observations
        ax.plot(t_range, observations[start_idx:stop_idx, d], "rx", label="obs", markersize=20)

        # Plot filtered and forecast ensembles
        for i, t in enumerate(range(start_idx, stop_idx - 1)):
            t0 = t_emissions[t]
            t1 = t_emissions[t+1]

            xf = x_ens_filtered[t, :, d]
            xp = x_ens_predicted[t+1, :, d]

            ax.scatter(np.full(N, t0), xf, color="tab:blue", alpha=0.5, s=15, label="filtered" if (d==0 and i==0) else "")
            ax.scatter(np.full(N, t1), xp, color="tab:orange", alpha=0.5, s=15, label="forecast" if (d==0 and i==0) else "")

            # connect filtered[t] to forecast[t+1] with faint lines
            for n in range(N):
                ax.plot([t0, t1], [xf[n], xp[n]], color="gray", alpha=0.2, linewidth=0.8)

        ax.set_ylabel(f"state {d}")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right")
    fig.suptitle("Particle diagnostics: forecasts vs updates")

    return fig

# ------------------------
# Inference/model helpers
# ------------------------
def get_or_sample(name, dist_obj, value):
    return (
        numpyro.sample(name, dist_obj)
        if value is None
        else numpyro.deterministic(name, value)
    )

def all_multi_indices(d, max_degree):
    out = []
    for total in range(max_degree + 1):
        for exps in itertools.product(range(total + 1), repeat=d):
            if sum(exps) == total:
                out.append(exps)
    return out

def build_exponents(input_dim, max_degree):
    idx = all_multi_indices(input_dim, max_degree)
    return jnp.array(idx, dtype=jnp.int32)

def eval_monomials(x, exponents):
    return jnp.prod(jnp.where(exponents==0, 1.0, x[None,:]**exponents), axis=1)

def poly_drift(x, weights, exponents):
    phi = eval_monomials(x, exponents)
    return weights @ phi

# -----------------------------
# Neural Network Drift Function
# -----------------------------
def make_nn_init_dict(key, state_dim, hidden_dims):
    """
    Create an initialization dictionary for NN weights/biases.
    Uses He-style scaling for weights, zeros for biases.
    """
    params = {}
    keys = iter(jr.split(key, len(hidden_dims) * 2 + 2))  # enough rng keys

    in_dim = state_dim
    for layer_idx, out_dim in enumerate(hidden_dims):
        kW, kb = next(keys), next(keys)
        params[f"W{layer_idx}"] = jnp.sqrt(2.0 / in_dim) * jr.normal(kW, (in_dim, out_dim))
        params[f"b{layer_idx}"] = jnp.zeros((out_dim,))
        in_dim = out_dim

    # Output layer
    kW_out, kb_out = next(keys), next(keys)
    params["W_out"] = jnp.sqrt(2.0 / in_dim) * jr.normal(kW_out, (in_dim, state_dim))
    params["b_out"] = jnp.zeros((state_dim,))

    return params

def make_bayesian_drift(state_dim, hidden_dims, prior="uniform", prior_scale=10.0, **kwargs):
    """Sample NN weights once, then return a deterministic drift(x)."""

    def sample_or_use(name, shape):
        if name in kwargs:
            val = kwargs[name]
            numpyro.deterministic(name, val)
            return val
        else:
            if prior == "uniform":
                return numpyro.sample(name, dist.Uniform(-prior_scale, prior_scale).expand(shape).to_event(len(shape)))
            elif prior == "normal":
                return numpyro.sample(name, dist.Normal(0.0, prior_scale).expand(shape).to_event(len(shape)))
            else:
                raise ValueError(f"Unknown prior: {prior}")

    # sample weights once
    params = {}
    in_dim = state_dim
    for layer_idx, out_dim in enumerate(hidden_dims):
        params[f"W{layer_idx}"] = sample_or_use(f"W{layer_idx}", (in_dim, out_dim))
        params[f"b{layer_idx}"] = sample_or_use(f"b{layer_idx}", (out_dim,))
        in_dim = out_dim
    params["W_out"] = sample_or_use("W_out", (in_dim, state_dim))
    params["b_out"] = sample_or_use("b_out", (state_dim,))

    # build deterministic forward
    def drift_fn(x):
        h = x
        in_dim = state_dim
        for layer_idx, out_dim in enumerate(hidden_dims):
            h = jnn.gelu(jnp.dot(h, params[f"W{layer_idx}"]) + params[f"b{layer_idx}"])
            in_dim = out_dim
        out = jnp.dot(h, params["W_out"]) + params["b_out"]
        return out

    return drift_fn

# ------------------------
# Filter hyperparameter helper
# ------------------------
def make_filter_hyperparams(cfg):
    diffeqsolve_settings = {
        # 'solver': eval(cfg.diffeqsolve_settings['solver']),
        # 'stepsize_controller': eval(cfg.diffeqsolve_settings['stepsize_controller']),
        # 'adjoint': eval(cfg.diffeqsolve_settings['adjoint']),
        # 'dt0': cfg.diffeqsolve_settings['dt0'],
        # 'tol_vbt': cfg.diffeqsolve_settings['tol_vbt'],
        'max_steps': cfg.diffeqsolve_max_steps,
    }
    if cfg.filter_type == "EnKF":
        return EnKFHyperParams(
            N_particles=cfg.N_particles,
            state_order=cfg.state_order,
            diffeqsolve_settings=diffeqsolve_settings,
            cov_rescaling=cfg.cov_rescaling,
            inflation_delta=cfg.inflation_delta,
        )
    elif cfg.filter_type == "EKF":
        return EKFHyperParams(
            state_order=cfg.state_order,
            diffeqsolve_settings=diffeqsolve_settings,
        )
    elif cfg.filter_type == "UKF":
        return UKFHyperParams(
            state_order=cfg.state_order,
            diffeqsolve_settings=diffeqsolve_settings,
        )
    else:
        raise ValueError(f"Unknown filter type: {cfg.filter_type}")
