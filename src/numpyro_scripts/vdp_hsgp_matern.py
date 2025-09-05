import sys
sys.path.append("../")
sys.path.append("../..")

import itertools
import numpy as np
import jax
# Make sure everything is 64bit (should prevent NaNs, but can be slow)
# Best to set this before importing jax.numpy or numpyro
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    SVI, Trace_ELBO, 
    init_to_median, init_to_value,
    Predictive,
)
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoMultivariateNormal
from numpyro.contrib.hsgp.approximation import (
    eigenfunctions,
    diag_spectral_density_matern,
)
from jax.scipy.linalg import cho_solve, cho_factor

from continuous_discrete_nonlinear_gaussian_ssm import (
    cdnlgssm_filter, ContDiscreteNonlinearGaussianSSM, 
    EnKFHyperParams, EKFHyperParams, UKFHyperParams
)

from numpyro_extension import build_params
from utils.diffrax_utils import adjust_rhs
from utils.optimize_utils import make_optimizer
from utils.simulation_utils import make_key_sequence
import wandb


# ------------------------
# Plot helpers
# ------------------------
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

# ------------------------
# Model helpers
# ------------------------
adjust_rhs_kwargs = {'lower_bound': -10.0, 
                     'upper_bound': 10.0, 
                     'lower_bound_derivative': -100.0,
                     'upper_bound_derivative': 100.0}

# ------------------------
# Main training entrypoint
# ------------------------
def main(**cfg):
    # Initialize a new W&B run.
    # - We pass in `cfg` (a dict from argparse defaults/CLI args) as the initial config. 
    #   This seeds wandb.config with *all arguments and their defaults*, so they appear
    #   in the run's Config tab even if they're not being swept over.
    # - When running under a sweep, the sweep controller will then overwrite any of these
    #   entries that are specified in the sweep YAML (e.g. num_epochs, init_lr, ...).
    run = wandb.init(config=cfg)

    # After wandb.init(), wandb.config now holds the *final merged config*:
    #   argparse defaults + CLI overrides + sweep overrides (if any).
    # This object behaves like a dict, but is also a special W&B container
    # that tracks hyperparameters and makes them visible in the UI.
    cfg = wandb.config

    # Global settings
    state_dim, emission_dim = 2, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    # Build polynomial exponents (up to quadratic)
    def true_drift(x):
        return jnp.array([x[1], 1.0*(1 - x[0]**2)*x[1] - x[0]])

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    
    # ----------------
    # HSGP-Matern setup
    # ---------------- 
    # Note: setting alpha=1.0, because it only supports isotropic covariance for now.
    # We do it ourselves by scaling the outputs later.
    SPD = jnp.sqrt(
            diag_spectral_density_matern(
                nu=cfg.nu, alpha=1.0, length=cfg.length_scale,
                ell=cfg.ell_box, m=cfg.m, dim=state_dim,
            )
    )
    print("Shape of SPD:", SPD.shape)  # should be (m*,)
    
    MSTAR = eigenfunctions(x=jnp.zeros((1, state_dim)), ell=cfg.ell_box, m=cfg.m).shape[-1]

    def compute_drift(x, beta, spd=SPD, alpha=cfg.alpha):

        x_ = jnp.atleast_2d(x)                                           # (N, state_dim)
        phi_x = eigenfunctions(x_, ell=cfg.ell_box, m=cfg.m)             # (N, m*)
        beta_scaled = beta * spd[:, None]                                # (m*, D)
        out = phi_x @ beta_scaled                                        # (N, D)
        out = out * jnp.sqrt(jnp.asarray(alpha))                         # broadcast per output
        return out[0] if x.ndim == 1 else out

    def compute_drift_covariance(x, beta_cov, spd=SPD, alpha=cfg.alpha):
        """
        Posterior covariance of f(x) given covariance of beta.

        Args:
            x: (N, state_dim)
            beta_cov: posterior covariance of beta, shape (m*D, m*D)
                    (flattened over basis × output dims)
            spd: (m*,)
            alpha: (D,)
        Returns:
            f_cov: (N, D, D) covariance matrices of f(x) for each input
        """
        x_ = jnp.atleast_2d(x)
        phi_x = eigenfunctions(x_, ell=cfg.ell_box, m=cfg.m)   # (N, m*)
        Phi_spd = phi_x * spd[None, :]                         # (N, m*)

        f_covs = []
        for n in range(x_.shape[0]):
            # expand Phi for all output dims: block-diagonal
            Phi_block = jnp.kron(jnp.eye(len(alpha)), Phi_spd[n, :])  # (D, m*D)
            cov_beta = Phi_block @ beta_cov @ Phi_block.T             # (D, D)

            # apply output scaling: diag(sqrt(alpha)) @ cov @ diag(sqrt(alpha))
            scale = jnp.sqrt(jnp.asarray(alpha))
            cov = (scale[:, None] * cov_beta) * scale[None, :]
            f_covs.append(cov)
        return jnp.stack(f_covs)  # (N, D, D)

    # Generic isotropic Matérn kernel
    def matern_kernel(x, y, nu, length):
        """Compute isotropic Matérn kernel with smoothness nu."""
        r = jnp.linalg.norm((x[:, None, :] - y[None, :, :]) / length, axis=-1)
        if nu == 0.5:  # exponential kernel
            return jnp.exp(-r)
        elif nu == 1.5:
            sqrt3r = jnp.sqrt(3.0) * r
            return (1 + sqrt3r) * jnp.exp(-sqrt3r)
        elif nu == 2.5:
            sqrt5r = jnp.sqrt(5.0) * r
            return (1 + sqrt5r + 5.0 * r**2 / 3.0) * jnp.exp(-sqrt5r)
        else:
            # fall back to general Matérn (slower, uses Bessel)
            from jax.scipy.special import kv, gamma
            factor = (2**(1 - nu)) / gamma(nu)
            scaled_r = jnp.sqrt(2 * nu) * r
            return factor * (scaled_r**nu) * kv(nu, scaled_r)

    def supervised_fit_drift(true_drift, spd=SPD, sigma2=None):
        # Training mesh
        x1 = jnp.linspace(-cfg.ell_box[0]/2, cfg.ell_box[0]/2, 20)
        x2 = jnp.linspace(-cfg.ell_box[1]/2, cfg.ell_box[1]/2, 20)
        X1, X2 = jnp.meshgrid(x1, x2, indexing="ij")
        X_train = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)  # (N,2)
        Y_train = jax.vmap(true_drift)(X_train)                 # (N,2)

        # Use emission noise variance as observation noise if not given
        if sigma2 is None:
            sigma2 = cfg.emission_cov

        # =========================================================
        # a) Exact GP regression (Matérn ν from cfg)
        # =========================================================
        gp_predictors = []
        for d in range(Y_train.shape[1]):
            alpha_d = cfg.alpha[d]
            K = alpha_d * matern_kernel(X_train, X_train, cfg.nu, cfg.length_scale)
            K = K + sigma2 * jnp.eye(len(X_train))
            cf = cho_factor(K)
            w = cho_solve(cf, Y_train[:, d])

            def gp_pred_single(xnew, w=w, d=d, alpha_d=alpha_d):
                k_star = alpha_d * matern_kernel(
                    X_train, jnp.atleast_2d(xnew), cfg.nu, cfg.length_scale
                )
                return (k_star.T @ w).squeeze()

            gp_predictors.append(gp_pred_single)

        def gp_predict(x):
            return jnp.stack([gp_predictors[d](x) for d in range(Y_train.shape[1])], axis=-1)

        # =========================================================
        # b) HSGP regression (prior-consistent, same nu as exact GP)
        # =========================================================
        Phi = eigenfunctions(X_train, ell=cfg.ell_box, m=cfg.m)   # (N, m*)
        Phi_scaled = Phi * spd[None, :]                           # (N, m*)

        prior_var = spd**2                                        # (m*,)
        reg_matrix = sigma2 * jnp.diag(1.0 / prior_var)           # (m*, m*)

        Y_train_scaled = Y_train / jnp.sqrt(jnp.asarray(cfg.alpha))
        beta_hsgp = jnp.linalg.solve(
            Phi_scaled.T @ Phi_scaled + reg_matrix,
            Phi_scaled.T @ Y_train_scaled
        )  # (m*, D)

        def hsgp_predict(x):
            phi = eigenfunctions(jnp.atleast_2d(x), ell=cfg.ell_box, m=cfg.m)
            out = phi @ (spd[:, None] * beta_hsgp)                # (N,D)
            out = out * jnp.sqrt(jnp.asarray(cfg.alpha))          # per-output scaling
            return out.squeeze()

        return gp_predict, hsgp_predict, beta_hsgp, (X_train, Y_train)

    # NumPyro model
    def model(t_emissions, emissions=None, store_filtered=False, store_grad=False, **kwargs):
        # If user supplies drift, use it; otherwise build HSGP drift
        lp = 0.0  # log prior
        if "drift" in kwargs and kwargs["drift"] is not None:
            drift = kwargs["drift"]
        else:
            if "beta" in kwargs and kwargs["beta"] is not None:
                beta = kwargs["beta"]
            else:
                # Sample coefficients once
                beta = numpyro.sample(
                    "beta",
                    dist.Normal(0., 1.).expand((MSTAR, state_dim))
                )  # -> shape (MSTAR, state_dim)

            drift = lambda x: compute_drift(x, beta)

            # Also return log prior
            lp = dist.Normal(0, 1).log_prob(beta).sum()

        # Build the model and its parameters
        cdnlgssm = ContDiscreteNonlinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim)
        params = build_params(
            state_dim=state_dim,
            emission_dim=emission_dim,
            initial_mean=jnp.zeros(state_dim), 
            initial_cov=cfg.initial_cov*jnp.eye(state_dim), # warning: choosing it too small, even if "true", can lead to numerical issues with the filter.
            drift=lambda x: adjust_rhs(x, drift(x), **adjust_rhs_kwargs),
            diffusion_coeff=cfg.diffusion_coeff * jnp.eye(state_dim),
            diffusion_cov=jnp.eye(state_dim),
            emission_function=lambda x: H @ x,
            emission_cov=cfg.emission_cov * jnp.eye(emission_dim),
        )
        # Sample emissions if not provided
        if emissions is None:
            states, emissions = cdnlgssm.sample(params=params, num_timesteps=t_emissions.shape[0], key=numpyro.prng_key(), t_emissions=t_emissions, transition_type="path")
            numpyro.deterministic("states", states); numpyro.deterministic("emissions", emissions)

        # Compute (approximate) marginal log likelihood via filtering
        filtered = cdnlgssm_filter(params=params, emissions=emissions, t_emissions=t_emissions, filter_hyperparams=FILTER_HYPERPARAMS)
        ll = filtered.marginal_loglik
        
        if store_filtered:
            # Store the filtering details for diagnostics
            numpyro.deterministic('filtered_means', filtered.filtered_means)
            numpyro.deterministic('filtered_covariances', filtered.filtered_covariances)
            numpyro.deterministic('predicted_means', filtered.predicted_means)
            numpyro.deterministic('predicted_covariances', filtered.predicted_covariances)
            numpyro.deterministic('neg_loglik_steps', -filtered.loglik_step)
            # numpyro.deterministic('S', filtered.S)
            # numpyro.deterministic('K', filtered.K)
            numpyro.deterministic('innovation', filtered.innovation)
            numpyro.deterministic('nis', filtered.nis)
            numpyro.deterministic('min_eig_S', filtered.min_eig_S)
            numpyro.deterministic('cond_S', filtered.cond_S)
            numpyro.deterministic('cond_K', filtered.cond_K)
            numpyro.deterministic('x_ens_filtered', filtered.x_ens_filtered)
            numpyro.deterministic('x_ens_predicted', filtered.x_ens_predicted)
        
        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)
        numpyro.factor("log_likelihood", ll)

        # Optionally store gradients (for diagnostics)
        if store_grad and "drift" not in kwargs:
            grad_beta = jax.grad(lambda _beta: cdnlgssm_filter(
                params=build_params(
                    state_dim=state_dim,
                    emission_dim=emission_dim,
                    initial_mean=jnp.zeros(state_dim), 
                    initial_cov=cfg.initial_cov*jnp.eye(state_dim),
                    drift=lambda x: compute_drift(x, _beta),
                    diffusion_coeff=cfg.diffusion_coeff * jnp.eye(state_dim),
                    diffusion_cov=jnp.eye(state_dim),
                    emission_function=lambda x: H @ x,
                    emission_cov=cfg.emission_cov * jnp.eye(emission_dim),
                ),
                emissions=emissions,
                t_emissions=t_emissions,
                filter_hyperparams=FILTER_HYPERPARAMS
            ).marginal_loglik)(beta)
            numpyro.deterministic("grad_neg_log_likelihood_beta", -grad_beta)
    

    # Define the true parameters for the model and its data generation
    true_values = {
        "drift": true_drift,  # <— include true weights
    }

    beta_init = jnp.zeros((MSTAR, state_dim))
    print("Neg-log-prior of beta_init:", -float(dist.Normal(0,1).log_prob(beta_init).sum().item()))
    f_init_base = lambda x: compute_drift(x=x, beta=beta_init)
    f_init = lambda x: adjust_rhs(x, f_init_base(x), **adjust_rhs_kwargs)
    fig = plot_drift_field(
        f_true=true_drift,
        f_learned=f_init,
        # x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        # x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50
    )
    # save the initial drift figure
    wandb.log({"fig/true_vs_initial_drift": wandb.Image(fig)})
    plt.close(fig)

    # Fit the GP directly to the true drift (for comparison)
    gp_predict, hsgp_predict, beta_supervised, (X_train, Y_train) = supervised_fit_drift(true_drift, sigma2=cfg.emission_cov)
    fig_exact_gp_supervised, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=gp_predict,
        # x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        # x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True
    )
    print(f"RMSE of exact GP drift fit to true drift: {rmse:.4f}")
    wandb.log({"fig/exact_gp_supervised": wandb.Image(fig_exact_gp_supervised)})
    plt.close(fig_exact_gp_supervised)
    
    fig_hsgp_supervised, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=hsgp_predict,
        # x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        # x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True
    )
    print(f"RMSE of HSGP drift fit to true drift: {rmse:.4f}")
    wandb.log({"fig/hsgp_supervised": wandb.Image(fig_hsgp_supervised)})
    plt.close(fig_hsgp_supervised)  # too many colorbars otherwise


    # Typically you keep these fixed/known during learning
    # know everything except "weights" # it is empty here, but useful when there are more params
    known_values = {key: value for key, value in true_values.items() if key not in ["drift"]}
    
    # Generate synthetic emissions
    t_emissions = jnp.arange(start=0.0, stop=cfg.T, step=cfg.dt).reshape(-1, 1)
    sim_data = Predictive(model, num_samples=1)(next(keys), t_emissions=t_emissions, store_filtered=True, **true_values)
    # Extract emissions from the simulation data
    emissions_obs = sim_data["emissions"].squeeze(0)

    # # Plot the emissions
    plt.figure(figsize=(10, 4))
    for d in range(emission_dim):
        plt.plot(t_emissions, emissions_obs[:, d], label=f"emission dim {d}")
    plt.xlabel("time")
    plt.ylabel("emission value")
    plt.title("Simulated emissions")
    plt.legend()
    plt.show()
    
    # plot the phase portrait of the emissions
    if emission_dim == 2:
        plt.figure(figsize=(6, 6))
        plt.plot(emissions_obs[:, 0], emissions_obs[:, 1], 'b-')
        plt.xlabel("emission dim 0")
        plt.ylabel("emission dim 1")
        plt.title("Phase portrait of emissions")
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
    # Log metrics
    wandb.log({
        "metrics/true/neg_log_prior": float(sim_data["neg_log_prior"].item()),
        "metrics/true/neg_log_lik": float(sim_data["neg_log_likelihood"].item()),
        "metrics/true/neg_log_joint": float(sim_data["neg_log_joint"].item()),
    })
    # Print the neg-log-likelihood values
    print(f"True model's neg_log_likelihood from filtering: {float(sim_data['neg_log_likelihood'].item())}")
        
    # Now plot the filtering ensemble trajectories for the true model
    if cfg.log_filtering:
        print("Plotting filtering ensemble trajectories for the true model...")
        fig = plot_particle_diagnostics(
            t_emissions=t_emissions.squeeze(),
            x_ens_filtered=sim_data["x_ens_filtered"][0],   # shape (T, N, D)
            x_ens_predicted=sim_data["x_ens_predicted"][0], # shape (T, N, D)
            observations=emissions_obs,
            figsize=(12, 8),
            start_idx=0,
            stop_idx=20,
        )
        wandb.log({"fig/true/ensembles_0to20": wandb.Image(fig)})
        plt.close(fig)

    # Log the trajectory from filtering
    if cfg.log_filtering:
        print("Logging trajectories from filtering/simulations with true and perturbed-truth models...")
        # Trajectories use "time" as their x-axis
        wandb.define_metric("filtering/true/true_state_*", step_metric="time")
        wandb.define_metric("filtering/true/pred_mean_*", step_metric="time")
        wandb.define_metric("filtering/true/filtered_mean_*", step_metric="time")
        wandb.define_metric("filtering/true/cond_K", step_metric="time")
        wandb.define_metric("filtering/true/cond_S", step_metric="time")
        wandb.define_metric("filtering/true/min_eig_S", step_metric="time")
        wandb.define_metric("filtering/true/nis", step_metric="time")
        wandb.define_metric("filtering/true/neg_loglik_steps", step_metric="time")
        for t in range(sim_data["predicted_means"].shape[1]):
            log_dict = {"time": t}
            log_dict["filtering/true/cond_S"] = float(sim_data["cond_S"][0, t])
            log_dict["filtering/true/min_eig_S"] = float(sim_data["min_eig_S"][0, t])
            log_dict["filtering/true/nis"] = float(sim_data["nis"][0, t])
            log_dict["filtering/true/neg_loglik_steps"] = float(sim_data["neg_loglik_steps"][0, t])
            log_dict["filtering/true/cond_K"] = float(sim_data["cond_K"][0, t])
            for d in range(state_dim):
                log_dict[f"filtering/true/true_state_dim{d}"] = float(sim_data["states"][0, t, d])
                log_dict[f"filtering/true/pred_mean_dim{d}"] = float(sim_data["predicted_means"][0, t, d])
                log_dict[f"filtering/true/filtered_mean_dim{d}"] = float(sim_data["filtered_means"][0, t, d])
            # Log the current time step
            wandb.log(log_dict)

    # Next, check the model at beta_supervised
    print("Checking supervised-fit predictive...")
    supervised_predictive = Predictive(model, num_samples=1)
    supervised_data = supervised_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, store_filtered=True, beta=beta_supervised)
    print(f"Supervised-fit model's neg_log_likelihood from filtering: {float(supervised_data['neg_log_likelihood'].item())}")
    print(f"Supervised-fit model's neg_log_prior: {float(supervised_data['neg_log_prior'].item())}")
    print(f"Supervised-fit model's neg_log_joint: {float(supervised_data['neg_log_joint'].item())}")
    # Log metrics
    wandb.log({
        "metrics/supervised/neg_log_prior": float(supervised_data["neg_log_prior"].item()),
        "metrics/supervised/neg_log_lik": float(supervised_data["neg_log_likelihood"].item()),
        "metrics/supervised/neg_log_joint": float(supervised_data["neg_log_joint"].item()),
    })

    # Now, generate a prior predictive conditioned on emissions/t_emissions (for diagnostics)
    print("Checking prior predictive...")
    prior_predictive = Predictive(model, num_samples=cfg.n_prior_samples)
    prior_data = prior_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, store_grad=True, **known_values)
    # Compute fraction of NaN values in the neg-log-likelihood computation
    nan_fraction = jnp.isnan(prior_data["neg_log_likelihood"]).mean()
    wandb.log({"metrics/prior_predictive/nan_fraction": nan_fraction})
    if nan_fraction > 0.0:
        print(f"Warning: {nan_fraction:.2%} of prior predictive neg_log_likelihood samples are NaN")
        # Show any one example weights matrix with NaN values
        j = jnp.where(jnp.isnan(prior_data["neg_log_likelihood"]))[0][0]
        print("Example weights with NaN in prior predictive neg_log_likelihood:")
        print(prior_data["beta"][j])
        # Optionally, you can raise an error here to stop execution        
        raise ValueError(f"NaN values found in prior predictive neg_log_likelihood: {nan_fraction:.2%} of samples are NaN. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    # Now compute fraction of NaN values in the grad_neg_log_likelihood_beta computation
    nan_fraction_grad = jnp.isnan(prior_data["grad_neg_log_likelihood_beta"]).mean()
    wandb.log({"metrics/prior_predictive/nan_fraction_grad": nan_fraction_grad})
    if nan_fraction_grad > 0.0:
        print(f"Warning: {nan_fraction_grad:.2%} of prior predictive grad_neg_log_likelihood_beta samples are NaN")
        # Show any one example weights matrix with NaN values
        j = jnp.where(jnp.isnan(prior_data["grad_neg_log_likelihood_beta"]))[0][0]
        print("Example weights with NaN in prior predictive grad_neg_log_likelihood_beta:")
        print(prior_data["beta"][j])
        # Optionally, you can raise an error here to stop execution        
        raise ValueError(f"NaN values found in prior predictive grad_neg_log_likelihood_beta: {nan_fraction_grad:.2%} of samples are NaN. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    # SVI
    optimizer = make_optimizer(
        initial_learning_rate=cfg.init_lr,
        decay_factor=cfg.decay_rate,
        epochs_per_step=cfg.epochs_per_step,
        num_epochs=cfg.num_epochs,
        use_lr_scheduler=cfg.use_lr_scheduler,
        clip_norm=cfg.clip_norm if cfg.clip_norm > 0.0 else None,
    )
    
    # First, check feasibility of the initialization
    # guide = AutoDelta(model, init_loc_fn=init_to_value(values={"beta": beta_init}))
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values={"beta": beta_init}))

    # Run model in predictive mode
    print("Checking initialization predictive...")
    init_predictive = Predictive(model, num_samples=1)
    init_data = init_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, store_filtered=True, beta=beta_init)

    if cfg.log_filtering:
        # Log a bunch of these trajectories to wandb
        wandb.define_metric("filtering/init/cond_K", step_metric="time")
        wandb.define_metric("filtering/init/cond_S", step_metric="time")
        wandb.define_metric("filtering/init/min_eig_S", step_metric="time")
        wandb.define_metric("filtering/init/nis", step_metric="time")
        wandb.define_metric("filtering/init/neg_loglik_steps", step_metric="time")
        for t in range(init_data["cond_S"].shape[1]):
            wandb.log({
                "time": t,
                "filtering/init/cond_S": float(init_data["cond_S"][0, t]),
                "filtering/init/min_eig_S": float(init_data["min_eig_S"][0, t]),
                "filtering/init/nis": float(init_data["nis"][0, t]),
                "filtering/init/neg_loglik_steps": float(init_data["neg_loglik_steps"][0, t]),
            })
        fig = plot_particle_diagnostics(
            t_emissions=t_emissions.squeeze(),
            x_ens_filtered=init_data["x_ens_filtered"][0],   # shape (T, N, D)
            x_ens_predicted=init_data["x_ens_predicted"][0], # shape (T, N, D)
            observations=emissions_obs,
            figsize=(12, 8),
            start_idx=0,
            stop_idx=20,
        )
        wandb.log({"fig/init/ensembles_0to20": wandb.Image(fig)})
        plt.close(fig)

    print(f"Initialization neg_log_likelihood from filtering: {float(init_data['neg_log_likelihood'].item())}")
    if jnp.isnan(init_data['neg_log_likelihood']):
        raise ValueError("NaN value found in initialization neg_log_likelihood. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    init_data = init_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, store_filtered=True, store_grad=True, beta=beta_init)
    if jnp.any(jnp.isnan(init_data['grad_neg_log_likelihood_beta'])):
        print(f"Initialization grad_neg_log_likelihood_beta from filtering: {init_data['grad_neg_log_likelihood_beta']}")
        raise ValueError("NaN value found in initialization grad_neg_log_likelihood_beta. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(next(keys), num_steps=cfg.num_epochs, t_emissions=t_emissions, emissions=emissions_obs, **known_values)

    # Log training curve
    # Loss curve uses "epoch" as its x-axis
    wandb.define_metric("svi/loss", step_metric="epoch")
    for step, loss in enumerate(svi_result.losses):
        wandb.log({"epoch": step, "svi/loss": float(loss)})

    # Log posterior predictive
    predictive_learned = Predictive(model, guide=guide, params=guide.median(svi_result.params), num_samples=1)
    learned_data = predictive_learned(next(keys), t_emissions=t_emissions, emissions=emissions_obs, store_filtered=True, **known_values)
    # Log metrics
    wandb.log({
        "metrics/learned/neg_log_prior": float(learned_data["neg_log_prior"].item()),
        "metrics/learned/neg_log_lik": float(learned_data["neg_log_likelihood"].item()),
        "metrics/learned/neg_log_joint": float(learned_data["neg_log_joint"].item()),
    })
    # Log figure
    # if AutoDelta, then beta_learned = guide.median(svi_result.params)["beta"]
    if isinstance(guide, AutoDelta):
        beta_learned = guide.median(svi_result.params)["beta"]
        beta_cov = None
        f_cov = None
        f_sd = None
    else:
        beta_posterior = guide.get_posterior(svi_result.params)
        beta_learned = beta_posterior.mean.reshape(MSTAR, state_dim)
        if hasattr(beta_posterior, "covariance_matrix"):
            # e.g. AutoMultivariateNormal
            beta_cov = beta_posterior.covariance_matrix
        else:
            # AutoDiagonalNormal
            beta_cov = jnp.diag(beta_posterior.variance.flatten())

        f_cov = lambda x: compute_drift_covariance(x, beta_cov)  # (N, D, D)
        f_sd = lambda x: jnp.sqrt(jnp.diagonal(f_cov(x), axis1=-2, axis2=-1))

        
    # beta_learned = guide.median(svi_result.params)["beta"]
    f_learned_base = lambda x: compute_drift(x=x, beta=beta_learned)
    f_learned = lambda x: adjust_rhs(x, f_learned_base(x), **adjust_rhs_kwargs)
    fig, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=f_learned,
        f_learned_sd=f_sd,
        # x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        # x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True,
    )
    wandb.log({"fig/true_vs_learned_drift": wandb.Image(fig)})
    plt.close(fig)
    
    # log the learned weights
    wandb.log({"weights/learned/beta": beta_learned})
    
    # Log the learned weights as line plots
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in range(beta_learned.shape[1]):
        ax.plot(beta_learned[:, d], label=f"dim {d}")
        ax.plot(beta_supervised[:, d], '--', label=f"dim {d} (supervised)")
    ax.legend()
    ax.set_title("Beta weights (per output dim)")
    ax.set_xlabel("basis index")
    ax.set_ylabel("weight value")
    ax.set_yscale("symlog")
    wandb.log({"fig/learned_beta_lines": wandb.Image(fig)})
    plt.close(fig)

    # Log the absolute error in weights
    wandb.log({"metrics/learned/rmse_error": float(rmse)})

    # Log the trajectory from filtering
    if cfg.log_filtering:
        print("Logging trajectories from filtering with learned model...")
        # Trajectories use "time" as their x-axis
        wandb.define_metric("filtering/learned/pred_mean_*", step_metric="time")
        wandb.define_metric("filtering/learned/filtered_mean_*", step_metric="time")
        for t in range(learned_data["predicted_means"].shape[1]):
            log_dict = {"time": t}
            for d in range(state_dim):
                log_dict[f"filtering/learned/pred_mean_dim{d}"] = float(learned_data["predicted_means"][0, t, d])
                log_dict[f"filtering/learned/filtered_mean_dim{d}"] = float(learned_data["filtered_means"][0, t, d])
            # Log the current time step
            wandb.log(log_dict)
        fig = plot_particle_diagnostics(
            t_emissions=t_emissions.squeeze(),
            x_ens_filtered=learned_data["x_ens_filtered"][0],   # shape (T, N, D)
            x_ens_predicted=learned_data["x_ens_predicted"][0], # shape (T, N, D)
            observations=emissions_obs,
            figsize=(12, 8),
            start_idx=0,
            stop_idx=20,
        )
        wandb.log({"fig/learned/ensembles_0to20": wandb.Image(fig)})
        plt.close(fig)

    run.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Randomization
    parser.add_argument("--seed", type=int, default=0)

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--use_lr_scheduler", type=int, default=0) # 1 for True, 0 for False
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--epochs_per_step", type=int, default=200)
    parser.add_argument("--clip_norm", type=float, default=0.0)  # 0.0 for no clipping, or a float value to clip.

    # Prior parameters
    parser.add_argument("--nu", type=float, default=1.5) # Matern smoothness parameter

    # NOTE: van der pol has x1dot \in [-4, 4], whereas x2dot \in [-50, 50] roughly.
    # Would be nice to have different variances per dimension, but for now we use the same.
    # Numpyro doesn't offer this natively, but it is just factoring out the alpha, so nothing too hard.
    # parser.add_argument("--alpha", type=float, default=1.0) # Matern variance parameter
    parser.add_argument("--alpha", type=float, nargs='+', default=[2.0**2, 10.0**2])
    parser.add_argument("--length_scale", type=float, default=1.0) # Matern length scale parameter
    parser.add_argument("--ell_box", type=float, nargs='+', default=[8.0, 8.0]) # size of the spectral box (one per input dimension)
    parser.add_argument("--m", type=int, default=20) # number of basis functions

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=25.0) # initial state covariance (times identity)
    parser.add_argument("--emission_cov", type=float, default=0.01) # emission noise covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=100) # final time
    parser.add_argument("--dt", type=float, default=0.01) # time step size
    parser.add_argument("--emission_dim", type=int, default=2) # observation dimension (default is to observe the first "emission_dim" states)

    # Filtering algorithm hyperparameters
    parser.add_argument("--filter_type", type=str, default="EnKF", choices=["EnKF", "EKF", "UKF"])  # Type of filter to use
    parser.add_argument("--N_particles", type=int, default=25)  # Number of particles for EnKF
    parser.add_argument("--state_order", type=str, default="first", choices=["discrete", "zeroth", "first", "second"])  # Order of the Taylor expansion for EKF/UKF
    parser.add_argument("--diffeqsolve_max_steps", type=int, default=100)  # Max steps for the ODE solver between filtered timesteps
    parser.add_argument("--cov_rescaling", type=float, default=1.0)  # Covariance rescaling factor for EnKF
    parser.add_argument("--inflation_delta", type=float, default=0.0)  # Inflation delta for EnKF

    # Diagnostics
    parser.add_argument("--n_prior_samples", type=int, default=2)

    # Logging
    parser.add_argument("--log_filtering", type=int, default=0)
    
    # Parse arguments
    args = parser.parse_args()
    
        
    main(**vars(args))
