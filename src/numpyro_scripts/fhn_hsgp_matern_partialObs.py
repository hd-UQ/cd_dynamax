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
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    SVI, Trace_ELBO, 
    init_to_median, init_to_value,
    Predictive,
)
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoMultivariateNormal
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.hsgp.approximation import (
    eigenfunctions,
    diag_spectral_density_matern,
)
from jax.scipy.linalg import cho_solve, cho_factor

from continuous_discrete_nonlinear_gaussian_ssm import (
    cdnlgssm_filter, ContDiscreteNonlinearGaussianSSM, 
)

from numpyro_extension import build_params
from utils.diffrax_utils import adjust_rhs
from utils.optimize_utils import make_optimizer
from utils.simulation_utils import make_key_sequence
import wandb
from _utils import *

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
    run = wandb.init(config=cfg, project=cfg["project"], name=cfg["run_name"], dir=cfg["dir"])

    # After wandb.init(), wandb.config now holds the *final merged config*:
    #   argparse defaults + CLI overrides + sweep overrides (if any).
    # This object behaves like a dict, but is also a special W&B container
    # that tracks hyperparameters and makes them visible in the UI.
    cfg = wandb.config

    # Global settings
    state_dim, emission_dim = 2, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    # Build FitzHugh-Nagumo true drift
    _a, _b, _c, _I = 0.08, 0.7, 0.8, 0.5
    # Equation 1: v' = v - (1/3)v^3 - w + I
    # Equation 2: w' = a*(v + b - c w) = a*v + a*b - a*c*w
    def true_drift(x):
        v, w = x[0], x[1]
        dv = v - (1/3)*v**3 - w + _I
        dw = _a* (v + _b - _c*w)
        return jnp.array([dv, dw])
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

    # Define the optimizer
    optimizer = make_optimizer(
        initial_learning_rate=cfg.init_lr,
        decay_factor=cfg.decay_rate,
        epochs_per_step=cfg.epochs_per_step,
        num_epochs=cfg.num_epochs,
        use_lr_scheduler=cfg.use_lr_scheduler,
        clip_norm=cfg.clip_norm if cfg.clip_norm > 0.0 else None,
    )

    # NumPyro model
    def model(t_emissions, emissions=None, supervised=False, state_prior=cfg.state_prior, **kwargs):
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
        
        if supervised:
            X_train = emissions
            Y_train = jax.vmap(true_drift)(X_train)
            Y_hat = jax.vmap(drift)(X_train)
            # Assume small observation noise for supervised fit
            # sigma2 = 1
            # obs_cov = jnp.eye(emission_dim)
            obs_cov = jnp.diag(jnp.array([1.0**2, 0.2**2]))  # to avoid numerical issues due to very small noise
            # mismatch between drift(X_train) and Y_train
            numpyro.sample("obs_supervised", dist.MultivariateNormal(Y_hat, obs_cov).to_event(1), obs=Y_train)

            # calculate the log-likelihood of the supervised data under the model
            ll = dist.MultivariateNormal(Y_hat, obs_cov).log_prob(Y_train).sum()
            numpyro.deterministic("neg_log_prior", -lp)
            numpyro.deterministic("neg_log_likelihood", -ll)
            numpyro.deterministic("neg_log_joint", -ll - lp)

            return

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

        if state_prior:
            # State means: [-0.20176842  0.53057752]
            # State stds: [1.42128004 0.57646959]
            lp_mean, lp_sd, mean_t, sd_t = moment_matching_prior(
                filtered.filtered_means,
                mu_0=jnp.array([-0.20176842, 0.53057752]),
                tau_0=0.1,
                mu_1=jnp.array([1.42128004, 0.57646959]),
                tau_1=0.1,
            )
            lp += lp_mean + lp_sd
            numpyro.deterministic("state_mean", mean_t)
            numpyro.deterministic("state_sd", sd_t)
            numpyro.deterministic("neg_log_prior_state_mean", -lp_mean)
            numpyro.deterministic("neg_log_prior_state_sd", -lp_sd)
        
        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)
        numpyro.factor("log_likelihood", ll)

    # Define the true parameters for the model and its data generation
    true_values = {
        "drift": true_drift,  # <— include true weights
    }

    # Initialize the HSGP weights to zero (prior mean) and plot the initial mean drift
    beta_init = jnp.zeros((MSTAR, state_dim))
    print("Neg-log-prior of beta_init:", -float(dist.Normal(0,1).log_prob(beta_init).sum().item()))
    f_init_base = lambda x: compute_drift(x=x, beta=beta_init)
    f_init = lambda x: adjust_rhs(x, f_init_base(x), **adjust_rhs_kwargs)
    fig = plot_drift_field(
        f_true=true_drift,
        f_learned=f_init,
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50
    )
    # save the initial drift figure
    wandb.log({"fig/true_vs_initial_drift": wandb.Image(fig)})
    plt.close(fig)

    # Fit the GP directly to the true drift (for comparison) either exactly (kernel trick) or with HSGP (basis regression)
    gp_predict, hsgp_predict, beta_supervised, (X_train, Y_train) = supervised_fit_drift(true_drift, sigma2=cfg.emission_cov)
    fig_exact_gp_supervised, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=gp_predict,
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True
    )
    print(f"RMSE of exact GP drift fit to true drift: {rmse:.4f}")
    wandb.log({"fig/exact_gp_supervised": wandb.Image(fig_exact_gp_supervised)})
    plt.close(fig_exact_gp_supervised)
    
    fig_hsgp_supervised, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=hsgp_predict,
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True
    )
    print(f"RMSE of HSGP drift fit to true drift: {rmse:.4f}")
    wandb.log({"fig/hsgp_cho_supervised": wandb.Image(fig_hsgp_supervised)})
    plt.close(fig_hsgp_supervised)  # too many colorbars otherwise


    # Values known to the model during training
    known_values = {}
    
    # Generate synthetic emissions
    t_emissions = jnp.arange(start=0.0, stop=cfg.T, step=cfg.dt).reshape(-1, 1)
    sim_data = Predictive(model, num_samples=1)(next(keys), t_emissions=t_emissions, **true_values)
    # Extract emissions from the simulation data
    emissions_obs = sim_data["emissions"].squeeze(0)
    
    fig = plot_simulated_data(
        t=t_emissions.squeeze(),
        states=sim_data["states"].squeeze(0),
        emissions=emissions_obs,
    )
    wandb.log({f"fig/simulated_data_trajectory": wandb.Image(fig)})
    plt.close(fig)
    
    # print mean and std of the states
    state_means = jnp.mean(sim_data['states'], axis=(0,1)) # (state_dim,)
    state_stds = jnp.std(sim_data['states'], axis=(0,1))   # (state_dim,)
    print(f"State means: {state_means}")
    print(f"State stds: {state_stds}")

    # Log metrics
    wandb.log({
        "metrics/true/neg_log_prior": float(sim_data["neg_log_prior"].item()),
        "metrics/true/neg_log_lik": float(sim_data["neg_log_likelihood"].item()),
        "metrics/true/neg_log_joint": float(sim_data["neg_log_joint"].item()),
    })
    # Print the neg-log-likelihood values
    print(f"True model's neg_log_likelihood from filtering: {float(sim_data['neg_log_likelihood'].item())}")

    # Next, check the model at beta_supervised
    print("Checking supervised-fit predictive...")
    supervised_predictive = Predictive(model, num_samples=1)
    supervised_data = supervised_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, beta=beta_supervised)
    print(f"Supervised-fit model's neg_log_likelihood from filtering: {float(supervised_data['neg_log_likelihood'].item())}")
    print(f"Supervised-fit model's neg_log_prior: {float(supervised_data['neg_log_prior'].item())}")
    print(f"Supervised-fit model's neg_log_joint: {float(supervised_data['neg_log_joint'].item())}")
    # Log metrics
    wandb.log({
        "metrics/supervised/neg_log_prior": float(supervised_data["neg_log_prior"].item()),
        "metrics/supervised/neg_log_lik": float(supervised_data["neg_log_likelihood"].item()),
        "metrics/supervised/neg_log_joint": float(supervised_data["neg_log_joint"].item()),
    })
    
    
    # Fit the drift in supervised mode, but use SVI + NUTS now
    print("Fitting drift in supervised mode using SVI + NUTS...")
    supervised_guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values={"beta": beta_init}))
    svi_supervised = SVI(model, supervised_guide, optimizer, loss=Trace_ELBO())
    print("Warning: using supervised=True in the model for supervised fitting, and feeding it simulated states as emissions.")
    # Run SVI
    svi_result_supervised = svi_supervised.run(next(keys), num_steps=cfg.num_epochs, t_emissions=t_emissions, emissions=sim_data["states"].squeeze(0), supervised=True, **known_values)
    # Log training curve
    wandb.define_metric("svi/supervised/loss", step_metric="epoch")
    for step, loss in enumerate(svi_result_supervised.losses):
        wandb.log({"epoch": step, "svi/supervised/loss": float(loss)})
    predictive_learned_super_svi = Predictive(model, guide=supervised_guide, params=supervised_guide.median(svi_result_supervised.params), num_samples=1)
    learned_data_super_svi = predictive_learned_super_svi(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    # Log metrics---computed using emissions_obs, not states, so comparable to the "true" model above.
    wandb.log({
        "metrics/svi/supervised/neg_log_prior": float(learned_data_super_svi["neg_log_prior"].item()),
        "metrics/svi/supervised/neg_log_lik": float(learned_data_super_svi["neg_log_likelihood"].item()),
        "metrics/svi/supervised/neg_log_joint": float(learned_data_super_svi["neg_log_joint"].item()),
    })
    print(f"Supervised-SVI-fit model's neg_log_prior: {float(learned_data_super_svi['neg_log_prior'].item())}")
    
    # Sample from the supervised SVI posterior and plot the learned drift
    supervised_beta_posterior = supervised_guide.get_posterior(svi_result_supervised.params)
    supervised_beta_learned_svi = supervised_beta_posterior.mean.reshape(MSTAR, state_dim)
    if hasattr(supervised_beta_posterior, "covariance_matrix"):
        # e.g. AutoMultivariateNormal
        supervised_beta_cov = supervised_beta_posterior.covariance_matrix
    else:
        # AutoDiagonalNormal
        supervised_beta_cov = jnp.diag(supervised_beta_posterior.variance.flatten())

    f_cov_supervised = lambda x: compute_drift_covariance(x, supervised_beta_cov)  # (N, D, D)
    f_sd_supervised = lambda x: jnp.sqrt(jnp.diagonal(f_cov_supervised(x), axis1=-2, axis2=-1))

    # beta_learned = guide.median(svi_result.params)["beta"]
    f_learned_base_sup = lambda x: compute_drift(x=x, beta=supervised_beta_learned_svi)
    f_learned_sup = lambda x: adjust_rhs(x, f_learned_base_sup(x), **adjust_rhs_kwargs)
    fig = plot_drift_field(
        f_true=true_drift,
        f_learned=f_learned_sup,
        f_learned_sd=f_sd_supervised,
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
    )
    wandb.log({"fig/supervised/hsgp_svi": wandb.Image(fig)})
    plt.close(fig)

    # Run NUTS initialized at median VI solution
    nuts_kernel = NUTS(model, 
                       init_strategy=init_to_value(values={"beta": supervised_guide.median(svi_result_supervised.params)["beta"]}),
                       max_tree_depth=cfg.max_tree_depth)
    mcmc = MCMC(nuts_kernel, num_warmup=cfg.nuts_warmup, num_samples=cfg.nuts_samples, num_chains=1)
    print("Again, warning, using supervised=True w/ NUTS in the model for supervised fitting, and feeding it simulated states as emissions.")
    mcmc.run(next(keys), t_emissions=t_emissions, emissions=sim_data["states"].squeeze(0), supervised=True, **known_values)
    mcmc_samples = mcmc.get_samples()
    print(mcmc.print_summary())
    
    # Use the ensemble of MCMC samples to compute a mean and sd for the drift
    beta_samples_mcmc = mcmc_samples["beta"]  # (num_samples, MSTAR, D)
    def f_learned_loc_sd(x):
        def one(p):
            # p should be (MSTAR, D)
            fx = compute_drift(x, p)  # (..., state_dim)
            return adjust_rhs(x, fx, **adjust_rhs_kwargs)  # (..., state_dim)
        vals = jax.vmap(one)(beta_samples_mcmc)                   # (N, ..., state_dim)
        mu = jnp.mean(vals, axis=0)                        # (..., state_dim)
        sd = jnp.std(vals, axis=0)                          # (..., state_dim)
        return mu, sd
    f_learned_sd = lambda x: f_learned_loc_sd(x)[1]
    f_learned = lambda x: f_learned_loc_sd(x)[0]

    fig, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=f_learned,
        f_learned_sd=f_learned_sd,
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True,
    )
    wandb.log({"fig/supervised/hsgp_NUTS": wandb.Image(fig)})
    plt.close(fig)

    # Log the learned weights as line plots
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in range(supervised_beta_learned_svi.shape[1]):
        # plot the mean of weights for MCMC samples
        ax.plot(jnp.mean(beta_samples_mcmc[:, :, d], axis=0), label=f"dim {d} (supervised MCMC mean)")
        ax.plot(supervised_beta_learned_svi[:, d], ':', label=f"dim {d} (supervised SVI mean)")
        ax.plot(beta_supervised[:, d], '--', label=f"dim {d} (supervised chol MAP)")
    ax.legend()
    ax.set_title("Beta weights (per output dim)")
    ax.set_xlabel("basis index")
    ax.set_ylabel("weight value")
    ax.set_yscale("symlog")
    wandb.log({"fig/supervised/mean_beta": wandb.Image(fig)})
    plt.close(fig)

    # Forest plot
    for max_params in [10, 50, 200]:
        fig = plot_forest(mcmc_samples, param_name="beta", max_params=max_params)
        wandb.log({f"fig/supervised/hsgp_NUTS_forest_{max_params}": wandb.Image(fig)})
        plt.close(fig)

        # Violin plots
        fig = plot_violin(mcmc_samples, param_name="beta", max_params=max_params)
        wandb.log({f"fig/supervised/hsgp_NUTS_violin_{max_params}": wandb.Image(fig)})
        plt.close(fig)

        # Correlation heatmap
        fig = plot_correlation_heatmap(mcmc_samples, param_name="beta", max_params=max_params)
        wandb.log({f"fig/supervised/hsgp_NUTS_corr_{max_params}": wandb.Image(fig)})
        plt.close(fig)

    # PCA scatter
    fig = plot_pca_scatter(mcmc_samples, param_name="beta")
    wandb.log({"fig/supervised/hsgp_NUTS_pca": wandb.Image(fig)})
    plt.close(fig)

    ################
    # Now, fit the drift in filtered mode using SVI, then NUTS
    ################
    print("Fitting drift in filtered mode using SVI + NUTS...")
    
    # First, check feasibility of the initialization
    # guide = AutoDelta(model, init_loc_fn=init_to_value(values={"beta": beta_init}))
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values={"beta": beta_init}))

    # Run model in predictive mode
    print("Checking initialization predictive...")
    init_predictive = Predictive(model, num_samples=1)
    init_data = init_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, beta=beta_init)

    print(f"Initialization neg_log_likelihood from filtering: {float(init_data['neg_log_likelihood'].item())}")
    if jnp.isnan(init_data['neg_log_likelihood']):
        raise ValueError("NaN value found in initialization neg_log_likelihood. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(next(keys), num_steps=cfg.num_epochs, t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    # Log training curve
    # Loss curve uses "epoch" as its x-axis
    wandb.define_metric("svi/filtered/loss", step_metric="epoch")
    for step, loss in enumerate(svi_result.losses):
        wandb.log({"epoch": step, "svi/filtered/loss": float(loss)})

    # Log posterior predictive
    predictive_learned = Predictive(model, guide=guide, params=guide.median(svi_result.params), num_samples=1)
    learned_data = predictive_learned(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    # Log metrics
    wandb.log({
        "metrics/filtered/neg_log_prior": float(learned_data["neg_log_prior"].item()),
        "metrics/filtered/neg_log_lik": float(learned_data["neg_log_likelihood"].item()),
        "metrics/filtered/neg_log_joint": float(learned_data["neg_log_joint"].item()),
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
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True,
    )
    wandb.log({"fig/filtered/hsgp_svi": wandb.Image(fig)})
    plt.close(fig)
    
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
    wandb.log({"fig/mean_beta_svi": wandb.Image(fig)})
    plt.close(fig)

    # Run NUTS initialized at median VI solution
    nuts_kernel = NUTS(model, init_strategy=init_to_value(values={"beta": guide.median(svi_result.params)["beta"]}), max_tree_depth=cfg.max_tree_depth)
    mcmc = MCMC(nuts_kernel, num_warmup=cfg.nuts_warmup, num_samples=cfg.nuts_samples, num_chains=1)
    mcmc.run(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    mcmc_samples = mcmc.get_samples()
    print(mcmc.print_summary())

    # Use the ensemble of MCMC samples to compute a mean and sd for the drift
    beta_samples_mcmc = mcmc_samples["beta"]  # (num_samples, MSTAR, D)
    def f_learned_loc_sd(x):
        def one(p):
            # p should be (MSTAR, D)
            fx = compute_drift(x, p)  # (..., state_dim)
            return adjust_rhs(x, fx, **adjust_rhs_kwargs)  # (..., state_dim)
        vals = jax.vmap(one)(beta_samples_mcmc)                   # (N, ..., state_dim)
        mu = jnp.mean(vals, axis=0)                        # (..., state_dim)
        sd = jnp.std(vals, axis=0)                          # (..., state_dim)
        return mu, sd
    f_learned_sd = lambda x: f_learned_loc_sd(x)[1]
    f_learned = lambda x: f_learned_loc_sd(x)[0]

    fig, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=f_learned,
        f_learned_sd=f_learned_sd,
        x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True,
    )
    wandb.log({"fig/filtered/hsgp_NUTS": wandb.Image(fig)})
    plt.close(fig)

    # Log the learned weights as line plots
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in range(supervised_beta_learned_svi.shape[1]):
        # plot the mean of weights for MCMC samples
        ax.plot(jnp.mean(beta_samples_mcmc[:, :, d], axis=0), label=f"dim {d} (filtered MCMC mean)")
        ax.plot(supervised_beta_learned_svi[:, d], ':', label=f"dim {d} (supervised SVI mean)")
        ax.plot(beta_supervised[:, d], '--', label=f"dim {d} (supervised chol MAP)")
    ax.legend()
    ax.set_title("Beta weights (per output dim)")
    ax.set_xlabel("basis index")
    ax.set_ylabel("weight value")
    ax.set_yscale("symlog")
    wandb.log({"fig/filtered/mean_beta": wandb.Image(fig)})
    plt.close(fig)

    # Forest plot
    for max_params in [10, 50, 200]:
        fig = plot_forest(mcmc_samples, param_name="beta", max_params=max_params)
        wandb.log({f"fig/filtered/hsgp_NUTS_forest_{max_params}": wandb.Image(fig)})
        plt.close(fig)

        # Violin plots
        fig = plot_violin(mcmc_samples, param_name="beta", max_params=max_params)
        wandb.log({f"fig/filtered/hsgp_NUTS_violin_{max_params}": wandb.Image(fig)})
        plt.close(fig)

        # Correlation heatmap
        fig = plot_correlation_heatmap(mcmc_samples, param_name="beta", max_params=max_params)
        wandb.log({f"fig/filtered/hsgp_NUTS_corr_{max_params}": wandb.Image(fig)})
        plt.close(fig)

    # PCA scatter
    fig = plot_pca_scatter(mcmc_samples, param_name="beta")
    wandb.log({"fig/filtered/hsgp_NUTS_pca": wandb.Image(fig)})
    plt.close(fig)

    print("Completed all tasks.")
    run.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Randomization
    parser.add_argument("--seed", type=int, default=0)

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--use_lr_scheduler", type=int, default=0) # 1 for True, 0 for False
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--epochs_per_step", type=int, default=200)
    parser.add_argument("--clip_norm", type=float, default=0.0)  # 0.0 for no clipping, or a float value to clip.

    # MCMC settings
    parser.add_argument("--nuts_warmup", type=int, default=200)
    parser.add_argument("--nuts_samples", type=int, default=1000)
    parser.add_argument("--max_tree_depth", type=int, default=10)

    # Prior parameters
    parser.add_argument("--nu", type=float, default=1.5) # Matern smoothness parameter

    # NOTE: van der pol has x1dot \in [-4, 4], whereas x2dot \in [-50, 50] roughly.
    # Would be nice to have different variances per dimension, but for now we use the same.
    # Numpyro doesn't offer this natively, but it is just factoring out the alpha, so nothing too hard.
    # parser.add_argument("--alpha", type=float, default=1.0) # Matern variance parameter
    parser.add_argument("--alpha", type=float, nargs='+', default=[2.0**2, 0.2**2]) #2^2 and 0.2^2 are good for FHN...verified via SVI/MCMC fits in supervised mode.
    parser.add_argument("--length_scale", type=float, default=1.0) # Matern length scale parameter
    parser.add_argument("--ell_box", type=float, nargs='+', default=[4.0, 4.0]) # size of the spectral box (one per input dimension)
    parser.add_argument("--m", type=int, default=10) # number of basis functions (20 is a bit better for this problem)

    # Prior parameters for state moments (used if cfg.state_prior=True)
    parser.add_argument("--state_prior", type=int, default=1) # 1 for
    
    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=1.0) # initial state covariance (times identity)
    parser.add_argument("--emission_cov", type=float, default=0.01) # emission noise covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=100) # final time
    parser.add_argument("--dt", type=float, default=0.1) # time step size
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

    # Wandb settings
    parser.add_argument("--project", type=str, default="FHN-HSGP")
    parser.add_argument("--run_name", type=str, default=None) # Allows you to custom-specify wandb run name (else uses default)
    parser.add_argument("--dir", type=str, default=None)  # Allows you to custom-specify wandb directory (else uses default)

    # Test run
    parser.add_argument("--test", type=int, default=0) # 1 for True, 0 for False
    
    # Parse arguments
    args = parser.parse_args()

    if args.test:
        # Override some settings for a quick test run
        args.num_epochs = 10
        args.nuts_warmup = 10
        args.nuts_samples = 10
        args.max_tree_depth = 3
        if args.run_name is None:
            args.run_name = "test_run_deleteme"

    main(**vars(args))
