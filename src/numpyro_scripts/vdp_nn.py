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
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoMultivariateNormal, AutoLaplaceApproximation
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
    state_dim, emission_dim = cfg.state_dim, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    # Build polynomial exponents (up to quadratic)
    def true_drift(x):
        return jnp.array([x[1], 1.0*(1 - x[0]**2)*x[1] - x[0]])

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    NN_HYPERPARAMS = {
        'state_dim': cfg.state_dim,
        'hidden_dims': cfg.hidden_dims,
        'prior': 'uniform',
        'prior_scale': 10.0,
    }
    
    def model(t_emissions, emissions=None, **kwargs):

        if "drift" in kwargs and kwargs["drift"] is not None:
            # this is to allow passing known drifts during predictive calls
            drift = lambda x: adjust_rhs(x, kwargs["drift"](x), **adjust_rhs_kwargs)
        else:
            # Sample or use provided parameters
            drift_base = make_bayesian_nn_drift(**NN_HYPERPARAMS, **kwargs)
            # note that this drift takes **kwargs to allow passing known drift-params during predictive calls.
            # scale_drift_output = jnp.array(cfg.scale_drift_output)
            drift = lambda x: adjust_rhs(x, 
                                        drift_base(x),
                                         **adjust_rhs_kwargs)

        # Build the model and its parameters
        cdnlgssm = ContDiscreteNonlinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim)
        params = build_params(
            state_dim=state_dim,
            emission_dim=emission_dim,
            initial_mean=jnp.zeros(state_dim), 
            initial_cov=cfg.initial_cov*jnp.eye(state_dim), # warning: choosing it too small, even if "true", can lead to numerical issues with the filter.
            drift=drift,  # including **kwargs allows passing known params during predictive calls.
            diffusion_coeff=cfg.diffusion_coeff * jnp.eye(state_dim),
            diffusion_cov=jnp.eye(state_dim),
            emission_function=lambda x: H @ x,
            emission_cov=cfg.emission_cov*jnp.eye(emission_dim),
        )
        # Sample emissions if not provided
        if emissions is None:
            states, emissions = cdnlgssm.sample(params=params, num_timesteps=t_emissions.shape[0], key=numpyro.prng_key(), t_emissions=t_emissions, transition_type="path")
            numpyro.deterministic("states", states); numpyro.deterministic("emissions", emissions)

        # Compute (approximate) marginal log likelihood via filtering
        filtered = cdnlgssm_filter(params=params, emissions=emissions, t_emissions=t_emissions, filter_hyperparams=FILTER_HYPERPARAMS)
        ll = filtered.marginal_loglik

        # Custom-compute and store the log prior (used only for diagnostics, not for learning)
        lp = 0 # currently uniform prior on weights/biases, so while not 0, it is constant across models

        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)

        # Use the filtering based log likelihood for the model's inference.
        numpyro.factor("log_likelihood", ll)
    
    # Define the true parameters for the model and its data generation
    true_values = {"drift": true_drift}
    # Typically you keep these fixed/known during learning
    # know everything except "weights" # it is empty here, but useful when there are more params
    known_values = {key: value for key, value in true_values.items() if key not in ["drift"]}

    
    # Generate synthetic emissions
    t_emissions = sample_t_emissions(start=0.0, stop=cfg.T, dt=cfg.dt, regular=cfg.t_regular, key=next(keys))
    sim_data = Predictive(model, num_samples=1)(next(keys), t_emissions=t_emissions, **true_values)
    # Extract emissions from the simulation data
    emissions_obs = sim_data["emissions"].squeeze(0)
        
    # Log metrics
    wandb.log({
        "metrics/true/neg_log_prior": float(sim_data["neg_log_prior"].item()),
        "metrics/true/neg_log_lik": float(sim_data["neg_log_likelihood"].item()),
        "metrics/true/neg_log_joint": float(sim_data["neg_log_joint"].item()),
    })
    # Print the neg-log-likelihood values
    print(f"True model's neg_log_likelihood from filtering: {float(sim_data['neg_log_likelihood'].item())}")

    # SVI
    optimizer = make_optimizer(
        initial_learning_rate=cfg.init_lr,
        decay_factor=cfg.decay_rate,
        epochs_per_step=cfg.epochs_per_step,
        num_epochs=cfg.num_epochs,
        use_lr_scheduler=cfg.use_lr_scheduler,
        clip_norm=cfg.clip_norm if cfg.clip_norm > 0.0 else None,
    )

    # Initialize at zero
    init_params_dict = make_nn_init_dict(next(keys), cfg.state_dim, cfg.hidden_dims, eps=1e-3)
    # Now run SVI for MAP inference!
    guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=init_params_dict))

    # plot the initial drift
    f_init_base = make_bayesian_nn_drift(**NN_HYPERPARAMS, **init_params_dict)
    f_init = lambda x: adjust_rhs(x, f_init_base(x), **adjust_rhs_kwargs)  # (..., state_dim)
    fig = plot_drift_field(
        f_true=true_drift,
        f_learned=f_init,
        # x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        # x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
    )
    # Log the initial drift figure
    wandb.log({"figures/initial_drift": wandb.Image(fig)})
    plt.close(fig)
    
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(next(keys), num_steps=cfg.num_epochs, t_emissions=t_emissions, emissions=emissions_obs, **known_values)

    # Log training curve
    # Loss curve uses "epoch" as its x-axis
    wandb.define_metric("svi/loss", step_metric="epoch")
    for step, loss in enumerate(svi_result.losses):
        wandb.log({"epoch": step, "svi/loss": float(loss)})

    # Log posterior predictive
    predictive_learned = Predictive(model, guide=guide, params=guide.median(svi_result.params), num_samples=1)
    learned_data = predictive_learned(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    # Log metrics
    wandb.log({
        "metrics/learned/neg_log_prior": float(learned_data["neg_log_prior"].item()),
        "metrics/learned/neg_log_lik": float(learned_data["neg_log_likelihood"].item()),
        "metrics/learned/neg_log_joint": float(learned_data["neg_log_joint"].item()),
    })

    #### Now, let's quantify uncertainty in the learned drift function
    # scale_drift_output = jnp.array(cfg.scale_drift_output)
    if isinstance(guide, AutoDelta):
        params_sample = guide.median(svi_result.params)
        f = make_bayesian_nn_drift(**NN_HYPERPARAMS, **params_sample)
        f_learned = lambda x: adjust_rhs(x, f(x), **adjust_rhs_kwargs)  # (..., state_dim)
        f_learned_sd = None
    else:
        # Sample from the posterior over weights/biases
        params_samples = guide.sample_posterior(params=svi_result.params, sample_shape=(cfg.n_uq_samples,), rng_key=next(keys))
        # Now, compute the mean and sds of the sampled functions
        # Be sure to wrap each sampled drift in adjust_rhs to keep it consistent with the learned drift
        def f_learned_loc_sd(x):
            def one(p):
                f = make_bayesian_nn_drift(**NN_HYPERPARAMS, **p)
                return adjust_rhs(x, f(x), **adjust_rhs_kwargs)  # (..., state_dim)
            vals = jax.vmap(one)(params_samples)                   # (N, ..., state_dim)
            mu = jnp.mean(vals, axis=0)                        # (..., state_dim)
            sd = jnp.std(vals, axis=0)                          # (..., state_dim)
            return mu, sd
        f_learned_sd = lambda x: f_learned_loc_sd(x)[1]
        f_learned = lambda x: f_learned_loc_sd(x)[0]

    fig, rmse = plot_drift_field(
        f_true=true_drift,
        f_learned=f_learned,
        f_learned_sd=f_learned_sd,
        # x1_range=(-cfg.ell_box[0]/2, cfg.ell_box[0]/2),
        # x2_range=(-cfg.ell_box[1]/2, cfg.ell_box[1]/2),
        num_points=50,
        return_rmse=True,
    )
    wandb.log({"figures/true_vs_learned_drift": wandb.Image(fig)})
    plt.close(fig)

    # Log the absolute error in weights
    wandb.log({"metrics/learned/rmse_error": float(rmse)})

    # Generate a long sequence of emissions from the learned and true models
    print("Generating long trajectory from learned model...")
    long_t_emissions = jnp.arange(start=0.0, stop=10*cfg.T, step=cfg.dt).reshape(-1, 1)
    long_learned_data = predictive_learned(next(keys), t_emissions=long_t_emissions, **known_values) # no emissions provided here, so it will sample them
    print("Generating long trajectory from true model...")
    long_true_data = Predictive(model, num_samples=1)(next(keys), t_emissions=long_t_emissions, **true_values)
    burnin_frac = 0.5
    burnin_idx = int(burnin_frac * long_t_emissions.shape[0])
    fig = plot_traj_kde({"Observed traj": long_true_data["emissions"].squeeze(0)[burnin_idx:], "Long learned traj": long_learned_data["emissions"].squeeze(0)[burnin_idx:]})
    wandb.log({"figures/emissions_kde": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_traj_kde({"True states": long_true_data["states"].squeeze(0)[burnin_idx:], "Learned states": long_learned_data["states"].squeeze(0)[burnin_idx:]})
    wandb.log({"figures/states_kde": wandb.Image(fig)})
    plt.close(fig)


    run.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Randomization
    parser.add_argument("--seed", type=int, default=0)

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=5e-3)
    parser.add_argument("--use_lr_scheduler", type=int, default=0) # 1 for True, 0 for False
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--epochs_per_step", type=int, default=200)
    parser.add_argument("--clip_norm", type=float, default=1.0)  # 0.0 for no clipping, or a float value to clip.

    # Prior parameters
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[50])  # hidden layer sizes for the NN drift
    # parser.add_argument("--scale_drift_output", type=float, nargs='+', default=[1.0, 1.0])  # signal scaling for the NN drift

    # UQ parameters
    parser.add_argument("--n_uq_samples", type=int, default=100)  #

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=25.0) # initial state covariance (times identity)
    parser.add_argument("--emission_cov", type=float, default=0.01) # emission noise covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=100) # final time
    parser.add_argument("--dt", type=float, default=0.01) # time step size
    parser.add_argument("--t_regular", type=int, default=1) # 1 for True (regular sampling), 0 for False (uniform random sampling)
    parser.add_argument("--state_dim", type=int, default=2, choices=[2]) # state dimension (it is 2 for VDP)
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

    # Wandb settings
    parser.add_argument("--project", type=str, default="vdp-nn-test")  # Wandb project name
    parser.add_argument("--run_name", type=str, default=None) # Allows you to custom-specify wandb run name (else uses default)
    parser.add_argument("--dir", type=str, default=None)  # Allows you to custom-specify wandb directory (else uses default)

    # Parse arguments
    args = parser.parse_args()
    
        
    main(**vars(args))
