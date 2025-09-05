import sys
sys.path.append("../")
sys.path.append("../..")

import numpy as np
import jax
# Make sure everything is 64bit (should prevent NaNs, but can be slow)
# Best to set this before importing jax.numpy or numpyro
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    SVI,
    Trace_ELBO,
    init_to_value,
    Predictive,
)
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta

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
# Main training entrypoint
# ------------------------
def main(**cfg):
    # Initialize a new W&B run.
    run = wandb.init(config=cfg, project=cfg["project"])
    cfg = wandb.config

    # Global settings
    state_dim, emission_dim = cfg.state_dim, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    def true_drift(x):
        sigma = 10.0; rho = 28.0; beta = 8.0/3.0
        return jnp.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    # NumPyro model
    def model(t_emissions, emissions=None, **kwargs):
        
        if "drift" in kwargs and kwargs["drift"] is not None:
            # this is to allow passing known drifts during predictive calls
            drift = lambda x: adjust_rhs(x, kwargs["drift"](x))
        else:
            # Sample or use provided parameters
            drift_base = make_bayesian_drift(
                state_dim=cfg.state_dim,
                hidden_dims=cfg.hidden_dims,   # e.g., [64, 64]
                prior="uniform",
                prior_scale=10.0,
            ) # note that this drift takes **kwargs to allow passing known drift-params during predictive calls.
            drift = lambda x: adjust_rhs(x, drift_base(x, **kwargs))

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
            emission_cov=jnp.eye(emission_dim),
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
    true_values = {
        "drift": true_drift,
    }

    # Typically you keep these fixed/known during learning
    # know everything except "weights" # it is empty here, but useful when there are more params
    known_values = {key: value for key, value in true_values.items() if key not in ["drift"]}
    
    # Generate synthetic emissions
    t_emissions = jnp.arange(start=0.0, stop=cfg.T, step=cfg.dt).reshape(-1, 1)
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
    init_params_dict = make_nn_init_dict(next(keys), cfg.state_dim, cfg.hidden_dims)
    # Now run SVI for MAP inference!
    guide = AutoDelta(model, init_loc_fn=init_to_value(values=init_params_dict))
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

    # Generate a long sequence of emissions from the learned model
    long_t_emissions = jnp.arange(start=0.0, stop=10*cfg.T, step=cfg.dt).reshape(-1, 1)
    long_learned_data = predictive_learned(next(keys), t_emissions=long_t_emissions, **known_values) # no emissions provided here, so it will sample them
    
    fig = plot_traj_kde({"Observed traj": emissions_obs, "Long learned traj": long_learned_data["emissions"].squeeze(0)})
    wandb.log({"figures/emissions_kde": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_traj_kde({"True states": sim_data["states"].squeeze(0), "Learned states": long_learned_data["states"].squeeze(0)})
    wandb.log({"figures/states_kde": wandb.Image(fig)})
    plt.close(fig)

    # Finish the W&B run
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
    parser.add_argument("--epochs_per_step", type=int, default=1500)
    parser.add_argument("--clip_norm", type=float, default=1.0)  # 0.0 for no clipping, or a float value to clip.

    # Prior parameters
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[100])  # hidden layer sizes for the NN drift

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=100.0) # initial state covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=40) # final time
    parser.add_argument("--dt", type=float, default=0.01) # time step size
    parser.add_argument("--emission_dim", type=int, default=3) # observation dimension (default is to observe the first "emission_dim" states)
    parser.add_argument("--state_dim", type=int, choices=[3], default=3) # state dimension (only 3 is supported now)

    # Filtering algorithm hyperparameters
    parser.add_argument("--filter_type", type=str, default="EnKF", choices=["EnKF", "EKF", "UKF"])  # Type of filter to use
    parser.add_argument("--N_particles", type=int, default=25)  # Number of particles for EnKF
    parser.add_argument("--state_order", type=str, default="first", choices=["discrete", "zeroth", "first", "second"])  # Order of the Taylor expansion for EKF/UKF
    parser.add_argument("--diffeqsolve_max_steps", type=int, default=100)  # Max steps for the ODE solver between filtered timesteps
    parser.add_argument("--cov_rescaling", type=float, default=1.0)  # Covariance rescaling factor for EnKF
    parser.add_argument("--inflation_delta", type=float, default=0.0)  # Inflation delta for EnKF
    
    # Wandb settings
    parser.add_argument("--project", type=str, default="l63_nn")

    # Parse arguments
    args = parser.parse_args()
    
        
    main(**vars(args))
