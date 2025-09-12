import sys
sys.path.append("../")
sys.path.append("../..")

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
    SVI,
    Trace_ELBO,
    init_to_value,
    init_to_median,
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
    run = wandb.init(config=cfg, project=cfg["project"], name=cfg["run_name"], dir=cfg["dir"])
    cfg = wandb.config

    # Global settings
    state_dim, emission_dim = cfg.state_dim, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    def stabliize(A):
        
        # Compute the singular values
        singular_vals = jnp.linalg.svd(A, compute_uv=False)
        spectral_norm = singular_vals.max()   # largest singular value
        # If the spectral norm is greater than max_spectral_norm, rescale A
        A_stable = jnp.where(spectral_norm > cfg.max_spectral_norm, A * (cfg.max_spectral_norm / spectral_norm), A)

        return A_stable

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    adjust_rhs_kwargs = {'lower_bound': cfg.state_bound_low,
                        'upper_bound': cfg.state_bound_high,
                        'lower_bound_derivative': cfg.derivative_clip_low,
                        'upper_bound_derivative': cfg.derivative_clip_high}
                         
    # NumPyro model
    def model(t_emissions, emissions=None, **kwargs):

        # normalize time to (T,1)
        if t_emissions.ndim == 1:
            t_emissions = t_emissions[:, None]
        elif t_emissions.shape[-1] != 1:
            t_emissions = t_emissions.reshape(-1, 1)
        T = t_emissions.shape[0]
        
        # A ~ U(-A_bound, A_bound) elementwise prior on the weights
        weights = get_or_sample("weights", dist.Uniform(-cfg.a, cfg.a).expand((state_dim, state_dim)).to_event(2), kwargs.get("weights"))
        bias = get_or_sample("bias", dist.Uniform(cfg.b_low, cfg.b_high).expand((state_dim,)).to_event(1), kwargs.get("bias"))
        # compute spectral norm of A, then rescale if needed
        # singular_vals = jnp.linalg.svd(A, compute_uv=False)
        # spectral_norm = singular_vals.max()   # largest singular value
        # A_learned = jnp.where(spectral_norm > cfg.max_spectral_norm, A * (cfg.max_spectral_norm / spectral_norm), A)
        weights_stable = stabliize(weights)

        # Build the drift function using the sampled/learned A
        drift_base = lambda x: weights_stable @ (x - bias)
        drift = lambda x: adjust_rhs(x, drift_base(x), **adjust_rhs_kwargs)

        # Custom-compute and store the log prior (used only for diagnostics, not for learning)
        lp = dist.Uniform(-cfg.a, cfg.a).log_prob(weights).sum()

        # Build the model and its parameters
        cdnlgssm = ContDiscreteNonlinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim) # observation matrix (observing the first "emission_dim" states)
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
            keys = jax.random.split(numpyro.prng_key(), cfg.N_trajectories)
            def _sample_one(k):
                states, ems = cdnlgssm.sample(params=params,
                                            num_timesteps=T,
                                            key=k,
                                            t_emissions=t_emissions,   # shared (T,1)
                                            transition_type="path")
                return states, ems
            states_B, ems_B = jax.vmap(_sample_one)(keys)         # (N,T,⋯)
            emissions = ems_B
            numpyro.deterministic("states", states_B)
            numpyro.deterministic("emissions", ems_B)
        else:
            if emissions.ndim == 2:      # (T,D) -> add N=1
                emissions = emissions[None, ...]
            assert emissions.ndim == 3 and emissions.shape[1] == T

        def _filter_one(ems_i):
            out = cdnlgssm_filter(params=params,
                                emissions=ems_i,                # (T,D)
                                t_emissions=t_emissions,        # shared (T,1)
                                filter_hyperparams=FILTER_HYPERPARAMS)
            return out.marginal_loglik

        # Vectorized over the batch of emissions; each is independent given the params
        ll_B = jax.vmap(_filter_one)(emissions)   # (N,)
        numpyro.deterministic("neg_log_likelihood_per_traj", -ll_B)

        # Use the sum (i.i.d. trajectories) as the model log-likelihood
        ll = ll_B.sum()                           # scalar

        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)

        # Use the filtering based log likelihood for the model's inference.
        numpyro.factor("log_likelihood", ll)
    
    # Define the true parameters for the model and its data generation
    # Sample from the prior to create a "true" model
    bias_true = jr.uniform(next(keys), shape=(state_dim), minval=1, maxval=10)
    
    omega = 2*jnp.pi/6.0
    alpha = -0.02
    weights_true = jnp.array([
        [ 0.06, 0.00, 0.00,   0.00,  0.00],
        [ 0.00,-0.06, 0.00,   0.00,  0.00],
        [ 0.00, 0.00, alpha, -omega, 0.00],
        [ 0.00, 0.00, omega,  alpha, 0.00],
        [ 0.00, 0.00, 0.00,   0.00, -0.02],
    ])

    weights_true *= 2.0

    true_values = {
        "weights": weights_true,
        "bias": bias_true,
    }

    # Typically you keep these fixed/known during learning
    # know everything except "weights" # it is empty here, but useful when there are more params
    known_values = {key: value for key, value in true_values.items() if key not in ["weights", "bias"]}

    # Generate synthetic emissions
    # All trajectories share the same observation times in this example
    t_emissions = sample_t_emissions(start=0.0, stop=cfg.T, dt=cfg.dt, regular=cfg.t_regular, key=next(keys)).reshape(-1, 1)  # (T,1)
    # Simulate data from the true model
    sim_data = Predictive(model, num_samples=1)(next(keys), t_emissions=t_emissions, **true_values)
    # Extract emissions from the simulation data
    emissions_obs = sim_data["emissions"].squeeze(0)

    # Plot the states and emissions for each trajectory
    for i in range(cfg.N_trajectories):
        fig = plot_simulated_data(t=t_emissions, states=sim_data["states"].squeeze(0)[i], emissions=emissions_obs[i])
        wandb.log({f"figures/true_states_traj{i}": wandb.Image(fig)})
        plt.close(fig)

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

    # Initialize at -1 * I (stable)
    # Now run SVI for MAP inference!
    guide = AutoDelta(model, init_loc_fn=init_to_value(values={
        "weights": -0.1 * jnp.eye(state_dim),
        "bias": jnp.ones(state_dim),
    }))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(next(keys), num_steps=cfg.num_epochs, t_emissions=t_emissions, emissions=emissions_obs, **known_values)

    # Log training curve
    # Loss curve uses "epoch" as its x-axis
    wandb.define_metric("svi/loss", step_metric="epoch")
    for step, loss in enumerate(svi_result.losses):
        wandb.log({"epoch": step, "svi/loss": float(loss)})

    # Plot the learned A matrix
    weights_learned = guide.median(svi_result.params)["weights"]
    fig = plot_coeff_heatmaps(weights_true, weights_learned)
    wandb.log({"figures/A_heatmaps": wandb.Image(fig)})
    plt.close(fig)

    bias_learned = guide.median(svi_result.params)["bias"]
    fig = plot_coeff_heatmaps(bias_true.reshape(-1,1), bias_learned.reshape(-1,1))
    wandb.log({"figures/bias_heatmaps": wandb.Image(fig)})
    plt.close(fig)

    # Log posterior predictive
    predictive_learned = Predictive(model, guide=guide, params=guide.median(svi_result.params), num_samples=1)
    learned_data = predictive_learned(next(keys), t_emissions=t_emissions, emissions=emissions_obs, store_filtered=True, **known_values)
    # Log metrics
    wandb.log({
        "metrics/learned/neg_log_prior": float(learned_data["neg_log_prior"].item()),
        "metrics/learned/neg_log_lik": float(learned_data["neg_log_likelihood"].item()),
        "metrics/learned/neg_log_joint": float(learned_data["neg_log_joint"].item()),
    })


    # Generate a long sequence of emissions from the learned and true models
    print("Generating long trajectory from learned model...")
    long_t_emissions = jnp.arange(start=0.0, stop=100*cfg.T, step=cfg.dt).reshape(-1, 1)
    long_learned_data = predictive_learned(next(keys), t_emissions=long_t_emissions, **known_values) # no emissions provided here, so it will sample them
    print("Generating long trajectory from true model...")
    long_true_data = Predictive(model, num_samples=1)(next(keys), t_emissions=long_t_emissions, **true_values)
    burnin_frac = 0.5
    burnin_idx = int(burnin_frac * long_t_emissions.shape[0])
    idx_traj = 0 # arbitary since they are i.i.d.
    fig = plot_traj_kde({"Observed traj": long_true_data["emissions"].squeeze(0)[idx_traj, burnin_idx:], "Long learned traj": long_learned_data["emissions"].squeeze(0)[idx_traj, burnin_idx:]})
    wandb.log({"figures/emissions_kde": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_traj_kde({"True states": long_true_data["states"].squeeze(0)[idx_traj, burnin_idx:], "Learned states": long_learned_data["states"].squeeze(0)[idx_traj, burnin_idx:]})
    wandb.log({"figures/states_kde": wandb.Image(fig)})
    plt.close(fig)

    print("Completed!")

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

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=1000.0) # initial state covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=3) # final time (hours)
    parser.add_argument("--dt", type=float, default=0.5) # time step size (minutes)
    parser.add_argument("--t_regular", type=int, default=1) # 1 for True (regular sampling), 0 for False (uniform random sampling)
    parser.add_argument("--emission_dim", type=int, default=5) # observation dimension (default is to observe the first "emission_dim" states)
    parser.add_argument("--state_dim", type=int, default=5) # state dimension (only 3 is supported now)
    parser.add_argument("--N_trajectories", type=int, default=30) # number of independent trajectories to simulate

    # Prior parameters
    parser.add_argument("--max_spectral_norm", type=float, default=200.0)
    parser.add_argument("--a", type=float, default=2.0) # bound for the uniform prior on the weights of A
    parser.add_argument("--b_low", type=float, default=0.1) # lower bound for the uniform prior on the bias term b
    parser.add_argument("--b_high", type=float, default=10.0) # upper bound for the uniform prior on the bias term b

    # Parameters for constraining the RHS of the drift function
    # these are useful to prevent numerical issues with the ODE solver
    # If state values exceed [state_bound_low, state_bound_high], the drift is automatically set to -x.
    # This protects against "BAD" choices of drift (A matrix) that can lead to numerical issues before the inference machinery can see that that it is BAD.
    parser.add_argument("--state_bound_low", type=float, default=-1000.0) # lower bound for state values (for clipping)
    parser.add_argument("--state_bound_high", type=float, default=1000.0) # upper bound for state values (for clipping)

    # Bounds for the derivative values (the drift output). 
    # This is to prevent extreme derivative values that can lead to numerical issues with the ODE solver.
    # Note that this is different from the state bounds above.
    # If the derivative exceeds [derivative_clip_low, derivative_clip_high], it is clipped to that range.
    parser.add_argument("--derivative_clip_low", type=float, default=-2000) # lower bound for derivative values (for clipping)
    parser.add_argument("--derivative_clip_high", type=float, default=2000) # upper bound for derivative values (for clipping)

    # Filtering algorithm hyperparameters
    parser.add_argument("--filter_type", type=str, default="EKF", choices=["EnKF", "EKF", "UKF"])  # Type of filter to use
    parser.add_argument("--state_order", type=str, default="zeroth", choices=["discrete", "zeroth", "first", "second"])  # Order of the Taylor expansion for EKF/UKF

    # parser.add_argument("--N_particles", type=int, default=25)  # Number of particles for EnKF
    parser.add_argument("--diffeqsolve_max_steps", type=int, default=100)  # Max steps for the ODE solver between filtered timesteps
    parser.add_argument("--cov_rescaling", type=float, default=1.0)  # Covariance rescaling factor for EnKF
    # parser.add_argument("--inflation_delta", type=float, default=0.0)  # Inflation delta for EnKF
    
    # Wandb settings
    parser.add_argument("--project", type=str, default="LinearGaussian_MultiTraj_example")
    parser.add_argument("--run_name", type=str, default=None) # Allows you to custom-specify wandb run name (else uses default)
    parser.add_argument("--dir", type=str, default=None)  # Allows you to custom-specify wandb directory (else uses default)

    # Parse arguments
    args = parser.parse_args()
    
        
    main(**vars(args))
