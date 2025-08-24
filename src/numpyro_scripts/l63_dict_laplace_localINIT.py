import sys
sys.path.append("../")
sys.path.append("../..")

import itertools
import numpy as np
import jax
import jax.numpy as jnp
# Make sure everything is 64bit (should prevent NaNs, but can be slow)
# jax.config.update("jax_enable_x64", True)
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
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta

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

    rel_err = jnp.abs(W_learned - W_true) / jnp.maximum(jnp.abs(W_true), 1e-8)
    im2 = axes[2].imshow(rel_err, aspect="auto", cmap="viridis")
    axes[2].set_title("Relative error"); axes[2].set_xlabel("Term index"); axes[2].set_ylabel("State index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

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
            cov_rescaling=cfg.cov_rescaling
        )
    elif cfg.filter_type == "EFK":
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
    state_dim, emission_dim = 3, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    # Build polynomial exponents (up to quadratic)
    EXPONENTS = build_exponents(state_dim, 2)  # poly degree fixed=2 for Lorenz63

    # Build true Lorenz63 weights matrix
    def idx_of(alpha_tuple):
        matches = jnp.all(EXPONENTS == jnp.array(alpha_tuple, dtype=jnp.int32), axis=1)
        return int(jnp.argmax(matches))
    i_x, i_y, i_z = idx_of((1,0,0)), idx_of((0,1,0)), idx_of((0,0,1))
    i_xy, i_xz   = idx_of((1,1,0)), idx_of((1,0,1))
    W_true = jnp.zeros((state_dim, EXPONENTS.shape[0]))
    W_true = W_true.at[0, i_x].set(-10.0).at[0, i_y].set(+10.0)
    W_true = W_true.at[1, i_x].set(28.0).at[1, i_xz].set(-1.0).at[1, i_y].set(-1.0)
    W_true = W_true.at[2, i_xy].set(+1.0).at[2, i_z].set(-8.0/3.0)

    # CORRUPT W_TRUE WITH NOISE
    eps = 0 #0.01
    flat_W = W_true.ravel()
    noise = jr.normal(next(keys), shape=flat_W.shape) * jnp.sqrt(eps)
    W_true = (flat_W + noise).reshape(W_true.shape)

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    # NumPyro model
    def model(t_emissions, emissions=None, store_filtered=False, **kwargs):
        
        # Sample or use provided parameters
        weights = get_or_sample("weights", dist.Laplace(0.0, cfg.laplace_scale).expand((state_dim, EXPONENTS.shape[0])).to_event(2), kwargs.get("weights"))
        exponents = numpyro.deterministic("exponents", EXPONENTS)

        # Build the model and its parameters
        cdnlgssm = ContDiscreteNonlinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim)
        params = build_params(
            state_dim=state_dim,
            emission_dim=emission_dim,
            initial_mean=jnp.zeros(state_dim), 
            initial_cov=cfg.initial_cov*jnp.eye(state_dim), # warning: choosing it too small, even if "true", can lead to numerical issues with the filter.
            drift=lambda x: adjust_rhs(x, poly_drift(x, weights, exponents)),
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
        
        if store_filtered:
            # Store the filtering details for diagnostics
            numpyro.deterministic('filtered_means', filtered.filtered_means)
            numpyro.deterministic('filtered_covariances', filtered.filtered_covariances)
            numpyro.deterministic('predicted_means', filtered.predicted_means)
            numpyro.deterministic('predicted_covariances', filtered.predicted_covariances)

        # Custom-compute and store the log prior (used only for diagnostics, not for learning)
        lp = dist.Laplace(0.0, cfg.laplace_scale).log_prob(weights).sum()
        numpyro.deterministic("neg_log_prior", -lp)
        
        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)
        numpyro.factor("log_likelihood", ll)


    # Define the true parameters for the model and its data generation
    true_values = {
        "weights": W_true,  # <— include true weights
    }

    # Typically you keep these fixed/known during learning
    # know everything except "weights" # it is empty here, but useful when there are more params
    known_values = {key: value for key, value in true_values.items() if key not in ["weights"]}
    
    # Generate synthetic emissions
    t_emissions = jnp.arange(start=0.0, stop=cfg.T, step=cfg.dt).reshape(-1, 1)
    sim_data = Predictive(model, num_samples=1)(next(keys), t_emissions=t_emissions, store_filtered=True, **true_values)
    # Extract emissions from the simulation data
    emissions_obs = sim_data["emissions"].squeeze(0)

    # Log metrics
    wandb.log({
        "metrics/true/neg_log_prior": float(sim_data["neg_log_prior"]),
        "metrics/true/neg_log_lik": float(sim_data["neg_log_likelihood"]),
        "metrics/true/neg_log_joint": float(sim_data["neg_log_joint"]),
    })
    # Print the neg-log-likelihood values
    print(f"True model's neg_log_likelihood from filtering: {float(sim_data['neg_log_likelihood'])}")
    
    # Next, check that the model is well-posed epsilon-close to the truth
    print("Checking predictive/ filtering performance at true parameter + small noise...")
    W_true_noisy = W_true + 0.1 * jr.normal(next(keys), shape=W_true.shape)
    perturbed_truth_predictive = Predictive(model, num_samples=1)
    perturbed_data = perturbed_truth_predictive(next(keys),
                                                t_emissions=t_emissions,
                                                emissions=emissions_obs,
                                                store_filtered=True,
                                                weights=W_true_noisy,
                                                **known_values)
    wandb.log({
        "metrics/perturbed/neg_log_prior": float(perturbed_data["neg_log_prior"]),
        "metrics/perturbed/neg_log_lik": float(perturbed_data["neg_log_likelihood"]),
        "metrics/perturbed/neg_log_joint": float(perturbed_data["neg_log_joint"]),
    })
    print(f"True+epsilon (perturbed) model's neg_log_likelihood from filtering: {float(perturbed_data['neg_log_likelihood'])}")

    # Log the trajectory from filtering
    print("Logging trajectories from filtering/simulations with true and perturbed-truth models...")
    # Trajectories use "time" as their x-axis
    wandb.define_metric("filtering/true/true_state_*", step_metric="time")
    wandb.define_metric("filtering/true/pred_mean_*", step_metric="time")
    wandb.define_metric("filtering/true/filtered_mean_*", step_metric="time")
    wandb.define_metric("filtering/perturbed/pred_mean_*", step_metric="time")
    wandb.define_metric("filtering/perturbed/filtered_mean_*", step_metric="time")
    for t in range(sim_data["predicted_means"].shape[1]):
        log_dict = {"time": t}
        for d in range(state_dim):
            log_dict[f"filtering/true/true_state_dim{d}"] = float(sim_data["states"][0, t, d])
            log_dict[f"filtering/true/pred_mean_dim{d}"] = float(sim_data["predicted_means"][0, t, d])
            log_dict[f"filtering/true/filtered_mean_dim{d}"] = float(sim_data["filtered_means"][0, t, d])
            log_dict[f"filtering/perturbed/pred_mean_dim{d}"] = float(perturbed_data["predicted_means"][0, t, d])
            log_dict[f"filtering/perturbed/filtered_mean_dim{d}"] = float(perturbed_data["filtered_means"][0, t, d])
        # Log the current time step
        wandb.log(log_dict)

    
    # Now, generate a prior predictive conditioned on emissions/t_emissions (for diagnostics)
    prior_predictive = Predictive(model, num_samples=cfg.n_prior_samples)
    prior_data = prior_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    # Compute fraction of NaN values in the neg-log-likelihood computation
    nan_fraction = jnp.isnan(prior_data["neg_log_likelihood"]).mean()
    wandb.log({"metrics/prior_predictive/nan_fraction": nan_fraction})
    if nan_fraction > 0.0:
        print(f"Warning: {nan_fraction:.2%} of prior predictive neg_log_likelihood samples are NaN")
        # Show any one example weights matrix with NaN values
        j = jnp.where(jnp.isnan(prior_data["neg_log_likelihood"]))[0][0]
        print("Example weights with NaN in prior predictive neg_log_likelihood:")
        print(prior_data["weights"][j])
        # Optionally, you can raise an error here to stop execution        
        raise ValueError(f"NaN values found in prior predictive neg_log_likelihood: {nan_fraction:.2%} of samples are NaN. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    # SVI
    optimizer = make_optimizer(
        initial_learning_rate=cfg.init_lr,
        decay_factor=cfg.decay_rate,
        epochs_per_step=cfg.epochs_per_step,
        num_epochs=cfg.num_epochs,
        use_lr_scheduler=cfg.use_lr_scheduler,
        clip_norm=cfg.clip_norm,
    )
    
    def make_init_loc_fn(W_true, eps, key):
        flat_W = W_true.ravel()
        noise = jr.normal(key, shape=flat_W.shape) * jnp.sqrt(eps)
        init_val = flat_W + noise
        return init_to_value(values={"weights": init_val.reshape(W_true.shape)})

    init_loc_fn = make_init_loc_fn(W_true, eps=1, key=next(keys))
    # init_loc_fn = init_to_median()
    # init_loc_fn = init_to_value(values={"weights": W_true})
    guide = AutoDelta(model, init_loc_fn=init_loc_fn)
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
        "metrics/learned/neg_log_prior": float(learned_data["neg_log_prior"]),
        "metrics/learned/neg_log_lik": float(learned_data["neg_log_likelihood"]),
        "metrics/learned/neg_log_joint": float(learned_data["neg_log_joint"]),
    })
    # Log figure
    W_learned = guide.median(svi_result.params)["weights"]
    fig = plot_coeff_heatmaps(W_true, W_learned, EXPONENTS)
    wandb.log({"fig/W_true_vs_W_learned": wandb.Image(fig)})
    plt.close(fig)

    # Log the trajectory from filtering
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


    run.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Randomization
    parser.add_argument("--seed", type=int, default=0)

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--use_lr_scheduler", type=int, default=1) # 1 for True, 0 for False
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--epochs_per_step", type=int, default=200)
    parser.add_argument("--clip_norm", type=float, default=1.0)  # None for no clipping, or a float value

    # Prior parameters
    parser.add_argument("--laplace_scale", type=float, default=0.5)

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=100.0) # initial state covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=0.01)
    parser.add_argument("--T", type=int, default=100) # final time
    parser.add_argument("--dt", type=float, default=0.01) # time step size
    parser.add_argument("--emission_dim", type=int, default=3) # observation dimension (default is to observe the first "emission_dim" states)

    # Filtering algorithm hyperparameters
    parser.add_argument("--filter_type", type=str, default="EnKF")  # "EnKF", "EFK", "UKF", "PF"
    parser.add_argument("--N_particles", type=int, default=25)  # Number of particles for EnKF
    parser.add_argument("--state_order", type=str, default="zeroth")  # "zeroth", "first", "second"
    parser.add_argument("--diffeqsolve_max_steps", type=int, default=1000)  # Max steps for the ODE solver between filtered timesteps
    parser.add_argument("--cov_rescaling", type=float, default=1000.0)  # Covariance rescaling factor for EnKF
    
    # Diagnostics
    parser.add_argument("--n_prior_samples", type=int, default=100)

    # Parse arguments
    args = parser.parse_args()
        
    main(**vars(args))
