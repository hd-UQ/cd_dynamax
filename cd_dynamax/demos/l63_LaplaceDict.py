import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_value, Predictive
from numpyro.infer.autoguide import AutoDelta
import wandb

# everything from your own package
from cd_dynamax import (
    ContDiscreteNonlinearGaussianSSM,
    build_params,
    adjust_rhs,
    make_optimizer,
    make_key_sequence,
)

from cd_dynamax.src.utils.demo_utils import (
    build_exponents,
    poly_drift,
    get_or_sample,
    make_filter_hyperparams,
    sample_t_emissions,
    plot_traj_kde,
    plot_simulated_data,
    plot_coeff_heatmaps,
    plot_param_recovery
)


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
    
    # Build polynomial exponents (up to quadratic)
    EXPONENTS = build_exponents(state_dim, cfg.poly_degree)  # poly degree fixed=2 for Lorenz63

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

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    # NumPyro model
    def model(t_emissions, emissions=None, **kwargs):

        exponents = numpyro.deterministic("exponents", EXPONENTS)
        
        # Sample or use provided parameters
        weights = get_or_sample("weights", dist.Laplace(0.0, cfg.laplace_scale).expand((state_dim, exponents.shape[0])).to_event(2), kwargs.get("weights"))

        # Noise params (positive)
        emission_sd = get_or_sample(
            "emission_sd",
            dist.Uniform(cfg.emission_sd_low, cfg.emission_sd_high),
            kwargs.get("emission_sd", None),
        )
        diffusion_coeff = get_or_sample(
            "diffusion_coeff",
            dist.Uniform(cfg.diffusion_low, cfg.diffusion_high),
            kwargs.get("diffusion_coeff", None),
        )

        # Build the model and its parameters
        cdnlgssm = ContDiscreteNonlinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim)
        params = build_params(
            state_dim=state_dim,
            emission_dim=emission_dim,
            initial_mean=jnp.zeros(state_dim), 
            initial_cov=cfg.initial_cov*jnp.eye(state_dim), # warning: choosing it too small, even if "true", can lead to numerical issues with the filter.
            drift=lambda x: adjust_rhs(x, poly_drift(x, weights, exponents)),
            diffusion_coeff=diffusion_coeff * jnp.eye(state_dim),
            diffusion_cov=jnp.eye(state_dim),
            emission_function=lambda x: H @ x,
            emission_cov=(emission_sd**2)*jnp.eye(emission_dim),  # isotropic emission noise covariance
        )
        # Sample emissions if not provided
        if emissions is None:
            states, emissions = cdnlgssm.sample(params=params, num_timesteps=t_emissions.shape[0], key=numpyro.prng_key(), t_emissions=t_emissions, transition_type="path")
            numpyro.deterministic("states", states); numpyro.deterministic("emissions", emissions)

        # Compute (approximate) marginal log likelihood via filtering
        filtered = cdnlgssm.filter(params=params, emissions=emissions, t_emissions=t_emissions, filter_hyperparams=FILTER_HYPERPARAMS)
        ll = filtered.marginal_loglik

        # Custom-compute and store the log prior (used only for diagnostics, not for learning)
        lp = 0.0
        if "weights" in cfg.learnables:
            lp += dist.Laplace(0.0, cfg.laplace_scale).log_prob(weights).sum()
        if "emission_sd" in cfg.learnables:
            lp += dist.Uniform(cfg.emission_sd_low, cfg.emission_sd_high).log_prob(emission_sd)
        if "diffusion_coeff" in cfg.learnables:
            lp += dist.Uniform(cfg.diffusion_low, cfg.diffusion_high).log_prob(diffusion_coeff)

        
        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)

        # Use the filtering based log likelihood for the model's inference.
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
    W_init = jnp.zeros_like(W_true)  # try zero initialization
    fig = plot_coeff_heatmaps(W_true, W_init, EXPONENTS)
    wandb.log({"fig/W_true_vs_W_init": wandb.Image(fig)})
    plt.close(fig)

    # Now run SVI for MAP inference!
    guide = AutoDelta(model, init_loc_fn=init_to_value(values={"weights": W_init}))
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
    W_learned = guide.median(svi_result.params)["weights"]
    fig = plot_coeff_heatmaps(W_true, W_learned, EXPONENTS)
    wandb.log({"fig/W_true_vs_W_learned": wandb.Image(fig)})
    plt.close(fig)
    print("W_true: ", W_true)
    print("W_learned: ", W_learned)

    # Log the mean absolute error in weights
    wandb.log({"metrics/learned/abs_err_weights_mean": float(jnp.abs(W_learned - W_true).mean())})

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

    # Initialization parameters
    parser.add_argument("--emission_sd_init", type=float, default=10.0)
    parser.add_argument("--diffusion_init", type=float, default=10.0)

    # Prior parameters
    parser.add_argument("--laplace_scale", type=float, default=0.5)
    parser.add_argument("--poly_degree", type=int, default=2)  # polynomial degree for the dictionary
    parser.add_argument("--emission_sd_low", type=float, default=1e-3)
    parser.add_argument("--emission_sd_high", type=float, default=50.0)
    parser.add_argument("--diffusion_low", type=float, default=1e-3)
    parser.add_argument("--diffusion_high", type=float, default=50.0)

    # Which parameters to learn
    parser.add_argument("--learnables", type=str, nargs="+",
                    default=["weights", "emission_sd", "diffusion_coeff"]) 
    # Default is to learn everything; you can also do any subset, e.g. ["weights", "bias"], or ["emission_sd", "diffusion_coeff"], etc.

    
    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=100.0) # initial state covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=1.0)
    parser.add_argument("--emission_sd", type=float, default=1.0)    # emission noise std (used only for data generation, not for learning)
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
    parser.add_argument("--project", type=str, default="l63_dict_laplace")
    parser.add_argument("--run_name", type=str, default=None) # Allows you to custom-specify wandb run name (else uses default)
    parser.add_argument("--dir", type=str, default=None)  # Allows you to custom-specify wandb directory (else uses default)

    # Parse arguments
    args = parser.parse_args()
    
        
    main(**vars(args))
