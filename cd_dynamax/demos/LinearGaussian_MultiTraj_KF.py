import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value, Predictive
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoMultivariateNormal, AutoLowRankMultivariateNormal
import wandb

# everything from your own package
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    adjust_rhs,
    make_key_sequence,
    make_optimizer
)

from cd_dynamax.src.utils.demo_utils import (
    get_or_sample,
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
    wandb_mode = "online" if cfg["online"] else "offline"
    run = wandb.init(config=cfg, project=cfg["project"], name=cfg["run_name"], dir=cfg["dir"], mode=wandb_mode)
    cfg = wandb.config

    # Global settings
    state_dim, emission_dim = cfg.state_dim, cfg.emission_dim

    # An iterable of PRNG (use is next(keys) to get the next key)
    keys = make_key_sequence(cfg.seed)
    
    def stabliize(A):  # (keeping your original name)
        singular_vals = jnp.linalg.svd(A, compute_uv=False)
        spectral_norm = singular_vals.max()
        A_stable = jnp.where(spectral_norm > cfg.max_spectral_norm,
                             A * (cfg.max_spectral_norm / spectral_norm),
                             A)
        return A_stable

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
        
        # Priors and (optional) learning for weights/bias/emission_sd/diffusion_coeff
        weights = get_or_sample(
            "weights",
            dist.Uniform(-cfg.a, cfg.a).expand((state_dim, state_dim)).to_event(2),
            kwargs.get("weights")
        )
        bias = get_or_sample(
            "bias",
            dist.Uniform(cfg.b_low, cfg.b_high).expand((state_dim,)).to_event(1),
            kwargs.get("bias")
        )

        ### NEW: noise params as learnable options (positive)
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

        # stabilize weights for drift
        weights_stable = stabliize(weights)

        # Build the drift function
        drift_base = lambda x: weights_stable @ (x - bias)
        drift = lambda x: adjust_rhs(x, drift_base(x), **adjust_rhs_kwargs)

        # Accumulate log-priors for diagnostics
        lp = 0.0
        if "weights" in cfg.learnables:
            lp = lp + dist.Uniform(-cfg.a, cfg.a).log_prob(weights).sum()
        if "bias" in cfg.learnables:
            lp = lp + dist.Uniform(cfg.b_low, cfg.b_high).log_prob(bias).sum()
        if "emission_sd" in cfg.learnables:
            lp = lp + dist.Uniform(cfg.emission_sd_low, cfg.emission_sd_high).log_prob(emission_sd)
        if "diffusion_coeff" in cfg.learnables:
            lp = lp + dist.Uniform(cfg.diffusion_low, cfg.diffusion_high).log_prob(diffusion_coeff)

        # Build the model and its parameters
        cdlgssm = ContDiscreteLinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim)  # observe first "emission_dim" states

        ### NEW: use emission_sd^2 and diffusion_coeff * I
        params = cdlgssm.build_params(
            x0_mean=jnp.zeros(state_dim), 
            x0_cov=cfg.initial_cov*jnp.eye(state_dim),
            dynamics_drift_weights=weights_stable,
            dynamics_bias=bias,
            diffusion_coeff=diffusion_coeff * jnp.eye(state_dim),
            diffusion_cov=jnp.eye(state_dim),
            emission_weights=H,
            emission_cov=(emission_sd**2) * jnp.eye(emission_dim),
        )

        # Sample emissions if not provided
        if emissions is None:
            keys_ = jax.random.split(numpyro.prng_key(), cfg.N_trajectories)
            def _sample_one(k):
                states, ems = cdlgssm.sample(params=params,
                                              num_timesteps=T,
                                              key=k,
                                              t_emissions=t_emissions)
                                            #   transition_type="path")
                return states, ems
            states_B, ems_B = jax.vmap(_sample_one)(keys_)
            emissions = ems_B
            numpyro.deterministic("states", states_B)
            numpyro.deterministic("emissions", ems_B)
        else:
            if emissions.ndim == 2:
                emissions = emissions[None, ...]
            assert emissions.ndim == 3 and emissions.shape[1] == T

        def _filter_one(ems_i):
            out = cdlgssm.filter(params=params,
                                  emissions=ems_i,
                                  t_emissions=t_emissions)
            return out.marginal_loglik

        ll_B = jax.vmap(_filter_one)(emissions)   # (N,)
        numpyro.deterministic("neg_log_likelihood_per_traj", -ll_B)

        ll = ll_B.sum()

        # Diagnostics
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)

        # Use the filtering likelihood
        numpyro.factor("log_likelihood", ll)
    
    # Define the true parameters for the model and its data generation
    # bias ~ U(b_low, b_high)
    bias_true = jr.uniform(next(keys), shape=(state_dim,), minval=cfg.b_low, maxval=cfg.b_high)
    
    omega = 2*jnp.pi/6.0
    alpha = -0.02
    weights_true = jnp.array([
        [ 0.06, 0.00, 0.00,   0.00,  0.00],
        [ 0.00,-0.06, 0.00,   0.00,  0.00],
        [ 0.00, 0.00, alpha, -omega, 0.00],
        [ 0.00, 0.00, omega,  alpha, 0.00],
        [ 0.00, 0.00, 0.00,   0.00, -0.02],
    ]) * 2.0

    ### NEW: true noise params
    emission_sd_true = jnp.array(cfg.emission_sd, dtype=jnp.float64)
    diffusion_coeff_true = jnp.array(cfg.diffusion_coeff, dtype=jnp.float64)

    true_values = {
        "weights": weights_true,
        "bias": bias_true,
        "emission_sd": emission_sd_true,           # NEW
        "diffusion_coeff": diffusion_coeff_true,   # NEW
    }

    # Typically you keep these fixed/known during learning unless requested
    known_values = {key: value for key, value in true_values.items() if key not in cfg.learnables}

    # Generate synthetic emissions
    t_emissions = sample_t_emissions(start=0.0, stop=cfg.T, dt=cfg.dt, regular=cfg.t_regular, key=next(keys)).reshape(-1, 1)
    sim_data = Predictive(model, num_samples=1)(next(keys), t_emissions=t_emissions, **true_values)
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

    # Allows you to choose different guides from config
    autoguide = {
        "AutoDelta": AutoDelta,
        "AutoNormal": AutoDiagonalNormal,
        "AutoMultivariateNormal": AutoMultivariateNormal,
        "AutoLowRankMultivariateNormal": AutoLowRankMultivariateNormal,
    }[cfg.svi_guide]

    # Initialize at conservative values (only for params we learn)
    init_dict = {
        "weights": -0.1 * jnp.eye(state_dim),
        "bias": jnp.ones(state_dim),
        ### NEW: initial guesses for noise params (if being learned)
        "emission_sd": jnp.array(cfg.emission_sd_init),
        "diffusion_coeff": jnp.array(cfg.diffusion_init),
    }
    init_values_dict = {key: value for key, value in init_dict.items() if key in cfg.learnables}
    guide = autoguide(model, init_loc_fn=init_to_value(values=init_values_dict))
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(next(keys), num_steps=cfg.num_epochs, t_emissions=t_emissions, emissions=emissions_obs, **known_values)

    # Log training curve
    wandb.define_metric("svi/loss", step_metric="epoch")
    for step, loss in enumerate(svi_result.losses):
        wandb.log({"epoch": step, "svi/loss": float(loss)})

    # Keep the weights heatmaps (as requested)
    if "weights" in cfg.learnables:
        weights_learned = guide.median(svi_result.params)["weights"]
        fig = plot_coeff_heatmaps(weights_true, weights_learned)
        wandb.log({"figures/A_heatmaps": wandb.Image(fig)})
        plt.close(fig)

    # Optional: bias heatmap (still handy)
    if "bias" in cfg.learnables:
        bias_learned = guide.median(svi_result.params)["bias"]
        fig = plot_coeff_heatmaps(bias_true.reshape(-1,1), bias_learned.reshape(-1,1))
        wandb.log({"figures/bias_heatmaps": wandb.Image(fig)})
        plt.close(fig)

    ### NEW: param recovery scatter plots (weights / bias / noise)
    scatter_groups = [
        ("weights", ["weights"]),
        ("bias", ["bias"]),
        ("noise", ["emission_sd", "diffusion_coeff"]),
    ]
    fig = plot_param_recovery(true_values, guide=guide, svi_result=svi_result, groups=scatter_groups, title="SVI: learned vs true")
    wandb.log({"figures/param_recovery_svi": wandb.Image(fig)})
    plt.close(fig)
    
    # Posterior predictive with learned params
    predictive_learned = Predictive(model, guide=guide, params=guide.median(svi_result.params), num_samples=1)
    learned_data = predictive_learned(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    wandb.log({
        "metrics/learned/neg_log_prior": float(learned_data["neg_log_prior"].item()),
        "metrics/learned/neg_log_lik": float(learned_data["neg_log_likelihood"].item()),
        "metrics/learned/neg_log_joint": float(learned_data["neg_log_joint"].item()),
    })

    # Generate long sequences
    print("Generating long trajectory from learned model...")
    long_t_emissions = jnp.arange(start=0.0, stop=cfg.T_long, step=cfg.dt).reshape(-1, 1)
    long_learned_data = predictive_learned(next(keys), t_emissions=long_t_emissions, **known_values)
    print("Generating long trajectory from true model...")
    long_true_data = Predictive(model, num_samples=1)(next(keys), t_emissions=long_t_emissions, **true_values)
    burnin_frac = 0.5
    burnin_idx = int(burnin_frac * long_t_emissions.shape[0])
    idx_traj = 0
    fig = plot_traj_kde({"Observed traj": long_true_data["emissions"].squeeze(0)[idx_traj, burnin_idx:], "Long learned traj": long_learned_data["emissions"].squeeze(0)[idx_traj, burnin_idx:]})
    wandb.log({"figures/emissions_kde": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_traj_kde({"True states": long_true_data["states"].squeeze(0)[idx_traj, burnin_idx:], "Learned states": long_learned_data["states"].squeeze(0)[idx_traj, burnin_idx:]})
    wandb.log({"figures/states_kde": wandb.Image(fig)})
    plt.close(fig)


    ## If you want, you can get better Uncertainty Quantification by running MCMC (e.g. NUTS);
    # Recommended is to initialize NUTS at the SVI solution.
    if cfg.run_mcmc:
        nuts_kernel = NUTS(model, 
                        init_strategy=init_to_value(values=guide.median(svi_result.params)),
                        max_tree_depth=cfg.max_tree_depth)
        mcmc = MCMC(nuts_kernel, num_warmup=cfg.nuts_warmup, num_samples=cfg.nuts_samples, num_chains=1)
        mcmc.run(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
        mcmc_samples = mcmc.get_samples()
        print(mcmc.print_summary())

        fig = plot_param_recovery(true_values, mcmc=mcmc, groups=scatter_groups, title="MCMC: learned vs true")
        wandb.log({"figures/param_recovery_mcmc": wandb.Image(fig)})
        plt.close(fig)

    print("Completed!")
    run.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Randomization
    parser.add_argument("--seed", type=int, default=0)

    # SVI settings [This is the fast optimization part of the script]
    parser.add_argument("--svi_guide", type=str, default="AutoDelta", choices=["AutoDelta", "AutoNormal", "AutoMultivariateNormal", "AutoLowRankMultivariateNormal"])

    # MCMC settings 
    # [This is the slow part of the script---skip it until SVI is giving reasonable answers.
    parser.add_argument("--run_mcmc", type=int, default=0) # 1 for True, 0 for False
    parser.add_argument("--nuts_warmup", type=int, default=200)
    parser.add_argument("--nuts_samples", type=int, default=100)
    parser.add_argument("--max_tree_depth", type=int, default=10)
    # The max number of steps per NUTS "iteration" is 2^max_tree_depth, but has a lot of BANG for its buck so no fear! 10 is NUTS default.

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--use_lr_scheduler", type=int, default=0) # 1 for True, 0 for False
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--epochs_per_step", type=int, default=1500)
    parser.add_argument("--clip_norm", type=float, default=1.0)  # 0.0 for no clipping, or a float value to clip.

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=1000.0)
    parser.add_argument("--diffusion_coeff", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--t_regular", type=int, default=1)
    parser.add_argument("--emission_dim", type=int, default=5)
    parser.add_argument("--state_dim", type=int, default=5)
    parser.add_argument("--N_trajectories", type=int, default=30)
    parser.add_argument("--emission_sd", type=float, default=1.0)

    # Evaluation parameters
    parser.add_argument("--T_long", type=int, default=300) # final time for long rollouts

    # Prior/bounds for learnable parameters
    parser.add_argument("--max_spectral_norm", type=float, default=200.0)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b_low", type=float, default=0.1)
    parser.add_argument("--b_high", type=float, default=10.0)

    ### NEW: bounds for noise params if learned
    parser.add_argument("--emission_sd_low", type=float, default=1e-3)
    parser.add_argument("--emission_sd_high", type=float, default=50.0)
    parser.add_argument("--diffusion_low", type=float, default=1e-3)
    parser.add_argument("--diffusion_high", type=float, default=50.0)

    ### NEW: init values if learning noise params
    parser.add_argument("--emission_sd_init", type=float, default=10.0)
    parser.add_argument("--diffusion_init", type=float, default=10.0)

    # Which parameters to learn
    parser.add_argument("--learnables", type=str, nargs="+",
                        default=["weights", "bias", "emission_sd", "diffusion_coeff"]) 
    # Default is to learn everything; you can also do any subset, e.g. ["weights", "bias"], or ["emission_sd", "diffusion_coeff"], etc.

    # Drift RHS constraints
    parser.add_argument("--state_bound_low", type=float, default=-1000.0)
    parser.add_argument("--state_bound_high", type=float, default=1000.0)
    parser.add_argument("--derivative_clip_low", type=float, default=-2000)
    parser.add_argument("--derivative_clip_high", type=float, default=2000)
    
    # Wandb settings
    parser.add_argument("--project", type=str, default="LinearGaussian_MultiTraj_exampleKF")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--dir", type=str, default="demo_outputs/LinearGaussian_MultiTraj_KF")  # Allows you to custom-specify wandb directory (else uses default)
    parser.add_argument("--online", type=int, default=0) # 0 for offline, 1 for online (need to configure wandb account for online)

    args = parser.parse_args()
    main(**vars(args))
