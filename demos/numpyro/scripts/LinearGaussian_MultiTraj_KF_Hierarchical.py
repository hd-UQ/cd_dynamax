import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value, init_to_median, Predictive
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoMultivariateNormal, AutoLowRankMultivariateNormal

# everything from your own package
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    make_key_sequence,
    make_optimizer
)

from cd_dynamax.src.utils.demo_utils import (
    get_or_sample,
    sample_t_emissions,
    plot_traj_kde,
    plot_simulated_data,
    plot_coeff_heatmaps,
    plot_param_recovery,
    plot_loss_curve,
)

# ------------------------
# Main training entrypoint
# ------------------------
def main(cfg):

    # Make output directory
    os.makedirs(cfg.dir, exist_ok=True)

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

    # ----------------
    # NumPyro model
    # ----------------
    def model(t_emissions, emissions=None, **kwargs):
        # normalize time to (T,1)
        if t_emissions.ndim == 1:
            t_emissions = t_emissions[:, None]
        elif t_emissions.shape[-1] != 1:
            t_emissions = t_emissions.reshape(-1, 1)
        T = t_emissions.shape[0]

        # Determine batch size (patients)
        if emissions is None:
            N = cfg.N_trajectories
        else:
            if emissions.ndim == 2:
                emissions = emissions[None, ...]
            assert emissions.ndim == 3 and emissions.shape[1] == T
            N = emissions.shape[0]

        
        # -------- global/shared priors (weights/noise) ----------        
        dist_weights = dist.Uniform(-cfg.a, cfg.a).expand((state_dim, state_dim)).to_event(2)
        weights = get_or_sample(
            "weights",
            dist_weights,
            kwargs.get("weights")
        )

        dist_emission_sd = dist.Uniform(cfg.emission_sd_low, cfg.emission_sd_high)
        emission_sd = get_or_sample(
            "emission_sd",
            dist_emission_sd,
            kwargs.get("emission_sd", None),
        )
        
        dist_diffusion_coeff = dist.Uniform(cfg.diffusion_low, cfg.diffusion_high)
        diffusion_coeff = get_or_sample(
            "diffusion_coeff",
            dist_diffusion_coeff,
            kwargs.get("diffusion_coeff", None),
        )

        # --------- population hyperpriors for biases ----------
        # mu_d ~ TruncatedNormal(m_mu, s_mu; lower=eps)
        dist_mu = dist.TruncatedNormal(loc=jnp.full((state_dim,), cfg.m_mu),
                                        scale=jnp.full((state_dim,), cfg.s_mu),
                                        low=jnp.full((state_dim,), cfg.eps),
                                        ).to_event(1)
        mu = get_or_sample("mu", dist_mu, kwargs.get("mu", None))
        
            
        # sigma_d ~ TruncatedNormal(m_sigma, s_sigma; lower=eps)
        dist_sigma = dist.TruncatedNormal(loc=jnp.full((state_dim,), cfg.m_sigma),
                                          scale=jnp.full((state_dim,), cfg.s_sigma),
                                          low=jnp.full((state_dim,), cfg.eps),
                                          ).to_event(1)
        sigma = get_or_sample("sigma", dist_sigma, kwargs.get("sigma", None))

        # --------- individual biases (per patient) ----------
        # b_i ~ TruncatedNormal(mu, sigma; lower=eps), independent across dims
        dist_biases = dist.TruncatedNormal(loc=jnp.broadcast_to(mu, (N, state_dim)),
                                            scale=jnp.broadcast_to(sigma, (N, state_dim)),
                                            low=jnp.full((N, state_dim), cfg.eps),
                                            ).to_event(2)
        biases = get_or_sample("biases", dist_biases, kwargs.get("biases", None))

        # Accumulate the negative log prior for logging
        lp = 0.0
        if "weights" in cfg.learnables:
            lp = lp + dist_weights.log_prob(weights)
        if "emission_sd" in cfg.learnables:
            lp = lp + dist_emission_sd.log_prob(emission_sd)
        if "diffusion_coeff" in cfg.learnables:
            lp = lp + dist_diffusion_coeff.log_prob(diffusion_coeff)
        if "mu" in cfg.learnables:
            lp = lp + dist_mu.log_prob(mu)
        if "sigma" in cfg.learnables:
            lp = lp + dist_sigma.log_prob(sigma)
        if "biases" in cfg.learnables:
            lp = lp + dist_biases.log_prob(biases)

        # stabilize weights for drift
        weights_stable = stabliize(weights)

        # Build the SSM object once; per-patient params later
        cdlgssm = ContDiscreteLinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
        H = jnp.eye(emission_dim, state_dim)  # observe first "emission_dim" states

        def _build_params_for_bias(b_vec):
            return cdlgssm.build_params(
                x0_mean=jnp.zeros(state_dim),
                x0_cov=cfg.initial_cov*jnp.eye(state_dim),
                dynamics_drift_weights=weights_stable,
                dynamics_bias=b_vec,
                diffusion_coeff=diffusion_coeff * jnp.eye(state_dim),
                diffusion_cov=jnp.eye(state_dim),
                emission_weights=H,
                emission_cov=(emission_sd**2) * jnp.eye(emission_dim),
            )

        # Sample emissions if not provided
        if emissions is None:
            keys_ = jax.random.split(numpyro.prng_key(), N)

            def _sample_one(k, b_vec):
                params_i = _build_params_for_bias(b_vec)
                states, ems = cdlgssm.sample(
                    params=params_i,
                    num_timesteps=T,
                    key=k,
                    t_emissions=t_emissions
                )
                return states, ems

            states_B, ems_B = jax.vmap(_sample_one)(keys_, biases)
            emissions = ems_B
            numpyro.deterministic("states", states_B)
            numpyro.deterministic("emissions", ems_B)
        else:
            # emissions provided; ensure batch dimension N is consistent
            assert emissions.shape[0] == N

        # Filtering log-likelihood per patient with their own bias
        def _filter_one(ems_i, b_vec):
            params_i = _build_params_for_bias(b_vec)
            out = cdlgssm.filter(params=params_i, emissions=ems_i, t_emissions=t_emissions)
            return out.marginal_loglik

        ll_B = jax.vmap(_filter_one)(emissions, biases)  # (N,)
        numpyro.deterministic("neg_log_likelihood_per_traj", -ll_B)
        ll = ll_B.sum()

        # Use the filtering likelihood
        numpyro.factor("log_likelihood", ll)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_prior", -lp)
        numpyro.deterministic("neg_log_joint", -lp - ll)

    # -----------------------
    # Define TRUE parameters
    # -----------------------
    # Keep global weights_true as before
    omega = 2*jnp.pi/6.0
    alpha = -0.02
    weights_true = jnp.array([
        [ 0.06, 0.00, 0.00,   0.00,  0.00],
        [ 0.00,-0.06, 0.00,   0.00,  0.00],
        [ 0.00, 0.00, alpha, -omega, 0.00],
        [ 0.00, 0.00, omega,  alpha, 0.00],
        [ 0.00, 0.00, 0.00,   0.00, -0.02],
    ]) * 2.0

    emission_sd_true = jnp.array(cfg.emission_sd, dtype=jnp.float64)
    diffusion_coeff_true = jnp.array(cfg.diffusion_coeff, dtype=jnp.float64)

    # Generate synthetic emissions via PRIOR PREDICTIVE:
    # - Fix weights/noise to truth
    # - Let (mu, sigma, biases) be drawn from their priors
    t_emissions = sample_t_emissions(start=0.0, stop=cfg.T, dt=cfg.dt,
                                     regular=cfg.t_regular, key=next(keys)).reshape(-1, 1)

    true_fixed = {
        "weights": weights_true,
        "emission_sd": emission_sd_true,
        "diffusion_coeff": diffusion_coeff_true,
    }

    prior_pred = Predictive(model, num_samples=1)
    sim_data = prior_pred(next(keys), t_emissions=t_emissions, **true_fixed)

    emissions_obs = sim_data["emissions"].squeeze(0)  # (N, T, E)
    states_true = sim_data["states"].squeeze(0)       # (N, T, D)
    mu_true = sim_data["mu"].squeeze(0)               # (D,)
    sigma_true = sim_data["sigma"].squeeze(0)         # (D,)
    biases_true = sim_data["biases"].squeeze(0)       # (N, D)

    true_values = {
        "weights": weights_true,
        "emission_sd": emission_sd_true,
        "diffusion_coeff": diffusion_coeff_true,
        "mu": mu_true,
        "sigma": sigma_true,
        "biases": biases_true,
    }

    # Plot the states and emissions for each trajectory
    n_plotted_traj = min(cfg.N_trajectories, 5)
    if cfg.N_trajectories > 5:
        print(f"Plotting the first {n_plotted_traj} of {cfg.N_trajectories} trajectories...")
    for i in range(n_plotted_traj):
        fig = plot_simulated_data(t=t_emissions, states=states_true[i], emissions=emissions_obs[i])
        fig.savefig(os.path.join(cfg.dir, f"true_states_traj{i}.png"))
        plt.close(fig)

    # Log metrics
    neg_log_names = ["neg_log_prior", "neg_log_likelihood", "neg_log_joint"]
    for name in neg_log_names:
        print(f"True model's {name} from filtering: {float(sim_data[name].item())}")

    # ----------------
    # SVI (fit model)
    # ----------------
    # Typically you keep these fixed/known during learning unless requested
    known_values = {key: value for key, value in true_values.items() if key not in cfg.learnables}


    optimizer = make_optimizer(
        initial_learning_rate=cfg.init_lr,
        decay_factor=cfg.decay_rate,
        epochs_per_step=cfg.epochs_per_step,
        num_epochs=cfg.num_epochs,
        use_lr_scheduler=cfg.use_lr_scheduler,
        clip_norm=cfg.clip_norm if cfg.clip_norm > 0.0 else None,
    )

    # Allows you to choose different guides from config
    autoguide = eval(cfg.svi_guide)
    guide = autoguide(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(next(keys),
                         num_steps=cfg.num_epochs,
                         t_emissions=t_emissions,
                         emissions=emissions_obs,
                         **known_values)

    # Log training curve
    fig = plot_loss_curve(svi_result.losses)
    fig.savefig(os.path.join(cfg.dir, "svi_loss_curve.png"))
    plt.close(fig)

    # ----------------
    # Plots / recovery
    # ----------------
    # Weights heatmaps
    weights_learned = guide.median(svi_result.params).get("weights", None)
    if weights_learned is not None:
        fig = plot_coeff_heatmaps(weights_true, weights_learned)
        fig.savefig(os.path.join(cfg.dir, "weights_heatmaps.png"))
        plt.close(fig)

    # Population mu and sigma heatmaps (treat as column vectors)
    learned = guide.median(svi_result.params)
    if "mu" in learned:
        fig = plot_coeff_heatmaps(mu_true.reshape(-1, 1), learned["mu"].reshape(-1, 1))
        fig.savefig(os.path.join(cfg.dir, "mu_heatmaps.png"))
        plt.close(fig)
    if "sigma" in learned:
        fig = plot_coeff_heatmaps(sigma_true.reshape(-1, 1), learned["sigma"].reshape(-1, 1))
        fig.savefig(os.path.join(cfg.dir, "sigma_heatmaps.png"))
        plt.close(fig)

    # Param recovery scatter groups
    scatter_groups = [
        ("weights", ["weights"]),
        ("bias_population_mu", ["mu"]),
        ("bias_population_sigma", ["sigma"]),
        ("bias_individuals", ["biases"]),
        ("noise", ["emission_sd", "diffusion_coeff"]),
    ]
    fig = plot_param_recovery(true_values, guide=guide, svi_result=svi_result,
                              groups=scatter_groups)
    fig.savefig(os.path.join(cfg.dir, "param_recovery_svi.png"))
    plt.close(fig)

    # Posterior predictive at learned medians (optional diagnostics)
    predictive_learned = Predictive(model, guide=guide, params=learned, num_samples=1)
    learned_data = predictive_learned(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
    for name in neg_log_names:
        print(f"Learned model's {name} from filtering: {float(learned_data[name].item())}")

    # Long rollouts for a single trajectory index
    print("Generating long trajectory from learned model...")
    long_t_emissions = jnp.arange(start=0.0, stop=cfg.T_long, step=cfg.dt).reshape(-1, 1)
    long_learned_data = predictive_learned(next(keys), t_emissions=long_t_emissions, **known_values)
    print("Generating long trajectory from true model...")
    long_true_data = Predictive(model, num_samples=1)(next(keys), t_emissions=long_t_emissions, **known_values)

    burnin_frac = 0.5
    burnin_idx = int(burnin_frac * long_t_emissions.shape[0])
    idx_traj = 0
    fig = plot_traj_kde({
        "Observed traj": long_true_data["emissions"].squeeze(0)[idx_traj, burnin_idx:],
        "Long learned traj": long_learned_data["emissions"].squeeze(0)[idx_traj, burnin_idx:]
    })
    fig.savefig(os.path.join(cfg.dir, "long_traj_emissions_kde.png"))
    plt.close(fig)

    fig = plot_traj_kde({
        "True states": long_true_data["states"].squeeze(0)[idx_traj, burnin_idx:],
        "Learned states": long_learned_data["states"].squeeze(0)[idx_traj, burnin_idx:]
    })
    fig.savefig(os.path.join(cfg.dir, "long_traj_states_kde.png"))
    plt.close(fig)

    # Optional: NUTS
    if cfg.run_mcmc:
        nuts_kernel = NUTS(model,
                           init_strategy=init_to_value(values=learned),
                           max_tree_depth=cfg.max_tree_depth)
        mcmc = MCMC(nuts_kernel, num_warmup=cfg.nuts_warmup, num_samples=cfg.nuts_samples, num_chains=1)
        mcmc.run(next(keys), t_emissions=t_emissions, emissions=emissions_obs, **known_values)
        print(mcmc.print_summary())

        fig = plot_param_recovery(true_values, mcmc=mcmc, groups=scatter_groups)
        fig.savefig(os.path.join(cfg.dir, "param_recovery_mcmc.png"))
        plt.close(fig)

    print("Completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Randomization
    parser.add_argument("--seed", type=int, default=0)

    # SVI settings
    parser.add_argument("--svi_guide", type=str, default="AutoMultivariateNormal", choices=["AutoDelta", "AutoDiagonalNormal", "AutoMultivariateNormal", "AutoLowRankMultivariateNormal"])

    # Which parameters to learn
    parser.add_argument("--learnables", type=str, nargs="+",
                        default=["weights", "mu", "sigma", "biases", "emission_sd", "diffusion_coeff"]) 

    # MCMC settings
    parser.add_argument("--run_mcmc", type=int, default=0)
    parser.add_argument("--nuts_warmup", type=int, default=200)
    parser.add_argument("--nuts_samples", type=int, default=100)
    parser.add_argument("--max_tree_depth", type=int, default=10)

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=1e-2)
    parser.add_argument("--use_lr_scheduler", type=int, default=0)
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--epochs_per_step", type=int, default=1500)
    parser.add_argument("--clip_norm", type=float, default=1.0)

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
    parser.add_argument("--T_long", type=int, default=300)

    # Prior/bounds for learnable parameters (weights)
    parser.add_argument("--max_spectral_norm", type=float, default=200.0)
    parser.add_argument("--a", type=float, default=3.0)

    # Output settings
    parser.add_argument("--dir", type=str, default="demo_outputs/LinearGaussian_MultiTraj_KF_HierTN")

    # --- NEW: Hyperprior controls for hierarchical biases ---
    parser.add_argument("--m_mu", type=float, default=2.5)     # mean for mu_d TN
    parser.add_argument("--s_mu", type=float, default=2.5)     # std  for mu_d TN
    parser.add_argument("--m_sigma", type=float, default=0.35) # mean for sigma_d TN
    parser.add_argument("--s_sigma", type=float, default=0.25) # std  for sigma_d TN
    parser.add_argument("--eps", type=float, default=1e-2)     # lower truncation bound

    # Noise param prior bounds (kept from your version)
    parser.add_argument("--emission_sd_low", type=float, default=1e-3)
    parser.add_argument("--emission_sd_high", type=float, default=50.0)
    parser.add_argument("--diffusion_low", type=float, default=1e-3)
    parser.add_argument("--diffusion_high", type=float, default=50.0)
    parser.add_argument("--emission_sd_init", type=float, default=10.0)
    parser.add_argument("--diffusion_init", type=float, default=10.0)

    args = parser.parse_args()
    main(args)
