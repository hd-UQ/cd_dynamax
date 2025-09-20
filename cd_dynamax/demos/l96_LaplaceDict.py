import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jr
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
    plot_param_recovery,
    plot_particle_diagnostics
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
    
    # Build polynomial exponents (up to quadratic)
    EXPONENTS = build_exponents(state_dim, 2)  # poly degree fixed=2 for Lorenz96

    # Build true Lorenz96 weights matrix
    def idx_of(alpha_tuple):
        matches = jnp.all(EXPONENTS == jnp.array(alpha_tuple, dtype=jnp.int32), axis=1)
        return int(jnp.argmax(matches))

    F = 8.0  # forcing term
    W_true = np.zeros((state_dim, EXPONENTS.shape[0]))
    for i in range(state_dim):
        # Quadratic terms
        alpha = np.zeros(state_dim, dtype=int)
        alpha[(i-1) % state_dim] += 1
        alpha[(i+1) % state_dim] += 1
        W_true[i, idx_of(tuple(alpha))] = 1.0
        
        alpha = np.zeros(state_dim, dtype=int)
        alpha[(i-1) % state_dim] += 1
        alpha[(i-2) % state_dim] += 1
        W_true[i, idx_of(tuple(alpha))] = -1.0
        
        # Linear term -x_i
        alpha = np.zeros(state_dim, dtype=int)
        alpha[i] = 1
        W_true[i, idx_of(tuple(alpha))] = -1.0
        
        # Constant forcing F
        W_true[i, idx_of(tuple([0] * state_dim))] = F
    
    W_true = jnp.array(W_true)
    print("W_true: ", W_true)

    # Build filter hyperparameters
    FILTER_HYPERPARAMS = make_filter_hyperparams(cfg)

    # NumPyro model
    def model(t_emissions, emissions=None, store_filtered=False, store_grad=False, **kwargs):
        
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
            emission_cov=cfg.emission_cov * jnp.eye(emission_dim),
        )
        # Sample emissions if not provided
        if emissions is None:
            states, emissions = cdnlgssm.sample(params=params, num_timesteps=t_emissions.shape[0], key=numpyro.prng_key(), t_emissions=t_emissions, transition_type="path")
            numpyro.deterministic("states", states); numpyro.deterministic("emissions", emissions)

        # Compute (approximate) marginal log likelihood via filtering
        filtered = cdnlgssm.filter(params=params, emissions=emissions, t_emissions=t_emissions, filter_hyperparams=FILTER_HYPERPARAMS)
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

        # Custom-compute and store the log prior (used only for diagnostics, not for learning)
        lp = dist.Laplace(0.0, cfg.laplace_scale).log_prob(weights).sum()
        numpyro.deterministic("neg_log_prior", -lp)
        
        # Store the log probs (used only for diagnostics, not for learning)
        numpyro.deterministic("neg_log_likelihood", -ll)
        numpyro.deterministic("neg_log_joint", -ll - lp)
        numpyro.factor("log_likelihood", ll)

        # Optionally store gradients (for diagnostics)
        if store_grad:
            grad_w = jax.grad(lambda w: cdnlgssm.filter(
                params=build_params(
                    state_dim=state_dim,
                    emission_dim=emission_dim,
                    initial_mean=jnp.zeros(state_dim), 
                    initial_cov=cfg.initial_cov*jnp.eye(state_dim),
                    drift=lambda x: adjust_rhs(x, poly_drift(x, w, exponents)),
                    diffusion_coeff=cfg.diffusion_coeff * jnp.eye(state_dim),
                    diffusion_cov=jnp.eye(state_dim),
                    emission_function=lambda x: H @ x,
                    emission_cov=cfg.emission_cov * jnp.eye(emission_dim),
                ),
                emissions=emissions,
                t_emissions=t_emissions,
                filter_hyperparams=FILTER_HYPERPARAMS
            ).marginal_loglik)(weights)
            numpyro.deterministic("grad_neg_log_likelihood_w", grad_w)
    

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

    # Plot the states and emissions for each trajectory
    fig = plot_simulated_data(t=t_emissions, states=sim_data["states"].squeeze(0), emissions=emissions_obs)
    wandb.log({f"figures/true_states_traj": wandb.Image(fig)})
    plt.close(fig)

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
        wandb.log({"figures/true/ensembles_0to20": wandb.Image(fig)})
        plt.close(fig)

    # Next, check that the model is well-posed epsilon-close to the truth
    eps_perturb = cfg.eps_perturb
    print("Checking predictive/ filtering performance at true parameter + small noise...")
    W_true_noisy = W_true + eps_perturb * jr.normal(next(keys), shape=W_true.shape)
    perturbed_truth_predictive = Predictive(model, num_samples=1)
    perturbed_data = perturbed_truth_predictive(next(keys),
                                                t_emissions=t_emissions,
                                                emissions=emissions_obs,
                                                store_filtered=True,
                                                weights=W_true_noisy,
                                                **known_values)
    wandb.log({
        "metrics/perturbed/neg_log_prior": float(perturbed_data["neg_log_prior"].item()),
        "metrics/perturbed/neg_log_lik": float(perturbed_data["neg_log_likelihood"].item()),
        "metrics/perturbed/neg_log_joint": float(perturbed_data["neg_log_joint"].item()),
    })

    # Log the trajectory from filtering
    if cfg.log_filtering:
        print(f"True+epsilon (perturbed) model's neg_log_likelihood from filtering: {float(perturbed_data['neg_log_likelihood'].item())}")
        fig = plot_particle_diagnostics(
            t_emissions=t_emissions.squeeze(),
            x_ens_filtered=perturbed_data["x_ens_filtered"][0],   # shape (T, N, D)
            x_ens_predicted=perturbed_data["x_ens_predicted"][0], # shape (T, N, D)
            observations=emissions_obs,
            figsize=(12, 8),
            start_idx=0,
            stop_idx=20,
        )
        wandb.log({"figures/perturbed/ensembles_0to20": wandb.Image(fig)})
        plt.close(fig)

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
        wandb.define_metric("filtering/perturbed/pred_mean_*", step_metric="time")
        wandb.define_metric("filtering/perturbed/filtered_mean_*", step_metric="time")
        wandb.define_metric("filtering/perturbed/cond_K", step_metric="time")
        wandb.define_metric("filtering/perturbed/cond_S", step_metric="time")
        wandb.define_metric("filtering/perturbed/min_eig_S", step_metric="time")
        wandb.define_metric("filtering/perturbed/nis", step_metric="time")
        wandb.define_metric("filtering/perturbed/neg_loglik_steps", step_metric="time")
        for t in range(sim_data["predicted_means"].shape[1]):
            log_dict = {"time": t}
            log_dict["filtering/true/cond_S"] = float(sim_data["cond_S"][0, t])
            log_dict["filtering/true/min_eig_S"] = float(sim_data["min_eig_S"][0, t])
            log_dict["filtering/true/nis"] = float(sim_data["nis"][0, t])
            log_dict["filtering/true/neg_loglik_steps"] = float(sim_data["neg_loglik_steps"][0, t])
            log_dict["filtering/true/cond_K"] = float(sim_data["cond_K"][0, t])
            log_dict["filtering/perturbed/cond_K"] = float(perturbed_data["cond_K"][0, t])
            log_dict["filtering/perturbed/cond_S"] = float(perturbed_data["cond_S"][0, t])
            log_dict["filtering/perturbed/min_eig_S"] = float(perturbed_data["min_eig_S"][0, t])
            log_dict["filtering/perturbed/nis"] = float(perturbed_data["nis"][0, t])
            log_dict["filtering/perturbed/neg_loglik_steps"] = float(perturbed_data["neg_loglik_steps"][0, t])
            for d in range(state_dim):
                log_dict[f"filtering/true/true_state_dim{d}"] = float(sim_data["states"][0, t, d])
                log_dict[f"filtering/true/pred_mean_dim{d}"] = float(sim_data["predicted_means"][0, t, d])
                log_dict[f"filtering/true/filtered_mean_dim{d}"] = float(sim_data["filtered_means"][0, t, d])
                log_dict[f"filtering/perturbed/pred_mean_dim{d}"] = float(perturbed_data["predicted_means"][0, t, d])
                log_dict[f"filtering/perturbed/filtered_mean_dim{d}"] = float(perturbed_data["filtered_means"][0, t, d])
            # Log the current time step
            wandb.log(log_dict)

    
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
        print(prior_data["weights"][j])
        # Optionally, you can raise an error here to stop execution        
        raise ValueError(f"NaN values found in prior predictive neg_log_likelihood: {nan_fraction:.2%} of samples are NaN. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    # Now compute fraction of NaN values in the grad_neg_log_likelihood_w computation
    nan_fraction_grad = jnp.isnan(prior_data["grad_neg_log_likelihood_w"]).mean()
    wandb.log({"metrics/prior_predictive/nan_fraction_grad": nan_fraction_grad})
    if nan_fraction_grad > 0.0:
        print(f"Warning: {nan_fraction_grad:.2%} of prior predictive grad_neg_log_likelihood_w samples are NaN")
        # Show any one example weights matrix with NaN values
        j = jnp.where(jnp.isnan(prior_data["grad_neg_log_likelihood_w"]))[0][0]
        print("Example weights with NaN in prior predictive grad_neg_log_likelihood_w:")
        print(prior_data["weights"][j])
        # Optionally, you can raise an error here to stop execution        
        raise ValueError(f"NaN values found in prior predictive grad_neg_log_likelihood_w: {nan_fraction_grad:.2%} of samples are NaN. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

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
    W_init = jnp.zeros_like(W_true)  # try zero initialization
    guide = AutoDelta(model, init_loc_fn=init_to_value(values={"weights": W_init}))
    print("W_init: ", W_init)

    fig = plot_coeff_heatmaps(W_true, W_init, EXPONENTS)
    wandb.log({"figures/W_true_vs_W_init": wandb.Image(fig)})
    plt.close(fig)
    
    # Run model in predictive mode
    print("Checking initialization predictive...")
    init_predictive = Predictive(model, num_samples=1)
    init_data = init_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, weights=W_init, store_filtered=True)

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
        wandb.log({"figures/init/ensembles_0to20": wandb.Image(fig)})
        plt.close(fig)

    print(f"Initialization neg_log_likelihood from filtering: {float(init_data['neg_log_likelihood'].item())}")
    if jnp.isnan(init_data['neg_log_likelihood']):
        print("Initialization weights: ", W_init)
        raise ValueError("NaN value found in initialization neg_log_likelihood. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

    init_data = init_predictive(next(keys), t_emissions=t_emissions, emissions=emissions_obs, weights=W_init, store_filtered=True, store_grad=True)
    print(f"Initialization grad_neg_log_likelihood_w from filtering: {init_data['grad_neg_log_likelihood_w']}")
    if jnp.any(jnp.isnan(init_data['grad_neg_log_likelihood_w'])):
        print("Initialization weights: ", W_init)
        raise ValueError("NaN value found in initialization grad_neg_log_likelihood_w. Training is TOO DANGEROUS to proceed; please refine your prior, protect the RHS from large jumps, ensure covariances are PSD, and/or try 64bit precision.")

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
    wandb.log({"figures/W_true_vs_W_learned": wandb.Image(fig)})
    plt.close(fig)
    print("W_true: ", W_true)
    print("W_learned: ", W_learned)

    # Log the absolute error in weights
    wandb.log({"metrics/learned/abs_err_weights_mean": float(jnp.abs(W_learned - W_true).mean())})

    # Generate long sequences
    print("Generating long trajectory from learned model...")
    long_t_emissions = jnp.arange(start=0.0, stop=cfg.T_long, step=cfg.dt).reshape(-1, 1)
    long_learned_data = predictive_learned(next(keys), t_emissions=long_t_emissions, **known_values)
    print("Generating long trajectory from true model...")
    long_true_data = Predictive(model, num_samples=1)(next(keys), t_emissions=long_t_emissions, **true_values)
    burnin_frac = 0.5
    burnin_idx = int(burnin_frac * long_t_emissions.shape[0])
    fig = plot_traj_kde({"Observed traj": long_true_data["emissions"].squeeze(0)[burnin_idx:], "Long learned traj": long_learned_data["emissions"].squeeze(0)[burnin_idx:]}, per_dim=False)
    wandb.log({"figures/emissions_kde": wandb.Image(fig)})
    plt.close(fig)

    fig = plot_traj_kde({"True states": long_true_data["states"].squeeze(0)[burnin_idx:], "Learned states": long_learned_data["states"].squeeze(0)[burnin_idx:]}, per_dim=False)
    wandb.log({"figures/states_kde": wandb.Image(fig)})
    plt.close(fig)


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
        wandb.log({"figures/learned/ensembles_0to20": wandb.Image(fig)})
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
    parser.add_argument("--laplace_scale", type=float, default=0.5)
    parser.add_argument("--eps_perturb", type=float, default=0.1) # noise level for initializing near the truth

    # True system parameters
    parser.add_argument("--initial_cov", type=float, default=25.0) # initial state covariance (times identity)
    parser.add_argument("--emission_cov", type=float, default=1.0) # emission noise covariance (times identity)
    parser.add_argument("--diffusion_coeff", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=40) # final time
    parser.add_argument("--dt", type=float, default=0.01) # time step size
    parser.add_argument("--state_dim", type=int, default=5) # number of states in the Lorenz96 model
    parser.add_argument("--emission_dim", type=int, default=5) # observation dimension (default is to observe the first "emission_dim" states)

    # Evaluation parameters
    parser.add_argument("--T_long", type=int, default=4000) # final time for long rollouts
    
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
    parser.add_argument("--project", type=str, default="l96_dict_laplace")
    parser.add_argument("--run_name", type=str, default=None) # Allows you to custom-specify wandb run name (else uses default)
    parser.add_argument("--dir", type=str, default="demo_outputs/l96_LaplaceDict") # wandb logging directory
    parser.add_argument("--online", type=int, default=0) # 0 for offline, 1 for online (need to configure wandb account for online)

    # Parse arguments
    args = parser.parse_args()
    
        
    main(**vars(args))
