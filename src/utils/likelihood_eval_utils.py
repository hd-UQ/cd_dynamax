import sys
from itertools import count
import argparse

import datetime
import numpy as np

# To be able to debug
import os, jax
# os.environ["JAX_DISABLE_JIT"] = "1"
# os.environ["EQX_ON_ERROR"] = "breakpoint"
# os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = "100"
# jax.config.update("jax_traceback_filtering", "off")

# Import jax and utils
from jax import numpy as jnp
import jax.random as jr
from typing import Union
from jaxtyping import Float, Array

# Additional, custom codebase
sys.path.append("../")
sys.path.append("../..")

# Import dynamax
from dynamax.parameters import ParameterProperties

# Our own custom src codebase
# continuous-discrete nonlinear Gaussian SSM codebase
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm import cdnlgssm_filter
# Load models
from continuous_discrete_nonlinear_gaussian_ssm.models import *

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
# Our own custom plotting codebase
# Useful utility functions
from utils.plotting_utils import *
sys.path.append("../notebooks/tutorial")
from simulation_utils import *
from lorenz_plotting import *


# JAX device check
print("************* Checking JAX device *************")
print("Running on jax device:{}".format(jax.devices()))
print("Running on jax device platform:{}".format(jax.devices()[0].platform))
print("***********************************************")


def experiment(
    times,
    emissions,
    model,
    true_params_dict,
    drift_class,  # must only have .params field, which is a vector
    param_sweep_1,
    param_sweep_2,
    sweep_indices=(0, 1), # only for drift_class.params
    hyperparam_dict={"EKF 1st order": EKFHyperParams(state_order="first")},
    output_dir=".",
    plot_truth=True,
    show=False,
):

    # Set up seed for simulation
    keys = map(jr.PRNGKey, count())

    true_params, _ = model.initialize(next(keys), **true_params_dict)

    def get_params_with_drift_component_changed(new_value_1, new_value_2):
        '''
        Returns a new set of parameters with the drift component changed to the new values.
        '''
        new_drift_params = true_params.dynamics.drift.params.copy()
        # use .at instead of [] to avoid in-place modification
        new_drift_params = new_drift_params.at[sweep_indices[0]].set(new_value_1)
        new_drift_params = new_drift_params.at[sweep_indices[1]].set(new_value_2)

        new_drift = {
            "params": drift_class(params=new_drift_params),
            "props": drift_class(params=ParameterProperties()),
        }

        params_dict = true_params_dict.copy()
        params_dict['dynamics_drift'] = new_drift
        new_params, _ = model.initialize(next(keys), **params_dict)
        return new_params

    def compute_log_likelihood_grad_and_covariances(new_value_1, new_value_2, hyperparams):
        def marginal_loglik_fn(values):
            # Extract values for new_value_1 and new_value_2
            v1, v2 = values
            new_params = get_params_with_drift_component_changed(v1, v2)
            filtered = cdnlgssm_filter(new_params, emissions, times, hyperparams=hyperparams)
            return filtered.marginal_loglik

        # Get value, gradient with respect to new_value_1 and new_value_2, and predicted covariances
        values = jnp.array([new_value_1, new_value_2])
        mll_value, grad_values = jax.value_and_grad(marginal_loglik_fn)(values)

        # Compute predicted covariances
        new_params = get_params_with_drift_component_changed(new_value_1, new_value_2)
        filtered = cdnlgssm_filter(new_params, emissions, times, hyperparams=hyperparams)
        predicted_covariances = filtered.predicted_covariances

        return mll_value, grad_values, predicted_covariances

    # Now, sweep over the two parameter values using each hyperparameter setting
    for hyperparam_name, hyperparam in hyperparam_dict.items():
        print(f"Running experiment for hyperparameter setting: {hyperparam_name}")
        # Compute the TRUE log-likelihood at the true parameter values
        true_ll, true_grad, true_pred_cov = compute_log_likelihood_grad_and_covariances(
            true_params.dynamics.drift.params[sweep_indices[0]],
            true_params.dynamics.drift.params[sweep_indices[1]],
            hyperparam,
        )
        is_psd_batched = jax.vmap(lambda x: jnp.all(jnp.linalg.eigvalsh(x) >= 0), in_axes=0)(true_pred_cov)
        true_non_psd_count = jnp.sum(~is_psd_batched)

        log_likelihoods = np.zeros((len(param_sweep_1), len(param_sweep_2)))
        gradients = np.zeros((len(param_sweep_1), len(param_sweep_2), 2))
        non_psd_counts = np.zeros((len(param_sweep_1), len(param_sweep_2)))

        # Define a single computation function for a pair (value_1, value_2)
        def compute_single(value_1, value_2, hyperparam):
            mll, grad_value, predicted_covariances = compute_log_likelihood_grad_and_covariances(
                value_1, value_2, hyperparam
            )
            is_psd_batched = jax.vmap(lambda x: jnp.all(jnp.linalg.eigvalsh(x) >= 0))(predicted_covariances)
            non_psd_count = jnp.sum(~is_psd_batched)
            return mll, grad_value, non_psd_count

        # Vectorize over both param_sweep_1 and param_sweep_2
        def compute_full(param_sweep_1, param_sweep_2, hyperparam):
            # Combine param_sweep_1 and param_sweep_2 into a grid
            param_grid_1, param_grid_2 = jnp.meshgrid(param_sweep_1, param_sweep_2, indexing='ij')

            # Flatten the grids for batching
            flat_param_grid_1 = param_grid_1.ravel()
            flat_param_grid_2 = param_grid_2.ravel()

            # Apply vmap to compute results for each pair
            results = jax.vmap(
                lambda v1, v2: compute_single(v1, v2, hyperparam)
            )(flat_param_grid_1, flat_param_grid_2)

            # Reshape results back to the grid shape
            mll_values = results[0].reshape(param_grid_1.shape)
            gradients = results[1].reshape(param_grid_1.shape + (2,))
            non_psd_counts = results[2].reshape(param_grid_1.shape)

            return mll_values, gradients, non_psd_counts

        # Call the vectorized function
        log_likelihoods, gradients, non_psd_counts = compute_full(param_sweep_1, param_sweep_2, hyperparam)

        path_name = os.path.join(output_dir, f"{hyperparam_name.replace(' ', '_')}_results.npz")
        np.savez_compressed(
            path_name,
            param_sweep_1=param_sweep_1,
            param_sweep_2=param_sweep_2,
            log_likelihoods=log_likelihoods,
            gradients=gradients,
            non_psd_counts=non_psd_counts,
            true_param_1=true_params.dynamics.drift.params[sweep_indices[0]],
            true_param_2=true_params.dynamics.drift.params[sweep_indices[1]],
            true_ll=true_ll,
            true_grad=true_grad,
            true_non_psd_count=true_non_psd_count,
            hyperparam_name=hyperparam_name,
        )

        # Plot the results
        plot_results(output_dir, plot_truth=plot_truth, show=show, path_name=path_name)


def plot_results(output_dir, plot_truth=True, show=False, path_name=None):
    import glob

    # Find all result files in the output directory
    if path_name is None:
        result_files = glob.glob(os.path.join(output_dir, "*_results.npz"))
    else:
        result_files = [path_name]

    for result_file in result_files:
        data = np.load(result_file)
        param_sweep_1 = data['param_sweep_1']
        param_sweep_2 = data['param_sweep_2']
        log_likelihoods = data['log_likelihoods']
        gradients = data['gradients']
        non_psd_counts = data['non_psd_counts']
        true_param_1 = data['true_param_1'].item()  # Extract scalar values
        true_param_2 = data['true_param_2'].item()
        true_ll = data['true_ll'].item()
        true_grad = data['true_grad']
        true_non_psd_count = data['true_non_psd_count'].item()
        hyperparam_name = data['hyperparam_name'].item()  # Extract string

        make_2d_plots(param_sweep_1, param_sweep_2, log_likelihoods, gradients, non_psd_counts,
                      hyperparam_name, true_param_1, true_param_2, output_dir, true_ll, plot_truth=plot_truth,
                      show=show)


def make_2d_plots(
    param_sweep_1,
    param_sweep_2,
    lls,
    gradients,
    non_psd_counts,
    hyperparam_name,
    true_param_1,
    true_param_2,
    output_dir,
    true_ll,
    plot_truth=True,
    show=False,
):
    # Mask NaN values
    lls_masked = np.ma.masked_invalid(lls)
    non_psd_counts_masked = np.ma.masked_invalid(non_psd_counts)

    # Create colormaps with transparent 'bad' values
    lls_cmap = plt.cm.viridis.copy()
    lls_cmap.set_bad(color="white", alpha=0)

    non_psd_cmap = plt.cm.plasma.copy()
    non_psd_cmap.set_bad(color="white", alpha=0)

    # Find the index of the maximum value in the likelihood surface, ignoring NaNs
    max_idx = np.unravel_index(np.nanargmax(lls), lls.shape)
    max_param_1 = param_sweep_1[max_idx[0]]
    max_param_2 = param_sweep_2[max_idx[1]]

    # Prepare grid for arrows
    x, y = np.meshgrid(param_sweep_1, param_sweep_2, indexing="ij")
    grad_x = gradients[..., 0]
    grad_y = gradients[..., 1]

    # Normalize gradients for arrow scaling
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_x_normalized = grad_x / (grad_magnitude + 1e-8)
    grad_y_normalized = grad_y / (grad_magnitude + 1e-8)

    # Original units plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Log-likelihoods
    # imshow takes X: (M, N) where M is the number of rows and N is the number of columns
    # That means that M will be the vertical axis and N will be the horizontal axis
    ll_surface = axs[0].imshow(
        lls_masked.T,
        extent=[param_sweep_1[0], param_sweep_1[-1], param_sweep_2[0], param_sweep_2[-1]],
        origin="lower",
        aspect="auto",
        cmap=lls_cmap,
    )
    axs[0].quiver(x, y, grad_x_normalized, grad_y_normalized, color="black", alpha=0.8, scale=20, width=0.003)
    if plot_truth:
        axs[0].scatter(
            true_param_1, true_param_2, color="red", marker="X", s=100, label=f"True Parameter (LL={true_ll:.2f})"
        )
    axs[0].scatter(
        max_param_1, max_param_2, color="blue", marker="o", s=100, label=f"Max Likelihood (LL={np.nanmax(lls):.2f})"
    )
    axs[0].set_xlabel("Parameter 1")
    axs[0].set_ylabel("Parameter 2")
    axs[0].set_title(f"Log-Likelihood Surface ({hyperparam_name}) - Original Units. True LL: {true_ll:.2f}")
    axs[0].legend()
    fig.colorbar(ll_surface, ax=axs[0], format="%.1e")

    # Plot 2: Number of non-PSD covariances
    non_psd_surface = axs[1].imshow(
        non_psd_counts_masked.T,
        extent=[param_sweep_1[0], param_sweep_1[-1], param_sweep_2[0], param_sweep_2[-1]],
        origin="lower",
        aspect="auto",
        cmap=non_psd_cmap,
    )
    if plot_truth:
        axs[1].scatter(true_param_1, true_param_2, color="red", marker="X", s=100, label="True Parameter")
    axs[1].scatter(max_param_1, max_param_2, color="blue", marker="o", s=100, label="Max Likelihood")
    axs[1].set_xlabel("Parameter 1")
    axs[1].set_ylabel("Parameter 2")
    axs[1].set_title(f"Non-PSD Predicted Covariances ({hyperparam_name}) - Original Units")
    axs[1].legend()
    fig.colorbar(non_psd_surface, ax=axs[1], format="%.1e")

    plt.savefig(
        os.path.join(output_dir, f"{hyperparam_name.replace(' ', '_')}_plots_original_units_with_gradients.png")
    )
    if show:
        plt.show()
    plt.close()

    # Log-scaled plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Log-likelihoods (log scale)
    ll_norm = matplotlib.colors.SymLogNorm(linthresh=1e-2, linscale=1, vmin=np.nanmin(lls), vmax=np.nanmax(lls))
    ll_surface_log = axs[0].imshow(
        lls_masked.T,
        extent=[param_sweep_1[0], param_sweep_1[-1], param_sweep_2[0], param_sweep_2[-1]],
        origin="lower",
        aspect="auto",
        cmap=lls_cmap,
        norm=ll_norm,
    )
    axs[0].quiver(x, y, grad_x_normalized, grad_y_normalized, color="black", alpha=0.8, scale=20, width=0.003)
    if plot_truth:
        axs[0].scatter(
            true_param_1, true_param_2, color="red", marker="X", s=100, label=f"True Parameter (LL={true_ll:.2f})"
        )
    axs[0].scatter(
        max_param_1, max_param_2, color="blue", marker="o", s=100, label=f"Max Likelihood (LL={np.nanmax(lls):.2f})"
    )
    axs[0].set_xlabel("Parameter 1")
    axs[0].set_ylabel("Parameter 2")
    axs[0].set_title(f"Log-Likelihood Surface ({hyperparam_name}) - Log Scale. True LL: {true_ll:.2f}")
    axs[0].legend()
    fig.colorbar(ll_surface_log, ax=axs[0], format="%.1e")

    # Plot 2: Number of non-PSD covariances (log scale)
    non_psd_norm = matplotlib.colors.SymLogNorm(
        linthresh=1e-2, linscale=1, vmin=np.nanmin(non_psd_counts), vmax=np.nanmax(non_psd_counts)
    )
    non_psd_surface_log = axs[1].imshow(
        non_psd_counts_masked.T,
        extent=[param_sweep_1[0], param_sweep_1[-1], param_sweep_2[0], param_sweep_2[-1]],
        origin="lower",
        aspect="auto",
        cmap=non_psd_cmap,
        norm=non_psd_norm,
    )
    if plot_truth:
        axs[1].scatter(true_param_1, true_param_2, color="red", marker="X", s=100, label="True Parameter")
    axs[1].scatter(max_param_1, max_param_2, color="blue", marker="o", s=100, label="Max Likelihood")
    axs[1].set_xlabel("Parameter 1")
    axs[1].set_ylabel("Parameter 2")
    axs[1].set_title(f"Non-PSD Predicted Covariances ({hyperparam_name}) - Log Scale")
    axs[1].legend()
    fig.colorbar(non_psd_surface_log, ax=axs[1], format="%.1e")

    plt.savefig(os.path.join(output_dir, f"{hyperparam_name.replace(' ', '_')}_plots_log_scale_with_gradients.png"))
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run experiments and/or plot results.')
    parser.add_argument('--run_experiments', action='store_true', help='Run the experiments.')
    parser.add_argument('--plot_results', action='store_true', help='Plot the results.')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for the results.')

    args = parser.parse_args()

    if args.run_experiments:
        # Set the output directory
        if args.output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Figures/experiment_{timestamp}"
        else:
            output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        #### Build the system
        # Set up seed for simulation
        keys = map(jr.PRNGKey, count())

        next(keys)

        T_total = 50
        num_timesteps_total = 10000

        # Generate synthetic data
        times, _, _, _, _, _ = generate_irregular_t_emissions(T_total=T_total,
                                                            num_timesteps=num_timesteps_total,
                                                            T_filter=T_total,
                                                            key=next(keys))

        # Define the true parameters of the Lorenz 63 system
        true_l63_drift_params=jnp.array([10.0, 28.0, 8 / 3])
        class lorenz63_drift(LearnableFunction):
            params: Union[Float[Array, "state_dim"], ParameterProperties]

            def f(self, x, u=None, t=None):
                foo = jnp.array(
                    [
                        self.params[0] * (x[1] - x[0]),
                        self.params[1] * x[0] - x[1] - x[0] * x[2],
                        -self.params[2] * x[2] + x[0] * x[1],
                    ]
                )
                return foo

        true_drift = {
            "params": lorenz63_drift(params=true_l63_drift_params),
            "props": lorenz63_drift(params=ParameterProperties()),
        }

        # Define other model parameters
        state_noise_sd = 1.0
        obs_noise_sd = 1.0
        init_state_sd = 30.0

        state_dim = 3
        emission_dim = 1
        true_diffusion_cov = {
            "params": LearnableMatrix(params=jnp.eye(state_dim)),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector())),
        }
        true_diffusion_coefficient = {
            "params": LearnableMatrix(
                params=state_noise_sd**2 * jnp.eye(state_dim)
            ),
            "props": LearnableMatrix(
                params=ParameterProperties()
            ),
        }
        true_emission = {
            "params": LearnableLinear(weights=jnp.array([[1.0, 0.0, 0.0]]), bias=jnp.zeros(emission_dim)),
            "props": LearnableLinear(weights=ParameterProperties(), bias=ParameterProperties()),
        }
        true_emission_cov = {
            "params": LearnableMatrix(params=obs_noise_sd**2 * jnp.eye(emission_dim)),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector())),
        }
        true_initial_mean = {
            "params": LearnableVector(params=jnp.zeros(state_dim)),
            "props": LearnableVector(params=ParameterProperties()),
        }
        true_initial_cov = {
            "params": LearnableMatrix(params=init_state_sd**2 * jnp.eye(state_dim)),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector())),
        }

        # Concatenate all parameters for the model
        true_params_dict = {
            'initial_mean': true_initial_mean,
            'initial_cov': true_initial_cov,
            'dynamics_drift': true_drift,
            'dynamics_diffusion_cov': true_diffusion_cov,
            'dynamics_diffusion_coefficient': true_diffusion_coefficient,
            'emission_function': true_emission,
            'emission_cov': true_emission_cov,
        }

        # Create the model
        model = ContDiscreteNonlinearGaussianSSM(state_dim, emission_dim)
        true_params, _ = model.initialize(next(keys), **true_params_dict)

        # Sample true states and emissions
        states, emissions = model.sample(true_params, next(keys), len(times), times, transition_type="path")

        # Run experiments
        # n_per_dim = 20
        n_per_dim = 20
        p0 = jnp.linspace(0, 18.0, n_per_dim)
        p1 = jnp.linspace(20, 50, n_per_dim)
        # p0 = jnp.linspace(5, 15.0, n_per_dim)
        # p1 = jnp.linspace(20, 40, n_per_dim)
        p2 = jnp.linspace(-1, 20, n_per_dim)

        experiment(
            times,
            emissions,
            model,
            true_params_dict,
            drift_class=lorenz63_drift,  # must only have .params field, which is a vector
            param_sweep_1=p0,
            param_sweep_2=p1,
            sweep_indices=(0, 1),  # only for drift_class.params
            hyperparam_dict={"EKF 1st order": EKFHyperParams(state_order="first")},
            output_dir=output_dir,
            plot_truth=True,
            show=False,
        )

    if args.plot_results:
        if args.output_dir is None:
            print("Please specify the output directory using --output_dir")
        else:
            plot_results(args.output_dir)

if __name__ == "__main__":
    main()
