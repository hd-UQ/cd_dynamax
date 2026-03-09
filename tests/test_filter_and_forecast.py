import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Dynamax Discrete-Discrete Linear Gaussian SSM (lgssm) model, filter and smoother
from cd_dynamax.dynamax.linear_gaussian_ssm import LinearGaussianSSM
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother

# CD-Dynamax cdlgssm model, filter and smoother
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    KFHyperParams,
    cdlgssm_filter,
    cdlgssm_forecast,
    cdlgssm_emissions
)

# Utilities for testing and comparing results
from cd_dynamax.src.utils.test_utils import compare, compare_structs

# Set this to True to plot the results of the tests for visual inspection
PLOT_TEST_RESULTS = False

# Use a fixed random seed for reproducibility in tests
@pytest.fixture
def rng_keys():
    return jr.split(jr.PRNGKey(0))

# Initialization of example cd-dynamax model
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector
def init_cdlgssm_model(key_init):
    """Helper function to initialize equivalent discrete LGSSM and a continuous-discrete LGSSM for testing.
        This equivalence is hardcoded for a single example, other equivalences are possible
    """
    # Model definition parameters: shared
    state_dim = 2
    emission_dim = 6

    # Continuous-discrete model setup with parameters that should closely match the discrete model dynamics and emissions
    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=state_dim,
        emission_dim=emission_dim
    )
    cd_params, cd_param_props = cd_model.initialize(
        key_init,
        initial_mean={
            "params": jnp.zeros(cd_model.state_dim),
            "props": ParameterProperties(),
        },
        initial_cov={
            "params": jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
        dynamics_weights={
            "params": -0.1 * jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(),
        },
        dynamics_bias={
            "params": jnp.zeros((cd_model.state_dim,)),
            "props": ParameterProperties(),
        },
        dynamics_diffusion_coefficient={
            "params": jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(),
        },
        dynamics_diffusion_cov={
            "params": (0.5 * 0.5) * 0.5 * jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
        emission_weights={
            "params": jr.normal(key_init, (cd_model.emission_dim, cd_model.state_dim)),
            "props": ParameterProperties(),
        },
        emission_bias={
            "params": jnp.zeros((cd_model.emission_dim,)),
            "props": ParameterProperties(),
        },
        emission_cov={
            "params": 0.1 * jnp.eye(cd_model.emission_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
    )

    # Return the initialized models and parameters for use in tests
    return cd_model, cd_params, cd_param_props

# Check whether continuous-discrete model filtering and forecasting run
def test_cdlgssm_filter_and_forecast(rng_keys):
    # Unpack RNG keys
    key_init, key_sample = rng_keys

    # Define discrete and continous-discrete models
    cd_model, cd_params, cd_param_props = init_cdlgssm_model(key_init)

    # Simulation params
    num_timesteps = 100
    forecast_timesteps = 25
    
    # Continuous-discrete model's t_emissions
    t_emissions = jnp.arange(num_timesteps)[:, None]
    t_forecast_emissions = jnp.arange(
        num_timesteps, num_timesteps + forecast_timesteps
    )[:, None]
    # For the forecast, we start from the last time point of the emissions,
    t_forecast_init = t_emissions[-1]
    
    # No inputs for now
    inputs = None

    ## Continous-discrete model
    # Sample from the continuous-discrete model
    cd_states, cd_emissions = cd_model.sample(
        cd_params,
        key_sample,
        num_timesteps=num_timesteps,
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Filter the discrete model emissions to get filtered posterior
    kf_hyperparams = KFHyperParams(dt_final=1.0)
    cd_filtered_posterior = cdlgssm_filter(
        cd_params,
        cd_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
    )

    # Check that the filtered posterior has the expected structure and dimensions
    assert cd_filtered_posterior.filtered_means.shape == (num_timesteps, cd_model.state_dim), \
        f"Expected filtered means shape {(num_timesteps, cd_model.state_dim)}, got {cd_filtered_posterior.filtered_means.shape}"
    assert cd_filtered_posterior.filtered_covariances.shape == (num_timesteps, cd_model.state_dim, cd_model.state_dim), \
        f"Expected filtered covariances shape {(num_timesteps, cd_model.state_dim, cd_model.state_dim)}, got {cd_filtered_posterior.filtered_covariances.shape}"

    # Check filtered means and covariances are finite
    assert jnp.all(jnp.isfinite(cd_filtered_posterior.filtered_means)), \
        "Filtered means contain non-finite values"
    assert jnp.all(jnp.isfinite(cd_filtered_posterior.filtered_covariances)), \
        "Filtered covariances contain non-finite values"

    # Check returned log-likelihood is a scalar floating JAX value and finite
    marginal_loglik = cd_filtered_posterior.marginal_loglik
    assert jnp.ndim(marginal_loglik) == 0, \
        f"Expected marginal log-likelihood to be scalar, got shape {jnp.shape(marginal_loglik)}"
    assert jnp.issubdtype(marginal_loglik.dtype, jnp.floating), \
        f"Expected floating dtype for marginal log-likelihood, got {marginal_loglik.dtype}"
    assert jnp.isfinite(marginal_loglik), \
        "Marginal log-likelihood is not finite"

    # If intersted, plot the results to visually inspect the filtering and forecasting performance
    if PLOT_TEST_RESULTS:
        import matplotlib.pyplot as plt

        for n_state in jnp.arange(cd_model.state_dim):
            plt.figure()
            plt.plot(
                t_emissions,
                cd_states[:, n_state],
                label="true discrete position",
                color="black",
            )
            plt.plot(
                t_emissions,
                cd_filtered_posterior.filtered_means[:, n_state],
                label="Continuous-Discrete filtered state",
                color="blue",
                marker="x",
            )
            plt.xlabel("time")
            plt.ylabel(f"x_{n_state}")
            plt.grid()
            plt.legend()
            plt.title("Filtered states after SGD optimization")
            plt.show()

    # Forecasting with the continuous-discrete model
    # New key for forecasting to ensure independence from sampling and filtering steps
    key_forecast, keys = jr.split(key_sample)

    # For the point forecast, we use the last filtered mean as the initial condition.
    cd_point_init_forecast = cd_filtered_posterior.filtered_means[-1, :]
    cd_point_forecasted = cdlgssm_forecast(
        params=cd_params,
        init_forecast=cd_point_init_forecast,
        t_init=t_forecast_init,
        t_forecast=t_forecast_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        key=key_forecast,
        diffeqsolve_settings={},
    )

    # Check forecasted state path has the expected structure and dimensions, and are finite
    assert cd_point_forecasted.forecasted_state_path.shape == (forecast_timesteps, cd_model.state_dim), \
        f"Expected forecasted state path shape {(forecast_timesteps, cd_model.state_dim)}, got {cd_point_forecasted.forecasted_state_path.shape}"
    assert jnp.all(jnp.isfinite(cd_point_forecasted.forecasted_state_path)), \
        "Forecasted state path contains non-finite values"

    # Emissions forecasts:
    # Use forecasted paths
    cd_point_emissions_forecasted_means, cd_point_emissions_forecasted_covariances = (
        cdlgssm_emissions(
            params=cd_params,
            t_states=t_forecast_emissions,
            state_means=cd_point_forecasted.forecasted_state_path,
            inputs=inputs,
        )
    )

    # Check forecasted emissions means and covariances have the expected structure and dimensions, and are finite
    assert cd_point_emissions_forecasted_means.shape == (forecast_timesteps, cd_model.emission_dim), \
        f"Expected forecasted emissions means shape {(forecast_timesteps, cd_model.emission_dim)}, got {cd_point_emissions_forecasted_means.shape}"
    assert cd_point_emissions_forecasted_covariances.shape == (forecast_timesteps, cd_model.emission_dim, cd_model.emission_dim)
    assert jnp.all(jnp.isfinite(cd_point_emissions_forecasted_means))
    assert jnp.all(jnp.isfinite(cd_point_emissions_forecasted_covariances))

    # For the distribution forecast,
    # we use the last filtered mean and covariance to define a Gaussian initial condition
    from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
    cd_dist_init_forecast = MVN(
        cd_filtered_posterior.filtered_means[-1, :],
        cd_filtered_posterior.filtered_covariances[-1, :],
    )
    cd_dist_forecasted = cdlgssm_forecast(
        params=cd_params,
        init_forecast=cd_dist_init_forecast,
        t_init=t_forecast_init,
        t_forecast=t_forecast_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        key=key_forecast,
        diffeqsolve_settings={},
    )

    # Check forecasted state means and covariances have the expected structure and dimensions, and are finite
    assert cd_dist_forecasted.forecasted_state_means.shape == (forecast_timesteps, cd_model.state_dim), \
        f"Expected forecasted state means shape {(forecast_timesteps, cd_model.state_dim)}, got {cd_dist_forecasted.forecasted_state_means.shape}"
    assert cd_dist_forecasted.forecasted_state_covariances.shape == (forecast_timesteps, cd_model.state_dim, cd_model.state_dim), \
        f"Expected forecasted state covariances shape {(forecast_timesteps, cd_model.state_dim, cd_model.state_dim)}, got {cd_dist_forecasted.forecasted_state_covariances.shape}"
    assert jnp.all(jnp.isfinite(cd_dist_forecasted.forecasted_state_means)), \
        "Forecasted state means contain non-finite values"
    assert jnp.all(jnp.isfinite(cd_dist_forecasted.forecasted_state_covariances)), \
        "Forecasted state covariances contain non-finite values"

    # Use distribution forecasts (means and covariances)
    cd_dist_emissions_forecasted_means, cd_dist_emissions_forecasted_covariances = (
        cdlgssm_emissions(
            params=cd_params,
            t_states=t_forecast_emissions,
            state_means=cd_dist_forecasted.forecasted_state_means,
            state_covs=cd_dist_forecasted.forecasted_state_covariances,
            inputs=inputs,
        )
    )

    # Check forecasted emissions means and covariances have the expected structure and dimensions, and are finite
    assert cd_dist_emissions_forecasted_means.shape == (forecast_timesteps, cd_model.emission_dim), \
        f"Expected forecasted emissions means shape {(forecast_timesteps, cd_model.emission_dim)}, got {cd_dist_emissions_forecasted_means.shape}"
    assert cd_dist_emissions_forecasted_covariances.shape == (forecast_timesteps, cd_model.emission_dim, cd_model.emission_dim), \
        f"Expected forecasted emissions covariances shape {(forecast_timesteps, cd_model.emission_dim, cd_model.emission_dim)}, got {cd_dist_emissions_forecasted_covariances.shape}"
    assert jnp.all(jnp.isfinite(cd_dist_emissions_forecasted_means)), \
        "Forecasted emissions means contain non-finite values"
    assert jnp.all(jnp.isfinite(cd_dist_emissions_forecasted_covariances)), \
        "Forecasted emissions covariances contain non-finite values"

    # If intersted, plot the results to visually inspect the filtering and forecasting performance
    if PLOT_TEST_RESULTS:
        import matplotlib.pyplot as plt

        for n_state in jnp.arange(cd_model.state_dim):
            plt.figure()
            plt.plot(
                t_emissions,
                cd_states[:, n_state],
                label="true discrete position",
                color="black",
            )
            plt.plot(
                t_emissions,
                cd_filtered_posterior.filtered_means[:, n_state],
                label="Continuous-Discrete filtered state",
                color="blue",
                marker="x",
            )
            plt.xlabel("time")
            plt.ylabel(f"x_{n_state}")
            plt.grid()
            plt.legend()
            plt.title("CD-LGSSM Filtered states")
            plt.show()

        for n_state in jnp.arange(cd_model.state_dim):
            plt.figure()
            plt.plot(
                t_forecast_emissions,
                cd_point_forecasted.forecasted_state_path[:, n_state],
                label="Forecasted path (point estimate)",
                color="black",
            )
            plt.plot(
                t_forecast_emissions,
                cd_dist_forecasted.forecasted_state_means[:, n_state],
                label="Forecasted state means (distribution)",
                color="orange",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.fill_between(
                t_forecast_emissions[:, 0],
                cd_dist_forecasted.forecasted_state_means[:, n_state]
                - jnp.sqrt(cd_dist_forecasted.forecasted_state_covariances[:, n_state, n_state]),
                cd_dist_forecasted.forecasted_state_means[:, n_state]
                + jnp.sqrt(cd_dist_forecasted.forecasted_state_covariances[:, n_state, n_state]),
                color="orange",
                alpha=0.2,
                label="Forecasted state uncertainty (1 std)",
            )
            plt.xlabel("Forecasted time")
            plt.ylabel(f"x_{n_state}")
            plt.grid()
            plt.legend()
            plt.title("Forecasted states")
            plt.show()

        for n_emission in jnp.arange(cd_model.emission_dim):
            plt.figure()
            plt.plot(
                t_forecast_emissions,
                cd_point_emissions_forecasted_means[:, n_emission],
                label="Forecasted emission path (point estimate)",
                color="black",
            )
            plt.plot(
                t_forecast_emissions,
                cd_dist_emissions_forecasted_means[:, n_emission],
                label="Forecasted emission means (distribution)",
                color="orange",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.fill_between(
                t_forecast_emissions[:, 0],
                cd_dist_emissions_forecasted_means[:, n_emission]
                - jnp.sqrt(
                    cd_dist_emissions_forecasted_covariances[:, n_emission, n_emission]
                ),
                cd_dist_emissions_forecasted_means[:, n_emission]
                + jnp.sqrt(
                    cd_dist_emissions_forecasted_covariances[:, n_emission, n_emission]
                ),
                color="orange",
                alpha=0.2,
                label="Forecasted emission uncertainty (1 std)",
            )
            plt.xlabel("Forecasted time")
            plt.ylabel(f"y_{n_emission}")
            plt.grid()
            plt.legend()
            plt.title("Forecasted emissions")
            plt.show()

if __name__ == "__main__":
    test_cdlgssm_filter_and_forecast(jr.split(jr.PRNGKey(0)))