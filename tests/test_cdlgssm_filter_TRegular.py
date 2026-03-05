import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    KFHyperParams,
    LinearGaussianSSM,
    cdlgssm_emissions,
    cdlgssm_forecast,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_filter
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.models import cdlgssm_filter
from cd_dynamax.src.utils.test_utils import compare, compare_structs
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

# Whether to plot test results or not - can be helpful for debugging but should be False for regular test runs
PLOT_TEST_RESULTS = False


# Use a fixed random seed for reproducibility in tests
@pytest.fixture
def rng_keys():
    return jr.split(jr.PRNGKey(0))


# Testing the filter and forecast functions together in a single test
# to check whether continuous-discrete model can recover discrete model (dynamax)
# filtering and forecasting results
def test_cdlgssm_filter_and_forecast_tregular(rng_keys):
    # Simple print statement to confirm test is running and show JAX device info
    print("Running on jax device:", jax.devices())

    # Unpack RNG keys
    key_init, key_sample = rng_keys

    # Model definition parameters: shared
    state_dim = 2
    emission_dim = 6
    num_timesteps = 100
    inputs = None

    # Discrete time model setup and filtering/forecasting
    # use dynamax to get reference results
    d_model = LinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
    d_params, d_param_props = d_model.initialize(
        key_init,
        dynamics_weights=0.9048373699188232421875 * jnp.eye(d_model.state_dim),
        dynamics_covariance=0.11329327523708343505859375 * jnp.eye(d_model.state_dim),
        dynamics_bias=jnp.zeros(d_model.state_dim),
        emission_bias=jnp.zeros(d_model.emission_dim),
    )

    # Sample from the discrete model to get emissions for filtering
    d_states, d_emissions = d_model.sample(
        d_params, key_sample, num_timesteps=num_timesteps, inputs=inputs
    )

    # Filter the discrete model emissions to get reference filtered posterior
    d_filtered_posterior = lgssm_filter(d_params, d_emissions, inputs)

    # Fit the discrete model using SGD to get fitted parameters and log-likelihoods
    d_sgd_fitted_params, d_sgd_lps = d_model.fit_sgd(
        d_params, d_param_props, d_emissions, inputs=inputs, num_epochs=10
    )

    # Filter the emissions with the fitted parameters to get the post-SGD fitted filtered posterior
    d_sgd_fitted_filtered_posterior = lgssm_filter(
        d_sgd_fitted_params, d_emissions, inputs
    )

    # Continuous-discrete model setup and filtering/forecasting
    t_emissions = jnp.arange(num_timesteps)[:, None]

    # cd-dynamax equivalence test: initialize cd model with parameters that should closely match the discrete model dynamics and emissions, then check that filtering and forecasting results are close to the discrete model results. This is a strong test of whether the continuous-discrete model can recover the discrete model behavior when initialized appropriately.
    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=state_dim, emission_dim=emission_dim
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

    # Sample from the continuous-discrete model using the same key
    # Based on the number of timesteps
    cd_num_timesteps_states, cd_num_timesteps_emissions = cd_model.sample(
        cd_params, key_sample, num_timesteps=num_timesteps, inputs=inputs
    )
    # Based on the emission timestamps
    cd_states, cd_emissions = cd_model.sample(
        cd_params,
        key_sample,
        num_timesteps=num_timesteps,
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Compare and check
    compare(cd_num_timesteps_states, cd_states)
    compare(cd_num_timesteps_emissions, cd_emissions)
    compare(d_states, cd_states)
    compare(d_emissions, cd_emissions)

    # Filter the continuous-discrete model emissions to get filtered posterior
    kf_hyperparams = KFHyperParams(dt_final=1.0)
    cd_filtered_posterior = cdlgssm_filter(
        cd_params,
        cd_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
    )

    # Compare the filtered posterior from the continuous-discrete model to the discrete model's filtered posterior
    compare_structs(d_filtered_posterior, cd_filtered_posterior)

    # Fit the continuous-discrete model using SGD to get fitted parameters and log-likelihoods, then filter again with the fitted parameters to get the post-SGD fitted filtered posterior
    cd_sgd_fitted_params, cd_sgd_lps = cd_model.fit_sgd(
        cd_params,
        cd_param_props,
        cd_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        num_epochs=10,
    )

    # Compare the SGD-fitted parameters and log-likelihoods between the discrete and continuous-discrete models.
    # We use accept_failure=True since we don't necessarily expect exact matches
    compare(cd_sgd_lps, d_sgd_lps)
    compare_structs(d_sgd_fitted_params, cd_sgd_fitted_params, accept_failure=True)

    # Filter with the SGD-fitted parameters to get the post-SGD fitted filtered posterior for the continuous-discrete model, and compare to the discrete model's post-SGD fitted filtered posterior
    cd_sgd_fitted_filtered_posterior = cdlgssm_filter(
        cd_sgd_fitted_params,
        cd_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
    )

    # Compare the post-SGD fitted filtered posterior from the continuous-discrete model to the discrete model's post-SGD fitted filtered posterior.
    ## We use accept_failure=True since we don't necessarily expect exact matches
    compare_structs(
        d_sgd_fitted_filtered_posterior,
        cd_sgd_fitted_filtered_posterior,
        accept_failure=True,
    )

    # Forecasting with the continuous-discrete model using both point and distribution initial conditions from the filtered posterior, and compare to the discrete model's forecasting results. This tests whether the continuous-discrete model can produce similar forecasts to the discrete model when initialized with the same filtered state estimates.
    forecast_timesteps = 25
    t_forecast_emissions = jnp.arange(
        num_timesteps, num_timesteps + forecast_timesteps
    )[:, None]
    # For the forecast, we start from the last time point of the emissions,
    # so we use the last timestamp from t_emissions as the initial time for the forecast
    init_time = t_emissions[-1]

    # New key for forecasting to ensure independence from sampling and filtering steps
    key_forecast = jr.split(key_sample)[0]

    # For the point forecast, we use the last filtered mean as the initial condition.
    cd_point_init_forecast = cd_filtered_posterior.filtered_means[-1, :]
    cd_point_forecasted = cdlgssm_forecast(
        params=cd_params,
        init_forecast=cd_point_init_forecast,
        t_init=init_time,
        t_forecast=t_forecast_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        key=key_forecast,
        diffeqsolve_settings={},
    )

    # For the distribution forecast,
    # we use the last filtered mean and covariance to define a Gaussian initial condition
    cd_dist_init_forecast = MVN(
        cd_filtered_posterior.filtered_means[-1, :],
        cd_filtered_posterior.filtered_covariances[-1, :],
    )
    cd_dist_forecasted = cdlgssm_forecast(
        params=cd_params,
        init_forecast=cd_dist_init_forecast,
        t_init=init_time,
        t_forecast=t_forecast_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        key=key_forecast,
        diffeqsolve_settings={},
    )

    # Emissions forecasts from the continuous-discrete model using the forecasted states
    # Use point forecasts
    cd_point_emissions_forecasted_means, cd_point_emissions_forecasted_covariances = (
        cdlgssm_emissions(
            params=cd_params,
            t_states=t_forecast_emissions,
            state_means=cd_point_forecasted.forecasted_state_path,
            inputs=inputs,
        )
    )

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

    # Checks
    assert cd_point_forecasted.forecasted_state_path.shape[0] == forecast_timesteps
    assert cd_dist_forecasted.forecasted_state_means.shape[0] == forecast_timesteps
    assert cd_point_emissions_forecasted_means.shape[0] == forecast_timesteps
    assert cd_dist_emissions_forecasted_means.shape[0] == forecast_timesteps

    # If intersted, plot the results to visually inspect the filtering and forecasting performance
    if PLOT_TEST_RESULTS:
        import matplotlib.pyplot as plt

        for n_state in jnp.arange(state_dim):
            plt.figure()
            plt.plot(
                t_emissions,
                d_states[:, n_state],
                label="true discrete position",
                color="black",
            )
            plt.plot(
                t_emissions,
                d_sgd_fitted_filtered_posterior.filtered_means[:, n_state],
                label="Post-SGD fit Discrete filtered state",
                color="orange",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.plot(
                t_emissions,
                cd_sgd_fitted_filtered_posterior.filtered_means[:, n_state],
                label="Post-SGD fit Continuous-Discrete filtered state",
                color="blue",
                marker="x",
            )
            plt.xlabel("time")
            plt.ylabel(f"x_{n_state}")
            plt.grid()
            plt.legend()
            plt.title("Filtered states after SGD optimization")
            plt.show()

        for n_state in jnp.arange(state_dim):
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
                - jnp.sqrt(
                    cd_dist_forecasted.forecasted_state_covariances[:, n_state, n_state]
                ),
                cd_dist_forecasted.forecasted_state_means[:, n_state]
                + jnp.sqrt(
                    cd_dist_forecasted.forecasted_state_covariances[:, n_state, n_state]
                ),
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

        for n_emission in jnp.arange(emission_dim):
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
