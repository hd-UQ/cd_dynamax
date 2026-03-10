import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# continuous-discrete nonlinear Gaussian SSM codebase
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm import (
    cdlgssm_filter,
    cdlgssm_forecast,
    ParamsCDLGSSM,
    KFHyperParams
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    ParamsCDNLGSSM,
    EKFHyperParams,
    UKFHyperParams,
    EnKFHyperParams,
)

from cd_dynamax.src.continuous_discrete_nonlinear_ssm import (
    ParamsCDNLSSM,
    DPFHyperParams,
)

# Cd-dynamax utilities
from cd_dynamax.src.utils.simulation_utils import (
    make_key_sequence,
    cddynamax_filter,
    cddynamax_forecast,
    cddynamax_emissions
)

from cd_dynamax.src.utils.experiment_utils import (
    create_cddynamax_model_from_config,
    generate_t_emissions
)

# Set this to True to plot the results of the tests for visual inspection
PLOT_TEST_RESULTS = False

# Use a fixed random seed for reproducibility in tests
@pytest.fixture
def seed():
    return 0

# Check whether continuous-discrete model filtering and forecasting run
def test_filter_forecast_emissions(seed):
    # Models, as defined in config files
    test_models= [
        'model/cdlgssm_x1',
        'model/true_l63_mech',
        'model/cdnlssm_oudrift_poissondata',
    ]

    # Iterate checking of filtering and forecasting for each model
    for model_config in test_models:
        print(f"Testing filtering and forecasting for model config {model_config}...")
        check_model_filter_forecast_emissions(
            model_config_file=model_config,
            key=seed,
        )
        print(f"... all tests passed for model config {model_config}!")
    
# Check whether continuous-discrete model filtering and forecasting run
def check_model_filter_forecast_emissions(
        model_config_file,
        key,
        config_path='../demos/python/configs/'
    ):

    # Sequence of keys
    keys = make_key_sequence(key)

    # Create and initialize the cd-dynamax model from the model config file
    cd_model, cd_params, cd_props = create_cddynamax_model_from_config(
        config_path=config_path,
        true_model_config_file=model_config_file,
        overrides=None,
    )

    # Simulation
    # Regularly sampled in [0,1]
    num_timesteps, t_all=generate_t_emissions(
        t0=0,
        t1=1,
        num_samples=100,
        irregular_samples = False,
        key=next(keys),
    )
    # Filtering Vs forecasting time points
    sample_idx = jnp.floor(num_timesteps * 0.8).astype(int)
    t_emissions = t_all[:sample_idx]
    t_forecast_emissions = t_all[sample_idx:]
    
    # No inputs for now
    inputs = None

    # Sample from the continuous-discrete model
    cd_states, cd_emissions = cd_model.sample(
        params=cd_params,
        key=next(keys),
        num_timesteps=len(t_emissions),
        t_emissions=t_emissions,
        inputs=inputs,
    )

    ### Filtering
    # To decide what filtering algorithm to use
    if isinstance(cd_params, ParamsCDLGSSM):
        # Linear case with Kalman filter
        filter_hyperparams_list = [KFHyperParams(dt_final=1.0)]
    elif isinstance(cd_params, ParamsCDNLGSSM):
        # Nonlinear Gaussian case
        # EKF, UKF and EnKF
        filter_hyperparams_list = [
            EKFHyperParams(),
            UKFHyperParams(),
            EnKFHyperParams()
        ]
    elif isinstance(cd_params, ParamsCDNLSSM):
        # Nonlinear case with DPF
        filter_hyperparams_list =[
            DPFHyperParams()
        ]
    else:
        raise ValueError(f"Unknown model parameters type {type(cd_params)}")
    
    # Iterate over filters
    for filter_hyperparams in filter_hyperparams_list:
        print(f"Testing filtering and forecasting for model config {model_config_file} with filter {type(filter_hyperparams)}...")
        # Filter the emissions to get filtered posterior
        filtered = cddynamax_filter(
            model_params=cd_params,
            filter_hyperparams=filter_hyperparams,
            t_emissions=t_emissions,
            emissions=cd_emissions,
            start_idx_filter=0,
            stop_idx_filter=num_timesteps-1,
            key=next(keys),
            filter_spec='model'
        )

        # Check that the filtered posterior has the expected structure and dimensions
    assert filtered.filtered_means.shape == (len(t_emissions), cd_model.state_dim), \
        f"Expected filtered means shape {(len(t_emissions), cd_model.state_dim)}, got {filtered.filtered_means.shape}"
    assert filtered.filtered_covariances.shape == (len(t_emissions), cd_model.state_dim, cd_model.state_dim), \
        f"Expected filtered covariances shape {(len(t_emissions), cd_model.state_dim, cd_model.state_dim)}, got {filtered.filtered_covariances.shape}"

    # Check filtered means and covariances are finite
    assert jnp.all(jnp.isfinite(filtered.filtered_means)), \
        "Filtered means contain non-finite values"
    assert jnp.all(jnp.isfinite(filtered.filtered_covariances)), \
        "Filtered covariances contain non-finite values"

    # Check returned log-likelihood is a scalar floating JAX value and finite
    marginal_loglik = filtered.marginal_loglik
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
                filtered.filtered_means[:, n_state],
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

    ### Forecasting
    # Initialize forecast with last filtered state
    init_time = t_emissions[num_timesteps - 1]
    if isinstance(cd_params, (ParamsCDLGSSM, ParamsCDNLGSSM)):
        from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
        init_forecast = MVN(
            filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :]
        )
    elif isinstance(cd_params, ParamsCDNLSSM):
        init_forecast = filtered.particles[-1, ...]

    # Forecast
    forecasted = cddynamax_forecast(
        model_params=cd_params,
        filter_hyperparams=filter_hyperparams,
        t_init=t_emissions[-1],
        init_forecast=init_forecast,
        t_forecast=t_forecast_emissions,
        key=next(keys),
        filter_spec='model'
    )

    # Check forecasted state means and covariances have the expected structure and dimensions, and are finite
    assert forecasted.forecasted_state_means.shape == (len(t_forecast_emissions), cd_model.state_dim), \
        f"Expected forecasted state means shape {(len(t_forecast_emissions), cd_model.state_dim)}, got {forecasted.forecasted_state_means.shape}"
    assert forecasted.forecasted_state_covariances.shape == (len(t_forecast_emissions), cd_model.state_dim, cd_model.state_dim), \
        f"Expected forecasted state covariances shape {(len(t_forecast_emissions), cd_model.state_dim, cd_model.state_dim)}, got {forecasted.forecasted_state_covariances.shape}"
    assert jnp.all(jnp.isfinite(forecasted.forecasted_state_means)), \
        "Forecasted state means contain non-finite values"
    assert jnp.all(jnp.isfinite(forecasted.forecasted_state_covariances)), \
        "Forecasted state covariances contain non-finite values"

    ### Emissions
    # Compute emissions for filtered and forecasted states
    filtered_emissions, forecasted_emissions = cddynamax_emissions(
        model=cd_model,
        model_params=cd_params,
        t_emissions_filter=t_emissions,
        filtered_state=filtered,
        t_emissions_forecast=t_forecast_emissions,
        forecasted_state=forecasted,
        filtering_inputs=None,
        forecasting_inputs=None,
        key=next(keys),
    )

    # Check forecasted emissions means and covariances have the expected structure and dimensions, and are finite
    assert filtered_emissions['mean'].shape == (len(t_emissions), cd_model.emission_dim), \
        f"Expected filtered emissions mean shape {(len(t_emissions), cd_model.emission_dim)}, got {filtered_emissions['mean'].shape}"
    assert filtered_emissions['cov'].shape == (len(t_emissions), cd_model.emission_dim, cd_model.emission_dim), \
        f"Expected filtered emissions covariances shape {(len(t_emissions), cd_model.emission_dim, cd_model.emission_dim)}, got {filtered_emissions['cov'].shape}"
    assert jnp.all(jnp.isfinite(filtered_emissions['mean'])), \
        "Filtered emissions means contain non-finite values"
    assert jnp.all(jnp.isfinite(filtered_emissions['cov'])), \
        "Filtered emissions covariances contain non-finite values"
    assert forecasted_emissions['mean'].shape == (len(t_forecast_emissions), cd_model.emission_dim), \
        f"Expected forecasted emissions means shape {(len(t_forecast_emissions), cd_model.emission_dim)}, got {forecasted_emissions['mean'].shape}"
    assert forecasted_emissions['cov'].shape == (len(t_forecast_emissions), cd_model.emission_dim, cd_model.emission_dim), \
        f"Expected forecasted emissions covariances shape {(len(t_forecast_emissions), cd_model.emission_dim, cd_model.emission_dim)}, got {forecasted_emissions['cov'].shape}"
    assert jnp.all(jnp.isfinite(forecasted_emissions['mean'])), \
        "Forecasted emissions means contain non-finite values"
    assert jnp.all(jnp.isfinite(forecasted_emissions['cov'])), \
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
                filtered.filtered_means[:, n_state],
                label="Continuous-Discrete filtered state",
                color="blue",
                marker="x",
            )
            plt.xlabel("time")
            plt.ylabel(f"x_{n_state}")
            plt.grid()
            plt.legend()
            plt.title("Filtered states")
            plt.show()

        for n_state in jnp.arange(cd_model.state_dim):
            plt.figure()
            plt.plot(
                t_forecast_emissions,
                forecasted.forecasted_state_means[:, n_state],
                label="Forecasted state means",
                color="orange",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.fill_between(
                t_forecast_emissions[:, 0],
                forecasted.forecasted_state_means[:, n_state]
                - jnp.sqrt(forecasted.forecasted_state_covariances[:, n_state, n_state]),
                forecasted.forecasted_state_means[:, n_state]
                + jnp.sqrt(forecasted.forecasted_state_covariances[:, n_state, n_state]),
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
                forecasted_emissions['mean'][:, n_emission],
                label="Forecasted emission means",
                color="orange",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.fill_between(
                t_forecast_emissions[:, 0],
                forecasted_emissions['mean'][:, n_emission]
                - jnp.sqrt(
                    forecasted_emissions['cov'][:, n_emission, n_emission]
                ),
                forecasted_emissions['mean'][:, n_emission]
                + jnp.sqrt(
                    forecasted_emissions['cov'][:, n_emission, n_emission]
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
    #test_filter_forecast_emissions(0)
    pass