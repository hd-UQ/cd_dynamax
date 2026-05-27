# Basic jax imports
import jax.numpy as jnp
import jax.random as jr
import pytest

# Dynamax Discrete-Discrete Linear Gaussian SSM (lgssm) model, filter and smoother
from cd_dynamax.dynamax.linear_gaussian_ssm import LinearGaussianSSM
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    lgssm_filter,
    lgssm_smoother,
)

# and additional utilities
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector

# CD-Dynamax cdlgssm model, filter and smoother
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    KFHyperParams,
    cdlgssm_filter,
    cdlgssm_smoother,
)

# Utilities for testing and comparing results
from cd_dynamax.src.utils.test_utils import compare, compare_structs

# Set this to True to plot the results of the tests for visual inspection
PLOT_TEST_RESULTS = False


# Use a fixed random seed for reproducibility in tests
@pytest.fixture
def rng_keys():
    return jr.split(jr.PRNGKey(0))


# Useful initialization of equivalent models
def init_cdlgssm_lgssm_models(key_init):
    """Helper function to initialize equivalent discrete LGSSM and a continuous-discrete LGSSM for testing.
    This equivalence is hardcoded for a single example, other equivalences are possible
    """
    # Model definition parameters: shared
    state_dim = 2
    emission_dim = 6

    # Discrete time model setup
    d_model = LinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
    d_params, d_param_props = d_model.initialize(
        key_init,
        dynamics_weights=0.9048373699188232421875 * jnp.eye(d_model.state_dim),
        dynamics_covariance=0.11329327523708343505859375 * jnp.eye(d_model.state_dim),
        dynamics_bias=jnp.zeros(d_model.state_dim),
        emission_bias=jnp.zeros(d_model.emission_dim),
    )

    # Continuous-discrete model setup with parameters that should closely match the discrete model dynamics and emissions
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

    # Return the initialized models and parameters for use in tests
    return d_model, d_params, d_param_props, cd_model, cd_params, cd_param_props


# Check whether continuous-discrete model definition (cd-dynamax) and sampling matches recover discrete model (dynamax)
def test_cdlgssm_lgssm_model_equivalence(rng_keys):
    # Unpack RNG keys
    key_init, key_sample = rng_keys

    # Define discrete and continous-discrete models
    d_model, d_params, d_param_props, cd_model, cd_params, cd_param_props = (
        init_cdlgssm_lgssm_models(key_init)
    )

    # Simulation params
    num_timesteps = 100
    # Continuous-discrete model's t_emissions
    t_emissions = jnp.arange(num_timesteps)[:, None]

    # No inputs for now
    inputs = None

    # Sample from the discrete model
    d_states, d_emissions = d_model.sample(
        d_params, key_sample, num_timesteps=num_timesteps, inputs=inputs
    )

    # Sample from the continuous-discrete model
    # Based on the emission timestamps
    cd_states, cd_emissions = cd_model.sample(
        cd_params,
        key_sample,
        num_timesteps=num_timesteps,
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Compare and check cd-dynamax matches dynamax
    compare(cd_states, d_states)
    compare(cd_emissions, d_emissions)


# Check whether continuous-discrete model filtering can recover discrete model's (dynamax) filtering
def test_cdlgssm_lgssm_filter_equivalence(rng_keys):
    # Unpack RNG keys
    key_init, key_sample = rng_keys

    # Define discrete and continous-discrete models
    d_model, d_params, d_param_props, cd_model, cd_params, cd_param_props = (
        init_cdlgssm_lgssm_models(key_init)
    )

    # Simulation params
    num_timesteps = 100
    # Continuous-discrete model's t_emissions
    t_emissions = jnp.arange(num_timesteps)[:, None]

    # No inputs for now
    inputs = None

    ## Discrete time model data and filtering as gold-standard
    # Sample from the discrete model
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

    ## Continous-discrete model: check filtering, and fitting
    # Filter the discrete model emissions to get filtered posterior
    kf_hyperparams = KFHyperParams(dt_final=1.0)
    cd_filtered_posterior = cdlgssm_filter(
        cd_params,
        d_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
    )

    # Compare the filtered posterior from the continuous-discrete model to the discrete model's filtered posterior
    compare_structs(d_filtered_posterior, cd_filtered_posterior)

    # Fit the continuous-discrete model using SGD to get fitted parameters and log-likelihoods
    cd_sgd_fitted_params, cd_sgd_lps = cd_model.fit_sgd(
        cd_params,
        cd_param_props,
        d_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        num_epochs=10,
    )

    # Compare the SGD-fitted parameters and log-likelihoods between the discrete and continuous-discrete models.
    # We use accept_failure=True since we don't necessarily expect exact matches
    compare(cd_sgd_lps, d_sgd_lps)
    compare_structs(d_sgd_fitted_params, cd_sgd_fitted_params, accept_failure=True)

    # Filter with the SGD-fitted parameters to get the post-SGD fitted filtered posterior for the continuous-discrete model,
    cd_sgd_fitted_filtered_posterior = cdlgssm_filter(
        cd_sgd_fitted_params,
        d_emissions,
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

    # If intersted, plot the results to visually inspect the filtering and forecasting performance
    if PLOT_TEST_RESULTS:
        import matplotlib.pyplot as plt

        for n_state in jnp.arange(cd_model.state_dim):
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


# Check whether continuous-discrete model smoothing can recover discrete model's (dynamax) smoothing
def test_cdlgssm_lgssm_smoother_equivalence(rng_keys):
    # Unpack RNG keys
    key_init, key_sample = rng_keys

    # Define discrete and continous-discrete models
    d_model, d_params, d_param_props, cd_model, cd_params, cd_param_props = (
        init_cdlgssm_lgssm_models(key_init)
    )

    # Simulation params
    num_timesteps = 100
    # Continuous-discrete model's t_emissions
    t_emissions = jnp.arange(num_timesteps)[:, None]

    # No inputs for now
    inputs = None

    # Sample from the discrete model
    d_states, d_emissions = d_model.sample(
        d_params, key_sample, num_timesteps=num_timesteps, inputs=inputs
    )

    # Discrete time smoothing
    print("Discrete time smoothing")
    d_smoother_posterior = lgssm_smoother(d_params, d_emissions, inputs)

    # Continuous-discrete smoothing
    # We set dt_final=1 so that predicted mean and covariance at the end of the sequence
    # match those of discrete filtering.
    kf_hyperparams = KFHyperParams(dt_final=1.0)

    # Iterate over both smoother types and compare results to discrete smoother
    for smoother_type in ["cd_smoother_1", "cd_smoother_2"]:
        print(f"Continuous-Discrete time KF smoothing {smoother_type}")
        cd_smoother_posterior = cdlgssm_smoother(
            cd_params,
            d_emissions,
            t_emissions,
            filter_hyperparams=kf_hyperparams,
            inputs=inputs,
            smoother_type=smoother_type,
        )

        print(f"Comparing {smoother_type} smoothed posteriors...")
        compare_structs(
            d_smoother_posterior, cd_smoother_posterior, accept_failure=True
        )

        print(
            f"All Discrete to Continuous-Discrete {smoother_type} smoothed posterior tests passed!"
        )

    # If interested, plot the results to visually inspect the impact of smoothing algorithm differences.
    if PLOT_TEST_RESULTS:
        print(
            "WARNING: plotting filtering results for understanding impact of smoothing algorithm differences."
        )
        import matplotlib.pyplot as plt

        for n_state in jnp.arange(d_model.state_dim):
            plt.figure()
            plt.plot(
                t_emissions,
                d_states[:, n_state],
                label="true discrete position",
                color="black",
            )
            plt.plot(
                t_emissions,
                d_smoother_posterior.filtered_means[:, n_state],
                label="Post-SGD fit Discrete filtered state",
                color="orange",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.plot(
                t_emissions,
                d_smoother_posterior.smoothed_means[:, n_state],
                label="Post-SGD fit Discrete smoothed state",
                color="red",
                marker="o",
                markerfacecolor="none",
                markeredgewidth=2,
                markersize=8,
            )
            plt.plot(
                t_emissions,
                cd_smoother_posterior.filtered_means[:, n_state],
                label="Post-SGD fit Continuous-Discrete filtered state",
                color="blue",
                marker="x",
            )
            plt.plot(
                t_emissions,
                cd_smoother_posterior.smoothed_means[:, n_state],
                label="Post-SGD fit Continuous-Discrete smoothed state",
                color="green",
                marker="x",
            )
            plt.xlabel("time")
            plt.ylabel("x_{}".format(n_state))
            plt.grid()
            plt.legend()
            plt.title("Filtered and smoothed states")
            plt.show()


def test_cdlgssm_smoother_verbosity(rng_keys, capsys):
    key_init, key_sample = rng_keys
    d_model, d_params, _, cd_model, cd_params, _ = init_cdlgssm_lgssm_models(key_init)

    num_timesteps = 10
    t_emissions = jnp.arange(num_timesteps)[:, None]
    _, d_emissions = d_model.sample(
        d_params, key_sample, num_timesteps=num_timesteps, inputs=None
    )
    kf_hyperparams = KFHyperParams(dt_final=1.0)

    cd_model.smoother(
        cd_params,
        d_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
    )
    assert capsys.readouterr().out == ""

    cd_model.smoother(
        cd_params,
        d_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        verbosity=2,
    )
    out = capsys.readouterr().out
    assert "Running KF smoother type = cd_smoother_1" in out
    assert "Running KF smoother type 1" in out


# Initialization of equivalent models with non-zero controls (inputs and bias)
def init_cdlgssm_lgssm_models_with_controls(key_init):
    """Initialize equivalent discrete LGSSM and CD-LGSSM with non-zero input weights and dynamics bias."""
    state_dim = 2
    emission_dim = 6
    input_dim = 3

    # Hardcoded discretization constants for F = -0.1*I, L = I, Qc = 0.125*I, dt = 1
    A_scalar = 0.9048373699188232421875
    Q_scalar = 0.11329327523708343505859375
    # C = \int e^{F (1-t)} dt = e^F \int e^{-F t} dt
    # = e^F (1 - e^{-F}) F^{-1} = (A - 1.0) / F * I.
    C_scalar = (A_scalar - 1.0) / (-0.1)

    # Continuous-time input weights and bias
    key_init, key_B = jr.split(key_init)
    B_cont = 0.1 * jr.normal(key_B, (state_dim, input_dim))
    b_cont = 0.1 * jnp.ones(state_dim)

    # Discrete model with discretized controls
    d_model = LinearGaussianSSM(
        state_dim=state_dim,
        emission_dim=emission_dim,
        input_dim=input_dim,
    )
    d_params, d_param_props = d_model.initialize(
        key_init,
        dynamics_weights=A_scalar * jnp.eye(state_dim),
        dynamics_covariance=Q_scalar * jnp.eye(state_dim),
        dynamics_input_weights=C_scalar * B_cont,
        dynamics_bias=C_scalar * b_cont,
        emission_bias=jnp.zeros(emission_dim),
    )

    # Continuous-discrete model
    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=state_dim,
        emission_dim=emission_dim,
        input_dim=input_dim,
    )
    cd_params, cd_param_props = cd_model.initialize(
        key_init,
        initial_mean={
            "params": jnp.zeros(state_dim),
            "props": ParameterProperties(),
        },
        initial_cov={
            "params": jnp.eye(state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
        dynamics_weights={
            "params": -0.1 * jnp.eye(state_dim),
            "props": ParameterProperties(),
        },
        dynamics_bias={
            "params": b_cont,
            "props": ParameterProperties(),
        },
        dynamics_input_weights={
            "params": B_cont,
            "props": ParameterProperties(trainable=False),
        },
        dynamics_diffusion_coefficient={
            "params": jnp.eye(state_dim),
            "props": ParameterProperties(),
        },
        dynamics_diffusion_cov={
            "params": (0.5 * 0.5) * 0.5 * jnp.eye(state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
        emission_weights={
            "params": jr.normal(key_init, (emission_dim, state_dim)),
            "props": ParameterProperties(),
        },
        emission_bias={
            "params": jnp.zeros(emission_dim),
            "props": ParameterProperties(),
        },
        emission_cov={
            "params": 0.1 * jnp.eye(emission_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
    )

    return d_model, d_params, d_param_props, cd_model, cd_params, cd_param_props


# Check that CD-LGSSM filtering with controls matches discrete LGSSM filtering
def test_cdlgssm_lgssm_filter_equivalence_with_controls(rng_keys):
    # Unpack RNG keys
    key_init, key_sample = rng_keys

    # Define discrete and continuous-discrete models with non-zero controls
    d_model, d_params, d_param_props, cd_model, cd_params, cd_param_props = (
        init_cdlgssm_lgssm_models_with_controls(key_init)
    )

    # Simulation params
    num_timesteps = 100
    t_emissions = jnp.arange(num_timesteps)[:, None]

    # Non-zero inputs
    inputs = 0.5 * jr.normal(key_sample, (num_timesteps, cd_model.input_dim))

    # Sample from the discrete model
    d_states, d_emissions = d_model.sample(
        d_params, key_sample, num_timesteps=num_timesteps, inputs=inputs
    )

    # Sample from the continuous-discrete model
    cd_states, cd_emissions = cd_model.sample(
        cd_params,
        key_sample,
        num_timesteps=num_timesteps,
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Compare samples
    compare(cd_states, d_states)
    compare(cd_emissions, d_emissions)

    # Filter discrete model emissions with both models
    d_filtered_posterior = lgssm_filter(d_params, d_emissions, inputs)

    kf_hyperparams = KFHyperParams(dt_final=1.0)
    cd_filtered_posterior = cdlgssm_filter(
        cd_params,
        d_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
    )

    # Compare filtered posteriors (including marginal log-likelihood)
    compare_structs(d_filtered_posterior, cd_filtered_posterior)
