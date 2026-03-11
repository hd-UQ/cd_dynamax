# Basic jax imports
import jax.numpy as jnp
import jax.random as jr
import pytest

# dynamax imports
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector

# cd-dynamax models
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
# continuous-discrete nonlinear Gaussian SSM codebase
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm import (
    ParamsCDLGSSM,
    KFHyperParams,
    cdlgssm_filter,
    cdlgssm_forecast,
    cdlgssm_smoother
)

from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    ParamsCDNLGSSM,
    EKFHyperParams,
    UKFHyperParams,
    EnKFHyperParams,
    cdnlgssm_smoother,
)

from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import (
    LearnableVector,
    LearnableMatrix,
    LearnableLinear,
)

from cd_dynamax.src.continuous_discrete_nonlinear_ssm import (
    ParamsCDNLSSM,
    DPFHyperParams,
)

# Gaussian initial and emimission distributions for CD-NLSSM
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.cdnlssm_utils import (
    StaticGaussianDistribution,
    LearnableGaussianEmission,
)

# Cd-dynamax utilities
from cd_dynamax.src.utils.simulation_utils import (
    make_key_sequence,
    cddynamax_filter,
    cddynamax_forecast,
    cddynamax_emissions
)

from cd_dynamax.src.utils.experiment_utils import (
    generate_t_emissions
)

# Utilities for testing and comparing results
from cd_dynamax.src.utils.test_utils import compare, compare_structs

# Set this to True to plot the results of the tests for visual inspection
PLOT_TEST_RESULTS = False

# Use a fixed random seed for reproducibility in tests
@pytest.fixture
def seed():
    return 0

# Initialization of example, linear Gaussian cd-dynamax model
def init_cdlgssm_model(key):
    """Helper function to initialize a continuous-discrete LGSSM for testing.
    """
    # Split key
    key_init, key_w=jr.split(key)

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
        ## Initial
        initial_mean={
            "params": jnp.zeros(cd_model.state_dim),
            "props": ParameterProperties(),
        },
        initial_cov={
            "params": jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
        ## Dynamics
        dynamics_weights={
            "params": -0.1 * jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(),
        },
        dynamics_bias={
            "params": jnp.zeros((cd_model.state_dim,)),
            "props": ParameterProperties(trainable=False),  # We do not learn bias term!
        },
        dynamics_diffusion_coefficient={
            "params": 0.1 * jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(),
        },
        dynamics_diffusion_cov={
            "params": 0.1 * jnp.eye(cd_model.state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
        ## Emission
        emission_weights={
            "params": jr.normal(key_w, (cd_model.emission_dim, cd_model.state_dim)),
            "props": ParameterProperties(),
        },
        emission_bias={
            "params": jnp.zeros((cd_model.emission_dim,)),
            "props": ParameterProperties(trainable=False),  # We do not learn bias term!
        },
        emission_cov={
            "params": 0.1 * jnp.eye(cd_model.emission_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()),
        },
    )
    # Return the initialized models and parameters for use in tests
    return cd_model, cd_params, cd_param_props, key_init

# Initialization of a cdlgssm equivalent cdnlgssm model
def init_cdnlgssm_equivalent_model(cdlgssm_model, cdlgssm_key, cdlgssm_params, dynamics_approx_order):
    cdnlgssm_model = ContDiscreteNonlinearGaussianSSM(
        state_dim=cdlgssm_model.state_dim,
        emission_dim=cdlgssm_model.emission_dim
    )

    # Initialize model with linear learnable functions, for path-based generation
    cdnlgssm_params, cdnlgssm_props = cdnlgssm_model.initialize(
        cdlgssm_key,
        initial_mean={
            "params": LearnableVector(params=jnp.zeros(cdlgssm_model.state_dim)),
            "props": LearnableVector(params=ParameterProperties()),
        },
        initial_cov={
            "params": LearnableMatrix(params=jnp.eye(cdlgssm_model.state_dim)),
            "props": LearnableMatrix(
                params=ParameterProperties(constrainer=RealToPSDBijector())
            ),
        },
        dynamics_drift={
            "params": LearnableLinear(
                weights=cdlgssm_params.dynamics.weights, bias=cdlgssm_params.dynamics.bias
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(),
                bias=ParameterProperties(trainable=False),  # We do not learn bias term!
            ),
        },
        dynamics_diffusion_coefficient={
            "params": LearnableMatrix(params=cdlgssm_params.dynamics.diffusion_coefficient),
            "props": LearnableMatrix(params=ParameterProperties()),
        },
        dynamics_diffusion_cov={
            "params": LearnableMatrix(params=cdlgssm_params.dynamics.diffusion_cov),
            "props": LearnableMatrix(
                params=ParameterProperties(constrainer=RealToPSDBijector())
            ),
        },
        dynamics_approx_order=dynamics_approx_order,
        emission_function={
            "params": LearnableLinear(
                weights=cdlgssm_params.emissions.weights, bias=cdlgssm_params.emissions.bias
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(),
                bias=ParameterProperties(trainable=False),  # We do not learn bias term!
            ),
        },
        emission_cov={
            "params": LearnableMatrix(params=0.1 * jnp.eye(cdlgssm_model.emission_dim)),
            "props": LearnableMatrix(
                params=ParameterProperties(constrainer=RealToPSDBijector())
            ),
        },
    )

    return cdnlgssm_model, cdnlgssm_params, cdnlgssm_props

# Initialization of a cdlgssm equivalent cdlgssm model
def init_cdnlssm_equivalent_model(cdlgssm_model, cdlgssm_key, cdlgssm_params):
    cdnlgssm_model = ContDiscreteNonlinearSSM(
        state_dim=cdlgssm_model.state_dim,
        emission_dim=cdlgssm_model.emission_dim
    )
    
    # Initialize the CD-NLSSM with the specified initial distribution, dynamics, and emission
    cdnlgssm_params, cdnlgssm_props= cdnlgssm_model.initialize(
        # Define the initial distribution as a StaticGaussianDistribution with the specified mean and covariance
        initial_distribution={
            "params": StaticGaussianDistribution(
                mean=jnp.zeros(cdlgssm_model.state_dim),
                cov=jnp.eye(cdlgssm_model.state_dim)
            ),
            "props": None,
        },
        # Define the dynamics with a linear drift and constant diffusion, matching the CD-LGSSM parameters
        dynamics_drift={
            "params": LearnableLinear(
                weights=cdlgssm_params.dynamics.weights,
                bias=cdlgssm_params.dynamics.bias
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(),
                bias=ParameterProperties(trainable=False),  # We do not learn bias term!
            ),
        },
        dynamics_diffusion_coefficient={
            "params": LearnableMatrix(params=cdlgssm_params.dynamics.diffusion_coefficient),
            "props": LearnableMatrix(params=ParameterProperties()),
        },
        dynamics_diffusion_cov={
            "params": LearnableMatrix(params=cdlgssm_params.dynamics.diffusion_cov),
            "props": LearnableMatrix(
                params=ParameterProperties(constrainer=RealToPSDBijector())
            ),
        },
        # The emission distribution is a Learnable Gaussian emission
        # with mean given by a linear function of the state and covariance matching the CD-LGSSM parameters
        emission_distribution={
            "params": LearnableGaussianEmission(
                emission_function=LearnableLinear(
                    weights=cdlgssm_params.emissions.weights, bias=cdlgssm_params.emissions.bias
                ),
                emission_cov=LearnableMatrix(params=0.1 * jnp.eye(cdlgssm_model.emission_dim))
            ),
            "props": None,  # Let us use the default parameter properties
        },
    )

    return cdnlgssm_model, cdnlgssm_params, cdnlgssm_props


# Check whether continuous-discrete model filtering (with forecasting and emissions) execute successfully
def test_cdnonlinear_filter_cdlinear_kf_match(seed):

    # Sequence of keys
    keys = make_key_sequence(seed)

    # Define discrete and continous-discrete models
    cdlgssm_model, cdlgssm_params, cdlgssm_props, cdlgssm_key = init_cdlgssm_model(next(keys))

    # Simulation from CD-LGSSM
    t0=0
    t1=1
    num_samples=100

    # Regularly sampled in [0,1]
    num_timesteps, t_all=generate_t_emissions(
        t0=t0,
        t1=t1,
        num_samples=num_samples,
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
    sampling_key=next(keys)
    cdlgssm_states, cdlgssm_emissions = cdlgssm_model.sample(
        params=cdlgssm_params,
        key=sampling_key,
        num_timesteps=len(t_emissions),
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Filter the cdlgssm emissions to get filtered posterior
    cdlgssm_filtered = cdlgssm_filter(
        cdlgssm_params,
        cdlgssm_emissions,
        t_emissions,
        filter_hyperparams=KFHyperParams(),
        inputs=inputs,
    )

    # Forecasting with cdlgssm model
    forecasting_key=next(keys)

    # For the distribution forecast,
    # we use the last filtered mean and covariance to define a Gaussian initial condition
    from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
    cdlgssm_init_forecast = MVN(
        cdlgssm_filtered.filtered_means[-1, :],
        cdlgssm_filtered.filtered_covariances[-1, :],
    )
    cdlgssm_forecasted = cdlgssm_forecast(
        params=cdlgssm_params,
        init_forecast=cdlgssm_init_forecast,
        t_init=t_emissions[-1],
        t_forecast=t_forecast_emissions,
        filter_hyperparams=KFHyperParams(),
        inputs=inputs,
        key=forecasting_key,
    )

    # CD-LGSSM Emissions
    emissions_key = next(keys)
    # Compute emissions for filtered and forecasted states
    cdlgssm_filtered_emissions, cdlgssm_forecasted_emissions = cddynamax_emissions(
        model=cdlgssm_model,
        model_params=cdlgssm_params,
        t_emissions_filter=t_emissions,
        filtered_state=cdlgssm_filtered,
        t_emissions_forecast=t_forecast_emissions,
        forecasted_state=cdlgssm_forecasted,
        filtering_inputs=None,
        forecasting_inputs=None,
        key=emissions_key,
    )
    
    # CD-NLGSSM
    # Define equivalent cdnlgssm model, with first and second order SDE approximation
    # both should be correct for linear models
    for dynamics_approx_order in [1.0, 2.0]:
        cdnlgssm_model, cdnlgssm_params, cdnlgssm_props = init_cdnlgssm_equivalent_model(
            cdlgssm_model,
            cdlgssm_key,
            cdlgssm_params,
            dynamics_approx_order
        )

        # Sample from the continuous-discrete nonlinear-gaussian model
        print("Sampling from CD-NLGSSM model...")
        cdnlgssm_states, cdnlgssm_emissions = cdnlgssm_model.sample(
            params=cdnlgssm_params,
            key=sampling_key,
            num_timesteps=len(t_emissions),
            t_emissions=t_emissions,
            inputs=inputs,
        )

        # Check that these are similar to samples from the cdlgssm model
        print("\tChecking states...")
        compare(cdnlgssm_states, cdlgssm_states)

        print("\tChecking emissions...")
        compare(cdnlgssm_emissions, cdlgssm_emissions)

        # Check match
        check_nonlinear_linear_filter_match(
            cddynamax_model=cdnlgssm_model,
            cddynamax_params=cdnlgssm_params,
            t_emissions=t_emissions,
            cdlgssm_emissions=cdlgssm_emissions,
            cdlgssm_filtered=cdlgssm_filtered,
            t_forecast_emissions=t_forecast_emissions,
            cdlgssm_forecasted=cdlgssm_forecasted,
            cdlgssm_filtered_emissions=cdlgssm_filtered_emissions,
            cdlgssm_forecasted_emissions=cdlgssm_forecasted_emissions,
            keys=(forecasting_key, emissions_key, keys),
        )
        print(f"... all filtering (with forecasting and emissions) tests passed for CD-NLGSSM model with dynamics_approx_order={dynamics_approx_order}")
    
    # CD-NLSSM
    # Define equivalent cdnlssm model
    cdnlssm_model, cdnlssm_params, cdnlssm_props = init_cdnlssm_equivalent_model(
        cdlgssm_model,
        cdlgssm_key,
        cdlgssm_params,
    )

    # Sample from the continuous-discrete nonlinear-gaussian model
    print("Sampling from CD-NLSSM model...")
    cdnlssm_states, cdnlssm_emissions = cdnlssm_model.sample(
        params=cdnlssm_params,
        key=sampling_key,
        num_timesteps=len(t_emissions),
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Check that these are similar to samples from the cdlgssm model
    print("\tChecking states...")
    compare(cdnlssm_states, cdlgssm_states)

    print("\tChecking emissions...")
    compare(cdnlssm_emissions, cdlgssm_emissions)

    # Check match
    check_nonlinear_linear_filter_match(
        cddynamax_model=cdnlssm_model,
        cddynamax_params=cdnlssm_params,
        t_emissions=t_emissions,
        cdlgssm_emissions=cdlgssm_emissions,
        cdlgssm_filtered=cdlgssm_filtered,
        t_forecast_emissions=t_forecast_emissions,
        cdlgssm_forecasted=cdlgssm_forecasted,
        cdlgssm_filtered_emissions=cdlgssm_filtered_emissions,
        cdlgssm_forecasted_emissions=cdlgssm_forecasted_emissions,
        keys=(forecasting_key, emissions_key, keys),
    )
    print("... all filtering (with forecasting and emissions) tests passed for CD-NLSSM model")
    
# Check whether continuous-discrete model filtering and forecasting run
def check_nonlinear_linear_filter_match(
        cddynamax_model,
        cddynamax_params,
        t_emissions,
        cdlgssm_emissions,
        cdlgssm_filtered,
        t_forecast_emissions,
        cdlgssm_forecasted,
        cdlgssm_filtered_emissions,
        cdlgssm_forecasted_emissions,
        keys,
    ):

    # Unravel keys
    forecasting_key, emissions_key, other_keys = keys
    
    ### Filtering
    # Decide what filtering algorithm to use
    if isinstance(cddynamax_params, ParamsCDNLGSSM):
        # Nonlinear Gaussian case
        # EKF, UKF and EnKF
        filter_hyperparams_list = [
            EKFHyperParams(),
            UKFHyperParams(),
            EnKFHyperParams(N_particles=100)
        ]
    elif isinstance(cddynamax_params, ParamsCDNLSSM):
        # Nonlinear case with DPF
        filter_hyperparams_list =[
            DPFHyperParams(N_particles=1000)
        ]
    else:
        raise ValueError(f"Unknown model parameters type {type(cddynamax_params)}")
    
    # Iterate over filters
    for filter_hyperparams in filter_hyperparams_list:
        print(f"Testing filtering for nonlinear model with filter {type(filter_hyperparams)}...")
        # Filter the emissions to get filtered posterior
        filtered = cddynamax_filter(
            model_params=cddynamax_params,
            filter_hyperparams=filter_hyperparams,
            t_emissions=t_emissions,
            emissions=cdlgssm_emissions,
            start_idx_filter=0,
            stop_idx_filter=len(cdlgssm_emissions),
            key=next(other_keys),
            filter_spec='model'
        )

        # Check that the filtered posterior has the expected structure and dimensions
        assert filtered.filtered_means.shape == (len(t_emissions), cddynamax_model.state_dim), \
            f"Expected filtered means shape {(len(t_emissions), cddynamax_model.state_dim)}, got {filtered.filtered_means.shape}"
        assert filtered.filtered_covariances.shape == (len(t_emissions), cddynamax_model.state_dim, cddynamax_model.state_dim), \
            f"Expected filtered covariances shape {(len(t_emissions), cddynamax_model.state_dim, cddynamax_model.state_dim)}, got {filtered.filtered_covariances.shape}"

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
        
        # Check nonlinear filter results are close to linear Kalman filter results
        print("\tChecking filtered means match Kalman filter results...")
        compare_structs(filtered, cdlgssm_filtered, accept_failure=True)

        # If intersted, plot the results to visually inspect the filtering and forecasting performance
        if PLOT_TEST_RESULTS:
            import matplotlib.pyplot as plt

            for n_state in jnp.arange(cddynamax_model.state_dim):
                plt.figure()
                plt.plot(
                    t_emissions,
                    cdlgssm_filtered.filtered_means[:, n_state],
                    label="KF-filtered state",
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
        if isinstance(cddynamax_params, (ParamsCDLGSSM, ParamsCDNLGSSM)):
            from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
            init_forecast = MVN(
                filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :]
            )
        elif isinstance(cddynamax_params, ParamsCDNLSSM):
            init_forecast = filtered.particles[-1, ...]
        # Forecast
        forecasted = cddynamax_forecast(
            model_params=cddynamax_params,
            filter_hyperparams=filter_hyperparams,
            t_init=t_emissions[-1],
            init_forecast=init_forecast,
            t_forecast=t_forecast_emissions,
            key=forecasting_key,
            filter_spec='model'
        )

        # Check forecasted state means and covariances have the expected structure and dimensions, and are finite
        assert forecasted.forecasted_state_means.shape == (len(t_forecast_emissions), cddynamax_model.state_dim), \
            f"Expected forecasted state means shape {(len(t_forecast_emissions), cddynamax_model.state_dim)}, got {forecasted.forecasted_state_means.shape}"
        assert forecasted.forecasted_state_covariances.shape == (len(t_forecast_emissions), cddynamax_model.state_dim, cddynamax_model.state_dim), \
            f"Expected forecasted state covariances shape {(len(t_forecast_emissions), cddynamax_model.state_dim, cddynamax_model.state_dim)}, got {forecasted.forecasted_state_covariances.shape}"
        assert jnp.all(jnp.isfinite(forecasted.forecasted_state_means)), \
            "Forecasted state means contain non-finite values"
        assert jnp.all(jnp.isfinite(forecasted.forecasted_state_covariances)), \
            "Forecasted state covariances contain non-finite values"

        # Check nonlinear forecast results are close to linear Kalman forecast results
        print("\tChecking forecasted states match Kalman forecast results...")
        compare_structs(forecasted, cdlgssm_forecasted, accept_failure=True)

        ### Emissions
        # Compute emissions for filtered and forecasted states
        filtered_emissions, forecasted_emissions = cddynamax_emissions(
            model=cddynamax_model,
            model_params=cddynamax_params,
            t_emissions_filter=t_emissions,
            filtered_state=filtered,
            t_emissions_forecast=t_forecast_emissions,
            forecasted_state=forecasted,
            filtering_inputs=None,
            forecasting_inputs=None,
            key=emissions_key,
        )

        # Check forecasted emissions means and covariances have the expected structure and dimensions, and are finite
        assert filtered_emissions['mean'].shape == (len(t_emissions), cddynamax_model.emission_dim), \
            f"Expected filtered emissions mean shape {(len(t_emissions), cddynamax_model.emission_dim)}, got {filtered_emissions['mean'].shape}"
        assert filtered_emissions['cov'].shape == (len(t_emissions), cddynamax_model.emission_dim, cddynamax_model.emission_dim), \
            f"Expected filtered emissions covariances shape {(len(t_emissions), cddynamax_model.emission_dim, cddynamax_model.emission_dim)}, got {filtered_emissions['cov'].shape}"
        assert jnp.all(jnp.isfinite(filtered_emissions['mean'])), \
            "Filtered emissions means contain non-finite values"
        assert jnp.all(jnp.isfinite(filtered_emissions['cov'])), \
            "Filtered emissions covariances contain non-finite values"
        assert forecasted_emissions['mean'].shape == (len(t_forecast_emissions), cddynamax_model.emission_dim), \
            f"Expected forecasted emissions means shape {(len(t_forecast_emissions), cddynamax_model.emission_dim)}, got {forecasted_emissions['mean'].shape}"
        assert forecasted_emissions['cov'].shape == (len(t_forecast_emissions), cddynamax_model.emission_dim, cddynamax_model.emission_dim), \
            f"Expected forecasted emissions covariances shape {(len(t_forecast_emissions), cddynamax_model.emission_dim, cddynamax_model.emission_dim)}, got {forecasted_emissions['cov'].shape}"
        assert jnp.all(jnp.isfinite(forecasted_emissions['mean'])), \
            "Forecasted emissions means contain non-finite values"
        assert jnp.all(jnp.isfinite(forecasted_emissions['cov'])), \
            "Forecasted emissions covariances contain non-finite values"

        # Check nonlinear model emissions are close to original cdlgssm emission
        print("\tChecking cd-dynamax nonlinear model filtered emissions match original cdlgssm emissions...")
        compare_structs(filtered_emissions, cdlgssm_filtered_emissions, accept_failure=True)
        print("\tChecking cd-dynamax nonlinear model forecasted emissions match original cdlgssm emissions...")
        compare_structs(forecasted_emissions, cdlgssm_forecasted_emissions, accept_failure=True)


# Check whether continuous-discrete model smoothing executes successfully
def test_cdnonlinear_smoother_cdlinear_ks_match(seed):

    # Sequence of keys
    keys = make_key_sequence(seed)

    # Define discrete and continous-discrete models
    cdlgssm_model, cdlgssm_params, cdlgssm_props, cdlgssm_key = init_cdlgssm_model(next(keys))

    # Simulation from CD-LGSSM
    t0=0
    t1=1
    num_samples=100

    # Regularly sampled in [0,1]
    num_timesteps, t_emissions=generate_t_emissions(
        t0=t0,
        t1=t1,
        num_samples=num_samples,
        irregular_samples = False,
        key=next(keys),
    )
    
    # No inputs for now
    inputs = None

    # Sample from the continuous-discrete model
    sampling_key=next(keys)
    cdlgssm_states, cdlgssm_emissions = cdlgssm_model.sample(
        params=cdlgssm_params,
        key=sampling_key,
        num_timesteps=len(t_emissions),
        t_emissions=t_emissions,
        inputs=inputs,
    )

    # Smooth the cdlgssm emissions to get smoothed posterior
    # There are 2 CD-LGSSM smoothers implemented, test both
    cdlgssm_smoothed_1 = cdlgssm_smoother(
        cdlgssm_params,
        cdlgssm_emissions,
        t_emissions,
        filter_hyperparams=KFHyperParams(),
        inputs=inputs,
        smoother_type='cd_smoother_1'
    )
    cdlgssm_smoothed_2 = cdlgssm_smoother(
        cdlgssm_params,
        cdlgssm_emissions,
        t_emissions,
        filter_hyperparams=KFHyperParams(),
        inputs=inputs,
        smoother_type='cd_smoother_2'
    )
    
    # CD-NLGSSM
    # Define equivalent cdnlgssm model, with first and second order SDE approximation
    # both should be correct for linear models
    for dynamics_approx_order in [1.0, 2.0]:
        cdnlgssm_model, cdnlgssm_params, cdnlgssm_props = init_cdnlgssm_equivalent_model(
            cdlgssm_model,
            cdlgssm_key,
            cdlgssm_params,
            dynamics_approx_order
        )

        # Check match
        check_nonlinear_linear_smoother_match(
            cddynamax_model=cdnlgssm_model,
            cddynamax_params=cdnlgssm_params,
            t_emissions=t_emissions,
            cdlgssm_emissions=cdlgssm_emissions,
            cdlgssm_smoothed_1=cdlgssm_smoothed_1,
            cdlgssm_smoothed_2=cdlgssm_smoothed_2,
            keys=keys,
        )
        print(f"... all smoothing tests passed for CD-NLGSSM model with dynamics_approx_order={dynamics_approx_order}")

# Check whether continuous-discrete model filtering and forecasting run
def check_nonlinear_linear_smoother_match(
        cddynamax_model,
        cddynamax_params,
        t_emissions,
        cdlgssm_emissions,
        cdlgssm_smoothed_1,
        cdlgssm_smoothed_2,
        keys,
    ):
    
    ### Filtering
    # Decide what filtering algorithm to use
    if isinstance(cddynamax_params, ParamsCDNLGSSM):
        # Nonlinear Gaussian case
        # EKF smoother only, 
        # with  first and second order state SDE approximation (both should be correct for linear models)
        filter_hyperparams_list = [
            EKFHyperParams(state_order="first"),
            EKFHyperParams(state_order="second"),
        ]

    else:
        raise ValueError(f"Unknown model parameters type {type(cddynamax_params)} for smoothing check")
    
    # Iterate over filters
    for filter_hyperparams in filter_hyperparams_list:
        # Smoothed the emissions to get filtered posterior
        print(f"Testing smoothing for nonlinear model with filter {type(filter_hyperparams)}...")
        smoothed = cdnlgssm_smoother(
            params=cddynamax_params,
            emissions=cdlgssm_emissions,
            t_emissions=t_emissions,
            filter_hyperparams=filter_hyperparams,
        )

        # Check that the smoothed posterior has the expected structure and dimensions
        assert smoothed.smoothed_means.shape == (len(t_emissions), cddynamax_model.state_dim), \
            f"Expected smoothed means shape {(len(t_emissions), cddynamax_model.state_dim)}, got {smoothed.smoothed_means.shape}"
        assert smoothed.smoothed_covariances.shape == (len(t_emissions), cddynamax_model.state_dim, cddynamax_model.state_dim), \
            f"Expected smoothed covariances shape {(len(t_emissions), cddynamax_model.state_dim, cddynamax_model.state_dim)}, got {smoothed.smoothed_covariances.shape}"

        # Check smoothed means and covariances are finite
        assert jnp.all(jnp.isfinite(smoothed.smoothed_means)), \
            "Smoothed means contain non-finite values"
        assert jnp.all(jnp.isfinite(smoothed.smoothed_covariances)), \
            "Smoothed covariances contain non-finite values"
        
        # Check nonlinear filter results are close to linear Kalman filter results
        print("\tChecking smoothed CD-NLGSSM match Kalman smoother type 1 results...")
        compare_structs(smoothed, cdlgssm_smoothed_1, accept_failure=True)
        print("\tChecking smoothed CD-NLGSSM match Kalman smoother type 2 results...")
        compare_structs(smoothed, cdlgssm_smoothed_2, accept_failure=True)

        # If intersted, plot the results to visually inspect the smoothing and forecasting performance
        if PLOT_TEST_RESULTS:
            import matplotlib.pyplot as plt

            for n_state in jnp.arange(cddynamax_model.state_dim):
                plt.figure()
                plt.plot(
                    t_emissions,
                    cdlgssm_smoothed_1.smoothed_means[:, n_state],
                    label="KF-smoothed Type 1 state",
                    color="black",
                )
                plt.plot(
                    t_emissions,
                    cdlgssm_smoothed_2.smoothed_means[:, n_state],
                    label="KF-smoothed Type 2 state",
                    color="gray",
                )
                plt.plot(
                    t_emissions,
                    smoothed.smoothed_means[:, n_state],
                    label="Continuous-Discrete smoothed state",
                    color="blue",
                    marker="x",
                )
                plt.xlabel("time")
                plt.ylabel(f"x_{n_state}")
                plt.grid()
                plt.legend()
                plt.title("Smoothed states after SGD optimization")
                plt.show()
        
if __name__ == "__main__":
    test_cdnonlinear_filter_cdlinear_kf_match(0)
    #test_cdnonlinear_smoother_cdlinear_ks_match(0)
    pass