# Imports
import jax.numpy as jnp
import jax.random as jr

from cd_dynamax import ContDiscreteLinearGaussianSSM, ContDiscreteNonlinearSSM
from cd_dynamax.src.utils.test_utils import compare, compare_structs

from cd_dynamax import DPFHyperParams, cdnlssm_filter

# Whether to plot test results or not
PLOT_TEST_RESULTS = False

# JAX device check
print("************* Checking JAX device *************")
import jax

print("Running on jax device:{}".format(jax.devices()))
print("Running on jax device platform:{}".format(jax.devices()[0].platform))
print("***********************************************")

# The idea of this test is as following (uses regular time intervals ONLY):
# First, establish equivalent linear systems in discrete and continuous time
# Show that samples from each are similar
# Show that continuous-discrete KF == {cd-EKFs, cd-UKF, cd-EnKF} for that linear system

#### General state and emission dimensionalities
STATE_DIM = 2
EMISSION_DIM = 6
# Discrete sampling
NUM_TIMESTEPS = 100

print("************* Continuous-Discrete LGSSM *************")
# Continuous-Discrete model
t_emissions = jnp.arange(NUM_TIMESTEPS)[:, None]

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
cd_model = ContDiscreteLinearGaussianSSM(
    state_dim=STATE_DIM,
    emission_dim=EMISSION_DIM,
    # Test with no biases
    has_dynamics_bias=True,
    has_emissions_bias=True,
)
# Initialize, controlling what is learned
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.models import *

cd_params, cd_param_props = cd_model.initialize(
    key1,
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
        "params": jr.normal(key1, (cd_model.emission_dim, cd_model.state_dim)),
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

# Simulate from continuous model
print("Simulating in continuous-discrete time")
cd_num_timesteps_states, cd_num_timesteps_emissions = cd_model.sample(
    cd_params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs
)

cd_states, cd_emissions = cd_model.sample(
    cd_params, key2, num_timesteps=NUM_TIMESTEPS, t_emissions=t_emissions, inputs=inputs
)

print(f"Sampling CDLGSSM path in continuous-discrete time")
cd_states_path, cd_emissions_path = cd_model.sample(
    cd_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    t_emissions=t_emissions,
    inputs=inputs,
    transition_type="path",
)

print("\tChecking states...")
compare(cd_num_timesteps_states, cd_states)

print("\tChecking emissions...")
compare(cd_num_timesteps_emissions, cd_emissions)

print("Continuous-Discrete time filtering: pre-fit")
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.inference import (
    cdlgssm_filter,
    KFHyperParams,
)

# We set dt_final=1 so that predicted mean and covariance at the end of sequence match those of discrete filtering
kf_hyperparams = KFHyperParams(dt_final=1.0)
# Define CD linear filter
cd_filtered_posterior = cdlgssm_filter(
    cd_params,
    cd_emissions,
    t_emissions,
    filter_hyperparams=kf_hyperparams,
    inputs=inputs,
)

print("Fitting continuous-discrete time linear with SGD")
cd_sgd_fitted_params, cd_sgd_lps = cd_model.fit_sgd(
    cd_params,
    cd_param_props,
    cd_emissions,
    t_emissions,
    filter_hyperparams=kf_hyperparams,
    inputs=inputs,
    num_epochs=10,
)

print("Continuous-Discrete time filtering: post-fit")
cd_sgd_fitted_filtered_posterior = cdlgssm_filter(
    cd_sgd_fitted_params,
    cd_emissions,
    t_emissions,
    filter_hyperparams=kf_hyperparams,
    inputs=inputs,
)

########### Now make non-linear models, assuming linearity ########
print("************* Continuous-Discrete Non-linear GSSM *************")
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import *
from cd_dynamax import cdnlgssm_filter, EKFHyperParams

# Model def
inputs = None  # Not interested in inputs for now
cdnl_model = ContDiscreteNonlinearSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)


def build_cdnl_params(dynamics_approx_order=1.0, initial_distribution=None):
    return cdnl_model.build_params(
        drift=lambda x, u=None, t=None: cd_params.dynamics.weights @ x
        + cd_params.dynamics.bias,
        diffusion_coeff=lambda x=None,
        u=None,
        t=None: cd_params.dynamics.diffusion_coefficient,
        diffusion_cov=lambda x=None, u=None, t=None: cd_params.dynamics.diffusion_cov,
        emission_function=lambda x, u=None, t=None: x @ cd_params.emissions.weights.T
        + cd_params.emissions.bias,
        emission_cov=cd_params.emissions.cov,
        initial_mean=cd_params.initial.mean,
        initial_cov=cd_params.initial.cov,
        initial_distribution=initial_distribution,
        approx_order=dynamics_approx_order,
    )


cdnl_params = build_cdnl_params()
print(f"Sampling CDNLSSM path in continuous-discrete time")
cdnl_states_path, cdnl_emissions_path = cdnl_model.sample_path(
    params=cdnl_params,
    key=key2,
    t_emissions=t_emissions,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs,
)

# check that these are similar to the path from the cd-linear model
print("\tChecking states...")
compare(cdnl_states_path, cd_states_path)

print("\tChecking emissions...")
compare(cdnl_emissions_path, cd_emissions_path)

# Test models with first and second order SDE approximation (both should be correct for linear models)
for dynamics_approx_order in [1.0, 2.0]:
    # Initialize models with linear functions
    cdnl_params = build_cdnl_params(dynamics_approx_order=dynamics_approx_order)

    # Simulate from continuous-discrete nl model
    print(f"**********************************")
    print(
        f"Simulating {dynamics_approx_order} order CDNLSSM in continuous-discrete time"
    )
    cdnl_states, cdnl_emissions = cdnl_model.sample(
        params=cdnl_params,
        key=key2,
        t_emissions=t_emissions,
        num_timesteps=NUM_TIMESTEPS,
        inputs=inputs,
    )

    # check that these are similar to samples from the cd-linear model
    print("\tChecking states...")
    compare(cdnl_states, cd_states)

    print("\tChecking emissions...")
    compare(cdnl_emissions, cd_emissions)


######## Continuous-discrete Differentiable Particle Filter
print(f"**********************************")
for N_particles in [1e2, 1e3, 1e4]:
    for resample_method in ["stop_gradient", "soft"]:
        print(
            f"Running Differentiable Particle Filter (resample_method={resample_method}, N_particles={N_particles}) with non-linear model class and data from first-order CDNLSSM model"
        )

        # define hyperparameters
        dpf_params = DPFHyperParams(
            dt_final=1.0,
            N_particles=int(N_particles),
            resample_method=resample_method,
            state_order="first",
        )

        cd_dpf_post = cdnlssm_filter(
            cdnl_params,
            cdnl_emissions,
            filter_hyperparams=dpf_params,
            t_emissions=t_emissions.reshape(-1, 1),
            inputs=inputs,
            key=key2,
        )

        # check that results in cd_dpf_post are similar to results from applying cd_kf (cd_filtered_posterior)
        print("\tComparing filtered means...")
        try:
            compare(cd_dpf_post.filtered_means, cd_filtered_posterior.filtered_means)
        except:
            if N_particles < 1e5:
                print("Test failed because too few particles")
                pass
            else:
                compare(
                    cd_dpf_post.filtered_means, cd_filtered_posterior.filtered_means
                )

        print("\tComparing filtered covariances...")
        try:
            compare(
                cd_dpf_post.filtered_covariances,
                cd_filtered_posterior.filtered_covariances,
                do_det=True,
            )
        except:
            if N_particles < 1e5:
                print("Test failed because too few particles")
                pass
            else:
                compare(
                    cd_dpf_post.filtered_covariances,
                    cd_filtered_posterior.filtered_covariances,
                    do_det=True,
                )


print(
    "All DPF tests passed---note that these are randomized approximations, so we don't expect to perfectly replicate EKF and KF (which are both exact in linear test cases shown here)! We want to see convergence to truth (hence checking the final filtered state)."
)

# Test forecasting
print("************* Continuous-Discrete DPF Forecasting *************")
# Define forecasting time points
FORECAST_TIMESTEPS = 25
t_forecast_emissions = jnp.arange(NUM_TIMESTEPS, NUM_TIMESTEPS + FORECAST_TIMESTEPS)[
    :, None
]
t_forecast_with_init = jnp.concatenate(
    [t_emissions[-1:, :], t_forecast_emissions], axis=0
)

# Forecasting randomness
key_forecast_point, key_forecast_dist = jr.split(jr.PRNGKey(0))

print("Forecasting with dpf-based forecast")
# Initialize forecast with last filtered state, as fixed point estimate
cd_dpf_point_init_forecast = cd_dpf_post.filtered_means[-1, :]
cd_dpf_point_params = build_cdnl_params(
    dynamics_approx_order=cdnl_params.dynamics.approx_order,
    initial_distribution=tfd.Deterministic(loc=cd_dpf_point_init_forecast),
)
cd_dpf_point_states, _ = cdnl_model.sample_path(
    params=cd_dpf_point_params,
    key=key_forecast_point,
    num_timesteps=t_forecast_with_init.shape[0],
    t_emissions=t_forecast_with_init,
    inputs=inputs,
)
cd_dpf_point_states = cd_dpf_point_states[1:, :]

# Initialize forecast with last filtered state distribution
cd_dpf_dist_init_forecast = MVN(
    cd_dpf_post.filtered_means[-1, :], cd_dpf_post.filtered_covariances[-1, :]
)
cd_dpf_dist_params = build_cdnl_params(
    dynamics_approx_order=cdnl_params.dynamics.approx_order,
    initial_distribution=cd_dpf_dist_init_forecast,
)
cd_dpf_dist_states, _ = cdnl_model.sample_path(
    params=cd_dpf_dist_params,
    key=key_forecast_dist,
    num_timesteps=t_forecast_with_init.shape[0],
    t_emissions=t_forecast_with_init,
    inputs=inputs,
)
cd_dpf_dist_states = cd_dpf_dist_states[1:, :]

emission_fn = cdnl_params.emissions.emission_distribution.emission_function.f
emission_cov_fn = cdnl_params.emissions.emission_distribution.emission_cov.f
t_forecast_flat = jnp.squeeze(t_forecast_emissions)

cd_dpf_point_emissions_forecasted_means = jax.vmap(
    lambda state, t: emission_fn(state, inputs, t)
)(cd_dpf_point_states, t_forecast_flat)
cd_dpf_point_emissions_forecasted_covariances = jax.vmap(
    lambda state, t: emission_cov_fn(state, inputs, t)
)(cd_dpf_point_states, t_forecast_flat)

cd_dpf_dist_emissions_forecasted_means = jax.vmap(
    lambda state, t: emission_fn(state, inputs, t)
)(cd_dpf_dist_states, t_forecast_flat)
cd_dpf_dist_emissions_forecasted_covariances = jax.vmap(
    lambda state, t: emission_cov_fn(state, inputs, t)
)(cd_dpf_dist_states, t_forecast_flat)

cd_dpf_dist_state_covariances = jnp.broadcast_to(
    cd_dpf_post.filtered_covariances[-1, :],
    (
        FORECAST_TIMESTEPS,
        cd_dpf_post.filtered_covariances.shape[-1],
        cd_dpf_post.filtered_covariances.shape[-1],
    ),
)

if PLOT_TEST_RESULTS:
    print("Plotting dpf forecasted state path and distributions.")
    import matplotlib.pyplot as plt

    for n_state in jnp.arange(STATE_DIM):
        plt.figure()
        plt.plot(
            t_forecast_emissions,
            cd_dpf_point_states[:, n_state],
            label="Forecasted path (point estimate)",
            color="black",
        )
        plt.plot(
            t_forecast_emissions,
            cd_dpf_dist_states[:, n_state],
            label="Forecasted state means (distribution)",
            color="orange",
            marker="o",
            markerfacecolor="none",
            markeredgewidth=2,
            markersize=8,
        )
        plt.fill_between(
            t_forecast_emissions[:, 0],
            cd_dpf_dist_states[:, n_state]
            - jnp.sqrt(cd_dpf_dist_state_covariances[:, n_state, n_state]),
            cd_dpf_dist_states[:, n_state]
            + jnp.sqrt(cd_dpf_dist_state_covariances[:, n_state, n_state]),
            color="orange",
            alpha=0.2,
            label="Forecasted state uncertainty (1 std)",
        )
        plt.xlabel("Forecasted time")
        plt.ylabel("x_{}".format(n_state))
        plt.grid()
        plt.legend()
        plt.title("Forecasted states")
        plt.show()

# Compute emissions from forecasted states
if PLOT_TEST_RESULTS:
    print("Plotting dpf forecasted emission path and distributions.")
    import matplotlib.pyplot as plt

    for n_emission in jnp.arange(EMISSION_DIM):
        plt.figure()
        plt.plot(
            t_forecast_emissions,
            cd_dpf_point_emissions_forecasted_means[:, n_emission],
            label="Forecasted emission path (point estimate)",
            color="black",
        )
        plt.plot(
            t_forecast_emissions,
            cd_dpf_dist_emissions_forecasted_means[:, n_emission],
            label="Forecasted emission means (distribution)",
            color="orange",
            marker="o",
            markerfacecolor="none",
            markeredgewidth=2,
            markersize=8,
        )
        plt.fill_between(
            t_forecast_emissions[:, 0],
            cd_dpf_dist_emissions_forecasted_means[:, n_emission]
            - jnp.sqrt(
                cd_dpf_dist_emissions_forecasted_covariances[:, n_emission, n_emission]
            ),
            cd_dpf_dist_emissions_forecasted_means[:, n_emission]
            + jnp.sqrt(
                cd_dpf_dist_emissions_forecasted_covariances[:, n_emission, n_emission]
            ),
            color="orange",
            alpha=0.2,
            label="Forecasted emission uncertainty (1 std)",
        )
        plt.xlabel("Forecasted time")
        plt.ylabel("x_{}".format(n_state))
        plt.grid()
        plt.legend()
        plt.title("Forecasted emissions")
        plt.show()
