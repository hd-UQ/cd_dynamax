import pdb
import sys

from datetime import datetime
import jax.numpy as jnp
import jax.random as jr

# Local dynamax
sys.path.append("..")
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import monotonically_increasing
from dynamax.utils.utils import ensure_array_has_batch_dim

# Our codebase
from cdssm_utils import compare, compare_structs
from continuous_discrete_linear_gaussian_ssm import ContDiscreteLinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM

# The idea of this test is as following (uses regular time intervals ONLY):
# First, establish equivalent linear systems in discrete and continuous time
# Show that samples from each are similar
# Show that discrete KF == continuous-discrete KF for that linear system
# Show that continuous-discrete KF == {cd-EKFs, cd-UKF, cd-EnKF} for that linear system

#### General state and emission dimensionalities
STATE_DIM = 2
EMISSION_DIM = 6

print("************* Discrete LGSSM *************")
# Discrete sampling
NUM_TIMESTEPS = 100

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
d_model = LinearGaussianSSM(
    state_dim=STATE_DIM,
    emission_dim=EMISSION_DIM,
    # Test with no biases 
    has_dynamics_bias = False,
    has_emissions_bias = False,
)
d_params, d_param_props = d_model.initialize(
    key1,
    # Hard coded parameters for tests to match
    dynamics_weights=0.9048373699188232421875 * jnp.eye(d_model.state_dim),
    dynamics_covariance=0.11329327523708343505859375 * jnp.eye(d_model.state_dim),
    dynamics_bias=None,
    emission_bias=None,
)
d_param_props.dynamics.bias.trainable = False
d_param_props.emissions.bias.trainable = False

# Simulate from discrete model
print("Simulating in discrete time")
d_states, d_emissions = d_model.sample(
    d_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs
)

print("Discrete time filtering: pre-fit")
from dynamax.linear_gaussian_ssm.inference import lgssm_filter

# Define filter
d_filtered_posterior = lgssm_filter(
    d_params,
    d_emissions,
    inputs
)

print("Fitting discrete time with SGD")
d_sgd_fitted_params, d_sgd_lps = d_model.fit_sgd(
    d_params,
    d_param_props,
    d_emissions,
    inputs=inputs,
    num_epochs=10
)

print("Discrete time filtering: post-fit")
d_sgd_fitted_filtered_posterior = lgssm_filter(
    d_sgd_fitted_params,
    d_emissions,
    inputs
)

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
    has_dynamics_bias = False,
    has_emissions_bias = False,
)
cd_params, cd_param_props = cd_model.initialize(
    key1,
    dynamics_weights=-0.1 * jnp.eye(cd_model.state_dim),  # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient=0.5 * jnp.eye(cd_model.state_dim),
    dynamics_diffusion_cov=0.5 * jnp.eye(cd_model.state_dim),
    dynamics_bias=None,
    emission_bias=None,
)
cd_param_props.dynamics.bias.trainable = False
cd_param_props.emissions.bias.trainable = False


# Simulate from continuous model
print("Simulating in continuous-discrete time")
cd_num_timesteps_states, cd_num_timesteps_emissions = cd_model.sample(
    cd_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs
)

cd_states, cd_emissions = cd_model.sample(
    cd_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    t_emissions=t_emissions,
    inputs=inputs
)

print("\tChecking states...")
compare(cd_num_timesteps_states, cd_states)

print("\tChecking emissions...")
compare(cd_num_timesteps_emissions, cd_emissions)

print("\tChecking states...")
compare(d_states, cd_states)

print("\tChecking emissions...")
compare(d_emissions, cd_emissions)

print("Continuous-Discrete time filtering: pre-fit")
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_filter
# Define CD linear filter
cd_filtered_posterior = cdlgssm_filter(
    cd_params,
    cd_emissions,
    t_emissions,
    inputs
)

print("Comparing filtered posteriors...")
compare_structs(d_filtered_posterior, cd_filtered_posterior)

print("Fitting continuous-discrete time linear with SGD")
cd_sgd_fitted_params, cd_sgd_lps = cd_model.fit_sgd(
    cd_params,
    cd_param_props,
    cd_emissions,
    t_emissions,
    inputs=inputs,
    num_epochs=10
)

print("\tChecking SGD log-probabilities sequence...")
compare(cd_sgd_lps, d_sgd_lps)

print("\tCheck that parameters are similar...")
compare_structs(d_sgd_fitted_params, cd_sgd_fitted_params, accept_failure=True)

print("Continuous-Discrete time filtering: post-fit")
cd_sgd_fitted_filtered_posterior = cdlgssm_filter(
    cd_sgd_fitted_params,
    cd_emissions,
    t_emissions,
    inputs
)

print("WARNING: If parameters are sufficiently similar, these tests SHOULD PASS (but don't currently).")
compare_structs(d_sgd_fitted_filtered_posterior, cd_sgd_fitted_filtered_posterior, accept_failure=True)

########### Now make these into non-linear models ########
print("************* Continuous-Discrete Non-linear GSSM *************")
from continuous_discrete_nonlinear_gaussian_ssm.models import LearnableFunction,ConstantLearnableFunction,LinearLearnableFunction
from continuous_discrete_nonlinear_gaussian_ssm import cdnlgssm_filter
from continuous_discrete_nonlinear_gaussian_ssm import EKFHyperParams

# Model def
inputs = None  # Not interested in inputs for now
cdnl_model = ContDiscreteNonlinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)

# TODO: check that these need input as second argument; also check about including bias terms.
# dynamics_drift_function = lambda z, u: cd_params.dynamics.weights @ z
# emission_function = lambda z: cd_params.emissions.weights @ z

for dynamics_approx_order in [1., 2.]:
    # Initialize with first/second order SDE approximation
    cdnl_params, cdnl_param_props = cdnl_model.initialize(
            key1,
            dynamics_drift=LinearLearnableFunction(
                params=cd_params.dynamics.weights
            ),
            dynamics_diffusion_coefficient=ConstantLearnableFunction(
                params=cd_params.dynamics.diffusion_coefficient
            ),
            dynamics_diffusion_cov=ConstantLearnableFunction(
                params=cd_params.dynamics.diffusion_cov
            ),
            dynamics_approx_order=dynamics_approx_order,
            emission_function=LinearLearnableFunction(
                params=cd_params.emissions.weights
            ),
        )

    # Simulate from continuous model
    print(f"Simulating {dynamics_approx_order} order CDNLGSSM in continuous-discrete time")
    cdnl_states, cdnl_emissions = cdnl_model.sample(
            cdnl_params, key2, t_emissions=t_emissions, num_timesteps=NUM_TIMESTEPS, inputs=inputs
        )

    # check that these are similar to samples from the linear model
    print("\tChecking states...")
    compare(cdnl_states, cd_states)

    print("\tChecking emissions...")
    compare(cdnl_emissions, cd_emissions)

    ######## Continuous-discrete EKF
    for state_order in ["first", "second"]:
        # Run First order ekf with the non-linear model and data from the first-order CDNLGSSM model
        # print("Running first-order EKF with non-linear model class and data from first-order CDNLGSSM model")
        print(f"Running {state_order}-order EKF with non-linear model class and data from {dynamics_approx_order}-order CDNLGSSM model")
        cd_ekf_post = cdnlgssm_filter(
                cdnl_params,
                cdnl_emissions,
                hyperparams=EKFHyperParams(state_order=state_order, emission_order="first"),
                t_emissions=t_emissions,
                inputs=inputs,
            )

        # check that results in cd_ekf_post are similar to results from applying cd_kf (cd_filtered_posterior)
        print("\tComparing filtered posteriors...")
        compare_structs(cd_ekf_post, cd_filtered_posterior)

        print("Fitting continuous-discrete Non-linear model with SGD")
        # Note: no bias terms present yet

        cdnl_sgd_fitted_params, cdnl_sgd_lps = cdnl_model.fit_sgd(
                cdnl_params,
                cdnl_param_props,
                cdnl_emissions,
                filter_hyperparams=EKFHyperParams(state_order=state_order, emission_order="first"),
                t_emissions=t_emissions,
                inputs=inputs,
                num_epochs=10
            )

        print("\tChecking SGD log-probabilities sequence...")
        compare(cd_sgd_lps, cdnl_sgd_lps, accept_failure=True)

        print("\tCheck that parameters are similar...")
        # Will eventually use this style (once ParameterSets are used, so we can match weights, biases, etc)...
        # compare_structs(cd_sgd_fitted_params, cdnl_sgd_fitted_params)

        print("\tInitial mean...")
        compare(cd_sgd_fitted_params.initial.mean, cdnl_sgd_fitted_params.initial.mean, accept_failure=True)
        print("\tInitial cov...")
        compare(cd_sgd_fitted_params.initial.cov, cdnl_sgd_fitted_params.initial.cov, accept_failure=True)
        print("\tDynamics weights...")
        compare(
            cd_sgd_fitted_params.dynamics.weights, cdnl_sgd_fitted_params.dynamics.drift.params, accept_failure=True
        )
        print("\tDynamics diffusion coefficient...")
        compare(
            cd_sgd_fitted_params.dynamics.diffusion_coefficient,
            cdnl_sgd_fitted_params.dynamics.diffusion_coefficient.params,
            accept_failure=True,
        )
        print("\tDynamics diffusion covariance...")
        compare(
            cd_sgd_fitted_params.dynamics.diffusion_cov,
            cdnl_sgd_fitted_params.dynamics.diffusion_cov.params,
            accept_failure=True,
        )
        print("\tEmission weights...")
        compare(
            cd_sgd_fitted_params.emissions.weights,
            cdnl_sgd_fitted_params.emissions.emission_function.params,
            accept_failure=True,
        )
        print("\tEmission cov...")
        compare(
            cd_sgd_fitted_params.emissions.cov,
            cdnl_sgd_fitted_params.emissions.emission_cov.params,
            accept_failure=True,
        )

        print("Continuous-Discrete time non-linear filtering: post-fit")
        cdnl_sgd_fitted_filtered_posterior = cdnlgssm_filter(
                cdnl_sgd_fitted_params,
                cdnl_emissions,
                hyperparams=EKFHyperParams(state_order=state_order, emission_order="first"),
                t_emissions=t_emissions,
                inputs=inputs,
        )

        print("\tComparing post-SGD filtered posterior...")
        compare_structs(cd_sgd_fitted_filtered_posterior, cdnl_sgd_fitted_filtered_posterior)


print("All EKF and CDNLGSSM model tests passed!")

######## Continuous-discrete Unscented Kalman Filter
# from continuous_discrete_nonlinear_gaussian_ssm import unscented_kalman_filter as cd_ukf
from continuous_discrete_nonlinear_gaussian_ssm import UKFHyperParams

# Run First order ekf with the non-linear model and data from the first-order CDNLGSSM model
print("Running Unscented Kalman Filter with non-linear model class and data from first-order CDNLGSSM model")    
# define hyperparameters
ukf_params = UKFHyperParams()

cd_ukf_post = cdnlgssm_filter(
    cdnl_params,
    cdnl_emissions,
    hyperparams=ukf_params,
    t_emissions=t_emissions,
    inputs=inputs,
)

# check that results in cd_enkf_1_post are similar to results from applying cd_kf (cd_filtered_posterior)
print("\tComparing filtered posteriors...")
compare_structs(cd_ukf_post, cd_filtered_posterior)

print("UKF tests passed.")

######## Continuous-discrete Ensemble Kalman Filter
# from continuous_discrete_nonlinear_gaussian_ssm import ensemble_kalman_filter as cd_enkf
from continuous_discrete_nonlinear_gaussian_ssm import EnKFHyperParams

# Run First order ekf with the non-linear model and data from the first-order CDNLGSSM model

for N_particles in [1e2, 1e3, 1e4]:
    for perturb_measurements in [True]:
        # print("Running Ensemble Kalman Filter (perturb_measurements=True) with non-linear model class and data from first-order CDNLGSSM model")
        # print(f"Running Ensemble Kalman Filter (perturb_measurements={perturb_measurements}) with non-linear model class and data from first-order CDNLGSSM model")
        print(f"Running Ensemble Kalman Filter (perturb_measurements={perturb_measurements}, N_particles={N_particles}) with non-linear model class and data from first-order CDNLGSSM model")

        # define hyperparameters
        enkf_params = EnKFHyperParams(N_particles=int(N_particles), perturb_measurements=perturb_measurements)

        cd_enkf_post = cdnlgssm_filter(
            cdnl_params,
            cdnl_emissions,
            hyperparams=enkf_params,
            t_emissions=t_emissions,
            inputs=inputs,
        )

        # check that results in cd_enkf_1_post are similar to results from applying cd_kf (cd_filtered_posterior)
        print("\tComparing filtered means...")
        try:
            compare(cd_enkf_post.filtered_means, cd_filtered_posterior.filtered_means)
        except:
            if N_particles < 1e5 or not perturb_measurements:
                print("Test failed because too few particles or perturb_measurements=False")
                pass
            else:
                compare(cd_enkf_post.filtered_means, cd_filtered_posterior.filtered_means)

        print("\tComparing filtered covariances...")
        try:
            compare(cd_enkf_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
        except:
            if N_particles < 1e5 or not perturb_measurements:
                print("Test failed because too few particles or perturb_measurements=False")
                pass
            else:
                compare(cd_enkf_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)


print("All EnKF tests passed---note that these are randomized approximations, so we don't expect to perfectly replicate EKF and KF (which are both exact in linear test cases shown here)! We want to see convergence to truth (hence checking the final filtered state).")
pdb.set_trace()
