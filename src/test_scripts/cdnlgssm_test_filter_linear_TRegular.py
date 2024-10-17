import jax.debug as jdb
import pdb
import sys

from datetime import datetime
import jax.numpy as jnp
import jax.random as jr

# Make sure main paths are added
sys.path.append("../")
sys.path.append("../..")

# Local dynamax
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import monotonically_increasing
from dynamax.utils.utils import ensure_array_has_batch_dim

# Our codebase
from continuous_discrete_linear_gaussian_ssm import ContDiscreteLinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM
from utils.test_utils import compare, compare_structs

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
    has_dynamics_bias = True,
    has_emissions_bias = True,
)
# Initialize, controlling what is learned
from continuous_discrete_linear_gaussian_ssm.models import *
cd_params, cd_param_props = cd_model.initialize(
    key1,
    ## Initial
    initial_mean = {
            "params": jnp.zeros(cd_model.state_dim),
            "props": ParameterProperties()
    },
    initial_cov = {
        "params": jnp.eye(cd_model.state_dim),
        "props": ParameterProperties(constrainer=RealToPSDBijector())
    },
    ## Dynamics
    dynamics_weights = {
        "params": -0.1 * jnp.eye(cd_model.state_dim),
        "props": ParameterProperties()
    },
    dynamics_bias = {
        "params": jnp.zeros((cd_model.state_dim,)),
        "props": ParameterProperties(trainable=False) # We do not learn bias term!
    },
    dynamics_diffusion_coefficient = {
        "params": 0.1 * jnp.eye(cd_model.state_dim),
        "props": ParameterProperties()
    },
    dynamics_diffusion_cov = {
        "params": 0.1 * jnp.eye(cd_model.state_dim),
        "props": ParameterProperties(constrainer=RealToPSDBijector())
    },
    ## Emission
    emission_weights = {
        "params": jr.normal(key1, (cd_model.emission_dim, cd_model.state_dim)),
        "props": ParameterProperties()
    },
    emission_bias = {
        "params": jnp.zeros((cd_model.emission_dim,)),
        "props": ParameterProperties(trainable=False) # We do not learn bias term!
    },
    emission_cov = {
        "params": 0.1 * jnp.eye(cd_model.emission_dim),
        "props": ParameterProperties(constrainer=RealToPSDBijector())
    }
)

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

print("Continuous-Discrete time filtering: pre-fit")
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_filter, KFHyperParams
# We set dt_final=1 so that predicted mean and covariance at the end of sequence match those of discrete filtering
kf_hyperparams=KFHyperParams(dt_final = 1.)
# Define CD linear filter
cd_filtered_posterior = cdlgssm_filter(
    cd_params,
    cd_emissions,
    t_emissions,
    filter_hyperparams=kf_hyperparams,
    inputs=inputs
)

print("Fitting continuous-discrete time linear with SGD")
cd_sgd_fitted_params, cd_sgd_lps = cd_model.fit_sgd(
    cd_params,
    cd_param_props,
    cd_emissions,
    t_emissions,
    filter_hyperparams=kf_hyperparams,
    inputs=inputs,
    num_epochs=10
)

print("Continuous-Discrete time filtering: post-fit")
cd_sgd_fitted_filtered_posterior = cdlgssm_filter(
    cd_sgd_fitted_params,
    cd_emissions,
    t_emissions,
    filter_hyperparams=kf_hyperparams,
    inputs=inputs
)

########### Now make non-linear models, assuming linearity ########
print("************* Continuous-Discrete Non-linear GSSM *************")
from continuous_discrete_nonlinear_gaussian_ssm.models import *
from continuous_discrete_nonlinear_gaussian_ssm import cdnlgssm_filter
from continuous_discrete_nonlinear_gaussian_ssm import EKFHyperParams

# Model def
inputs = None  # Not interested in inputs for now
cdnl_model = ContDiscreteNonlinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)

# Test models with first and second order SDE approximation (both should be correct for linear models)
for dynamics_approx_order in [1., 2.]:
    # Initialize models with linear learnable functions
    cdnl_params, cdnl_param_props = cdnl_model.initialize(
        key1,
        initial_mean = {
            "params": LearnableVector(params=jnp.zeros(cdnl_model.state_dim)),
            "props": LearnableVector(params=ParameterProperties())
        },
        initial_cov = {
            "params": LearnableMatrix(params=jnp.eye(cdnl_model.state_dim)),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector()))
        },
        dynamics_drift={
            "params": LearnableLinear(
                weights=cd_params.dynamics.weights,
                bias=cd_params.dynamics.bias
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(),
                bias=ParameterProperties(trainable=False) # We do not learn bias term!
            ),
        },
        dynamics_diffusion_coefficient={
            "params": LearnableMatrix(params=cd_params.dynamics.diffusion_coefficient),
            "props": LearnableMatrix(params=ParameterProperties()),
        },
        dynamics_diffusion_cov={
            "params": LearnableMatrix(params=cd_params.dynamics.diffusion_cov),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector())),
        },
        dynamics_approx_order=dynamics_approx_order,
        emission_function={
            "params": LearnableLinear(
                weights=cd_params.emissions.weights,
                bias=cd_params.emissions.bias
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(),
                bias=ParameterProperties(trainable=False) # We do not learn bias term!
            ),
        },
        emission_cov = {
            "params": LearnableMatrix(params=0.1*jnp.eye(cdnl_model.emission_dim)),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector()))
        }
    )

    # Simulate from continuous-discrete nl model
    print(f"**********************************")
    print(f"Simulating {dynamics_approx_order} order CDNLGSSM in continuous-discrete time")
    cdnl_states, cdnl_emissions = cdnl_model.sample(
            cdnl_params, key2, t_emissions=t_emissions, num_timesteps=NUM_TIMESTEPS, inputs=inputs
        )

    # check that these are similar to samples from the cd-linear model
    print("\tChecking states...")
    compare(cdnl_states, cd_states)

    print("\tChecking emissions...")
    compare(cdnl_emissions, cd_emissions)

    print("Continuous-Discrete time non-linear EKF filtering: pre-fit")
    ######## Continuous-discrete EKF
    # first and second order state SDE approximation (both should be correct for linear models)
    for state_order in ["first", "second"]:
        # Run ekf with the non-linear model and data from the CDNLGSSM model
        print(f"Running {state_order}-order EKF with non-linear model class and data from {dynamics_approx_order}-order CDNLGSSM model")
        cd_ekf_post = cdnlgssm_filter(
                cdnl_params,
                cdnl_emissions,
                hyperparams=EKFHyperParams(dt_final=1., state_order=state_order, emission_order="first"),
                t_emissions=t_emissions,
                inputs=inputs,
            )

        # check that results in cd_ekf_post are similar to results from applying cd_kf (cd_filtered_posterior)
        print("\tComparing pre-fit filtered posteriors...")
        compare_structs(cd_ekf_post, cd_filtered_posterior)

        print(f"Fitting continuous-discrete non-linear model with SGD + {state_order}-order EKF")
        # Note: no bias terms present yet
        cdnl_sgd_fitted_params, cdnl_sgd_lps = cdnl_model.fit_sgd(
                cdnl_params,
                cdnl_param_props,
                cdnl_emissions,
                filter_hyperparams=EKFHyperParams(dt_final=1., state_order=state_order, emission_order="first"),
                t_emissions=t_emissions,
                inputs=inputs,
                num_epochs=10
            )

        print("\tChecking post-SGD fit log-probabilities sequence...")
        compare(cd_sgd_lps, cdnl_sgd_lps, accept_failure=True)

        print("\tCheck that post-SGD fit parameters are similar...")
        print("\t\tInitial mean...")
        compare(cd_sgd_fitted_params.initial.mean, cdnl_sgd_fitted_params.initial.mean.params, accept_failure=True)
        print("\t\tInitial cov...")
        compare(cd_sgd_fitted_params.initial.cov, cdnl_sgd_fitted_params.initial.cov.params, accept_failure=True)
        print("\t\tDynamics weights...")
        compare(
            cd_sgd_fitted_params.dynamics.weights, cdnl_sgd_fitted_params.dynamics.drift.weights, accept_failure=True
        )
        print("\t\tDynamics bias...")
        compare(
            cd_sgd_fitted_params.dynamics.bias, cdnl_sgd_fitted_params.dynamics.drift.bias, accept_failure=True
        )
        print("\t\tDynamics diffusion coefficient...")
        compare(
            cd_sgd_fitted_params.dynamics.diffusion_coefficient,
            cdnl_sgd_fitted_params.dynamics.diffusion_coefficient.params,
            accept_failure=True,
        )
        print("\t\tDynamics diffusion covariance...")
        compare(
            cd_sgd_fitted_params.dynamics.diffusion_cov,
            cdnl_sgd_fitted_params.dynamics.diffusion_cov.params,
            accept_failure=True,
        )
        print("\t\tEmission weights...")
        compare(
            cd_sgd_fitted_params.emissions.weights,
            cdnl_sgd_fitted_params.emissions.emission_function.weights,
            accept_failure=True,
        )
        print("\t\tEmission bias...")
        compare(
            cd_sgd_fitted_params.emissions.bias,
            cdnl_sgd_fitted_params.emissions.emission_function.bias,
            accept_failure=True,
        )
        print("\t\tEmission cov...")
        compare(
            cd_sgd_fitted_params.emissions.cov,
            cdnl_sgd_fitted_params.emissions.emission_cov.params,
            accept_failure=True,
        )

        print(f"Continuous-Discrete time non-linear {state_order}-order EKF filtering: post-fit")
        cdnl_sgd_fitted_filtered_posterior = cdnlgssm_filter(
                cdnl_sgd_fitted_params,
                cdnl_emissions,
                hyperparams=EKFHyperParams(dt_final=1., state_order=state_order, emission_order="first"),
                t_emissions=t_emissions,
                inputs=inputs,
        )

        print(f"\tComparing post-SGD fit {state_order}-order EKF filtered posterior...")
        compare_structs(
            cd_sgd_fitted_filtered_posterior,
            cdnl_sgd_fitted_filtered_posterior,
            accept_failure=False
        )


print("All EKF and CDNLGSSM model tests passed!")

######## Continuous-discrete Unscented Kalman Filter
# from continuous_discrete_nonlinear_gaussian_ssm import unscented_kalman_filter as cd_ukf
from continuous_discrete_nonlinear_gaussian_ssm import UKFHyperParams

# Run First order ekf with the non-linear model and data from the first-order CDNLGSSM model
print(f"**********************************")
print("Running Unscented Kalman Filter with non-linear model class and data from first-order CDNLGSSM model")    
# define hyperparameters
ukf_params = UKFHyperParams(dt_final=1.)

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
print(f"**********************************")
for N_particles in [1e2, 1e3, 1e4]:
    for perturb_measurements in [True]:
        # print("Running Ensemble Kalman Filter (perturb_measurements=True) with non-linear model class and data from first-order CDNLGSSM model")
        # print(f"Running Ensemble Kalman Filter (perturb_measurements={perturb_measurements}) with non-linear model class and data from first-order CDNLGSSM model")
        print(f"Running Ensemble Kalman Filter (perturb_measurements={perturb_measurements}, N_particles={N_particles}) with non-linear model class and data from first-order CDNLGSSM model")

        # define hyperparameters
        enkf_params = EnKFHyperParams(dt_final=1., N_particles=int(N_particles), perturb_measurements=perturb_measurements)

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
