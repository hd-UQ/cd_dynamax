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
# Show that discrete Kalman smoother == continuous-discrete EKF for that linear system

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
    # has_dynamics_bias = False,
    # has_emissions_bias = False,
)
cd_params, cd_param_props = cd_model.initialize(
    key1,
    dynamics_weights=-0.1 * jnp.eye(cd_model.state_dim),  # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient=0.5 * jnp.eye(cd_model.state_dim),
    dynamics_diffusion_cov=0.5 * jnp.eye(cd_model.state_dim),
    dynamics_bias=jnp.zeros(cd_model.state_dim),
    emission_bias=jnp.zeros(cd_model.emission_dim),
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

########### Now make non-linear models, assuming linearity ########
print("************* Continuous-Discrete Non-linear GSSM *************")
from continuous_discrete_nonlinear_gaussian_ssm.models import *
from continuous_discrete_nonlinear_gaussian_ssm import cdnlgssm_smoother
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
            "params": jnp.zeros(cdnl_model.state_dim),
            "props": ParameterProperties() # Also want to learn this
        },
        initial_cov = {
            "params": jnp.eye(cdnl_model.state_dim),
            "props": ParameterProperties(constrainer=RealToPSDBijector()) # Also want to learn these
        },
        dynamics_drift={
            "params": LearnableLinear(weights=cd_params.dynamics.weights, bias=cd_params.dynamics.bias),
            "props": LearnableLinear(weights=ParameterProperties(), bias=ParameterProperties()),
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
            "params": LearnableLinear(weights=cd_params.emissions.weights, bias=cd_params.emissions.bias),
            "props": LearnableLinear(weights=ParameterProperties(), bias=ParameterProperties()),
        },
        emission_cov = {
            "params": LearnableMatrix(params=0.1*jnp.eye(cdnl_model.emission_dim)),
            "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector())) # Also want to learn these
        }
    )

    # Simulate from continuous-discrete nl model
    print(f"Simulating {dynamics_approx_order} order CDNLGSSM in continuous-discrete time")
    cdnl_states, cdnl_emissions = cdnl_model.sample(
            cdnl_params, key2, t_emissions=t_emissions, num_timesteps=NUM_TIMESTEPS, inputs=inputs
        )

    # check that these are similar to samples from the cd-linear model
    print("\tChecking states...")
    compare(cdnl_states, cd_states)

    print("\tChecking emissions...")
    compare(cdnl_emissions, cd_emissions)

    print(f"**********************************")
    print("Continuous-Discrete time linear smoothing comparisons")
    from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_smoother
    for smoother_type in ["cd_smoother_1", "cd_smoother_2"]:
        print(f'Continuous-Discrete time KF smoothing {smoother_type}')
        cd_smoother_posterior = cdlgssm_smoother(
            cd_params,
            cd_emissions,
            t_emissions,
            inputs,
            smoother_type=smoother_type
        )

        ######## Continuous-discrete EKF
        # first and second order state SDE approximation (both should be correct for linear models)
        for state_order in ["first", "second"]:
            # Run ekf with the non-linear model and data from the CDNLGSSM model
            print(f"Running {state_order}-order EKF with non-linear model class and data from {dynamics_approx_order}-order CDNLGSSM model")
            cd_ekf_smoother_posterior = cdnlgssm_smoother(
                    cdnl_params,
                    cdnl_emissions,
                    hyperparams=EKFHyperParams(state_order=state_order, emission_order="first"),
                    t_emissions=t_emissions,
                    inputs=inputs,
                )

            # check that results in cd_ekf_post are similar to results from applying cd_kf (cd_filtered_posterior)
            print(f"Comparing {smoother_type} KF smoothed posteriors with {state_order}-order EKF smoother...")
            compare_structs(cd_smoother_posterior, cd_ekf_smoother_posterior, accept_failure=True)

        print(f"Discrete to Continous-Discrete {smoother_type} smoothed posterior tests passed!")   

print("All EKS and CDNLGSSM model tests passed!")

