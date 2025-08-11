import jax
jax.config.update("jax_platform_name", "cpu")

# JAX device check
print("************* Checking JAX device *************")
print('Running on jax device:{}'.format(
        jax.devices()
    )
)
print('Running on jax device platform:{}'.format(
        jax.devices()[0].platform
    )
)
print("***********************************************")

# Main imports
import sys
from itertools import count

# Import jax and utils
from jax import numpy as jnp
from jax import vmap
import jax.random as jr
from jax import jit, vmap
import diffrax as dfx

# Additional, custom codebase
sys.path.append("../")
sys.path.append("../..")

# Our own custom src codebase
# Continuous-discrete linear Gaussian SSM codebase
# CD-LGSSM models
from continuous_discrete_linear_gaussian_ssm import ContDiscreteLinearGaussianSSM
from continuous_discrete_linear_gaussian_ssm.models import *
from continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import *

# continuous-discrete nonlinear Gaussian SSM codebase
# CD-Nonlinear Gaussian models
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm.models import *
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import *

# Useful utility functions
from utils.simulation_utils import *

# Plotting
# Our own custom plotting codebase
from utils.plotting_utils import *

##################### SIMULATION #####################
### Simulation set-up

# Randomness
keys = map(jr.PRNGKey, count())

# Simulation parameters
T_total = 10
T_filter = int(0.8 * T_total)
num_timesteps_total = int(T_total / 0.01)

## Train set
(
    t_emissions_train,
    t_filter_train,
    t_forecast_train,
    num_timesteps_train,
    num_timesteps_filter_train,
    num_timesteps_forecast_train,
) = generate_irregular_t_emissions(
    T_total=T_total,
    num_timesteps=num_timesteps_total,
    T_filter=T_filter,
    key=next(keys)
)

###################### CD-LGSSM ######################
## Main settings
state_dim = 3
emission_dim = 3

# Model definition: MySSM is a CD-LGSSM
true_model = ContDiscreteLinearGaussianSSM(state_dim, emission_dim)

# Get the default parameters
true_model_params, true_model_param_props=create_cdlgssm_params_and_props(
    params=true_model._default_cdlgssm_params()
)

# Prior definition
from utils.prior_utils import CDLGSSM_WeightPrior
weight_prior = CDLGSSM_WeightPrior(
    params=true_model_params
)

# Initialize the model with new, default parameters
new_model_params_dict = true_model._default_cdlgssm_params()
# Because we are using a prior, we need to indicate that these are trainable parameters before computing the prior averaged Fisher information matrix
new_model_params_dict['dynamics_weights']['props'].trainable=True

# Prior averaged FIM
expected_prior_FIM_cdlgssm = true_model.prior_averaged_fisher(
    prior = weight_prior,
    initial_params = new_model_params_dict, # Initialize with dictionaries, not ParamsCDLGSSM
    num_timesteps= num_timesteps_train,
    t_emissions=t_emissions_train,
    inputs=None,
    filter_hyperparams = None,
    transition_type = "distribution",
    n_samples = 10,
    key = next(keys)
)

pdb.set_trace()

###################### CD-NLGSSM ######################
# Load the Lorenz 63 drift function definition
from utils.physics_based_models import LearnableLorenz63_Drift
# Define the true parameters of the drift function
true_l63_drift_params = jnp.array([10.0, 28.0, 8 / 3])
# And the corresponding Lorenz 63 system
true_drift = {
    "params": LearnableLorenz63_Drift(
        sigma=true_l63_drift_params[0],
        rho=true_l63_drift_params[1],
        beta=true_l63_drift_params[2]
    ),
    "props": LearnableLorenz63_Drift(
        sigma=ParameterProperties(trainable=False),
        rho=ParameterProperties(trainable=False),
        beta=ParameterProperties(trainable=False)
    ),
}

# Define the true parameters of the diffusion function
true_diffusion_cov = {
    "params": LearnableMatrix(
        params=jnp.eye(state_dim)
    ),
    "props": LearnableMatrix(
        params=ParameterProperties(
            trainable=False,
            constrainer=RealToPSDBijector()
        )
    ),
}

# Define the true parameters of the diffusion function
true_diffusion_coefficient_param = 1.0
true_diffusion_coefficient = {
    "params": LearnableMatrix(
        params=true_diffusion_coefficient_param * jnp.eye(state_dim)
    ),
    "props": LearnableMatrix(
        params=ParameterProperties(
            trainable=False
        )
    ),
}

# Define the true parameters of the emission function
# Full observability
H=jnp.eye(emission_dim,state_dim)
true_emission = {
    "params": LearnableLinear(
        weights=H,
        bias=jnp.zeros(emission_dim)
    ),
    "props": LearnableLinear(
        weights=ParameterProperties(
            trainable=False,
        ),
        bias=ParameterProperties(
            trainable=False,
        )
    ),
}

# Define the true parameters of the emission covariance
R=jnp.eye(emission_dim)
true_emission_cov = {
    "params": LearnableMatrix(
        params=R
    ),
    "props": LearnableMatrix(
        params=ParameterProperties(
            trainable=False,
            constrainer=RealToPSDBijector()
        )
    ),
}

# Define the true initial mean and covariance
true_initial_mean = {
    "params": LearnableVector(
        params=jnp.zeros(state_dim)
    ),
    "props": LearnableVector(
        params=ParameterProperties(
            trainable=False
        )
    ),
}

true_initial_cov_param = 400.0 # approximate variance of the Lorenz attractor
true_initial_cov = {
    "params": LearnableMatrix(
        params=true_initial_cov_param*jnp.eye(state_dim)
    ),
    "props": LearnableMatrix(
        params=ParameterProperties(
            trainable=False,
            constrainer=RealToPSDBijector()
        )
    ),
}

true_dynamics_approx_order = {
            "params": 2.,
            "props": ParameterProperties(trainable=False), # never trainable, no constraints to apply.
        }
# Concatenate all parameters in dictionary, for later easy use
all_true_params = {
    'initial_mean': true_initial_mean,
    'initial_cov': true_initial_cov,
    'dynamics_drift': true_drift,
    'dynamics_diffusion_coefficient': true_diffusion_coefficient,
    'dynamics_diffusion_cov': true_diffusion_cov,
    'dynamics_approx_order': true_dynamics_approx_order,
    'emission_function': true_emission,
    'emission_cov': true_emission_cov,
}


### Model creation: object instantiation
hifi_forward_settings = {
}  # empty uses default settings

# Create CD-NLGSSM model
true_model = ContDiscreteNonlinearGaussianSSM(
    state_dim,
    emission_dim,
    diffeqsolve_settings=hifi_forward_settings
)
true_params, _ = true_model.initialize(next(keys), **all_true_params)

# Load the prior from utils
from utils.prior_utils import CDNLGSSM_DynamicDrift_L63ParamsPrior

# Prior for the dynamic drift parameters
new_prior = CDNLGSSM_DynamicDrift_L63ParamsPrior(params=true_params)

# Initialize the new model with different L63 parameters ---to be replaced with the prior
# Define the true parameters of the drift function
init_l63_drift_params = jnp.array([1.0, 2.0, 3.0])
init_drift = {
    "params": LearnableLorenz63_Drift(
        sigma=init_l63_drift_params[0],
        rho=init_l63_drift_params[1],
        beta=init_l63_drift_params[2]
    ),
    "props": LearnableLorenz63_Drift(
        sigma=ParameterProperties(trainable=True), # Set this to true to allow training/prior averaged computation
        rho=ParameterProperties(trainable=True), # Set this to true to allow training/prior averaged computation
        beta=ParameterProperties(trainable=True), # Set this to true to allow training/prior averaged computation
    ),
}

# Concatenate all parameters in dictionary, for later easy use
all_init_params = {
    'initial_mean': true_initial_mean,
    'initial_cov': true_initial_cov,
    'dynamics_drift': init_drift,
    'dynamics_diffusion_coefficient': true_diffusion_coefficient,
    'dynamics_diffusion_cov': true_diffusion_cov,
    'dynamics_approx_order': true_dynamics_approx_order,
    'emission_function': true_emission,
    'emission_cov': true_emission_cov,
}

# Prior averaged FIM
expected_prior_FIM_cdnlgssm = true_model.prior_averaged_fisher(
    prior = new_prior,
    initial_params = all_init_params, # Initialize with dictionaries, not ParamsCDNLGSSM
    num_timesteps= num_timesteps_train,
    t_emissions=t_emissions_train,
    inputs=None,
    filter_hyperparams = None,
    transition_type = "distribution",
    n_samples = 10,
    key = next(keys)
)

pdb.set_trace()