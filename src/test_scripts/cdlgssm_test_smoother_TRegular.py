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
# Show that discrete KS == continuous-discrete KS for that linear system

#### Randomness
key_init, key_sample = jr.split(jr.PRNGKey(0))

#### General state and emission dimensionalities
STATE_DIM = 2
EMISSION_DIM = 6

print("************* Discrete LGSSM *************")
# Discrete sampling
NUM_TIMESTEPS = 100

# Model def
inputs = None  # Not interested in inputs for now
d_model = LinearGaussianSSM(
    state_dim=STATE_DIM,
    emission_dim=EMISSION_DIM,
)
d_params, d_param_props = d_model.initialize(
    key_init,
    # Hard coded parameters for tests to match
    dynamics_weights=0.9048373699188232421875 * jnp.eye(d_model.state_dim),
    dynamics_covariance=0.11329327523708343505859375 * jnp.eye(d_model.state_dim),
    dynamics_bias=jnp.zeros(d_model.state_dim),
    emission_bias=jnp.zeros(d_model.emission_dim),
)

# Simulate from discrete model
print("Simulating in discrete time")
d_states, d_emissions = d_model.sample(
    d_params,
    key_sample,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs
)

from dynamax.linear_gaussian_ssm.inference import lgssm_smoother
print('Discrete time smoothing')
d_smoother_posterior = lgssm_smoother(d_params, d_emissions, inputs)

print("************* Continuous-Discrete LGSSM *************")
# Continuous-Discrete model
t_emissions = jnp.arange(NUM_TIMESTEPS)[:, None]

# Model def
inputs = None  # Not interested in inputs for now
cd_model = ContDiscreteLinearGaussianSSM(
    state_dim=STATE_DIM,
    emission_dim=EMISSION_DIM,
)

# Initialize, controlling what is learned
from continuous_discrete_linear_gaussian_ssm.models import *
cd_params, cd_param_props = cd_model.initialize(
    key_init,
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
        "props": ParameterProperties()
    },
    dynamics_diffusion_coefficient = {
        "params": 0.5 * jnp.eye(cd_model.state_dim),
        "props": ParameterProperties()
    },
    dynamics_diffusion_cov = {
        "params": 0.5 * jnp.eye(cd_model.state_dim),
        "props": ParameterProperties(constrainer=RealToPSDBijector())
    },
    ## Emission
    emission_weights = {
        "params": jr.normal(key_init, (cd_model.emission_dim, cd_model.state_dim)),
        "props": ParameterProperties()
    },
    emission_bias = {
        "params": jnp.zeros((cd_model.emission_dim,)),
        "props": ParameterProperties()
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
    key_sample,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs
)

cd_states, cd_emissions = cd_model.sample(
    cd_params,
    key_sample,
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

from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_smoother, KFHyperParams
# We set dt_final=1 so that predicted mean and covariance at the end of sequence match those of discrete filtering
kf_hyperparams=KFHyperParams(dt_final = 1.)

for smoother_type in ["cd_smoother_1", "cd_smoother_2"]:
    print(f'Continuous-Discrete time KF smoothing {smoother_type}')
    cd_smoother_posterior = cdlgssm_smoother(
        cd_params,
        cd_emissions,
        t_emissions,
        filter_hyperparams=kf_hyperparams,
        inputs=inputs,
        smoother_type=smoother_type
    )

    print(f"Comparing {smoother_type} smoothed posteriors...")
    compare_structs(d_smoother_posterior, cd_smoother_posterior, accept_failure=True)

    print(f"All Discrete to Continous-Discrete {smoother_type} smoothed posterior tests passed!")

print("WARNING: plotting filtering results for understanding impact of smoothing algorithm differences.")
import matplotlib.pyplot as plt
for n_state in jnp.arange(STATE_DIM):
    plt.figure()
    plt.plot(
        t_emissions,
        d_states[:, n_state],
        label="true discrete position",
        color="black"
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
        marker="x"
    )
    plt.plot(
        t_emissions,
        cd_smoother_posterior.smoothed_means[:, n_state],
        label="Post-SGD fit Continuous-Discrete smoothed state",
        color="green",
        marker="x"
    )
    plt.xlabel("time")
    plt.ylabel("x_{}".format(n_state))
    plt.grid()
    plt.legend()
    plt.title("Filtered and smoothed states")
    plt.show()


