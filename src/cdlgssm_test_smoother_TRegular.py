import jax.debug as jdb
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

# Equivalent initializations
cd_params, cd_param_props = cd_model.initialize(
    key_init,
    dynamics_weights=-0.1 * jnp.eye(cd_model.state_dim),  # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient=0.5*jnp.eye(cd_model.state_dim),
    dynamics_diffusion_cov=0.5*jnp.eye(cd_model.state_dim),
    dynamics_bias=jnp.zeros(d_model.state_dim),
    emission_bias=jnp.zeros(d_model.emission_dim),
)

cd_params, cd_param_props = cd_model.initialize(
    key_init,
    dynamics_weights=-0.1 * jnp.eye(cd_model.state_dim),  # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient=jnp.eye(cd_model.state_dim),
    dynamics_diffusion_cov=(0.5*0.5)*0.5*jnp.eye(cd_model.state_dim),
    dynamics_bias=jnp.zeros(d_model.state_dim),
    emission_bias=jnp.zeros(d_model.emission_dim),
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
        label="Post-SGD fit Discrete filtered state",
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
        label="Post-SGD fit Continuous-Discrete filtered state",
        color="green",
        marker="x"
    )
    plt.xlabel("time")
    plt.ylabel("x_{}".format(n_state))
    plt.grid()
    plt.legend()
    plt.title("Filtered and smoothed states")
    plt.show()
    
pdb.set_trace()
