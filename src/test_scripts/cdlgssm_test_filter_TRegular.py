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
# Show that discrete KF == continuous-discrete KF for that linear system

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

print("\tSGD works!")

print("\tChecking SGD log-probabilities sequence...")
compare(cd_sgd_lps, d_sgd_lps)

print("WARNING: Please take the following comparisons with a grain of salt.")
print("Check that parameters are similar...")
compare_structs(d_sgd_fitted_params, cd_sgd_fitted_params, accept_failure=True)

print("Continuous-Discrete time filtering: post-fit")
cd_sgd_fitted_filtered_posterior = cdlgssm_filter(
    cd_sgd_fitted_params,
    cd_emissions,
    t_emissions,
    inputs
)

print("WARNING: Please take the following comparisons with a grain of salt.")
compare_structs(d_sgd_fitted_filtered_posterior, cd_sgd_fitted_filtered_posterior, accept_failure=True)

print("WARNING: plotting filtering results for understanding impact of SGD differences.")
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
        marker="x"
    )
    plt.xlabel("time")
    plt.ylabel("x_{}".format(n_state))
    plt.grid()
    plt.legend()
    plt.title("Filtered states after SGD optimization")
    plt.show()

print("All Discrete to Continous-Discrete model and filtering tests passed!")
