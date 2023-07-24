import pdb
import sys

from datetime import datetime
import jax.numpy as jnp
import jax.random as jr

# Local dynamax 
sys.path.append('..')
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import monotonically_increasing

# Our codebase
from continuous_discrete_linear_gaussian_ssm import ContDiscreteLinearGaussianSSM

# Discrete sampling
NUM_TIMESTEPS = 100

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
d_model=LinearGaussianSSM(state_dim=2, emission_dim=5)
d_params, d_param_props = d_model.initialize(key1)

# Discrete sampling
NUM_TIMESTEPS = 100

# Simulate from discrete model
print('Simulating in discrete time')
d_states, d_emissions = d_model.sample(
    d_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs
)

print('Discrete time smoothing')
from dynamax.linear_gaussian_ssm.inference import lgssm_smoother
d_smoother_posterior = lgssm_smoother(d_params, d_emissions, inputs)

# Continuous-Discrete model
NUM_TIMESTEPS = 100
t_emissions = jnp.arange(NUM_TIMESTEPS)[:,None]

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
cd_model=ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=5)
cd_params, cd_param_props = cd_model.initialize(key1)

# Simulate from continuous model
print('Simulating in continuous-discrete time')
cd_states, cd_emissions = cd_model.sample(
    cd_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    t_emissions=t_emissions,
    inputs=inputs
)
assert jnp.allclose(d_states, cd_states)
assert jnp.allclose(d_emissions, cd_emissions)

print('Continuous-Discrete time smoothing')
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_smoother
cd_smoothed_posterior = cdlgssm_smoother(cd_params, cd_emissions, t_emissions, inputs)

print('Checking that smoother posterior properties are close...')
assert jnp.allclose(cd_smoothed_posterior.marginal_loglik,
                    d_smoother_posterior.marginal_loglik)

assert jnp.allclose(cd_smoothed_posterior.filtered_means,
                    d_smoother_posterior.filtered_means)

assert jnp.allclose(cd_smoothed_posterior.filtered_covariances,
                    d_smoother_posterior.filtered_covariances)

assert jnp.allclose(cd_smoothed_posterior.smoothed_means,
                    d_smoother_posterior.smoothed_means)

assert jnp.allclose(cd_smoothed_posterior.smoothed_covariances,
                    d_smoother_posterior.smoothed_covariances)

assert jnp.allclose(cd_smoothed_posterior.smoothed_cross_covariances,
                    d_smoother_posterior.smoothed_cross_covariances)

print('All tests passed!')

pdb.set_trace()
