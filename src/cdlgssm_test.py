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
model=LinearGaussianSSM(state_dim=2, emission_dim=5)
params, param_props = model.initialize(key1)

# Discrete sampling
NUM_TIMESTEPS = 100

# Simulate from discrete model
print('Simulating in discrete time')
d_states, d_emissions = model.sample(
    params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    inputs=inputs
)

pdb.set_trace()
print('Fitting discrete time with SGD')
d_sgd_fitted_params, d_sgd_lps = model.fit_sgd(
    params,
    param_props,
    d_emissions,
    inputs=inputs,
    num_epochs=10
)

pdb.set_trace()
# Continuous-Discrete model
NUM_TIMESTEPS = 100
t_emissions = jnp.arange(NUM_TIMESTEPS)[:,None]

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
cdmodel=ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=5)
cdparams, cdparam_props = cdmodel.initialize(key1)

# Simulate from continuous model
print('Simulating in continuous-discrete time')
cd_states, cd_emissions = cdmodel.sample(
    cdparams,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    t_emissions=t_emissions,
    inputs=inputs
)

pdb.set_trace()
assert jnp.allclose(d_states, cd_states)
assert jnp.allclose(d_emissions, cd_emissions)

pdb.set_trace()
print('Fitting continuous-discrete time with SGD')
cd_sgd_fitted_params, cd_sgd_lps = cdmodel.fit_sgd(
    cdparams,
    cdparam_props,
    cd_emissions,
    t_emissions,
    inputs=inputs,
    num_epochs=10
)

pdb.set_trace()

print('Checking that discrete and continuous-discrete models computed similar log probabilities')
assert jnp.allclose(d_sgd_lps, cd_sgd_lps)

# print('Checking that discrete and continuous-discrete models computed similar parameters')
print('Checking that dynamics weights are close...')
assert jnp.allclose(d_sgd_fitted_params.dynamics.weights,
                    cd_sgd_fitted_params.dynamics.weights)

print('Checking that dynamics biases are close...')
assert jnp.allclose(d_sgd_fitted_params.dynamics.bias,
                     cd_sgd_fitted_params.dynamics.bias)

print('Checking that emission weights are close...')
assert jnp.allclose(d_sgd_fitted_params.emission.weights,
                    cd_sgd_fitted_params.emission.weights)

print('Checking that emission biases are close...')
assert jnp.allclose(d_sgd_fitted_params.emission.bias,
                        cd_sgd_fitted_params.emission.bias)

print('Checking that initial mean is close...')
assert jnp.allclose(d_sgd_fitted_params.initial_mean,
                    cd_sgd_fitted_params.initial_mean)

print('Checking that initial covariance is close...')
assert jnp.allclose(d_sgd_fitted_params.initial_covariance,
                    cd_sgd_fitted_params.initial_covariance)

print('Checking that dynamics covariance is close...')
assert jnp.allclose(d_sgd_fitted_params.dynamics_covariance,
                    cd_sgd_fitted_params.dynamics_covariance)

print('Checking that emission covariance is close...')
assert jnp.allclose(d_sgd_fitted_params.emission_covariance,
                    cd_sgd_fitted_params.emission_covariance)

print('All tests passed!')



pdb.set_trace()
