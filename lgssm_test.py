import pdb
import sys

from datetime import datetime
import jax.numpy as jnp
import jax.random as jr

# Our dynamax
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import monotonically_increasing

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
print('Fitting discrete time with EM')
d_em_fitted_params, d_em_lps = model.fit_em(
    params,
    param_props,
    d_emissions,
    inputs=inputs,
    num_iters=10
)
assert monotonically_increasing(d_em_lps)
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
# Continuous Sampling
NUM_TIMESTEPS = 100
t_emissions = jnp.arange(NUM_TIMESTEPS)

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
model=LinearGaussianSSM(state_dim=2, emission_dim=5)
params, param_props = model.initialize(key1)

# Simulate from continuous model
print('Simulating in continuous time')
c_states, c_emissions = model.sample(
    params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    t_emissions=t_emissions,
    inputs=inputs
)

pdb.set_trace()
assert jnp.allclose(d_states, c_states)
assert jnp.allclose(d_emissions, c_emissions)

pdb.set_trace()
# TODO: check batched approach to c_emissions and how to deal with t_emissions
print('Fitting continuous time with EM')
c_em_fitted_params, c_em_lps = model.fit_em(
    params,
    param_props,
    c_emissions,
    t_emissions, 
    inputs=inputs,
    num_iters=10
)
assert monotonically_increasing(c_em_lps)

pdb.set_trace()
print('Fitting continuous time with SGD')
c_sgd_fitted_params, c_sgd_lps = model.fit_sgd(
    params,
    param_props,
    c_emissions,
    t_emissions,
    inputs=inputs,
    num_epochs=10
)

pdb.set_trace()
