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
print('Simulating in continuous time')
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

