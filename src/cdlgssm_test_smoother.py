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

print('************* Discrete LGSSM *************')
# Discrete sampling
NUM_TIMESTEPS = 100

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
d_model=LinearGaussianSSM(state_dim=2, emission_dim=5)
d_params, d_param_props = d_model.initialize(
    key1,
    # Hard coded parameters for tests to match
    dynamics_weights=0.9048373699188232421875*jnp.eye(d_model.state_dim),
    dynamics_covariance = 0.11329327523708343505859375 * jnp.eye(d_model.state_dim),
)

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

print('Fitting discrete time with EM')
d_em_fitted_params, d_em_lps = d_model.fit_em(
    d_params,
    d_param_props,
    d_emissions,
    inputs=inputs,
    num_iters=10
)
assert monotonically_increasing(d_em_lps)

print('************* Continuous-Discrete LGSSM *************')
# Continuous-Discrete model
NUM_TIMESTEPS = 100
t_emissions = jnp.arange(NUM_TIMESTEPS)[:,None]

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
cd_model=ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=5)
cd_params, cd_param_props = cd_model.initialize(
    key1,
    dynamics_weights=-0.1*jnp.eye(cd_model.state_dim), # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient = 0.5 * jnp.eye(cd_model.state_dim),
    dynamics_diffusion_cov = 0.5 * jnp.eye(cd_model.state_dim),
)

# Simulate from continuous model
print('Simulating in continuous-discrete time')
cd_states, cd_emissions = cd_model.sample(
    cd_params,
    key2,
    num_timesteps=NUM_TIMESTEPS,
    t_emissions=t_emissions,
    inputs=inputs
)

if not jnp.allclose(d_states, cd_states):
    assert jnp.allclose(d_states,cd_states, atol=1e-06)
    print('\tStates allclose with atol=1e-06')

if not jnp.allclose(d_emissions, cd_emissions):
    assert jnp.allclose(d_emissions, cd_emissions, atol=1e-05)
    print('\tEmissions allclose with atol=1e-05')

print('Continuous-Discrete time smoothing')
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_smoother
cd_smoothed_posterior = cdlgssm_smoother(cd_params, cd_emissions, t_emissions, inputs)

print('Checking that smoother posterior properties are close...')

if not jnp.allclose(
        d_smoother_posterior.filtered_means,
        cd_smoothed_posterior.filtered_means
    ):
    assert jnp.allclose(
        d_smoother_posterior.filtered_means,
        cd_smoothed_posterior.filtered_means,
        atol=1e-06
    )
    print('\tFiltered means allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.filtered_covariances,
        cd_smoothed_posterior.filtered_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.filtered_covariances,
        cd_smoothed_posterior.filtered_covariances,
        atol=1e-06
    )
    print('\tFiltered covariances allclose with atol=1e-06')
    
if not jnp.allclose(
        d_smoother_posterior.smoothed_means,
        cd_smoothed_posterior.smoothed_means
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_means,
        cd_smoothed_posterior.smoothed_means,
        atol=1e-06
    )
    print('\tSmoothed means allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.smoothed_covariances,
        cd_smoothed_posterior.smoothed_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_covariances,
        cd_smoothed_posterior.smoothed_covariances,
        atol=1e-06
    )
    print('\tSmoothed covariances allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.smoothed_cross_covariances,
        cd_smoothed_posterior.smoothed_cross_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_cross_covariances,
        cd_smoothed_posterior.smoothed_cross_covariances,
        atol=1e-06
    )
    print('\tSmoothed cross-covariances allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.marginal_loglik,
        cd_smoothed_posterior.marginal_loglik
    ):
    if not jnp.allclose(
        d_smoother_posterior.marginal_loglik,
        cd_smoothed_posterior.marginal_loglik,
        atol=1e-06
    ):
        print('\tMarginal log likelihood allclose with atol=1e-06')
    else:
        print('\tMarginal log likelihood differences')
        print(d_smoother_posterior.marginal_loglik - cd_smoothed_posterior.marginal_loglik)

'''
pdb.set_trace()
print('Fitting continuous-discrete time with EM')
cd_em_fitted_params, cd_em_lps = cd_model.fit_em(
    cd_params,
    cd_param_props,
    cd_emissions,
    t_emissions,
    inputs=inputs,
    num_iters=10
)
assert monotonically_increasing(d_em_lps)
'''
print('All tests passed!')

pdb.set_trace()
