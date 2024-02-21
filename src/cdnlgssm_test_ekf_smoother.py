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
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM

STATE_DIM = 2
EMISSION_DIM = 6

print('************* Discrete LGSSM *************')
# Discrete sampling
NUM_TIMESTEPS = 100

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
d_model=LinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)
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

'''
print('Fitting discrete time with EM')
d_em_fitted_params, d_em_lps = d_model.fit_em(
    d_params,
    d_param_props,
    d_emissions,
    inputs=inputs,
    num_iters=10
)
assert monotonically_increasing(d_em_lps)
'''

print('************* Continuous-Discrete LGSSM *************')
# Continuous-Discrete model
t_emissions = jnp.arange(NUM_TIMESTEPS)[:,None]

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
cd_model=ContDiscreteLinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)
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

print('Continuous-Discrete time smoothing: type 1')
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_smoother
cd_smoothed_posterior_1 = cdlgssm_smoother(
    cd_params,
    cd_emissions,
    t_emissions,
    inputs,
    smoother_type='cd_smoother_1'
)

print('Checking that smoother type 1 posterior properties are close...')

if not jnp.allclose(
        d_smoother_posterior.filtered_means,
        cd_smoothed_posterior_1.filtered_means
    ):
    assert jnp.allclose(
        d_smoother_posterior.filtered_means,
        cd_smoothed_posterior_1.filtered_means,
        atol=1e-06
    )
    print('\tFiltered means allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.filtered_covariances,
        cd_smoothed_posterior_1.filtered_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.filtered_covariances,
        cd_smoothed_posterior_1.filtered_covariances,
        atol=1e-06
    )
    print('\tFiltered covariances allclose with atol=1e-06')
    
if not jnp.allclose(
        d_smoother_posterior.smoothed_means,
        cd_smoothed_posterior_1.smoothed_means
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_means,
        cd_smoothed_posterior_1.smoothed_means,
        atol=1e-06
    )
    print('\tSmoothed means allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.smoothed_covariances,
        cd_smoothed_posterior_1.smoothed_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_covariances,
        cd_smoothed_posterior_1.smoothed_covariances,
        atol=1e-06
    )
    print('\tSmoothed covariances allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.smoothed_cross_covariances,
        cd_smoothed_posterior_1.smoothed_cross_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_cross_covariances,
        cd_smoothed_posterior_1.smoothed_cross_covariances,
        atol=1e-06
    )
    print('\tSmoothed cross-covariances allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.marginal_loglik,
        cd_smoothed_posterior_1.marginal_loglik
    ):
    if not jnp.allclose(
        d_smoother_posterior.marginal_loglik,
        cd_smoothed_posterior_1.marginal_loglik,
        atol=1e-06
    ):
        print('\tMarginal log likelihood allclose with atol=1e-06')
    else:
        print('\tMarginal log likelihood differences')
        print(d_smoother_posterior.marginal_loglik - cd_smoothed_posterior_1.marginal_loglik)

print('All KF smoother type 1 tests passed!')

print('Continuous-Discrete time smoothing: type 2')
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_smoother
cd_smoothed_posterior_2 = cdlgssm_smoother(
    cd_params,
    cd_emissions,
    t_emissions,
    inputs,
    smoother_type='cd_smoother_2'
)

print('Checking that smoother type 2 posterior properties are close...')

if not jnp.allclose(
        cd_smoothed_posterior_1.filtered_means,
        cd_smoothed_posterior_2.filtered_means
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.filtered_means,
        cd_smoothed_posterior_2.filtered_means,
        atol=1e-06
    )
    print('\tFiltered means allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_1.filtered_covariances,
        cd_smoothed_posterior_2.filtered_covariances
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.filtered_covariances,
        cd_smoothed_posterior_2.filtered_covariances,
        atol=1e-06
    )
    print('\tFiltered covariances allclose with atol=1e-06')
    
if not jnp.allclose(
        cd_smoothed_posterior_1.smoothed_means,
        cd_smoothed_posterior_2.smoothed_means
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.smoothed_means,
        cd_smoothed_posterior_2.smoothed_means,
        atol=1e-06
    )
    print('\tSmoothed means allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_1.smoothed_covariances,
        cd_smoothed_posterior_2.smoothed_covariances
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.smoothed_covariances,
        cd_smoothed_posterior_2.smoothed_covariances,
        atol=1e-06
    )
    print('\tSmoothed covariances allclose with atol=1e-06')

print('\tSmoothed cross-covariances for type 2 need to be computed')

if not jnp.allclose(
        cd_smoothed_posterior_1.marginal_loglik,
        cd_smoothed_posterior_2.marginal_loglik
    ):
    if not jnp.allclose(
        cd_smoothed_posterior_1.marginal_loglik,
        cd_smoothed_posterior_2.marginal_loglik,
        atol=1e-06
    ):
        print('\tMarginal log likelihood allclose with atol=1e-06')
    else:
        print('\tMarginal log likelihood differences')
        print(cd_smoothed_posterior_1.marginal_loglik - cd_smoothed_posterior_2.marginal_loglik)

print('All KF smoother type 2 tests passed!')

pdb.set_trace()

########### Now make these into non-linear models ########
print("************* Continuous-Discrete Non-linear GSSM *************")
# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs=None # Not interested in inputs for now
cdnl_model = ContDiscreteNonlinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)

# TODO: check that these need input as second argument; also check about including bias terms.
dynamics_drift_function = lambda z, u: cd_params.dynamics.weights @ z
emission_function = lambda z: cd_params.emissions.weights @ z

# Initialize with first order SDE approximation
cdnl_params_1, cdnl_param_props_1 = cdnl_model.initialize(
    key1,
    dynamics_drift_function=dynamics_drift_function,
    dynamics_diffusion_coefficient=cd_params.dynamics.diffusion_coefficient,
    dynamics_diffusion_cov=cd_params.dynamics.diffusion_cov,
    dynamics_approx_type="first",
    emission_function=emission_function,
)

# Simulate from continuous model
print("Simulating first order CDNLGSSM in continuous-discrete time")
cdnl_states_1, cdnl_emissions_1 = cdnl_model.sample(
    cdnl_params_1, key2, t_emissions=t_emissions, num_timesteps=NUM_TIMESTEPS, inputs=inputs
)

# check that these are similar to samples from the linear model
if not jnp.allclose(cdnl_states_1, cd_states):
    assert jnp.allclose(cdnl_states_1, cd_states, atol=1e-05)
    print("\tStates allclose with atol=1e-05")

if not jnp.allclose(cdnl_emissions_1, cd_emissions):
    assert jnp.allclose(cdnl_emissions_1, cd_emissions, atol=1e-05)
    print("\tEmissions allclose with atol=1e-05")

# Initialize with second order SDE approximation
cdnl_params_2, cdnl_param_props_2 = cdnl_model.initialize(
    key1,
    dynamics_drift_function=dynamics_drift_function,
    dynamics_diffusion_coefficient=cd_params.dynamics.diffusion_coefficient,
    dynamics_diffusion_cov=cd_params.dynamics.diffusion_cov,
    dynamics_approx_type="second",
    emission_function=emission_function,
)

# Simulate from continuous model
print("Simulating second order CDNLGSSM in continuous-discrete time")
cdnl_states_2, cdnl_emissions_2 = cdnl_model.sample(
    cdnl_params_2, key2, t_emissions=t_emissions, num_timesteps=NUM_TIMESTEPS, inputs=inputs
)

# check that these are similar to samples from the linear model
if not jnp.allclose(cdnl_states_2, cd_states):
    assert jnp.allclose(cdnl_states_2, cd_states, atol=1e-05)
    print("\tStates allclose with atol=1e-05")

if not jnp.allclose(cdnl_emissions_2, cd_emissions):
    assert jnp.allclose(cdnl_emissions_2, cd_emissions, atol=1e-05)
    print("\tEmissions allclose with atol=1e-05")

######## Continuous-discrete Non linearEKF
from continuous_discrete_nonlinear_gaussian_ssm import extended_kalman_smoother as cdnlgssm_ekf_smoother
from continuous_discrete_nonlinear_gaussian_ssm import EKFHyperParams

print('Continuous-Discrete time EKF smoothing')
cdnlgssm_smoothed_posterior_1 = cdnlgssm_ekf_smoother(
    cdnl_params_1,
    cdnl_emissions_1,
    t_emissions=t_emissions,
    hyperparams=EKFHyperParams(state_order="first", emission_order="first", smooth_order='first'),
    inputs=inputs,
)

print('Checking that CD-Nonlinear smoother posterior properties are close to Discrete KF smoother...')

if not jnp.allclose(
        d_smoother_posterior.filtered_means,
        cdnlgssm_smoothed_posterior_1.filtered_means
    ):
    assert jnp.allclose(
        d_smoother_posterior.filtered_means,
        cdnlgssm_smoothed_posterior_1.filtered_means,
        atol=1e-05
    )
    print('\tFiltered means allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.filtered_covariances,
        cdnlgssm_smoothed_posterior_1.filtered_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.filtered_covariances,
        cdnlgssm_smoothed_posterior_1.filtered_covariances,
        atol=1e-05
    )
    print('\tFiltered covariances allclose with atol=1e-06')
    
if not jnp.allclose(
        d_smoother_posterior.smoothed_means,
        cdnlgssm_smoothed_posterior_1.smoothed_means
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_means,
        cdnlgssm_smoothed_posterior_1.smoothed_means,
        atol=1e-05
    )
    print('\tSmoothed means allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.smoothed_covariances,
        cdnlgssm_smoothed_posterior_1.smoothed_covariances
    ):
    assert jnp.allclose(
        d_smoother_posterior.smoothed_covariances,
        cdnlgssm_smoothed_posterior_1.smoothed_covariances,
        atol=1e-06
    )
    print('\tSmoothed covariances allclose with atol=1e-06')

if not jnp.allclose(
        d_smoother_posterior.marginal_loglik,
        cdnlgssm_smoothed_posterior_1.marginal_loglik
    ):
    if not jnp.allclose(
        d_smoother_posterior.marginal_loglik,
        cdnlgssm_smoothed_posterior_1.marginal_loglik,
        atol=1e-06
    ):
        print('\tMarginal log likelihood allclose with atol=1e-06')
    else:
        print('\tMarginal log likelihood differences')
        print(d_smoother_posterior.marginal_loglik - cd_smoothed_posterior.marginal_loglik)

print('All Discrete Linear to Continuous-Discrete NonLinear smoothing tests passed!')

print('Checking that CD-Nonlinear smoother posterior properties are close to CD-KF smoother type 1...')

if not jnp.allclose(
        cd_smoothed_posterior_1.filtered_means,
        cdnlgssm_smoothed_posterior_1.filtered_means
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.filtered_means,
        cdnlgssm_smoothed_posterior_1.filtered_means,
        atol=1e-05
    )
    print('\tFiltered means allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_1.filtered_covariances,
        cdnlgssm_smoothed_posterior_1.filtered_covariances
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.filtered_covariances,
        cdnlgssm_smoothed_posterior_1.filtered_covariances,
        atol=1e-05
    )
    print('\tFiltered covariances allclose with atol=1e-06')
    
if not jnp.allclose(
        cd_smoothed_posterior_1.smoothed_means,
        cdnlgssm_smoothed_posterior_1.smoothed_means
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.smoothed_means,
        cdnlgssm_smoothed_posterior_1.smoothed_means,
        atol=1e-05
    )
    print('\tSmoothed means allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_1.smoothed_covariances,
        cdnlgssm_smoothed_posterior_1.smoothed_covariances
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_1.smoothed_covariances,
        cdnlgssm_smoothed_posterior_1.smoothed_covariances,
        atol=1e-06
    )
    print('\tSmoothed covariances allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_1.marginal_loglik,
        cdnlgssm_smoothed_posterior_1.marginal_loglik
    ):
    if not jnp.allclose(
        cd_smoothed_posterior_1.marginal_loglik,
        cdnlgssm_smoothed_posterior_1.marginal_loglik,
        atol=1e-06
    ):
        print('\tMarginal log likelihood allclose with atol=1e-06')
    else:
        print('\tMarginal log likelihood differences')
        print(cd_smoothed_posterior_1.marginal_loglik - cd_smoothed_posterior.marginal_loglik)

print('All CD-Linear type 1 to Continuous-Discrete NonLinear smoothing tests passed!')

print('Checking that CD-Nonlinear smoother posterior properties are close to CD-KF smoother type II...')

if not jnp.allclose(
        cd_smoothed_posterior_2.filtered_means,
        cdnlgssm_smoothed_posterior_1.filtered_means
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_2.filtered_means,
        cdnlgssm_smoothed_posterior_1.filtered_means,
        atol=1e-05
    )
    print('\tFiltered means allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_2.filtered_covariances,
        cdnlgssm_smoothed_posterior_1.filtered_covariances
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_2.filtered_covariances,
        cdnlgssm_smoothed_posterior_1.filtered_covariances,
        atol=1e-05
    )
    print('\tFiltered covariances allclose with atol=1e-06')
    
if not jnp.allclose(
        cd_smoothed_posterior_2.smoothed_means,
        cdnlgssm_smoothed_posterior_1.smoothed_means
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_2.smoothed_means,
        cdnlgssm_smoothed_posterior_1.smoothed_means,
        atol=1e-05
    )
    print('\tSmoothed means allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_2.smoothed_covariances,
        cdnlgssm_smoothed_posterior_1.smoothed_covariances
    ):
    assert jnp.allclose(
        cd_smoothed_posterior_2.smoothed_covariances,
        cdnlgssm_smoothed_posterior_1.smoothed_covariances,
        atol=1e-06
    )
    print('\tSmoothed covariances allclose with atol=1e-06')

if not jnp.allclose(
        cd_smoothed_posterior_2.marginal_loglik,
        cdnlgssm_smoothed_posterior_1.marginal_loglik
    ):
    if not jnp.allclose(
        cd_smoothed_posterior_2.marginal_loglik,
        cdnlgssm_smoothed_posterior_1.marginal_loglik,
        atol=1e-06
    ):
        print('\tMarginal log likelihood allclose with atol=1e-06')
    else:
        print('\tMarginal log likelihood differences')
        print(cd_smoothed_posterior_2.marginal_loglik - cd_smoothed_posterior.marginal_loglik)

print('All CD-Linear type II to Continuous-Discrete NonLinear smoothing tests passed!')
