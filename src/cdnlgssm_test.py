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
from continuous_discrete_linear_gaussian_ssm import ContDiscreteLinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM

print("************* Discrete LGSSM *************")
# Discrete sampling
NUM_TIMESTEPS = 100

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
d_model = LinearGaussianSSM(state_dim=2, emission_dim=5)
d_params, d_param_props = d_model.initialize(
    key1,
    # Hard coded parameters for tests to match
    dynamics_weights=0.9048373699188232421875 * jnp.eye(d_model.state_dim),
    dynamics_covariance=0.11329327523708343505859375 * jnp.eye(d_model.state_dim),
)

# Simulate from discrete model
print("Simulating in discrete time")
d_states, d_emissions = d_model.sample(d_params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs)

print("Discrete time filtering: pre-fit")
from dynamax.linear_gaussian_ssm.inference import lgssm_filter

d_filtered_posterior = lgssm_filter(d_params, d_emissions, inputs)

print("Fitting discrete time with SGD")
d_sgd_fitted_params, d_sgd_lps = d_model.fit_sgd(d_params, d_param_props, d_emissions, inputs=inputs, num_epochs=10)

print("Discrete time filtering: post-fit")
d_sgd_fitted_filtered_posterior = lgssm_filter(d_sgd_fitted_params, d_emissions, inputs)

print("************* Continuous-Discrete LGSSM *************")
# Continuous-Discrete model
NUM_TIMESTEPS = 100
t_emissions = jnp.arange(NUM_TIMESTEPS)[:, None]

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
cd_model = ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=5)
cd_params, cd_param_props = cd_model.initialize(
    key1,
    dynamics_weights=-0.1 * jnp.eye(cd_model.state_dim),  # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient=0.5 * jnp.eye(cd_model.state_dim),
    dynamics_diffusion_covariance=0.5 * jnp.eye(cd_model.state_dim),
)

# Simulate from continuous model
print("Simulating in continuous-discrete time")
cd_num_timesteps_states, cd_num_timesteps_emissions = cd_model.sample(
    cd_params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs
)

cd_states, cd_emissions = cd_model.sample(
    cd_params, key2, num_timesteps=NUM_TIMESTEPS, t_emissions=t_emissions, inputs=inputs
)

if not jnp.allclose(cd_num_timesteps_states, cd_states):
    assert jnp.allclose(cd_num_timesteps_states, cd_states, atol=1e-06)
    print("\tStates allclose with atol=1e-06")

if not jnp.allclose(cd_num_timesteps_emissions, cd_emissions):
    assert jnp.allclose(cd_num_timesteps_emissions, cd_emissions, atol=1e-05)
    print("\tEmissions allclose with atol=1e-05")

if not jnp.allclose(d_states, cd_states):
    assert jnp.allclose(d_states, cd_states, atol=1e-06)
    print("\tStates allclose with atol=1e-06")

if not jnp.allclose(d_emissions, cd_emissions):
    assert jnp.allclose(d_emissions, cd_emissions, atol=1e-05)
    print("\tEmissions allclose with atol=1e-05")

print("Continuous-Discrete time filtering: pre-fit")
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_filter

cd_filtered_posterior = cdlgssm_filter(cd_params, cd_emissions, t_emissions, inputs)

if not jnp.allclose(d_filtered_posterior.filtered_means, cd_filtered_posterior.filtered_means):
    assert jnp.allclose(d_filtered_posterior.filtered_means, cd_filtered_posterior.filtered_means, atol=1e-06)
    print("\tFiltered means allclose with atol=1e-06")

if not jnp.allclose(d_filtered_posterior.filtered_covariances, cd_filtered_posterior.filtered_covariances):
    assert jnp.allclose(
        d_filtered_posterior.filtered_covariances, cd_filtered_posterior.filtered_covariances, atol=1e-06
    )
    print("\tFiltered covariances allclose with atol=1e-06")

print("Fitting continuous-discrete time with SGD")
cd_sgd_fitted_params, cd_sgd_lps = cd_model.fit_sgd(
    cd_params, cd_param_props, cd_emissions, t_emissions, inputs=inputs, num_epochs=10
)

print("Continuous-Discrete time filtering: post-fit")
cd_sgd_fitted_filtered_posterior = cdlgssm_filter(cd_sgd_fitted_params, cd_emissions, t_emissions, inputs)

print("Checking that discrete and continuous-discrete models trained via SGD are similar")
if not jnp.allclose(d_sgd_fitted_filtered_posterior.filtered_means, cd_sgd_fitted_filtered_posterior.filtered_means):
    if jnp.allclose(
        d_sgd_fitted_filtered_posterior.filtered_means, cd_sgd_fitted_filtered_posterior.filtered_means, atol=1e-06
    ):
        print("\tFiltered means allclose with atol=1e-06")
    else:
        print("\tFiltered mean differences")
        print(d_sgd_fitted_filtered_posterior.filtered_means - cd_sgd_fitted_filtered_posterior.filtered_means)

if not jnp.allclose(
    d_sgd_fitted_filtered_posterior.filtered_covariances, cd_sgd_fitted_filtered_posterior.filtered_covariances
):
    if jnp.allclose(
        d_sgd_fitted_filtered_posterior.filtered_covariances,
        cd_sgd_fitted_filtered_posterior.filtered_covariances,
        atol=1e-06,
    ):
        print("\tFiltered covariances allclose with atol=1e-06")
    else:
        print("\tFiltered covariance differences")
        print(
            d_sgd_fitted_filtered_posterior.filtered_covariances - cd_sgd_fitted_filtered_posterior.filtered_covariances
        )

print("Checking that discrete and continuous-discrete models computed similar log probabilities")
if not jnp.allclose(d_sgd_lps, cd_sgd_lps):
    print("\tFiltered covariance differences")
    print(d_sgd_lps - cd_sgd_lps)

print("Checking that discrete and continuous-discrete models computed similar parameters")
print("Checking that initial mean is close...")
if not jnp.allclose(d_sgd_fitted_params.initial.mean, cd_sgd_fitted_params.initial.mean):
    if jnp.allclose(d_sgd_fitted_params.initial.mean, cd_sgd_fitted_params.initial.mean, atol=1e-06):
        print("\tInitial mean allclose with atol=1e-06")
    else:
        print("\tInitial mean differences")
        print(d_sgd_fitted_params.initial.mean - cd_sgd_fitted_params.initial.mean)

print("Checking that initial covariance is close...")
if not jnp.allclose(d_sgd_fitted_params.initial.cov, cd_sgd_fitted_params.initial.cov):
    if jnp.allclose(d_sgd_fitted_params.initial.cov, cd_sgd_fitted_params.initial.cov, atol=1e-06):
        print("\tInitial covariance allclose with atol=1e-06")
    else:
        print("\tInitial covariance differences")
        print(d_sgd_fitted_params.initial.cov - cd_sgd_fitted_params.initial.cov)

"""
print('Checking that dynamics weights are close...')
# Do these make sense?
assert jnp.allclose(d_sgd_fitted_params.dynamics.weights,
                    cd_sgd_fitted_params.dynamics.weights)

print('Checking that dynamics biases are close...')
assert jnp.allclose(d_sgd_fitted_params.dynamics.bias,
                     cd_sgd_fitted_params.dynamics.bias)
print('Checking that dynamics covariance is close...')
assert jnp.allclose(d_sgd_fitted_params.dynamics.cov,
                    cd_sgd_fitted_params.dynamics.diff_cov)
"""

print("Checking that emission weights are close...")
if not jnp.allclose(d_sgd_fitted_params.emissions.weights, cd_sgd_fitted_params.emissions.weights):
    if jnp.allclose(d_sgd_fitted_params.emissions.weights, cd_sgd_fitted_params.emissions.weights, atol=1e-06):
        print("\tEmission weights allclose with atol=1e-06")
    else:
        print("\tEmission weights differences")
        print(d_sgd_fitted_params.emissions.weights - cd_sgd_fitted_params.emissions.weights)

print("Checking that emission biases are close...")
if not jnp.allclose(d_sgd_fitted_params.emissions.bias, cd_sgd_fitted_params.emissions.bias):
    if jnp.allclose(d_sgd_fitted_params.emissions.bias, cd_sgd_fitted_params.emissions.bias, atol=1e-06):
        print("\tEmission biases allclose with atol=1e-06")
    else:
        print("\tEmission bias differences")
        print(d_sgd_fitted_params.emissions.bias - cd_sgd_fitted_params.emissions.bias)

print("Checking that emission covariance is close...")
if not jnp.allclose(d_sgd_fitted_params.emissions.cov, cd_sgd_fitted_params.emissions.cov):
    if jnp.allclose(d_sgd_fitted_params.emissions.cov, cd_sgd_fitted_params.emissions.cov, atol=1e-06):
        print("\tEmission covariance allclose with atol=1e-06")
    else:
        print("\tEmission covariance differences")
        print(d_sgd_fitted_params.emissions.cov - cd_sgd_fitted_params.emissions.cov)

print("All tests passed!")

pdb.set_trace()


########### Now make these into non-linear models ########
print("************* Continuous-Discrete Non-linear GSSM *************")

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
cdnl_model = ContDiscreteNonlinearGaussianSSM(state_dim=2, emission_dim=5)

# TODO: check that these need input as second argument; also check about including bias terms.
dynamics_function = lambda z,u: cd_params.dynamics.weights @ z
emission_function = lambda z: cd_params.emissions.weights @ z
pdb.set_trace()
# Initialize
cdnl_params, cdnl_param_props = cdnl_model.initialize(
    key1,
    dynamics_function=dynamics_function,
    dynamics_diffusion_coefficient=cd_params.dynamics.diff_coeff,
    dynamics_diffusion_covariance=cd_params.dynamics.diff_cov,
    dynamics_covariance_order = 'first',
    emission_function=emission_function,
)

# Simulate from continuous model
print("Simulating in continuous-discrete time")
cdnl_states, cdnl_emissions = cdnl_model.sample(cdnl_params, key2, 
                                                t_emissions=t_emissions,
                                                num_timesteps=NUM_TIMESTEPS,
                                                inputs=inputs)

# check that these are similar to samples from the linear model
if not jnp.allclose(cdnl_states, cd_states):
    assert jnp.allclose(cdnl_states, cd_states, atol=1e-06)
    print("\tStates allclose with atol=1e-06")

if not jnp.allclose(cdnl_emissions, cd_emissions):
    assert jnp.allclose(cdnl_emissions, cd_emissions, atol=1e-05)
    print("\tEmissions allclose with atol=1e-05")


# Now, run ukf with the non-linear model and data from the linear model
print("Running UKF with non-linear model and data from linear model")
from continuous_discrete_nonlinear_gaussian_ssm import unscented_kalman_filter as cd_ukf

cd_ukf_post = cd_ukf(cdnl_params, cd_emissions, t_emissions, inputs)

# check that results in cd_ukf_post are similar to results from applying cd_kf (cd_filtered_posterior)
if not jnp.allclose(cd_ukf_post.filtered_means, cd_filtered_posterior.filtered_means):
    assert jnp.allclose(cd_ukf_post.filtered_means, cd_filtered_posterior.filtered_means, atol=1e-06)
    print("\tFiltered means allclose with atol=1e-06")

if not jnp.allclose(cd_ukf_post.filtered_covariances, cd_filtered_posterior.filtered_covariances):
    assert jnp.allclose(cd_ukf_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, atol=1e-06)
    print("\tFiltered covariances allclose with atol=1e-06")

print("All tests passed!")
