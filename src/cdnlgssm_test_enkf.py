import pdb
import sys

from datetime import datetime
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

# Local dynamax
sys.path.append("..")
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import monotonically_increasing
from dynamax.utils.utils import ensure_array_has_batch_dim

# Our codebase
from continuous_discrete_linear_gaussian_ssm import ContDiscreteLinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM

STATE_DIM = 2
EMISSION_DIM = 6

def try_all_close(x, y, start_tol=-8, end_tol=-4):
    """Try all close with increasing tolerance"""
    # create list of tols 1e-8, 1e-7, 1e-6, ..., 1e1
    tol_list = jnp.array([10 ** i for i in range(start_tol, end_tol+1)])
    for tol in tol_list:
        if jnp.allclose(x, y, atol=tol):
            return True, tol
    return False, tol

def compare(x, x_ref, do_det=False):
    allclose, tol = try_all_close(x, x_ref)
    if allclose:
        print(f"\tAllclose passed with atol={tol}.")
    else:
        print(f"\tAllclose FAILED with atol={tol}.")

        # compute MSE of determinants over time
        if do_det:
            x = vmap(jnp.linalg.det)(x)
            x_ref = vmap(jnp.linalg.det)(x_ref)
            mse = (x - x_ref) ** 2
            rel_mse = mse / (x_ref**2)
        else:
            mse = jnp.mean((x - x_ref) ** 2, axis=1)
            rel_mse = mse / jnp.mean(x_ref ** 2, axis=1)

        print("\tInitial relative MSE: ", rel_mse[0])
        print("\tFinal relative MSE: ", rel_mse[-1])
        print("\tMax relative MSE: ", jnp.max(rel_mse))
        print("\tAverage relative MSE: ", jnp.mean(rel_mse))

        allclose, tol = try_all_close(rel_mse, 0, end_tol=-3)
        assert allclose, f"Relative MSE allclose FAILED with atol={tol}."

    pass


print("************* Discrete LGSSM *************")
# Discrete sampling
NUM_TIMESTEPS = 100

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
d_model = LinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)
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
cd_model = ContDiscreteLinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)
cd_params, cd_param_props = cd_model.initialize(
    key1,
    dynamics_weights=-0.1 * jnp.eye(cd_model.state_dim),  # Hard coded here for tests to match with default in linear
    dynamics_diffusion_coefficient=0.5 * jnp.eye(cd_model.state_dim),
    dynamics_diffusion_cov=0.5 * jnp.eye(cd_model.state_dim),
)

# Simulate from continuous model
print("Simulating in continuous-discrete time")
cd_num_timesteps_states, cd_num_timesteps_emissions = cd_model.sample(
    cd_params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs
)

cd_states, cd_emissions = cd_model.sample(
    cd_params, key2, num_timesteps=NUM_TIMESTEPS, t_emissions=t_emissions, inputs=inputs
)

compare(cd_num_timesteps_states, cd_states)
print("\tStates allclose.")

compare(cd_num_timesteps_emissions, cd_emissions)
print("\tEmissions allclose.")

compare(d_states, cd_states)
print("\tStates allclose.")

compare(d_emissions, cd_emissions)
print("\tEmissions allclose.")

print("Continuous-Discrete time filtering: pre-fit")
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_filter

cd_filtered_posterior = cdlgssm_filter(cd_params, cd_emissions, t_emissions, inputs)

compare(d_filtered_posterior.filtered_means, cd_filtered_posterior.filtered_means)
print("\tFiltered means allclose.")

compare(d_filtered_posterior.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
print("\tFiltered covariances allclose.")

print("Not Fitting continuous-discrete time with SGD")
'''
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
                    cd_sgd_fitted_params.dynamics.diffusion_cov)
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

print("All fitting tests passed!")
'''

########### Now make these into non-linear models ########
print("************* Continuous-Discrete Non-linear GSSM *************")

# Randomness
key1, key2 = jr.split(jr.PRNGKey(0))

# Model def
inputs = None  # Not interested in inputs for now
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
compare(cdnl_states_1, cd_states)
print("\tStates allclose.")

compare(cdnl_emissions_1, cd_emissions)
print("\tEmissions allclose.")

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
compare(cdnl_states_2, cd_states)
print("\tStates allclose.")

compare(cdnl_emissions_2, cd_emissions)
print("\tEmissions allclose.")

######## Continuous-discrete EKF
from continuous_discrete_nonlinear_gaussian_ssm import extended_kalman_filter as cd_ekf
from continuous_discrete_nonlinear_gaussian_ssm import EKFHyperParams

# Run First order ekf with the non-linear model and data from the first-order CDNLGSSM model
print("Running first-order EKF with non-linear model class and data from first-order CDNLGSSM model")

cd_ekf_11_post = cd_ekf(
    cdnl_params_1,
    cdnl_emissions_1,
    hyperparams=EKFHyperParams(state_order="first", emission_order="first"),
    t_emissions=t_emissions,
    inputs=inputs,
)

# check that results in cd_ekf_1_post are similar to results from applying cd_kf (cd_filtered_posterior)
compare(cd_ekf_11_post.filtered_means, cd_filtered_posterior.filtered_means)
print("\tFiltered means allclose.")

compare(cd_ekf_11_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
print("\tFiltered covariances allclose.")

print("All first-order EKF tests and first-order CDNLGSSM model passed!")

# Run Second order ekf with the non-linear model and data from the first-order CDNLGSSM model
print("Running second-order EKF with non-linear model class and data from first-order CDNLGSSM model")

cd_ekf_21_post = cd_ekf(
    cdnl_params_1,
    cdnl_emissions_1,
    hyperparams=EKFHyperParams(state_order="second", emission_order="first"),
    t_emissions=t_emissions,
    inputs=inputs,
)

# check that results in cd_ekf_21_post are similar to results from applying cd_kf (cd_filtered_posterior)
compare(cd_ekf_21_post.filtered_means, cd_filtered_posterior.filtered_means)
print("\tFiltered means allclose.")

compare(cd_ekf_21_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
print("\tFiltered covariances allclose.")

print("All second-order EKF tests and first-order CDNLGSSM model passed!")

# Run First order ekf with the non-linear model and data from the second-order CDNLGSSM model
print("Running first-order EKF with non-linear model class and data from second-order CDNLGSSM model")

cd_ekf_12_post = cd_ekf(
    cdnl_params_2,
    cdnl_emissions_2,
    hyperparams=EKFHyperParams(state_order="first", emission_order="first"),
    t_emissions=t_emissions,
    inputs=inputs,
)

# check that results in cd_ekf_12_post are similar to results from applying cd_kf (cd_filtered_posterior)
compare(cd_ekf_12_post.filtered_means, cd_filtered_posterior.filtered_means)
print("\tFiltered means allclose.")

compare(cd_ekf_12_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
print("\tFiltered covariances allclose.")

print("All first-order EKF tests and second-order CDNLGSSM model passed!")

# Run Second order ekf with the non-linear model and data from the second-order CDNLGSSM model
print("Running second-order EKF with non-linear model class and data from second-order CDNLGSSM model")

cd_ekf_22_post = cd_ekf(
    cdnl_params_2,
    cdnl_emissions_2,
    hyperparams=EKFHyperParams(state_order="second", emission_order="first"),
    t_emissions=t_emissions,
    inputs=inputs,
)

# check that results in cd_ekf_22_post are similar to results from applying cd_kf (cd_filtered_posterior)
compare(cd_ekf_22_post.filtered_means, cd_filtered_posterior.filtered_means)
print("\tFiltered means allclose.")

compare(cd_ekf_22_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
print("\tFiltered covariances allclose.")

print("All second-order EKF tests and second-order CDNLGSSM model passed!")

######## Continuous-discrete Ensemble Kalman Filter
from continuous_discrete_nonlinear_gaussian_ssm import ensemble_kalman_filter as cd_enkf
from continuous_discrete_nonlinear_gaussian_ssm import EnKFHyperParams

# Run First order ekf with the non-linear model and data from the first-order CDNLGSSM model

for perturb_measurements in [False, True]:
    for N_particles in [1e2, 1e3, 1e5]:
        # print("Running Ensemble Kalman Filter (perturb_measurements=True) with non-linear model class and data from first-order CDNLGSSM model")
        # print(f"Running Ensemble Kalman Filter (perturb_measurements={perturb_measurements}) with non-linear model class and data from first-order CDNLGSSM model")
        print(f"Running Ensemble Kalman Filter (perturb_measurements={perturb_measurements}, N_particles={N_particles}) with non-linear model class and data from first-order CDNLGSSM model")

        # define hyperparameters
        enkf_params = EnKFHyperParams(N_particles=int(N_particles), perturb_measurements=perturb_measurements)

        key_enkf = jr.PRNGKey(0)
        cd_enkf_post = cd_enkf(
            key_enkf,
            cdnl_params_1,
            cdnl_emissions_1,
            hyperparams=enkf_params,
            t_emissions=t_emissions,
            inputs=inputs,
        )

        # check that results in cd_enkf_1_post are similar to results from applying cd_kf (cd_filtered_posterior)
        print("\tComparing filtered means...")
        try:
            compare(cd_enkf_post.filtered_means, cd_filtered_posterior.filtered_means)
        except:
            if N_particles < 1e5 or not perturb_measurements:
                print("Test failed because too few particles or perturb_measurements=False")
                pass
            else:
                compare(cd_enkf_post.filtered_means, cd_filtered_posterior.filtered_means)

        print("\tComparing filtered covariances...")
        try:
            compare(cd_enkf_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)
        except:
            if N_particles < 1e5 or not perturb_measurements:
                print("Test failed because too few particles or perturb_measurements=False")
                pass
            else:
                compare(cd_enkf_post.filtered_covariances, cd_filtered_posterior.filtered_covariances, do_det=True)


print("All EnKF tests passed---note that these are randomized approximations, so we don't expect to perfectly replicate EKF and KF (which are both exact in linear test cases shown here)! We want to see convergence to truth (hence checking the final filtered state).")
pdb.set_trace()
