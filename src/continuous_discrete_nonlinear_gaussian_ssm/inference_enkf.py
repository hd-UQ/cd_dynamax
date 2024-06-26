import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, List

from jax.tree_util import tree_map
from dynamax.utils.utils import psd_solve

import jax.debug as jdb

# Dynamax shared code
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

# Our codebase
# CDNLGSSM param and function definition
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import *
# Diffrax based diff-eq solver
from utils.diffrax_utils import diffeqsolve

# Currently employing method from https://arxiv.org/abs/2205.02730 Neilsen et al. 2022

class EnKFHyperParams(NamedTuple):
    """Lightweight container for UKF hyperparameters.

    Default values taken from https://github.com/sbitzer/UKF-exposed
    """

    N_particles: float = 2000
    perturb_measurements: bool = True
    key: float = jr.PRNGKey(0)


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
# Using sum of _outers to compute covariance to avoid degenerate dimensionality cases

def _predict(
    key,
    x, # particles
    params: ParamsCDNLGSSM,  # All necessary CD dynamic params
    t0: Float,
    t1: Float,
    u,
):
    """Predict evolution of ensemble of particles through the nonlinear stochastic dynamics.

    Args:
        key: random key.
        x (N_particles, D_hid): particles at time t0.
        params: parameters of CD nonlinear dynamics, containing dynamics RHS function, coeff matrix and Brownian covariance matrix.
        t0: initial time-instant
        t1: final time-instant
        u (D_in,): inputs.

    Returns:
        x_pred (N_particles, D_hid): predicted particles

    """

    def drift(t, y, args):
        return params.dynamics.drift.f(y, u, t)

    def diffusion(t, y, args):
        Qc_t = params.dynamics.diffusion_cov.f(None, u, t)
        L_t = params.dynamics.diffusion_coefficient.f(None, u, t)
        Q_sqrt = jnp.linalg.cholesky(Qc_t)
        combined_diffusion = L_t @ Q_sqrt

        return combined_diffusion

    my_solve = lambda y0, key0: diffeqsolve(key=key0, drift=drift, diffusion=diffusion, t0=t0, t1=t1, y0=y0)
    key_array = jr.split(key, x.shape[0])
    sol = vmap(my_solve, in_axes=0)(x, key_array) # N_particles x 1 time x D_hid
    x_pred = sol[:, 0, :] # N_particles x D_hid

    return x_pred


def _condition_on(key, x, h, R, u, y, t, perturb_measurements=True):
    """Condition a Gaussian potential on a new observation

    Args:
        key: random key.
        x (N_particles, D_hid): prior particles.
        h (Callable): emission function.
        R (D_obs,D_obs): emssion covariance matrix
        u (D_in,): inputs.
        y (D_obs,): observation.black
        perturb_measurements: whether to perturb the measurements.
        t: time-instant of conditioning

    Returns:
        ll (float): log-likelihood of observation
        x_cond (N_particles, D_hid): filtered particles

    """
    n_particles, state_dim = x.shape

    # duplicate inputs for each particle
    u_s = jnp.array([u] * n_particles)

    # Propagate ensemble through emission function
    # The shape of y_ensemble is n_particles x Observation Dimensions
    y_ensemble = vmap(h, in_axes=(0, None, None))(x, u_s, t)

    ## These 2 computations should use deterministic observation ensemble, not perturbed
    # compute predicted mean of measurements
    y_pred_mean = jnp.mean(y_ensemble, axis=0)

    # compute predicted covariance of measurements as outer product of differences from mean
    # represents "HPH^T" in Kalman gain computation
    # y_pred_cov = jnp.cov(y_ensemble, rowvar=False)
    y_pred_cov = jnp.sum(_outer(y_ensemble - y_pred_mean, y_ensemble - y_pred_mean), axis=0) / (n_particles - 1)

    # Compute log-likelihood of observation
    ll = MVN(y_pred_mean, y_pred_cov).log_prob(y)

    ## Compute Kalman gain
    # make perturbed ensemble
    if perturb_measurements:
        # Add noise to the ensemble
        y_data_perturbed = jr.multivariate_normal(key=key, mean=y, cov=R, shape=(n_particles,))
    else:
        y_data_perturbed = y

    # compute cross_cov between x and y_data_perturbed
    # represents "PH^T" in Kalmna gain computation
    cross_cov = jnp.sum(_outer(x - jnp.mean(x, axis=0), y_ensemble - y_pred_mean), axis=0) / (n_particles - 1)
    S = y_pred_cov + R
    K = psd_solve(S, cross_cov.T).T

    # Updated the particles
    x_cond = x + (K @ (y_data_perturbed - y_ensemble).T).T

    return ll, x_cond


def ensemble_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    hyperparams: EnKFHyperParams = EnKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "filtered_means",
        "filtered_covariances",
        "predicted_means",
        "predicted_covariances",
    ],
    dt_final: Optional[float] = 1e-10,
) -> PosteriorGSSMFiltered:
    """Run a ensemble Kalman filter to produce the marginal likelihood and
    filtered state estimates.

    Args:
        key: random key.
        params: model parameters.
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        hyperparams: hyper-parameters.
        inputs: optional array of inputs.
        output_fields: list of fields to include in the output.
        dt_final: final time instant for the continuous-time solution (creates an unused prediction)

    Returns:
        filtered_posterior: posterior object.

    """
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[:, 0], t_emissions)
        t1 = tree_map(
            lambda x: jnp.concatenate(
                (t_emissions[1:, 0], jnp.array([t_emissions[-1, 0] + dt_final]))  # NB: t_{N+1} is simply t_{N}+dt_final
            ),
            t_emissions,
        )
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps)
        t1 = jnp.arange(1, num_timesteps + 1)

    t0_idx = jnp.arange(num_timesteps)

    # Only emission function
    h = params.emissions.emission_function.f
    # h = _process_fn(h, inputs)
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        ll, pred_x_ens = carry
        key, t0, t1, t0_idx = args

        # split key for (1) diffeqsolve, (2) perturbed measurements
        key_predict, key_filter = jr.split(key, 2)

        # Get parameters and inputs for time t0
        u = inputs[t0_idx]
        y = emissions[t0_idx]
        R = params.emissions.emission_cov.f(None, u, t0)

        # Condition on this emission
        log_likelihood, filtered_x_ens = _condition_on(
            key_filter, pred_x_ens, h, R, u, y, t0, hyperparams.perturb_measurements
        )

        # Update the log likelihood
        ll += log_likelihood

        # compute Gaussian statistics
        filtered_mean = jnp.mean(filtered_x_ens, axis=0)
        filtered_cov = jnp.sum(_outer(filtered_x_ens - filtered_mean, filtered_x_ens - filtered_mean), axis=0) / (
            hyperparams.N_particles - 1
        )
        # filtered_cov = jnp.cov(filtered_x_ens, rowvar=False)

        # Predict the next state, based on Ensemble prediction
        pred_x_ens = _predict(key_predict, filtered_x_ens, params, t0, t1, u)

        # compute Gaussian statistics
        pred_mean = jnp.mean(pred_x_ens, axis=0)
        pred_cov = jnp.sum(_outer(pred_x_ens - pred_mean, pred_x_ens - pred_mean), axis=0) / (
            hyperparams.N_particles - 1
        )
        # pred_cov = jnp.cov(pred_x_ens, rowvar=False)

        # Build carry and output states
        carry = (ll, pred_x_ens)
        outputs = {
            # TODO: if interested, save filtered/predicted particles here.
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": pred_mean,
            "predicted_covariances": pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}
        return carry, outputs

    # Build keys to be used to: (1) draw initial particles, (2) run each step of the filter
    keys = jr.split(hyperparams.key, num_timesteps + 1)
    key_init, key_times = keys[0], keys[1:]

    # Run the Ensemble Kalman Filter
    # draw initial particles from the prior
    x_ens_init = jr.multivariate_normal(
        key=key_init, mean=params.initial.mean, cov=params.initial.cov, shape=(hyperparams.N_particles,)
    )
    carry = (0.0, x_ens_init)

    # compute ll and outputs using a for loop instead of lax.scan to debug
    (ll, *_), outputs = lax.scan(_step, carry, (key_times, t0, t1, t0_idx))
    # for i in range(num_timesteps):
    # print(i)
    # carry, outputs = _step(carry, (key_times[i], t0[i], t1[i], t0_idx[i]))
    # ll, _, _ = carry

    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered
