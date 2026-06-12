# JAX imports
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import vmap
from jax.tree_util import tree_map

# Typing annotations
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, List, Tuple

# Distributions, compatible with JAX, from TensorFlow Probability
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

# Dynamax shared code
from cd_dynamax.dynamax.types import PRNGKey
from cd_dynamax.dynamax.utils.utils import psd_solve

# To avoid unnecessary redefinitions of code,
# We import those posterior filtering and smoothing equivalent classes that can be reused from dynamax
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    PosteriorGSSMFiltered,
)

# Our codebase
# CDLGSSM forecasting definition is reused
from ..continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import GSSMForecast

# CDNLGSSM specific param and function definitions
from ..continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import ParamsCDNLGSSM

# Diffrax based diff-eq solver
from ..utils.diffrax_utils import diffeqsolve

# Debugging utilities
from ..utils.debug_utils import psd, lax_scan

tfb = tfp.bijectors

DEBUG = False

#### Helper functions
# Helper functions --- from dynamax
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x

#### CD-NLGSSM filtering: Ensemble Kalman Filter (EnKF)


# Default EnKF filtering hyperparameters, as class
class EnKFHyperParams(NamedTuple):
    """Lightweight container for EnKF hyperparameters."""

    dt_final: float = (
        1e-4  # Small dt_final for predicted mean and covariance at the end of sequence
    )
    N_particles: float = 25
    perturb_measurements: bool = True
    cov_rescaling: float = 1.0
    inflation_delta: float = 0.0  # Inflation factor for ensemble spread
    diffeqsolve_settings: dict = {}
    state_order: str = "first"
    dt_average: float = 0.1  # Average timestep for discrete state order, if applicable


## CD-NLGSSM filtering key functions: predict and condition_on
# Currently employing method from https://arxiv.org/abs/2205.02730 Neilsen et al. 2022


# Predict next mean and covariance, under EnKF approximations
def _predict(
    key,
    x,  # particles
    params: ParamsCDNLGSSM,
    t0: Float,
    t1: Float,
    u,
    filter_hyperparams,
):
    """Predict evolution of ensemble of particles through the nonlinear stochastic dynamics.

    Args:
        key: random key.
        x (N_particles, D_hid): particles at time t0.
        params: CD-NLGSSM parameters, containing
            - dynamics RHS drift function
            - diffusion coefficient matrix L
            - Brownian covariance matrix Q
        t0: initial time-instant
        t1: final time-instant
        u (D_in,): inputs.
        filter_hyperparams: EnKF hyper-parameters

    Returns:
        x_pred (N_particles, D_hid): predicted particles

    """

    # Dynamics drift function
    def drift(t, y, args):
        return params.dynamics.drift.f(y, u, t)

    # Dynamics diffusion function
    if filter_hyperparams.state_order == "zeroth":
        # No diffusion in zeroth order EnKF
        diffusion = None
    else:
        # First order EnKF diffusion
        def diffusion(t, y, args):
            # Get parameters at time t and input u
            Qc_t = params.dynamics.diffusion_cov.f(y, u, t)
            L_t = (
                params.dynamics.diffusion_coefficient.f(y, u, t)
                * filter_hyperparams.cov_rescaling
            )
            Q_sqrt = jnp.linalg.cholesky(Qc_t)

            # Compute combined diffusion matrix
            combined_diffusion = L_t @ Q_sqrt

            return combined_diffusion

    # Solve the SDE for each particle
    # Define a function to solve the SDE for a single particle
    particle_solve = lambda y0, key0: diffeqsolve(
        key=key0,
        drift=drift,
        diffusion=diffusion,
        t0=t0,
        t1=t1,
        y0=y0,
        **filter_hyperparams.diffeqsolve_settings,
    )
    # Split keys for each particle
    key_array = jr.split(key, x.shape[0])
    # vmap over particles
    sol = vmap(particle_solve, in_axes=0)(x, key_array)  # N_particles x 1 time x D_hid
    # Extract final state for each particle
    x_pred = sol[:, 0, :]  # N_particles x D_hid

    # Add state noise according to state order
    if filter_hyperparams.state_order in ["zeroth", "discrete"]:
        # Predicted covariance
        if filter_hyperparams.state_order == "zeroth":
            # For zeroth order, we use the timestep at each step
            dt = t1 - t0
        else:
            # For discrete state order, we use a user-specified average timestep
            # This maps us into a discrete EnKF setting where the deterministic dynamics still map from t0 to t1
            # but the same amount of noise is added after each measurement.
            dt = filter_hyperparams.dt_average

        key_array = jr.split(key, x.shape[0])

        def _sample_noise(x_i, key_i):
            Qc_t = params.dynamics.diffusion_cov.f(x_i, u, t0)
            L_t = (
                params.dynamics.diffusion_coefficient.f(x_i, u, t0)
                * filter_hyperparams.cov_rescaling
            )
            state_noise_cov = dt * L_t @ Qc_t @ L_t.T  # D_hid x D_hid
            return jr.multivariate_normal(
                key=key_i, mean=jnp.zeros(x.shape[1]), cov=state_noise_cov
            )

        noise = vmap(_sample_noise, in_axes=(0, 0))(x_pred, key_array)

        # Add noise to predicted particles
        x_pred += noise

    # Return predicted particles
    return x_pred


# Condition on a new observation, under EnKF approximations
def _condition_on(
    key,
    x,
    h,
    R,
    u,
    y,
    t,
    perturb_measurements=True,
    inflation_delta=0.0,
    warn: bool = True,
):
    """Condition a Gaussian potential on a new observation
        using Ensemble Kalman Filter approximations.

    Args:
        key: random key.
        x (N_particles, D_hid): prior particles.
        h (Callable): emission function.
        R (D_obs,D_obs): emssion covariance matrix
        u (D_in,): inputs.
        y (D_obs,): observation.black
        t: time-instant of conditioning
        perturb_measurements: whether to perturb the measurements.
        inflation_delta: inflation factor for ensemble spread.
        warn: whether to warn about PSD issues.

    Returns:
        ll (float): log-likelihood of observation
        x_cond (N_particles, D_hid): filtered particles

    """

    # Number of particles and state dimension
    n_particles, state_dim = x.shape

    # Replicate inputs for each particle
    u_s = jnp.array([u] * n_particles)

    # Propagate ensemble through emission function
    # The shape of y_ensemble is n_particles x Observation Dimensions
    y_ensemble = vmap(h, in_axes=(0, None, None))(x, u_s, t)

    ## These 2 computations should use deterministic observation ensemble, not perturbed
    # compute predicted mean of measurements
    y_pred_mean = jnp.mean(y_ensemble, axis=0)

    # compute predicted covariance of measurements as outer product of differences from mean
    # Using sum of _outers to compute covariance to avoid degenerate dimensionality cases
    # represents "HPH^T" in Kalman gain computation
    y_pred_cov = psd(
        jnp.sum(_outer(y_ensemble - y_pred_mean, y_ensemble - y_pred_mean), axis=0)
        / (n_particles - 1),
        warn=warn,
    )

    # Compute log-likelihood of observation based on Gaussian distribution,
    # using predicted mean and covariance
    ll_step = MVN(y_pred_mean, y_pred_cov + R).log_prob(y)

    # === Inflate ensemble for assimilation if requested ===
    if inflation_delta > 0.0:
        x_mean = jnp.mean(x, axis=0)
        x_inflated = x_mean + (1.0 + inflation_delta) * (x - x_mean)
        # re-propagate inflated ensemble through emission fn for assimilation
        y_ensemble_infl = vmap(h, in_axes=(0, None, None))(x_inflated, u_s, t)
        y_pred_mean_infl = jnp.mean(y_ensemble_infl, axis=0)
        y_pred_cov_infl = psd(
            jnp.sum(
                _outer(
                    y_ensemble_infl - y_pred_mean_infl,
                    y_ensemble_infl - y_pred_mean_infl,
                ),
                axis=0,
            )
            / (n_particles - 1),
            warn=warn,
        )
    else:
        # No inflation, just use original ensemble
        x_inflated = x
        y_ensemble_infl = y_ensemble
        y_pred_mean_infl = y_pred_mean
        y_pred_cov_infl = y_pred_cov

    # compute cross_cov between x and y_data_perturbed
    # represents "PH^T" in Kalmna gain computation
    # cross-covariance using inflated ensemble
    cross_cov = jnp.sum(
        _outer(
            x_inflated - jnp.mean(x_inflated, axis=0),
            y_ensemble_infl - y_pred_mean_infl,
        ),
        axis=0,
    ) / (n_particles - 1)

    # Kalman gain
    S = y_pred_cov_infl + R
    K = psd_solve(S, cross_cov.T).T

    # make perturbed ensemble
    if perturb_measurements:
        # Add noise to the ensemble
        y_data_perturbed = jr.multivariate_normal(
            key=key, mean=y, cov=R, shape=(n_particles,)
        )
    else:
        y_data_perturbed = y

    # Updated the particles
    x_cond = x_inflated + (K @ (y_data_perturbed - y_ensemble_infl).T).T

    # --- Diagnostics ---
    innovation = y - y_pred_mean
    nis = innovation @ jnp.linalg.solve(S, innovation)  # normalized innovation squared
    eigvals_S = jnp.linalg.eigvalsh(S)
    min_eig_S = jnp.min(eigvals_S)
    max_eig_S = jnp.max(eigvals_S)
    cond_S = max_eig_S / min_eig_S

    return {
        "loglik_step": ll_step,
        "x_cond": x_cond,
        "y_ens_pred": y_ensemble,
        "S": S,
        "K": K,
        "innovation": innovation,
        "nis": nis,
        "min_eig_S": min_eig_S,
        "cond_S": cond_S,
    }


# EnKF filtering main function
def ensemble_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: EnKFHyperParams = EnKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "filtered_means",
        "filtered_covariances",
        "predicted_means",
        "predicted_covariances",
        "marginal_loglik",
        "posterior_extras",
    ],
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> PosteriorGSSMFiltered:
    r"""Run a ensemble Kalman filter
        to produce the marginal likelihood and filtered state estimates.

    Args:
        params: CD-NLGSSM parameters, containing
            - dynamics RHS drift function
            - diffusion coefficient matrix L
            - Brownian covariance matrix Q
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyper-parameters.
        inputs: optional array of inputs.
        output_fields: list of top-level posterior fields to include in the output.
            `posterior_extras` stores filter-specific diagnostics including
            `y_ens_pred`. The innovation covariance `S` is the observation-
            predictive covariance used for data scoring and assimilation,
            corresponding to the ensemble predictive covariance plus the
            observation covariance `R`.
        key: random generator key.
        warn: whether to warn about PSD issues.

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
                (
                    t_emissions[1:, 0],
                    jnp.array([t_emissions[-1, 0] + filter_hyperparams.dt_final]),
                )  # NB: t_{N+1} is simply t_{N}+dt_final
            ),
            t_emissions,
        )
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps)
        t1 = jnp.arange(1, num_timesteps + 1)

    # Set-up indexing
    t0_idx = jnp.arange(num_timesteps)

    # Process inputs
    inputs = _process_input(inputs, num_timesteps)

    # Only emission function
    h = params.emissions.emission_function.f

    # Define one step of EnKF filtering
    def _step(carry, args):
        # Unpack the inputs
        ll_cum, pred_x_ens = carry
        key, t0, t1, t0_idx = args

        # split key for (1) diffeqsolve, (2) perturbed measurements
        key_predict, key_filter = jr.split(key, 2)

        # Get parameters and inputs for time t0
        u = inputs[t0_idx]
        y = emissions[t0_idx]
        R = params.emissions.emission_cov.f(None, u, t0)

        # Condition on this emission
        cond_dict = _condition_on(
            key_filter,
            pred_x_ens,
            h,
            R,
            u,
            y,
            t0,
            filter_hyperparams.perturb_measurements,
            filter_hyperparams.inflation_delta,
            warn=warn,
        )
        # Filtered ensemble
        filtered_x_ens = cond_dict["x_cond"]

        # Update the log likelihood
        ll_cum += cond_dict["loglik_step"]

        # Compute filtered Gaussian statistics form ensemble
        filtered_mean = jnp.mean(filtered_x_ens, axis=0)
        filtered_cov = psd(
            jnp.sum(
                _outer(filtered_x_ens - filtered_mean, filtered_x_ens - filtered_mean),
                axis=0,
            )
            / (filter_hyperparams.N_particles - 1),
            warn=warn,
        )

        # Predict the next state, based on Ensemble prediction
        pred_x_ens = _predict(
            key_predict, filtered_x_ens, params, t0, t1, u, filter_hyperparams
        )

        # Compute predicted Gaussian statistics from predicted ensemble
        pred_mean = jnp.mean(pred_x_ens, axis=0)
        pred_cov = psd(
            jnp.sum(_outer(pred_x_ens - pred_mean, pred_x_ens - pred_mean), axis=0)
            / (filter_hyperparams.N_particles - 1),
            warn=warn,
        )

        # Build carry and output states
        carry = (ll_cum, pred_x_ens)

        # EnKF extras
        posterior_extras = {
            # Filtered/predicted particles here.
            "x_ens_filtered": filtered_x_ens,
            "x_ens_predicted": pred_x_ens,
            "y_ens_pred": cond_dict["y_ens_pred"],
            # Other diagnostics
            "loglik_step": cond_dict["loglik_step"],
            "S": cond_dict["S"],
            "K": cond_dict["K"],
            "innovation": cond_dict["innovation"],
            "nis": cond_dict["nis"],
            "min_eig_S": cond_dict["min_eig_S"],
            "cond_S": cond_dict["cond_S"],
            "cond_K": jnp.linalg.cond(cond_dict["K"]),
        }

        # Posterior output
        outputs = {
            # Default outputs
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": pred_mean,
            "predicted_covariances": pred_cov,
            # Extras
            "posterior_extras": posterior_extras,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        # Return carry and outputs
        return carry, outputs

    # Split keys to be used to:
    # (1) draw initial particles,
    # (2) run each step of the filter
    keys = jr.split(key, num_timesteps + 1)
    key_init, key_times = keys[0], keys[1:]

    # Initialize carry, by drawing particles from the initial distribution
    x_ens_init = jr.multivariate_normal(
        key=key_init,
        mean=params.initial.mean.f(),
        cov=params.initial.cov.f(),
        shape=(filter_hyperparams.N_particles,),
    )
    carry = (0.0, x_ens_init)
    # Run the Ensemble Kalman Filter, via lax.scan
    (ll_total, *_), outputs = lax.scan(_step, carry, (key_times, t0, t1, t0_idx))
    # Build and return posterior object
    outputs = {"marginal_loglik": ll_total, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


# CD-NLGSSM forecast function: Ensemble Kalman Filter Forecast
def forecast_ensemble_kalman_filter(
    params: ParamsCDNLGSSM,
    init_forecast: tfd.Distribution,
    t_init: Float[Array, "1 1"],
    t_forecast: Float[Array, "num_timesteps 1"],
    filter_hyperparams: EnKFHyperParams = EnKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "forecasted_state_means",
        "forecasted_state_covariances",
    ],
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> GSSMForecast:
    r"""Run an Ensemble Kalman filter to forecast states

    Args:
        params: CD-NLGSSM parameters, containing
            - dynamics RHS drift function
            - diffusion coefficient matrix L
            - Brownian covariance matrix Q
        init_forecast: initial distribution to forecast with.
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants to forecast
        filter_hyperparams: hyper-parameters of the EnKF, related to the approximation order
        inputs: optional array of inputs.
        output_fields: list of fields to return
        key: random key.
        warn: whether to warn about PSD issues.

    Returns:
        forecast: forecast object.

    """

    # Figure out timestamps, as vectors to scan over
    # t_forecast is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_forecast is not None:
        num_timesteps = t_forecast.shape[0]
        t0 = tree_map(
            lambda x: jnp.concatenate((t_init, t_forecast[:-1, 0])),
            t_forecast,
        )
        t1 = tree_map(lambda x: x[:, 0], t_forecast)
    else:
        raise ValueError("t_forecast must be provided for forecasting")

    # Set-up indexing and inputs
    t0_idx = jnp.arange(num_timesteps)
    inputs = _process_input(inputs, num_timesteps)

    # Define one step of EnKF forecasting
    def _step(carry, args):
        # Unpack the inputs
        current_x_ens = carry
        key, t0, t1, t0_idx = args

        # split key for prediction
        key_predict, _ = jr.split(key, 2)

        # Predict the next state, based on Ensemble prediction
        pred_x_ens = _predict(
            key_predict,
            current_x_ens,
            params,
            t0,
            t1,
            inputs[t0_idx],  # Inputs impact state at t0
            filter_hyperparams,
        )

        # Compute Gaussian statistics from predicted ensemble
        pred_state_mean = jnp.mean(pred_x_ens, axis=0)
        pred_state_cov = psd(
            jnp.sum(
                _outer(pred_x_ens - pred_state_mean, pred_x_ens - pred_state_mean),
                axis=0,
            )
            / (filter_hyperparams.N_particles - 1),
            warn=warn,
        )

        # Build carry and output states
        carry = pred_x_ens
        outputs = {
            "forecasted_state_means": pred_state_mean,
            "forecasted_state_covariances": pred_state_cov,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Build keys to be used to:
    # (1) draw initial particles,
    # (2) run each step of the filter
    keys = jr.split(key, num_timesteps + 1)
    key_init, key_times = keys[0], keys[1:]

    # Initialize the state,
    # by drawing particles from the provided initial distribution
    carry = init_forecast.sample(
        seed=key_init, sample_shape=filter_hyperparams.N_particles
    )

    # Run the Ensemble Kalman Filter, via lax.scan
    _, outputs = lax_scan(_step, carry, (key_times, t0, t1, t0_idx), debug=DEBUG)

    # Build the forecast object
    forecast = GSSMForecast(
        **outputs,
    )
    return forecast


# CD-NLGSSM emission function: Emissions from EnKF approximation
def emissions_ensemble_kalman_filter(
    params: ParamsCDNLGSSM,
    t_states: Float[Array, "num_timesteps 1"],
    state_means: Float[Array, "num_timesteps state_dim"],
    state_covs: Optional[Float[Array, "num_timesteps state_dim state_dim"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    filter_hyperparams: EnKFHyperParams = EnKFHyperParams(),
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> Tuple[
    Float[Array, "num_timesteps emission_dim"],
    Optional[Float[Array, "num_timesteps emission_dim emission_dim"]],
]:
    r"""Compute the emissions corresponding to the EnKF linearization of the model.

    Args:
        params: CD-NLGSSM parameters, containing
            - dynamics RHS drift function
            - diffusion coefficient matrix L
            - Brownian covariance matrix Q
        t_states: continuous-time specific time instants of states
        state_means: state means at time instants t_states, always required
        state_covs: state covariances at time instants t_states, optional
            - if None, then we assume that the states are point estimates, and simply push through emission function
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        filter_hyperparams: hyper-parameters of the filter
        key: random key.
        warn: whether to warn about PSD issues.

    Returns:
        emissions_mean: mean of emissions
        emissions_covariance: covariance of emissions, if available

    """

    # Figure out timestamps, as vectors to scan over
    # t_states is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_states is not None:
        num_timesteps = t_states.shape[0]
        t0 = tree_map(lambda x: x[:, 0], t_states)
    else:
        raise ValueError("t_states must be provided for forecasting")

    # Set-up indexing and inputs
    t0_idx = jnp.arange(num_timesteps)
    inputs = _process_input(inputs, num_timesteps)

    # Emission function
    h = params.emissions.emission_function.f

    # Define one step of EnKF emission computation
    def _step(carry, args):
        # Unpack the inputs
        key, state_mean, state_cov, t0, t0_idx = args

        # Draw ensemble from the state mean and covariance
        state_ens = MVN(state_mean, state_cov).sample(
            seed=key, sample_shape=filter_hyperparams.N_particles
        )

        # Emission covariance
        R = params.emissions.emission_cov.f(None, inputs[t0_idx], t0)

        # Propagate ensemble through emission function
        # Replicate inputs for each particle
        u_s = jnp.array([inputs[t0_idx]] * filter_hyperparams.N_particles)
        # Propagate ensemble via emission function, using vmap
        y_ensemble = vmap(h, in_axes=(0, None, None))(state_ens, u_s, t0)
        # The shape of y_ensemble is n_particles x Observation Dimensions

        ## These 2 computations should use deterministic observation ensemble, not perturbed
        # compute predicted mean of measurements
        emission_mean = jnp.mean(y_ensemble, axis=0)

        # compute predicted covariance of measurements as outer product of differences from mean
        # represents "HPH^T" in Kalman gain computation
        emission_cov = psd(
            R
            + jnp.sum(
                _outer(y_ensemble - emission_mean, y_ensemble - emission_mean), axis=0
            )
            / (filter_hyperparams.N_particles - 1),
            warn=warn,
        )

        # Return carry and output states
        return (state_mean, state_cov), (emission_mean, emission_cov)

    # Build keys to be used
    # to draw particles at each step of the filter
    keys = jr.split(key, num_timesteps)
    # Initialize the state,
    # based on provided initial distribution's mean and covariance
    carry = (state_means[0], state_covs[0])
    # Run the Ensemble Kalman filter, via lax.scan
    _, (emissions_mean, emissions_covariance) = lax_scan(
        _step, carry, (keys, state_means, state_covs, t0, t0_idx), debug=DEBUG
    )  # type: ignore

    # Return emissions mean and covariance
    return emissions_mean, emissions_covariance
