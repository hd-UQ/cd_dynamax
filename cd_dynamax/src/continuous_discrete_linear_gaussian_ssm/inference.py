# JAX imports
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.tree_util import tree_map

# Typing annotations
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union, Tuple, List
from functools import wraps
import inspect

# Distributions, compatible with JAX, from TensorFlow Probability
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)
import tensorflow_probability.substrates.jax.distributions as tfd

# Dynamax shared code
from cd_dynamax.dynamax.types import PRNGKey, Scalar
from cd_dynamax.dynamax.utils.utils import psd_solve

# To avoid unnecessary redefinitions of code,
# We import those classes that can be reused from LGSSM first, and define the rest later
# Filtering and smoothing classes are equivalent
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    PosteriorGSSMFiltered,
    PosteriorGSSMSmoothed,
)

# Initial and emission parameter classes are equivalent
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSMInitial,
    ParamsLGSSMEmissions,
)

# Our codebase
# CDLGSSM param and function definition
from ..continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import (
    ParamsCDLGSSM,
    ParamsCDLGSSMDynamics,
    GSSMForecast,
)

# Diffrax based diff-eq solver
from ..utils.diffrax_utils import diffeqsolve

# Debug utilities
from ..utils.debug_utils import psd

tfb = tfp.bijectors

DEBUG = False


#### Helper functions
# Helper functions --- modified from dynamax
def _get_params(x, dim, t):
    if callable(x):
        return x(t)
    # Added for non-existent biases
    elif x is None:
        return x
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x


_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)
_process_input = lambda x, y: jnp.zeros((y, 1)) if x is None else x


# CD-LGSSM preprocess parameters and inputs
def preprocess_params_and_inputs(params, num_timesteps, inputs):
    """Preprocess parameters in case some are set to None."""

    # Make sure all the required parameters are there
    assert params.initial.mean is not None
    assert params.initial.cov is not None
    assert params.dynamics.weights is not None
    assert params.dynamics.diffusion_coefficient is not None
    assert params.dynamics.diffusion_cov is not None
    assert params.emissions.weights is not None
    assert params.emissions.cov is not None

    # Get shapes
    emission_dim, state_dim = params.emissions.weights.shape[-2:]

    # Default the inputs to zero
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))
    input_dim = inputs.shape[-1]

    # Default other parameters to zero
    dynamics_input_weights = _zeros_if_none(
        params.dynamics.input_weights, (state_dim, input_dim)
    )
    dynamics_bias = _zeros_if_none(params.dynamics.bias, (state_dim,))
    emissions_input_weights = _zeros_if_none(
        params.emissions.input_weights, (emission_dim, input_dim)
    )
    emissions_bias = _zeros_if_none(params.emissions.bias, (emission_dim,))

    full_params = ParamsCDLGSSM(
        initial=ParamsLGSSMInitial(mean=params.initial.mean, cov=params.initial.cov),
        dynamics=ParamsCDLGSSMDynamics(
            weights=params.dynamics.weights,
            bias=dynamics_bias,
            input_weights=dynamics_input_weights,
            diffusion_coefficient=params.dynamics.diffusion_coefficient,
            diffusion_cov=params.dynamics.diffusion_cov,
        ),
        emissions=ParamsLGSSMEmissions(
            weights=params.emissions.weights,
            bias=emissions_bias,
            input_weights=emissions_input_weights,
            cov=params.emissions.cov,
        ),
    )
    return full_params, inputs


# CD-LGSSM preprocess_args decorator
def preprocess_args(f):
    """Preprocess the parameter and input arguments in case some are set to None."""
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Extract the arguments by name
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments["params"]
        emissions = bound_args.arguments["emissions"]
        t_emissions = bound_args.arguments["t_emissions"]
        filter_hyperparams = bound_args.arguments["filter_hyperparams"]
        inputs = bound_args.arguments["inputs"]
        warn = bound_args.arguments["warn"]

        num_timesteps = len(emissions)
        full_params, inputs = preprocess_params_and_inputs(
            params, num_timesteps, inputs
        )

        return f(
            full_params,
            emissions,
            t_emissions,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs,
            warn=warn,
        )

    return wrapper


#### CD-LGSSM push-forward function
# CD-LGSSM push-forward is computed
# based on the assumed continuous-discrete linear Gaussian SSM dynamics
# using an diffrax-based SDE solver
def compute_pushforward(
    params: ParamsCDLGSSM,
    t0: Float,
    t1: Float,
    diffeqsolve_settings: dict = {},
    warn: bool = True,
) -> Tuple[Float[Array, "state_dim state_dim"], Float[Array, "state_dim state_dim"]]:
    r"""Compute the push-forward of the continuous-time linear Gaussian dynamics
        based on the SDE definition $dz_t = F_t z_t dt + L_t dB_t$
        from time t0 to t1, returning the dynamics matrix A and covariance Q, such that
           $z_{t1} = A z_{t0} + \epsilon,  \epsilon \sim \mathcal{N}(0, Q)$

    Args:
        params: CD-LGSSM model parameters
        t0: initial time
        t1: final time
        diffeqsolve_settings: settings for the diff-eq solver
        warn: whether to issue warnings (e.g., about PSD issues).

    Returns:
        A: dynamics matrix from t0 to t1
        Q: dynamics covariance from t0 to t1
    """

    # Initial conditions
    state_dim = params.dynamics.weights.shape[0]
    A0 = jnp.eye(state_dim)
    Q0 = jnp.zeros((state_dim, state_dim))
    y0 = (A0, Q0)

    # Define the right-hand side of the SDEs for mean (A) and covariance (Q)
    def rhs_all(t, y, args):
        # Unpack
        A, Q = y

        # Possibly time-dependent linear matrices/weights
        F_t = _get_params(params.dynamics.weights, 2, t)
        Qc_t = _get_params(params.dynamics.diffusion_cov, 2, t)
        L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t)

        # Define the RHS of the SDE for A and Q
        dAdt = F_t @ A
        dQdt = F_t @ Q + Q @ F_t.T + L_t @ Qc_t @ L_t.T

        # Return the RHS
        return (dAdt, dQdt)

    # Solve the SDE as specified by rhs_all
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0, **diffeqsolve_settings)
    # Extract final A and Q, ensure PSD covariance
    A, Q = sol[0][-1], psd(sol[1][-1], warn=warn)
    # Return SDE's mean (A) and covariance (Q)
    return A, Q


#### CD-LGSSM filtering
# Default filtering hyperparameters, as class
class KFHyperParams(NamedTuple):
    """Lightweight container for filtering hyperparameters."""

    dt_final: float = (
        1e-10  # Small dt_final for predicted mean and covariance at the end of sequence
    )
    diffeqsolve_settings: dict = {}


# Predict next mean and covariance under a linear Gaussian model
def _predict(m, S, F, B, b, Q, u, warn: bool = True):
    r"""Predict next mean and covariance under a linear Gaussian model.

        p(z_{t+1}) = int N(z_t \mid m, S) N(z_{t+1} \mid Fz_t + Bu + b, Q)
                    = N(z_{t+1} \mid Fm + Bu, F S F^T + Q)

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        F (D_hid,D_hid): dynamics matrix.
        B (D_hid,D_in): dynamics input matrix.
        u (D_in,): inputs.
        Q (D_hid,D_hid): dynamics covariance matrix.
        b (D_hid,): dynamics bias.
        warn: whether to issue warnings (e.g., about PSD issues).

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the predicted mean and covariance
    mu_pred = F @ m + B @ u + b
    Sigma_pred = psd(F @ S @ F.T + Q, warn=warn)

    # Return them
    return mu_pred, Sigma_pred


# Condition on a new linear Gaussian observation
def _condition_on(m, P, H, D, d, R, u, y, warn: bool = True):
    r"""Condition a Gaussian potential on a new linear Gaussian observation
       p(z_t \mid y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t \mid y_{1:t-1}, u_{1:t-1}) p(y_t \mid z_t, u_t)
         = N(z_t \mid m, P) N(y_t \mid H_t z_t + D_t u_t + d_t, R_t)
         = N(z_t \mid mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = H*m + D*u + d
         S = (R + H * P * H')
         K = P * H' * S^{-1}
         PP = P - K S K' = Sigma_cond

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         H (D_obs,D_hid): emission matrix.
         D (D_obs,D_in): emission input weights.
         u (D_in,): inputs.
         d (D_obs,): emission bias.
         R (D_obs,D_obs): emission covariance matrix.
         y (D_obs,): observation.
         warn: whether to issue warnings (e.g., about PSD issues).

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the Kalman gain
    if R.ndim == 2:
        S = R + H @ P @ H.T
        K = psd_solve(S, H @ P).T
    else:
        # Optimization using Woodbury identity with A=R, U=H@chol(P), V=U.T, C=identity_matrix
        # (see https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
        identity_matrix = jnp.eye(P.shape[0])
        U = H @ jnp.linalg.cholesky(P)
        X = U / R[:, None]
        S_inv = jnp.diag(1.0 / R) - X @ psd_solve(identity_matrix + U.T @ X, X.T)
        """
        # Could alternatively use U=H and C=P
        R_inv = jnp.diag(1.0 / R)
        P_inv = psd_solve(P, jnp.eye(P.shape[0]))
        S_inv = R_inv - R_inv @ H @ psd_solve(P_inv + H.T @ R_inv @ H, H.T @ R_inv)
        """
        K = P @ H.T @ S_inv
        S = jnp.diag(R) + H @ P @ H.T

    # Compute the conditional mean and covariance
    Sigma_cond = psd(P - K @ S @ K.T, warn=warn)
    mu_cond = m + K @ (y - D @ u - d - H @ m)
    # Return them
    return mu_cond, Sigma_cond


# CD-LGSSM filtering implementation: Kalman filter
@preprocess_args
def cdlgssm_filter(
    params: ParamsCDLGSSM,
    emissions: Float[Array, "num_timesteps emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[KFHyperParams] = KFHyperParams(),
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    warn: bool = True,
) -> PosteriorGSSMFiltered:
    r"""Run a Continuous Discrete Kalman filter
        to produce the marginal likelihood and filtered state estimates.

    Args:
        params: CD-LGSSM parameters
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyperparameters for the filter.
        inputs: optional array of inputs.
        warn: whether to issue warnings during filtering (e.g., PSD issues).

    Returns:
        PosteriorGSSMFiltered: filtered posterior object
    """
    # Double-check filter_hyperparams is not None
    if filter_hyperparams is None:
        filter_hyperparams = KFHyperParams()

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
                    jnp.array(
                        [t_emissions[-1, 0] + filter_hyperparams.dt_final]
                    ),  # NB: t_{N+1} is simply t_{N}+dt_final
                )
            ),
            t_emissions,
        )
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps)
        t1 = jnp.arange(1, num_timesteps + 1)
    # Indices for emissions
    t0_idx = jnp.arange(num_timesteps)
    # Default inputs to zero if None
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Define the filtering step to be scanned over time
    def _step(carry, args):
        # Unpack the inputs
        ll, pred_mean, pred_cov = carry
        t0, t1, t0_idx = args

        # Possibly time-dependent weights and params
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        H = _get_params(params.emissions.weights, 2, t0)
        D = _get_params(params.emissions.input_weights, 2, t0)
        d = _get_params(params.emissions.bias, 1, t0)
        R = _get_params(params.emissions.cov, 2, t0)
        # Get inputs and emissions at this time
        u = inputs[t0_idx]
        y = emissions[t0_idx]

        # Update the log likelihood
        ll += MVN(H @ pred_mean + D @ u + d, H @ pred_cov @ H.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, H, D, d, R, u, y, warn=warn
        )

        # Predict to the next time instant
        # Get the matrices for the push-forward from t0 to t1
        F, Q = compute_pushforward(
            params,
            t0,
            t1,
            diffeqsolve_settings=filter_hyperparams.diffeqsolve_settings,
            warn=warn,
        )
        # Predict the next state
        pred_mean, pred_cov = _predict(
            filtered_mean, filtered_cov, F, B, b, Q, u, warn=warn
        )

        # Return the carry and outputs
        return (ll, pred_mean, pred_cov), (
            filtered_mean,
            filtered_cov,
            pred_mean,
            pred_cov,
        )

    # The Kalman filter
    # Initial carry
    carry = (0.0, params.initial.mean, params.initial.cov)
    # Scan over all time steps
    (ll, _, _), (filtered_means, filtered_covs, pred_means, pred_covs) = lax.scan(
        _step, carry, (t0, t1, t0_idx)
    )

    # Return the posterior object
    return PosteriorGSSMFiltered(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        predicted_means=pred_means,
        predicted_covariances=pred_covs,
    )


# Kalmam Smoothing equations, in continuous-time
def _smooth(
    m_filter,
    P_filter,  # Filtered mean and covariance
    m_smooth,
    P_smooth,  # Smoothed mean and covariance
    params: ParamsCDLGSSM,  # All necessary CD dynamic params
    t0: Float,
    t1: Float,
    u: Float[Array, " input_dim"],
    filter_hyperparams: Optional[KFHyperParams] = KFHyperParams(),
    warn: bool = True,
):
    r"""Smooth the next mean and covariance using EKF smoothing equations
        where the evolution of m and P are computed based on
            Equations 3.149 in Sarkka's Thesis

    Args:
        m_filter (D_hid,): filtered mean at t1.
        P_filter (D_hid,D_hid): filtered covariance at t1.
        m_smooth (D_hid,): smooth mean at t1.
        P_smooth (D_hid,D_hid): smoothed covariance at t1.
        params: parameters of CD nonlinear dynamics, containing dynamics RHS function, coeff matrix and Brownian covariance matrix.
        t0: initial time-instant
        t1: final time-instant
        u (D_in,): inputs
            NB: these are currently ignored, need to incorporate them in the dynamics RHS
        warn: whether to issue warnings (e.g., about PSD issues).

    Returns:
        mu_smooth (D_hid,): smoothed mean at t0.
        Sigma_smooth (D_hid,D_hid): smoothed covariance at t0.
    """

    # Initialize
    y0 = (m_smooth, P_smooth)

    # Smoothed mean and covariance evolution
    def rhs_all(t, y, args):
        # Unpack
        m_smooth, P_smooth = y
        m_filter, P_filter = args

        # possibly time-dependent weights
        F_t = _get_params(params.dynamics.weights, 2, t)
        Qc_t = _get_params(params.dynamics.diffusion_cov, 2, t)
        L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t)

        # Auxiliary matrix, used in both mean and covariance computations
        # Inverse product computed via psd_solve
        aux_matrix = psd_solve(P_filter, L_t @ Qc_t @ L_t.T).T

        # Mean evolution
        dmsmoothdt = F_t @ m_smooth + aux_matrix @ (m_smooth - m_filter)
        # Covariance evolution
        dPsmoothdt = (
            (F_t + aux_matrix) @ P_smooth
            + P_smooth @ (F_t + aux_matrix).T
            - L_t @ Qc_t @ L_t.T
        )

        return (dmsmoothdt, dPsmoothdt)

    # We solve the rhs in reverse:
    # from t1 to t0, BUT y0 contains initial conditions at t1
    sol = diffeqsolve(
        rhs_all,
        t0=t0,
        t1=t1,
        y0=y0,
        reverse=True,
        args=(m_filter, P_filter),
        **filter_hyperparams.diffeqsolve_settings,
    )
    # Extract and return smoothed mean and covariance, ensure PSD covariance
    m_smooth, P_smooth = sol[0][-1], psd(sol[1][-1], warn=warn)
    return m_smooth, P_smooth


# CD-LGSSM smoother implementation: Kalman smoother
# @preprocess_args: fix to accommodate smoother_type
def cdlgssm_smoother(
    params: ParamsCDLGSSM,
    emissions: Float[Array, "num_timesteps emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[KFHyperParams] = KFHyperParams(),
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    smoother_type: Optional[str] = "cd_smoother_1",
    warn: bool = True,
) -> PosteriorGSSMSmoothed:
    r"""Run forward-filtering, backward-smoother to compute expectations
        under the posterior distribution on latent states.
        Technically, this smoother implements two different types of smoothers

    Args:
        params: CD-LGSSM parameters
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        inputs: array of inputs.
        smoother_type:
            cd_smoother_1: Sarkka's Algorithm 3.17
            cd_smoother_2: Sarkka's Algorithm 3.18
        warn: whether to issue warnings during smoothing (e.g., PSD issues).

    Returns:
        PosteriorGSSMSmoothed: smoothed posterior object.
    """

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps-1 \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
        t1 = tree_map(lambda x: x[1:, 0], t_emissions)
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps - 1)
        t1 = jnp.arange(1, num_timesteps)

    # Indices for emissions
    t0_idx = jnp.arange(num_timesteps - 1)
    # Default inputs to zero if None
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = cdlgssm_filter(
        params, emissions, t_emissions, filter_hyperparams, inputs, warn=warn
    )
    # Unpack the filtered posterior
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    print("Running KF smoother type = {}".format(smoother_type))

    # Smoothing step, according to smoother type 1 (Sarkka's Algorithm 3.17)
    def _step_1(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t0, t1, t0_idx, filtered_mean, filtered_cov = args

        print("Running KF smoother type 1")
        # Get the discretization matrices
        F, Q = compute_pushforward(
            params,
            t0,
            t1,
            diffeqsolve_settings=filter_hyperparams.diffeqsolve_settings,
            warn=warn,
        )
        # Possibly time-dependent weights
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        # Get inputs at this time
        u = inputs[t0_idx]

        # Computation of C in CD-Kalman Smoother in Equation 3.148
        C = psd_solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance as in Equation 3.148
        if b is None:
            smoothed_mean = filtered_mean + C @ (
                smoothed_mean_next - F @ filtered_mean - B @ u
            )
        else:
            smoothed_mean = filtered_mean + C @ (
                smoothed_mean_next - F @ filtered_mean - B @ u - b
            )

        # Compute the smoothed covariance
        smoothed_cov = psd(
            filtered_cov + C @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ C.T,
            warn=warn,
        )

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = C @ smoothed_cov_next + jnp.outer(
            smoothed_mean, smoothed_mean_next
        )

        # Return the carry and outputs
        return (smoothed_mean, smoothed_cov), (
            smoothed_mean,
            smoothed_cov,
            smoothed_cross,
        )

    # Smoothing step, according to smoother type 2 (Sarkka's Algorithm 3.18)
    def _step_2(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t0, t1, t0_idx, filtered_mean, filtered_cov = args

        print("Running KF smoother type 2")
        # Compute the smoothed mean and covariance by solving Equation 3.149
        smoothed_mean, smoothed_cov = _smooth(
            m_filter=filtered_mean,
            P_filter=filtered_cov,  # Filtered
            m_smooth=smoothed_mean_next,
            P_smooth=smoothed_cov_next,  # Smoothed
            params=params,
            t0=t0,
            t1=t1,
            u=inputs[t0_idx],
            filter_hyperparams=filter_hyperparams,
            warn=warn,
        )

        # Smoothed cross-covariance is not computed in this smoother type
        smoothed_cross = jnp.nan * jnp.ones(filtered_cov.shape)

        # Return the carry and outputs
        return (smoothed_mean, smoothed_cov), (
            smoothed_mean,
            smoothed_cov,
            smoothed_cross,
        )

    # Select the smoothing step function with a Python branch.
    # This avoids tracing both branches (as lax.switch would do), which can
    # trigger misleading debug prints from both smoother implementations.
    if smoother_type == "cd_smoother_1":
        _step = _step_1
    elif smoother_type == "cd_smoother_2":
        _step = _step_2
    else:
        raise ValueError(
            f"Unknown smoother_type '{smoother_type}'. Expected "
            "'cd_smoother_1' or 'cd_smoother_2'."
        )

    # Run smoother steps via lax scan, using reverse mode
    _, (smoothed_means, smoothed_covs, smoothed_cross) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),
        (t0, t1, t0_idx, filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    # Return the posterior object
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        smoothed_cross_covariances=smoothed_cross,
    )


#### CD-LGSSM interface functions, outside the class definition, for convenience
# CD-LGSSM sampling function, based on the model distributions
def cdlgssm_joint_sample(
    params: ParamsCDLGSSM,
    key: PRNGKey,
    num_timesteps: int,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    diffeqsolve_settings={},
    warn: bool = True,
) -> Tuple[
    Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]
]:
    r"""Sample from the joint distribution to produce state and emission trajectories.

    Args:
        params: CD-LGSSM parameters
        key: random key
        num_timesteps: number of time steps to sample
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        inputs: optional array of inputs.
        diffeqsolve_settings: settings for the diff-eq solver
        warn: whether to issue warnings during sampling (e.g., PSD issues).

    Returns:
        latent states and observed emissions

    """
    # Preprocess parameters and inputs
    params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

    # Function to sample from transition distribution
    def _sample_transition(key, F, B, b, Q, x_tm1, u):
        # Compute the mean of the transition distribution
        mean = F @ x_tm1 + B @ u + b
        # Return a sample from the transition distribution
        return MVN(mean, Q).sample(seed=key)

    # Function to sample from emission distribution
    def _sample_emission(key, H, D, d, R, x, u):
        # Compute the mean of the emission distribution
        mean = H @ x + D @ u + d
        # Compute the covariance of the emission distribution
        R = jnp.diag(R) if R.ndim == 1 else R
        # Return a sample from the emission distribution
        return MVN(mean, R).sample(seed=key)

    # Function to sample initial state and emission
    def _sample_initial(key, params, inputs):
        # Split key for initial state and emission sampling
        key1, key2 = jr.split(key)

        # Sample initial state
        initial_state = MVN(params.initial.mean, params.initial.cov).sample(seed=key1)

        # Get initial emission parameters
        H0 = _get_params(params.emissions.weights, 2, 0)
        D0 = _get_params(params.emissions.input_weights, 2, 0)
        d0 = _get_params(params.emissions.bias, 1, 0)
        R0 = _get_params(params.emissions.cov, 2, 0)
        # Get initial input
        u0 = tree_map(lambda x: x[0], inputs)

        # Sample initial emission
        initial_emission = _sample_emission(key2, H0, D0, d0, R0, initial_state, u0)

        # Return initial state and emission
        return initial_state, initial_emission

    # Function to sample next state and emission given previous state
    def _step(prev_state, args):
        # Unpack the arguments
        key, t0, t1, inpt = args
        # Split key for state and emission sampling
        key1, key2 = jr.split(key, 2)

        # Get parameters and inputs for time index t
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        F, Q = compute_pushforward(params, t0, t1, diffeqsolve_settings, warn=warn)
        H = _get_params(params.emissions.weights, 2, t1)
        D = _get_params(params.emissions.input_weights, 2, t1)
        d = _get_params(params.emissions.bias, 1, t1)
        R = _get_params(params.emissions.cov, 2, t1)

        # Sample from transition and emission distributions
        state = _sample_transition(key1, F, B, b, Q, prev_state, inpt)
        emission = _sample_emission(key2, H, D, d, R, state, inpt)

        # Return the sampled state and emission
        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)

    # Sample initial state and emission
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Set keys for the remaining time steps
    next_keys = jr.split(key2, num_timesteps - 1)

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
        t1 = tree_map(lambda x: x[1:, 0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps - 1)
        t1 = jnp.arange(1, num_timesteps)

    # Get inputs for remaining time steps
    next_inputs = tree_map(lambda x: x[1:], inputs)

    # Sample the remaining emissions and states via scan
    _, (next_states, next_emissions) = lax.scan(
        _step, initial_state, (next_keys, t0, t1, next_inputs)
    )

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions


# CD-LGSSM path sampling function, based on the model distributions and SDE solver
def cdlgssm_path_sample(
    params: ParamsCDLGSSM,
    key: PRNGKey,
    num_timesteps: int,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    diffeqsolve_settings={},
    warn: bool = True,
) -> Tuple[
    Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]
]:
    r"""Sample from a forward path of the CD-LGSSM
        to produce state and emission trajectories.

    Args:
        params: CD-LGSSM parameters
        key: random number generator key
        num_timesteps: number of time steps to sample
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        inputs: optional array of inputs.
        diffeqsolve_settings: settings for the SDE solver

    Returns:
        latent states and observed emissions

    """
    # Preprocess parameters and inputs
    params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

    # Function to sample from emission distribution
    def _sample_emission(key, H, D, d, R, x, u):
        # Compute the mean of the emission distribution
        mean = H @ x + D @ u + d
        # Compute the covariance of the emission distribution
        R = jnp.diag(R) if R.ndim == 1 else R

        # Return a sample from the emission distribution
        return MVN(mean, R).sample(seed=key)

    # Function to sample initial state and emission
    def _sample_initial(key, params, inputs):
        # Split key for initial state and emission sampling
        key1, key2 = jr.split(key)

        # Sample initial state
        initial_state = MVN(params.initial.mean, params.initial.cov).sample(seed=key1)

        # Get initial emission parameters
        H0 = _get_params(params.emissions.weights, 2, 0)
        D0 = _get_params(params.emissions.input_weights, 2, 0)
        d0 = _get_params(params.emissions.bias, 1, 0)
        R0 = _get_params(params.emissions.cov, 2, 0)
        # Get initial inputs
        u0 = tree_map(lambda x: x[0], inputs)

        # Sample initial emission
        initial_emission = _sample_emission(key2, H0, D0, d0, R0, initial_state, u0)

        return initial_state, initial_emission

    # Function to path-sample next state and emission given previous state
    def _step(prev_state, args):
        # Unpack the arguments
        key, t0, t1, inpt = args
        # Split key for state and emission sampling
        key1, key2 = jr.split(key, 2)

        # get parameters and inputs for time index t
        F_t = _get_params(params.dynamics.weights, 2, t0)
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        Q_t = _get_params(params.dynamics.diffusion_cov, 2, t0)
        L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t0)

        # SDE definition as per the CD-LGSSM
        def drift(t, y, args):
            return F_t @ y + B @ inpt + b

        def diffusion(t, y, args):
            Q_sqrt = jnp.linalg.cholesky(Q_t)
            combined_diffusion = L_t @ Q_sqrt
            return combined_diffusion

        # Numerically solve the SDE, from t0 to t1
        state = diffeqsolve(
            key=key1,
            drift=drift,
            diffusion=diffusion,
            t0=t0,
            t1=t1,
            y0=prev_state,
            **diffeqsolve_settings,
        )[0]

        # Get emission parameters at time t1
        H = _get_params(params.emissions.weights, 2, t1)
        D = _get_params(params.emissions.input_weights, 2, t1)
        d = _get_params(params.emissions.bias, 1, t1)
        R = _get_params(params.emissions.cov, 2, t1)
        # Sample from emission distribution at time t1
        emission = _sample_emission(key2, H, D, d, R, state, inpt)

        return state, (state, emission)

    # Split key for initial state and emission sampling
    key1, key2 = jr.split(key)

    # Sample initial state and emission
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Set keys for the remaining time steps
    next_keys = jr.split(key2, num_timesteps - 1)

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
        t1 = tree_map(lambda x: x[1:, 0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps - 1)
        t1 = jnp.arange(1, num_timesteps)

    # Get inputs for remaining time steps
    next_inputs = tree_map(lambda x: x[1:], inputs)

    # Sample the remaining emissions and states via scan
    _, (next_states, next_emissions) = lax.scan(
        _step, initial_state, (next_keys, t0, t1, next_inputs)
    )

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions


# CD-LGSSM Posterior Sampling function
def cdlgssm_posterior_sample(
    key: PRNGKey,
    params: ParamsCDLGSSM,
    emissions: Float[Array, "num_timesteps emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[KFHyperParams] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    jitter: Optional[Scalar] = 0,
    warn: bool = True,
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples from $p(z_{1:T} \mid y_{1:T}, u_{1:T})$.

    Args:
        key: random number key.
        params: CD-LGSSM parameters.
        emissions: sequence of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        inputs: optional sequence of inptus.
        jitter: padding to add to the diagonal of the covariance matrix before sampling.
        warn: whether to issue warnings during smoothing (e.g., PSD issues).

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.

    """
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps-1 \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
        t1 = tree_map(lambda x: x[1:, 0], t_emissions)
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps - 1)
        t1 = jnp.arange(1, num_timesteps)

    # Indices for emissions
    t0_idx = jnp.arange(num_timesteps - 1)
    # Default inputs to zero if None
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = cdlgssm_filter(
        params, emissions, t_emissions, filter_hyperparams, inputs, warn=warn
    )
    # Unpack the filtered posterior
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Define the backward sampling step
    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        key, t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # get parameters and inputs for time index t
        F, Q = compute_pushforward(
            params,
            t0,
            t1,
            diffeqsolve_settings=filter_hyperparams.diffeqsolve_settings,
            warn=warn,
        )
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        # Get inputs at this time
        u = inputs[t0_idx]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(
            filtered_mean, filtered_cov, F, B, b, Q, u, next_state, warn=warn
        )
        # Sample the current state
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)

        return state, state

    # Split key for sampling the last state
    key, this_key = jr.split(key, 2)

    # Sample the last state from the filtered distribution at time T
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    # Sample the remaining states via scan, in reverse mode
    _, states = lax.scan(
        _step,
        last_state,
        (
            jr.split(key, num_timesteps - 1),
            t0,
            t1,
            t0_idx,
            filtered_means[:-1],
            filtered_covs[:-1],
        ),
        reverse=True,
    )

    # Concatenate the sampled states and return
    return jnp.vstack([states, last_state])


# CD-LGSSM Forecasting function
def cdlgssm_forecast(
    params: ParamsCDLGSSM,
    init_forecast: Union[tfd.Distribution, Float[Array, "state_dim 1"]],
    t_init: Float[Array, "1 1"],
    t_forecast: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[KFHyperParams] = KFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "forecasted_state_means",
        "forecasted_state_covariances",
    ],
    key: PRNGKey = jr.PRNGKey(0),
    diffeqsolve_settings: dict = {},
    warn: bool = True,
) -> GSSMForecast:
    r"""Run a Continuous Discrete Kalman filter
        to produce forecasted state estimates.

    Args:
        params: CD-LGSSM parameters.
        init_forecast: initial condition to start forecasting with:
            - if init_forecast is a distribution, then we forecast such distribution based on different filtering methods
            - if init_forecast is a point estimate of state, then we forecast a forward path starting at that state
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyper-parameters of the filter
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        output_fields: list of fields to return in posterior object.
            These can take the values
                If we forecast Gaussian distributions, based on filtering methods
                    "forecasted_state_means",
                    "forecasted_state_covariances",
                If we forecast paths, based on solving the SDE
                    "forecasted_state_path".
        key: random key (e.g., for Ensemble Kalman).
        diffeqsolve_settings: settings for the SDE solver
        warn: whether to issue warnings during filtering (e.g., PSD issues).

    Returns:
        post: forecasted object.

    """

    # Double-check filter_hyperparams is not None
    if filter_hyperparams is None:
        filter_hyperparams = KFHyperParams()

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

    # Default inputs to zero if None
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Check whether init_forecast is a distribution or a point estimate
    if isinstance(init_forecast, tfd.Distribution):
        # Forecasting a distribution, based on different filters

        # Define the function to scan over
        def _step(carry, args):
            # Unpack the inputs
            current_state_mean, current_state_cov = carry
            t0, t1, inpt = args

            # CD-LGSSM parameters: just the dynamics
            B = _get_params(params.dynamics.input_weights, 2, t0)
            b = _get_params(params.dynamics.bias, 1, t0)
            F, Q = compute_pushforward(
                params,
                t0,
                t1,
                diffeqsolve_settings=filter_hyperparams.diffeqsolve_settings,
                warn=warn,
            )

            # Predict the next state based on the KF
            pred_state_mean, pred_state_cov = _predict(
                current_state_mean,
                current_state_cov,
                F,
                B,
                b,
                Q,
                inpt,
                warn=warn,
            )

            # Build carry and output states
            carry = (pred_state_mean, pred_state_cov)
            outputs = {
                "forecasted_state_means": pred_state_mean,
                "forecasted_state_covariances": pred_state_cov,
            }
            outputs = {key: val for key, val in outputs.items() if key in output_fields}

            return carry, outputs

        # Initialize the state, based on provided initial distribution's mean and covariance
        carry = (init_forecast.mean(), init_forecast.covariance())

        # Run the Kalman filter steps via scan
        _, outputs = lax.scan(
            _step,
            carry,
            (t0, t1, inputs),
        )

        # Build the forecast object
        forecast = GSSMForecast(
            **outputs,
        )
    else:
        # Forecasting point estimates, based on pushing forward the model

        # Define the function to scan over
        def _step(prev_state, args):
            # Unpack the arguments
            key, t0, t1, inpt = args

            # Get parameters and inputs for time index t
            F_t = _get_params(params.dynamics.weights, 2, t0)
            B = _get_params(params.dynamics.input_weights, 2, t0)
            b = _get_params(params.dynamics.bias, 1, t0)
            Q_t = _get_params(params.dynamics.diffusion_cov, 2, t0)
            L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t0)

            # SDE definition as per the CD-LGSSM
            def drift(t, y, args):
                return F_t @ y + B @ inpt + b

            def diffusion(t, y, args):
                Q_sqrt = jnp.linalg.cholesky(Q_t)
                combined_diffusion = L_t @ Q_sqrt
                return combined_diffusion

            # Solve the SDE numerically, from t0 to t1
            state = diffeqsolve(
                key=key,
                drift=drift,
                diffusion=diffusion,
                t0=t0,
                t1=t1,
                y0=prev_state,
                **diffeqsolve_settings,
            )[0]

            # Return the state
            return state, (state)

        # Split keys for each time step
        next_keys = jr.split(key, num_timesteps)

        # Forecast states over time, via scan
        _, (next_states) = lax.scan(
            _step,
            init_forecast,
            (next_keys, t0, t1, inputs),
        )  # type: ignore

        # Build the forecast object
        forecast = GSSMForecast(forecasted_state_path=next_states)

    # Return the forecast object
    return forecast


# CD-LGSSM Emission function
def cdlgssm_emissions(
    params: ParamsCDLGSSM,
    t_states: Float[Array, "num_timesteps 1"],
    state_means: Float[Array, "num_timesteps state_dim"],
    state_covs: Optional[Float[Array, "num_timesteps state_dim state_dim"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> Tuple[
    Float[Array, "num_timesteps emission_dim"],
    Float[Array, "num_timesteps emission_dim emission_dim"],
]:
    r"""Compute the emissions corresponding to
        - a continuous-discrete linear model, as specified by params

    Args:
        params: model parameters.
        t_states: continuous-time specific time instants of states
        state_means: state means at time instants t_states, always required
        state_covs: state covariances at time instants t_states, optional
            - if None, then we assume that the states are point estimates, and simply push through emission function
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        key: random key for sampling
        warn: whether to issue warnings during filtering (e.g., PSD issues).

    Returns:
        emissions_mean: mean of emissions
        emissions_covariance: covariance of emissions
    """

    # Emissions based on a linear transformation of the state through

    # Figure out timestamps, as vectors to scan over
    # t_states is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_states is not None:
        num_timesteps = t_states.shape[0]
        t0 = tree_map(lambda x: x[:, 0], t_states)
    else:
        raise ValueError("t_states must be provided for forecasting")

    # Default inputs to zero if None
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Define the function to scan over
    def _step(state, args):
        # Unpack the arguments
        this_state, this_cov, t0, inpt = args

        # Emission function:
        # a linear Gaussian transformation of the state
        H = _get_params(params.emissions.weights, 2, t0)
        D = _get_params(params.emissions.input_weights, 2, t0)
        d = _get_params(params.emissions.bias, 1, t0)

        # Emission mean, as linear combination of state
        emission_mean = H @ this_state + D @ inpt + d

        # Emission covariance, as determined by the model
        R = _get_params(params.emissions.cov, 2, t0)
        if this_cov is not None:
            # If we have state covariances, then we compute the emission covariance
            emission_cov = psd(H @ this_cov @ H.T + +R, warn=warn)
        else:
            emission_cov = jnp.diag(R) if R.ndim == 1 else R

        # Return the state and emission's mean and covariance
        return this_state, (emission_mean, emission_cov)

    # Compute emissions, over time, via scan
    _, (emissions_mean, emissions_covariance) = lax.scan(
        _step,
        state_means[0],
        (state_means, state_covs, t0, inputs),
    )  # type: ignore

    # Return the emission mean and covariance
    return emissions_mean, emissions_covariance
