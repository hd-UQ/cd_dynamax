import jax.numpy as jnp
import jax.random as jr
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from functools import wraps
import inspect

from jax.tree_util import tree_map
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union, Tuple
from dynamax.utils.utils import psd_solve, symmetrize
from dynamax.parameters import ParameterProperties
from dynamax.types import PRNGKey, Scalar

import diffrax as dfx

import jax.debug as jdb

class ParamsCDLGSSMInitial(NamedTuple):
    r"""Parameters of the initial distribution

    $$p(z_1) = \mathcal{N}(z_1 \mid \mu_1, Q_1)$$

    The tuple doubles as a container for the ParameterProperties.

    :param mean: $\mu_1$
    :param cov: $Q_1$

    """
    mean: Union[Float[Array, "state_dim"], ParameterProperties]
    # unconstrained parameters are stored as a vector.
    cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]


class ParamsCDLGSSMDynamics(NamedTuple):
    r"""Parameters of the dynamics distribution

    $$p(z_{t+1} \mid z_t, u_t) = \mathcal{N}(z_{t+1} \mid A z_t + B u_t + b, Q)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: dynamics weights $F$ -> used to compute A based on ODE
    :param bias: dynamics bias $b$
    :param input_weights: dynamics input weights $B$
    :param cov: dynamics covariance $Q$

    """
    weights: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], ParameterProperties]
    bias: Union[Float[Array, "state_dim"], Float[Array, "ntime state_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "state_dim input_dim"], Float[Array, "ntime state_dim input_dim"], ParameterProperties]
    diff_coeff: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], ParameterProperties]
    diff_cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]
    
class ParamsCDLGSSMEmissions(NamedTuple):
    r"""Parameters of the emission distribution

    $$p(y_t \mid z_t, u_t) = \mathcal{N}(y_t \mid H z_t + D u_t + d, R)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: emission weights $H$
    :param bias: emission bias $d$
    :param input_weights: emission input weights $D$
    :param cov: emission covariance $R$

    """
    weights: Union[Float[Array, "emission_dim state_dim"], Float[Array, "ntime emission_dim state_dim"], ParameterProperties]
    bias: Union[Float[Array, "emission_dim"], Float[Array, "ntime emission_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "emission_dim input_dim"], Float[Array, "ntime emission_dim input_dim"], ParameterProperties]
    cov: Union[Float[Array, "emission_dim emission_dim"], Float[Array, "ntime emission_dim emission_dim"], Float[Array, "emission_dim_triu"], ParameterProperties]



class ParamsCDLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    """
    initial: ParamsCDLGSSMInitial
    dynamics: ParamsCDLGSSMDynamics
    emissions: ParamsCDLGSSMEmissions


class PosteriorCDLGSSMFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$

    """
    marginal_loglik: Union[Scalar, Float[Array, "ntime"]]
    filtered_means: Optional[Float[Array, "ntime state_dim"]] = None
    filtered_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None
    predicted_means: Optional[Float[Array, "ntime state_dim"]] = None
    predicted_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None


class PosteriorCDLGSSMSmoothed(NamedTuple):
    r"""Marginals of the Gaussian filtering and smoothing posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    :param smoothed_means: array of smoothed means $\mathbb{E}[z_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_covariances: array of smoothed marginal covariances, $\mathrm{Cov}[z_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_cross_covariances: array of smoothed cross products, $\mathbb{E}[z_t z_{t+1}^T \mid y_{1:T}, u_{1:T}]$

    """
    marginal_loglik: Scalar
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_covariances: Float[Array, "ntime state_dim state_dim"]
    smoothed_cross_covariances: Optional[Float[Array, "ntime_minus1 state_dim state_dim"]] = None


# Helper functions
# _get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
def _get_params(x, dim, t):
    if callable(x):
        return x(t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x
_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)


def diffeqsolve(
    rhs,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = dfx.Dopri5(),
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    dt0: float = 0.01,
) -> jnp.ndarray:
    return dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
    ).ys

def compute_pushforward(
    params: ParamsCDLGSSM,
    t0: Float,
    t1: Float,
) -> Tuple[Float[Array, "state_dim state_dim"], Float[Array, "state_dim state_dim"]]:

    # A and Q are computed based on Sarkka's thesis eq (3.135)
    state_dim = params.dynamics.weights.shape[0]
    A0 = jnp.eye(state_dim)
    Q0 = jnp.zeros((state_dim, state_dim))
    y0 = (A0, Q0)

    def rhs_all(t, y, args):
        A, Q = y

        # possibly time-dependent weights
        F_t = _get_params(params.dynamics.weights, 2, t)
        Qc_t = _get_params(params.dynamics.diff_cov, 2, t)
        L_t = _get_params(params.dynamics.diff_coeff, 2, t)

        dAdt = F_t @ A

        dQdt = F_t @ Q + Q @ F_t.T + L_t @ Qc_t @ L_t.T

        return (dAdt, dQdt)
    
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0)
    A, Q = sol[0][-1], sol[1][-1]
    
    # Original trick to pass discrete vs continuous tests, simply uncomment the below 2 lines.
    #A = params.dynamics.weights
    #Q = params.dynamics.diff_cov
    # Second trick to pass tests
    #jdb.breakpoint()
    #print('{:.32}'.format(A[0,0]))
    #print('{:.32}'.format(Q[0,0]))
    
    return A, Q

def make_cdlgssm_params(initial_mean,
                      initial_cov,
                      dynamics_weights,
                      dynamics_diffusion_coeff,
                      dynamics_diffusion_cov,
                      emissions_weights,
                      emissions_cov,
                      dynamics_bias=None,
                      dynamics_input_weights=None,
                      emissions_bias=None,
                      emissions_input_weights=None):
    """Helper function to construct a ParamsCDLGSSM object from arguments."""
    state_dim = len(initial_mean)
    emission_dim = emissions_cov.shape[-1]
    input_dim = max(dynamics_input_weights.shape[-1] if dynamics_input_weights is not None else 0,
                    emissions_input_weights.shape[-1] if emissions_input_weights is not None else 0)

    params = ParamsCDLGSSM(
        initial=ParamsCDLGSSMInitial(
            mean=initial_mean,
            cov=initial_cov
        ),
        dynamics=ParamsCDLGSSMDynamics(
            weights=dynamics_weights,
            bias=_zeros_if_none(dynamics_bias,state_dim),
            input_weights=_zeros_if_none(dynamics_input_weights, (state_dim, input_dim)),
            diff_coeff=dynamics_diffusion_coeff,
            diff_cov=dynamics_diffusion_cov,
        ),
        emissions=ParamsCDLGSSMEmissions(
            weights=emissions_weights,
            bias=_zeros_if_none(emissions_bias, emission_dim),
            input_weights=_zeros_if_none(emissions_input_weights, (emission_dim, input_dim)),
            cov=emissions_cov
        )
    )
    return params


def _predict(m, S, F, B, b, Q, u):
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

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    mu_pred = F @ m + B @ u + b
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, P, H, D, d, R, u, y):
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
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         H (D_obs,D_hid): emission matrix.
         D (D_obs,D_in): emission input weights.
         u (D_in,): inputs.
         d (D_obs,): emission bias.
         R (D_obs,D_obs): emission covariance matrix.
         y (D_obs,): observation.

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the Kalman gain
    S = R + H @ P @ H.T
    K = psd_solve(S, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - D @ u - d - H @ m)
    return mu_cond, symmetrize(Sigma_cond)


def preprocess_params_and_inputs(params, num_timesteps, inputs):
    """Preprocess parameters in case some are set to None."""

    # Make sure all the required parameters are there
    assert params.initial.mean is not None
    assert params.initial.cov is not None
    assert params.dynamics.weights is not None
    assert params.dynamics.diff_coeff is not None
    assert params.dynamics.diff_cov is not None
    assert params.emissions.weights is not None
    assert params.emissions.cov is not None

    # Get shapes
    emission_dim, state_dim = params.emissions.weights.shape[-2:]

    # Default the inputs to zero
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))
    input_dim = inputs.shape[-1]

    # Default other parameters to zero
    dynamics_input_weights = _zeros_if_none(params.dynamics.input_weights, (state_dim, input_dim))
    dynamics_bias = _zeros_if_none(params.dynamics.bias, (state_dim,))
    emissions_input_weights = _zeros_if_none(params.emissions.input_weights, (emission_dim, input_dim))
    emissions_bias = _zeros_if_none(params.emissions.bias, (emission_dim,))

    full_params = ParamsCDLGSSM(
        initial=ParamsCDLGSSMInitial(
            mean=params.initial.mean,
            cov=params.initial.cov),
        dynamics=ParamsCDLGSSMDynamics(
            weights=params.dynamics.weights,
            bias=dynamics_bias,
            input_weights=dynamics_input_weights,
            diff_coeff=params.dynamics.diff_coeff,
            diff_cov=params.dynamics.diff_cov),
        emissions=ParamsCDLGSSMEmissions(
            weights=params.emissions.weights,
            bias=emissions_bias,
            input_weights=emissions_input_weights,
            cov=params.emissions.cov)
        )
    return full_params, inputs


def preprocess_args(f):
    """Preprocess the parameter and input arguments in case some are set to None."""
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Extract the arguments by name
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments['params']
        emissions = bound_args.arguments['emissions']
        t_emissions = bound_args.arguments['t_emissions']
        inputs = bound_args.arguments['inputs']

        num_timesteps = len(emissions)
        full_params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

        return f(full_params, emissions, t_emissions, inputs=inputs)
    return wrapper

def cdlgssm_joint_sample(
    params: ParamsCDLGSSM,
    key: PRNGKey,
    num_timesteps: int,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
)-> Tuple[Float[Array, "num_timesteps state_dim"],
          Float[Array, "num_timesteps emission_dim"]]:
    r"""Sample from the joint distribution to produce state and emission trajectories.
    
    Args:
        params: model parameters
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional array of inputs.

    Returns:
        latent states and emissions

    """
    raise ValueError("Not implemented yet")

    params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

    def _sample_transition(key, F, B, b, Q, x_tm1, u):
        mean = F @ x_tm1 + B @ u + b
        return MVN(mean, Q).sample(seed=key)

    def _sample_emission(key, H, D, d, R, x, u):
        mean = H @ x + D @ u + d
        return MVN(mean, R).sample(seed=key)
    
    def _sample_initial(key, params, inputs):
        key1, key2 = jr.split(key)

        initial_state = MVN(params.initial.mean, params.initial.cov).sample(seed=key1)

        H0 = _get_params(params.emissions.weights, 2, 0)
        D0 = _get_params(params.emissions.input_weights, 2, 0)
        d0 = _get_params(params.emissions.bias, 1, 0)
        R0 = _get_params(params.emissions.cov, 2, 0)
        u0 = tree_map(lambda x: x[0], inputs)

        initial_emission = _sample_emission(key2, H0, D0, d0, R0, initial_state, u0)
        return initial_state, initial_emission

    def _step(prev_state, args):
        key, t0, t1, inpt = args
        key1, key2 = jr.split(key, 2)

        # Shorthand: get parameters and inputs for time index t
        B = _get_params(params.dynamics.input_weights, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        F, Q = compute_pushforward(params, t0, t1)
        H = _get_params(params.emissions.weights, 2, t)
        D = _get_params(params.emissions.input_weights, 2, t)
        d = _get_params(params.emissions.bias, 1, t)
        R = _get_params(params.emissions.cov, 2, t)

        # Sample from transition and emission distributions
        state = _sample_transition(key1, F, B, b, Q, prev_state, inpt)
        emission = _sample_emission(key2, H, D, d, R, state, inpt)

        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)
    
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Sample the remaining emissions and states
    next_keys = jr.split(key2, num_timesteps - 1)
    
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
        t1 = tree_map(lambda x: x[1:,0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps-1)
        t1 = jnp.arange(1,num_timesteps)
    
    t0_idx = jnp.arange(num_timesteps-1)

    # next_times = jnp.arange(1, num_timesteps)
    
    next_inputs = tree_map(lambda x: x[1:], inputs)
    _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, t0, t1, t0_idx, next_inputs))

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions

@preprocess_args
def cdlgssm_filter(
    params: ParamsCDLGSSM,
    emissions:  Float[Array, "num_timesteps emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
) -> PosteriorCDLGSSMFiltered:
    r"""Run a Continuous Discrete Kalman filter to produce the marginal likelihood and filtered state estimates.

    Args:
        params: model parameters
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional array of inputs.

    Returns:
        PosteriorCDLGSSMFiltered: filtered posterior object

    """
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[:,0], t_emissions)
        t1 = tree_map(
                lambda x: jnp.concatenate(
                    (
                        t_emissions[1:,0],
                        jnp.array([t_emissions[-1,0]+1]) # NB: t_{N+1} is simply t_{N}+1 
                    )
                ),
                t_emissions
            )
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps)
        t1 = jnp.arange(1,num_timesteps+1)
    
    t0_idx = jnp.arange(num_timesteps)

    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    def _step(carry, args):
        ll, pred_mean, pred_cov = carry
        t0, t1, t0_idx = args

        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        H = _get_params(params.emissions.weights, 2, t0)
        D = _get_params(params.emissions.input_weights, 2, t0)
        d = _get_params(params.emissions.bias, 1, t0)
        R = _get_params(params.emissions.cov, 2, t0)
        u = inputs[t0_idx]
        y = emissions[t0_idx]

        # Update the log likelihood
        ll += MVN(H @ pred_mean + D @ u + d, H @ pred_cov @ H.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # Predict the next state
        F, Q = compute_pushforward(params, t0, t1)
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params.initial.mean, params.initial.cov)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, (t0, t1, t0_idx))
    return PosteriorCDLGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


@preprocess_args
def cdlgssm_smoother(
    params: ParamsCDLGSSM,
    emissions: Float[Array, "num_timesteps emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
) -> PosteriorCDLGSSMSmoothed:
    r"""Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        params: an CDLGSSMParams instance (or object with the same fields)
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: array of inputs.

    Returns:
        PosteriorCDLGSSMSmoothed: smoothed posterior object.

    """
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps-1 \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
        t1 = tree_map(lambda x: x[1:,0], t_emissions)
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps-1)
        t1 = jnp.arange(1,num_timesteps)
    
    t0_idx = jnp.arange(num_timesteps-1)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = cdlgssm_filter(params, emissions, t_emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior
    
    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        F, Q = compute_pushforward(params, t0, t1)
        # TODO: when smoothing, shall we use t0 or t1 for inputs?
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        u = inputs[t0_idx]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        # TODO: make sure that computation of G is correct in CD-Kalman Smoother case
        G = psd_solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - F @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ G.T

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    
    args = (
        t0[::-1], t1[::-1],
        t0_idx[::-1],
        filtered_means[:-1][::-1], filtered_covs[:-1][::-1]
    )
    _, (smoothed_means, smoothed_covs, smoothed_cross) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    smoothed_cross = smoothed_cross[::-1]
    return PosteriorCDLGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        smoothed_cross_covariances=smoothed_cross,
    )

def cdlgssm_posterior_sample(
    key: PRNGKey,
    params: ParamsCDLGSSM,
    emissions:  Float[Array, "num_timesteps emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
    jitter: Optional[Scalar]=0
    
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples from $p(z_{1:T} \mid y_{1:T}, u_{1:T})$.

    Args:
        key: random number key.
        params: parameters.
        emissions: sequence of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional sequence of inptus.
        jitter: padding to add to the diagonal of the covariance matrix before sampling.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.
    """
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps-1 \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
        t1 = tree_map(lambda x: x[1:,0], t_emissions)
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps-1)
        t1 = jnp.arange(1,num_timesteps)
    
    t0_idx = jnp.arange(num_timesteps-1)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = cdlgssm_filter(params, emissions, t_emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Sample backward in time
    def _step(carry, args):
        next_state = carry
        key, t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        F, Q = compute_pushforward(params, t0, t1)
        # TODO: when smoothing, shall we use t0 or t1 for inputs?
        B = _get_params(params.dynamics.input_weights, 2, t0)
        b = _get_params(params.dynamics.bias, 1, t0)
        u = inputs[t0_idx]
        
        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, F, B, b, Q, u, next_state)
        smoothed_cov = smoothed_cov + jnp.eye(smoothed_cov.shape[-1]) * jitter
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)
    
    args = (
        jr.split(key, num_timesteps - 1),
        t0[::-1], t1[::-1], # jnp.arange(num_timesteps - 2, -1, -1),
        t0_idx[::-1],
        filtered_means[:-1][::-1], filtered_covs[:-1][::-1],
    )
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return states