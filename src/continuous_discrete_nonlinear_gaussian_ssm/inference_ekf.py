import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jacfwd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import NamedTuple, List, Optional

from jax.tree_util import tree_map
from dynamax.utils.utils import psd_solve, symmetrize
from dynamax.types import PRNGKey

# Dynamax shared code
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions, PosteriorGSSMFiltered, PosteriorGSSMSmoothed

# Our codebase
from continuous_discrete_nonlinear_gaussian_ssm.models import ParamsCDNLGSSM
from cdssm_utils import diffeqsolve

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x

class EKFHyperParams(NamedTuple):
    """Lightweight container for EKF hyperparameters.

    """

    state_order: str = 'second'
    emission_order: str = 'first'


def _predict(
    m, P, # Current mean and covariance
    params: ParamsCDNLGSSM,  # All necessary CD dynamic params
    t0: Float,
    t1: Float,
    u,
    hyperparams
    ):
    r"""Predict next mean and covariance using EKF equations
        p(z_{t+1}) = N(z_{t+1} | m_{t+1}, P_{t+1})
        
        where the evolution of m and P are computed based on
            First order approximation to model SDE as in Equation 3.158
            Second order approximation to model SDE as in Equation 3.159

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        params: parameters of CD nonlinear dynamics, containing dynamics RHS function, coeff matrix and Brownian covariance matrix.
        t0: initial time-instant
        t1: final time-instant
        u (D_in,): inputs.
        hyperparams: EKF hyperparameters

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    
    # Initialize
    y0 = (m, P)
    # Predicted mean and covariance evolution, by using the EKF state order approximations
    def rhs_all(t, y, args):
        x, P = y
        
        # possibly time-dependent functions
        # TODO: figure out how to use get_params functionality for time-varying functions
        #f_t = _get_params(params.dynamics_function, 2, t)
        f_t = params.dynamics_function
        Qc_t = _get_params(params.dynamics_covariance, 2, t)
        L_t = _get_params(params.dynamics_coefficients, 2, t)

       
        # following Sarkka thesis eq. 3.158
        if hyperparams.state_order=='first':
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t=params.dynamics_function_jacobian(x,u)
            
            # Mean evolution
            # TODO: figure out how to vectorize via vmap
            dxdt = f_t(x, u)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        
        # follow Sarkka thesis eq. 3.159
        elif hyperparams.state_order=='second':
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t=params.dynamics_function_jacobian(x,u)
            # Evaluate the Hessian of the dynamics function at x and inputs
            H_t=params.dynamics_function_hessian(x,u)
        
            # Mean evolution
            # TODO: figure out how to vectorize via vmap
            dxdt = f_t(x, u) + 0.5*jnp.trace(H_t @ P)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        else:
            raise ValueError('params.dynamics_approx = {} not implemented yet'.format(params.dynamics_approx))

        return (dxdt, dPdt)
    
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0)
    return sol[0][-1], sol[1][-1]
    
# Condition on observations for EKF
# Based on first order approximation, as in Equation 3.59
# TODO: implement second order EKF, as in Equation 3.63
def _condition_on(m, P, h, H, R, u, y, num_iter, hyperparams):
    r"""Condition a Gaussian potential on a new observation.

       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u)
        S = R + H_x @ prior_cov @ H_x.T
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, symmetrize(Sigma_cond)


def extended_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    num_iter: int = 1,
    hyperparams: EKFHyperParams = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        num_iter: number of linearizations around posterior for update step (default 1).
        hyperparams: hyper-parameters of the EKF, related to the approximation order
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

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
    
    # Only emission function
    h = params.emission_function
    # First order EKF update implemented for now
    # TODO: consider second-order EKF updates
    H = jacfwd(h)
    h, H = (_process_fn(fn, inputs) for fn in (h, H))
    inputs = _process_input(inputs, num_timesteps)
    
    def _step(carry, args):
        ll, pred_mean, pred_cov = carry
        t0, t1, t0_idx = args

        # TODO:
        # Get parameters and inputs for time t0
        Q = _get_params(params.dynamics_covariance, 2, t0)
        R = _get_params(params.emission_covariance, 2, t0)
        u = inputs[t0_idx]
        y = emissions[t0_idx]

        # Update the log likelihood
        # According to first order EKF update
        # TODO: incorporate second order EKF updates!
        H_x = H(pred_mean, u)
        ll += MVN(h(pred_mean, u), H_x @ pred_cov @ H_x.T + R).log_prob(jnp.atleast_1d(y))

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, h, H, R, u, y, num_iter, hyperparams)

        # Predict the next state based on EKF approximations
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, params, t0, t1, u, hyperparams)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": pred_mean,
            "predicted_covariances": pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, (t0, t1, t0_idx))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def iterated_extended_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    num_iter: int = 2,
    hyperparams: EKFHyperParams = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMFiltered:
    r"""Run an iterated extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        num_iter: number of linearizations around posterior for update step (default 2).
        hyperparams: hyper-parameters of the EKF, related to the approximation order
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    filtered_posterior = extended_kalman_filter(params, emissions, t_emissions, num_iter, hyperparams, inputs)
    return filtered_posterior


def extended_kalman_smoother(
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

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

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(params, emissions, t_emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # TODO: use compute_push_forward instead, and let autodiff do the magic? How about time-instants?
    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # TODO:
        # Get parameters and inputs for time t0
        Q = _get_params(params.dynamics_covariance, 2, t0)
        R = _get_params(params.emission_covariance, 2, t0)
        u = inputs[t0_idx]
        F_x = F(filtered_mean, u)

        # Prediction step
        m_pred = f(filtered_mean, u)
        S_pred = Q + F_x @ filtered_cov @ F_x.T
        G = psd_solve(S_pred, F_x @ filtered_cov).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the extended Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (
        t0[::-1], t1[::-1],
        t0_idx[::-1],
        filtered_means[:-1][::-1], filtered_covs[:-1][::-1]
    )
    _, (smoothed_means, smoothed_covs) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )


def extended_kalman_posterior_sample(
    key: PRNGKey,
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples.

    Args:
        key: random number key.
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional array of inputs.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.
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

    # Get filtered posterior
    filtered_posterior = extended_kalman_filter(params, emissions, t_emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # TODO: use compute_push_forward instead, and let autodiff do the magic? how do we incorporate time though?
    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        key, t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # TODO:
        # Get parameters and inputs for time t0
        Q = _get_params(params.dynamics_covariance, 2, t0)
        u = inputs[t0_idx]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, f, F, Q, u, next_state, 1)
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


def iterated_extended_kalman_smoother(
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an iterated extended Kalman smoother (IEKS).

    Args:
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """

    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = extended_kalman_smoother(params, emissions, t_emissions, smoothed_prior, inputs)
        return smoothed_posterior, None

    smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    return smoothed_posterior
