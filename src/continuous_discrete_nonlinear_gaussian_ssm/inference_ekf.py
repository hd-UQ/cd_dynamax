import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jacfwd,jacrev
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd

from jaxtyping import Array, Float
from typing import NamedTuple, List, Optional

from jax.tree_util import tree_map
from dynamax.utils.utils import psd_solve, symmetrize
from dynamax.types import PRNGKey

import jax.debug as jdb
from pdb import set_trace as bp

# Dynamax shared code
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

# Our codebase
# CDNLGSSM param and function definition
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import *
# Diffrax based diff-eq solver
from utils.diffrax_utils import diffeqsolve
from utils.debug_utils import lax_scan
DEBUG = False

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x

class EKFHyperParams(NamedTuple):
    """Lightweight container for EKF hyperparameters.

    """

    dt_final: float = 1e-10 # Small dt_final for predicted mean and covariance at the end of sequence 
    state_order: str = 'second'
    emission_order: str = 'first'
    smooth_order: str = 'first'
    cov_rescaling: float = 1.0
    
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

    # Predicted mean and covariance evolution, by using the EKF state order approximations
    def rhs_all(t, y, args):
        if hyperparams.state_order=='zeroth':
            m, = y
        else:
            m, P = y

        # TODO: possibly time- and parameter-dependent functions
        f=params.dynamics.drift.f

        # Get time-varying parameters
        Qc_t = params.dynamics.diffusion_cov.f(None,u,t)
        L_t = params.dynamics.diffusion_coefficient.f(None,u,t)
        # Get time-varying parameters
        # Qc_t = _get_params(params.dynamics.diffusion_cov, 2, t)
        # L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t)

        # following Sarkka thesis eq. 3.158

        # Evaluate the jacobian of the dynamics function at m and inputs
        F_t = jacfwd(f)(m,u,t)
        
        if hyperparams.state_order=='zeroth':
            # Mean evolution
            dmdt = f(m, u, t)

        elif hyperparams.state_order=='first':
            # Mean evolution
            dmdt = f(m, u,t)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T

        # follow Sarkka thesis eq. 3.159
        elif hyperparams.state_order=='second':
            # Evaluate the Hessian of the dynamics function at m and inputs
            # Based on these recommendationshttps://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev
            H_t=jacfwd(jacrev(f))(m,u,t)

            # Mean evolution
            dmdt = f(m,u,t) + 0.5*jnp.trace(H_t @ P)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        else:
            raise ValueError('EKF hyperparams.state_order = {} not implemented yet'.format(hyperparams.state_order))

        if hyperparams.state_order=='zeroth':
            return (dmdt, )
        else:
            return (dmdt, dPdt)

    # Zero-th approach, only mean is pushed via RHS ODE
    if hyperparams.state_order=='zeroth':
        # Initialize
        y0 = (m,)

        # Compute predicted mean
        sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0)
        m_final = sol[0][-1]

        # Predicted covariance
        dt = t1 - t0
        Qc_t = params.dynamics.diffusion_cov.f(None,u,t0)
        L_t = params.dynamics.diffusion_coefficient.f(None, u, t0) * hyperparams.cov_rescaling
        P_final = P + jnp.sqrt(dt) * L_t @ Qc_t @ L_t.T
    # Otherwise, both mean and covariance pushed via RHS ODE
    else:
        # Initialize
        y0 = (m, P)
        # Compute predicted mean and covariance
        sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0)
        m_final = sol[0][-1]
        P_final = sol[1][-1]

    return m_final, P_final

# Condition on observations for EKF
# Based on first order approximation, as in Equation 3.59
# TODO: implement second order EKF, as in Equation 3.63
def _condition_on(m, P, h, H, R, u, y, t, num_iter, hyperparams):
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
         t () : time instant to condition on
         num_iter (int): number of re-linearizations around posterior for update step.

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u, t)
        S = R + H_x @ prior_cov @ H_x.T
        # if not jnp.all(jnp.linalg.eigvals(S) > 0):
        #     print(f"Condition number of S: {jnp.linalg.cond(S)}")
        #     print(f"Most negative eigenvalue of S: {jnp.min(jnp.linalg.eigvals(S))}")
            # bp()
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u, t))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax_scan(_step, carry, jnp.arange(num_iter), debug=DEBUG)
    return mu_cond, symmetrize(Sigma_cond)


def extended_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    hyperparams: EKFHyperParams = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    num_iter: int = 1,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.
        Two implementations are available,
        based on first- and second-order approximations
            i.e. Algorithms 3.21 and 3.22 in Sarkka's thesis

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
                        jnp.array([t_emissions[-1,0]+hyperparams.dt_final]) # NB: t_{N+1} is simply t_{N}+dt_final
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
    h = params.emissions.emission_function.f
    # First order EKF update implemented for now
    # TODO: consider second-order EKF updates
    H = jacfwd(h)
    # h, H = (_process_fn(fn, inputs) for fn in (h, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        ll, pred_mean, pred_cov = carry
        t0, t1, t0_idx = args
        # print(f"t0: {t0}, t1: {t1}, t0_idx: {t0_idx}")

        # if pred_cov is not SPD, breakpoint
        # evals_pred_cov = jnp.linalg.eigvals(pred_cov)
        # if not jnp.all(evals_pred_cov > 0):
        #     print(f"pred_cov is not SPD. Most negative eigenvalue: {jnp.min(evals_pred_cov)} at t0: {t0}")
        #     # bp()

        # TODO:
        # Get parameters and inputs for time t0
        # Q = _get_params(params.dynamics.diffusion_cov, 2, t0)
        # R = _get_params(params.emissions.emission_cov, 2, t0)
        u = inputs[t0_idx]
        y = emissions[t0_idx]
        Q = params.dynamics.diffusion_cov.f(None,u,t0)
        R = params.emissions.emission_cov.f(None,u,t0)

        # Update the log likelihood
        # According to first order EKF update
        # TODO: incorporate second order EKF updates!
        H_x = H(pred_mean, u, t0)
        ll += MVN(h(pred_mean, u, t0), H_x @ pred_cov @ H_x.T + R).log_prob(jnp.atleast_1d(y))

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, h, H, R, u, y, t0, num_iter, hyperparams)

        # print condition number of filtered_cov
        # print(f"Condition number of filtered_cov: {jnp.linalg.cond(filtered_cov)}")

        # if filtered_cov is not SPD, breakpoint
        # evals_filtered_cov = jnp.linalg.eigvals(filtered_cov)
        # if not jnp.all(evals_filtered_cov > 0):
        #     print(f"filtered_cov is not SPD. Most negative eigenvalue: {jnp.min(evals_filtered_cov)} at t0: {t0}")
        # bp()

        # Predict the next state based on EKF approximations
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, params, t0, t1, u, hyperparams)

        # print condition number of pred_cov
        # print(f"Condition number of pred_cov: {jnp.linalg.cond(pred_cov)}")

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
    carry = (0.0, params.initial.mean.f(), params.initial.cov.f())
    (ll, *_), outputs = lax_scan(_step, carry, (t0, t1, t0_idx), debug=DEBUG)
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def iterated_extended_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    hyperparams: EKFHyperParams = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    num_iter: int = 2,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
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
    filtered_posterior = extended_kalman_filter(
        params,
        emissions,
        t_emissions,
        hyperparams,
        inputs,
        num_iter,
        output_fields
    )
    return filtered_posterior

def _smooth(
    m_filter, P_filter, # Filtered mean and covariance
    m_smooth, P_smooth, # Smoothed mean and covariance
    params: ParamsCDNLGSSM,  # All necessary CD dynamic params
    t0: Float,
    t1: Float,
    u,
    hyperparams
    ):
    r"""smooth the next mean and covariance using EKF smoothing equations
        where the evolution of m and P are computed based on
            First order smoothing approximation
                as in Algorithm 3.23 in Sarkka

    Args:
        m_filter (D_hid,): filtered mean at t1.
        P_filter (D_hid,D_hid): filtered covariance at t1.
        m_smooth (D_hid,): smooth mean at t1.
        P_smooth (D_hid,D_hid): smoothed covariance at t1.
        params: parameters of CD nonlinear dynamics, containing dynamics RHS function, coeff matrix and Brownian covariance matrix.
        t0: initial time-instant
        t1: final time-instant
        u (D_in,): inputs.
        hyperparams: EKF hyperparameters

    Returns:
        mu_smooth (D_hid,): smoothed mean at t0.
        Sigma_smooth (D_hid,D_hid): smoothed covariance at t0.
    """
    
    # Initialize
    y0 = (m_smooth, P_smooth)
    # Smoothed mean and covariance evolution
    # by using the EKF state order approximations
    def rhs_all(t, y, args):
        m_smooth, P_smooth = y
        m_filter, P_filter = args
        
        # TODO: possibly time- and parameter-dependent functions
        f=params.dynamics.drift.f

        # Get time-varying parameters
        Qc_t = params.dynamics.diffusion_cov.f(None,u,t)
        L_t = params.dynamics.diffusion_coefficient.f(None,u,t)
        #Qc_t = _get_params(params.dynamics.diffusion_cov, 2, t)
        #L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t)

        # following Sarkka thesis eq. 3.163
        if hyperparams.smooth_order=='first':
            # Evaluate the jacobian of the dynamics function at m and inputs
            F_t=jacfwd(f)(m_filter,u,t)
            
            '''
            # Direct implementation of Equations 3.163
            
            # Auxiliary matrix, used in both mean and covariance
            # Inverse product computed via psd_solve
            aux_matrix=psd_solve(P_filter, (P_filter @ F_t.T + L_t @ Qc_t @ L_t.T))

            # Mean evolution
            dmsmoothdt = f(m_filter,u,t) + aux_matrix.T @ (m_smooth-m_filter)
            # Covariance evolution
            dPsmoothdt = aux_matrix.T @ P_smooth + P_smooth @ aux_matrix - L_t @ Qc_t @ L_t.T
            '''
            
            # Revised implementation,
            # where we avoid numerical P @ P^{-1} computations
            
            # Auxiliary matrix, used in both mean and covariance
            # Inverse product computed via psd_solve
            aux_matrix=psd_solve(P_filter, L_t @ Qc_t @ L_t.T).T

            # Mean evolution
            dmsmoothdt = f(m_filter,u,t) + (F_t + aux_matrix) @ (m_smooth-m_filter)
            # Covariance evolution
            dPsmoothdt = (F_t + aux_matrix) @ P_smooth + P_smooth @ (F_t + aux_matrix).T - L_t @ Qc_t @ L_t.T
            
        else:
            raise ValueError('EKF hyperparams.smooth_order = {} not implemented yet'.format(hyperparams.smooth_order))

        return (dmsmoothdt, dPsmoothdt)

    # Recall that we solve the rhs in reverse:
    # from t1 to t0, BUT y0 contains initial conditions at t1
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0, reverse=True, args=(m_filter, P_filter))
    return sol[0][-1], sol[1][-1]

def extended_kalman_smoother(
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    hyperparams: EKFHyperParams = EKFHyperParams(),
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman smoother,
        as described in Algorithm 3.23 in Sarkka's thesis

    Args:
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        filtered_posterior: optional output from filtering step.
        hyperparams: hyper-parameters of the EKF, related to the approximation order
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

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

    # Get filtered EKF posterior
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(
            params,
            emissions,
            t_emissions=t_emissions,
            hyperparams=hyperparams,
            inputs=inputs
        )
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Process inputs
    inputs = _process_input(inputs, num_timesteps)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # Smooth mean and covariance based on continuous-time solution
        smoothed_mean, smoothed_cov = _smooth(
            m_filter=filtered_mean, P_filter=filtered_cov, # Filtered 
            m_smooth=smoothed_mean_next, P_smooth=smoothed_cov_next, # Smoothed 
            params=params,
            t0=t0,t1=t1,
            u = inputs[t0_idx],
            hyperparams = hyperparams,
        )

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

def iterated_extended_kalman_smoother(
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    hyperparams: EKFHyperParams = EKFHyperParams(),
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
        hyperparams: EKFHyperParams = EKFHyperParams(),
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """

    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = extended_kalman_smoother(
            params,
            emissions,
            hyperparams,
            t_emissions,
            smoothed_prior,
            inputs
        )
        return smoothed_posterior, None

    # TODO: replace single call with _step
    print("WARNING: We are not running iterated_EKF, until we figure out how to scan over _steps with different input-output carry variables")
    smoothed_posterior = extended_kalman_smoother(
            params,
            emissions,
            hyperparams,
            t_emissions,
            None,
            inputs
        )
    
    # However, this does not run, because
    # smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    # " the input carry carry is a <class 'NoneType'> but the corresponding component of the carry output is a <class 'dynamax.linear_gaussian_ssm.inference.PosteriorGSSMSmoothed'>, so their Python types differ"
    # i.e., can have None as first smoothed_prior
    
    return smoothed_posterior

def extended_kalman_posterior_sample(
    key: PRNGKey,
    params: ParamsCDNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples.

    Args:
        key: random number key.
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional array of inputs

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
                (t_emissions[1:, 0], jnp.array([t_emissions[-1, 0] + hyperparams.dt_final]))  # NB: t_{N+1} is simply t_{N}+dt_final
            ),
            t_emissions,
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
        # Q = _get_params(params.dynamics.diffusion_cov, 2, t0)
        u = inputs[t0_idx]
        Q = params.dynamics.diffusion_cov.f(None,u,t0)

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, f, F, Q, u, next_state, t0, 1)
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

def forecast_extended_kalman_filter(
    params: ParamsCDNLGSSM,
    init_forecast: tfd.Distribution,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    hyperparams: EKFHyperParams = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=[
        "forecasted_state_means",
        "forecasted_state_covariances",
        "forecasted_emission_means",
        "forecasted_emission_covariances",
    ],
) -> GSSMForecast:
    r"""Run an extended Kalman filter to forecast state and emissions.
        Two implementations are available,
        based on first- and second-order approximations
            i.e. Algorithms 3.21 and 3.22 in Sarkka's thesis

    Args:
        params: model parameters.
        init_forecast: initial distribution to forecast with.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        hyperparams: hyper-parameters of the EKF, related to the approximation order
        inputs: optional array of inputs.
        output_fields: list of fields to return 

    Returns:
        post: forecast object.

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
                        jnp.array([t_emissions[-1,0]+hyperparams.dt_final]) # NB: t_{N+1} is simply t_{N}+dt_final
                    )
                ),
                t_emissions
            )
    else:
        raise ValueError("t_emissions must be provided for forecasting")

    t0_idx = jnp.arange(num_timesteps)

    # Only emission function
    h = params.emissions.emission_function.f
    # First order EKF update implemented for now
    # TODO: consider second-order EKF updates
    H = jacfwd(h)
    # h, H = (_process_fn(fn, inputs) for fn in (h, H))
    inputs = _process_input(inputs, num_timesteps)
    
    def _step(carry, args):
        current_state_mean, current_state_cov = carry
        t0, t1, t0_idx = args
        # print(f"t0: {t0}, t1: {t1}, t0_idx: {t0_idx}")

        # TODO:
        # Get parameters and inputs for time t0
        # R = _get_params(params.emissions.emission_cov, 2, t0)
        u = inputs[t0_idx]
        R = params.emissions.emission_cov.f(None,u,t0)

        # Predict the next state based on EKF approximations
        pred_state_mean, pred_state_cov = _predict(current_state_mean, current_state_cov, params, t0, t1, u, hyperparams)

        # Corresponding emissions
        # According to first order EKF update
        # TODO: incorporate second order EKF updates!
        H_x = H(pred_state_mean, u, t0)
        pred_emission_mean = h(pred_state_mean, u, t0)
        pred_emission_cov = H_x @ pred_state_cov @ H_x.T + R
        
        # Build carry and output states
        carry = (pred_state_mean, pred_state_cov)
        outputs = {
            "forecasted_state_means": pred_state_mean,
            "forecasted_state_covariances": pred_state_cov,
            "forecasted_emission_means": pred_emission_mean,
            "forecasted_emission_covariances": pred_emission_cov,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Initialize the state, based on provided initial distribution's mean and covariance
    carry = (init_forecast.mean(), init_forecast.covariance())
    # Run the extended Kalman filter
    _, outputs = lax_scan(_step, carry, (t0, t1, t0_idx), debug=DEBUG)
    forecast = GSSMForecast(
        **outputs,
    )
    return forecast