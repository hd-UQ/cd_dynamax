import jax.numpy as jnp
from jax import lax
from jax import vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, List

from jax.tree_util import tree_map
from dynamax.utils.utils import psd_solve

# Dynamax shared code
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

# Our codebase
# CDNLGSSM param and function definition
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import *
# Diffrax based diff-eq solver
from utils.diffrax_utils import diffeqsolve
from utils.debug_utils import lax_scan
DEBUG = False

# We redefe UKFHyperParams, due to dt_final
# from dynamax.nonlinear_gaussian_ssm.inference_ukf import UKFHyperParams
class UKFHyperParams(NamedTuple):
    """Lightweight container for UKF hyperparameters.

    Default values taken from https://github.com/sbitzer/UKF-exposed
    """
    dt_final: float = 1e-10 # Small dt_final for predicted mean and covariance at the end of sequence 
    alpha: float = jnp.sqrt(3)
    beta: int = 2
    kappa: int = 1
    diffeqsolve_settings: dict = {}


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_compute_lambda = lambda x, y, z: x**2 * (y + z) - z


def _compute_sigmas(m, P, n, lamb):
    """Compute (2n+1) sigma points used for inputs to  unscented transform.
        These are independent of Continuous or Discrete solutions
    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        n (int): number of state dimensions.
        lamb (Scalar): unscented parameter lambda.

    Returns:
        sigmas (2*D_hid+1,): 2n+1 sigma points.
    """
    distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
    sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
    sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))


def _compute_weights(n, alpha, beta, lamb):
    """Compute weights used to compute predicted mean and covariance.
        These are different from Continuous and Discrete solutions
        We here use 
    Args:
        n (int): number of state dimensions.
        alpha (float): hyperparameter that determines the spread of sigma points
        beta (float): hyperparameter that incorporates prior information
        lamb (float): lamb = alpha**2 *(n + kappa) - n

    Returns:
        w_mean (2*n+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*n+1,): 2n+1 weights to compute predicted covariance.
        W_matrix (2*n+1,2*n+1): matrix of weights defined by combining w_mean and w_cov as in eq. 3.82 of Saarka's Thesis.
    """

    # These follow eq. 3.69-3.70 in Sarkka's thesis
    factor = 1 / (2 * (n + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
    w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - alpha**2 + beta)]), jnp.ones(2 * n) * factor))

    # These follow eq. 3.81-3.82 in Sarkka's thesis
    # W =  (I - [w_mean , \dots, w_mean]) diag(w_cov) (I - [w_mean , \dots, w_mean])^T
    I_w = jnp.eye(2 * n + 1) - w_mean[:, None]
    W_matrix = I_w @ jnp.diag(w_cov) @ I_w.T

    return w_mean, w_cov, W_matrix


# TODO: Revise and implement push-forward here
def _predict(
    m,
    P,  # priors
    params: ParamsCDNLGSSM,  # All necessary CD dynamic params
    t0: Float,
    t1: Float,
    lamb,
    w_mean,
    w_cov,
    W_matrix,
    u,
    hyperparams
):
    """Predict next mean and covariance using additive UKF

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        params: parameters of CD nonlinear dynamics, containing dynamics RHS function, coeff matrix and Brownian covariance matrix.
        t0: initial time-instant
        t1: final time-instant
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        W_matrix (2*D_hid+1,2*D_hid+1): matrix of weights defined by combining w_mean and w_cov as in eq. 3.82 of Saarka's Thesis.
        u (D_in,): inputs.

    Returns:
        m_pred (D_hid,): predicted mean.
        P_pred (D_hid,D_hid): predicted covariance.

    """
    n = len(m)

    # Sarkka Thesis's algo 3.24
    # weights are defined in eq. 3.69;
    # the related weight vector w_m and matrix W are defined in eq 3.81-3.82;
    def rhs_all(t, y, args):
        # TODO: reconsider when we want time-varying dynamics functions
        f = params.dynamics.drift.f
        Qc_t = params.dynamics.diffusion_cov.f(None, u, t)
        L_t = params.dynamics.diffusion_coefficient.f(None, u, t)

        # create sigma points X_t
        m_t, P_t = y
        X_t = _compute_sigmas(m_t, P_t, n, lamb)

        # TODO: add controls u
        # f_X_t = vmap(f_t, (0, 0), 0)(X_t, u)
        f_X_t = vmap(f, in_axes=(0, None, None))(X_t, u, t)
        # dimensions of f_X_t are (2*state_dim+1, state_dim)

        # dmdt = f_X_t  w_mean
        dmdt = f_X_t.T @ w_mean

        # dPdt = f_x W X^T + X W f_x^T + L Qc L^T (Transposes flipped due to shape of f_X_t and X_t)
        foo = f_X_t.T @ W_matrix @ X_t
        dPdt = foo + foo.T + L_t @ Qc_t @ L_t.T

        return (dmdt, dPdt)

    # solve Saarka's ODE 3.183 in thesis
    y0 = (m, P)
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0, **hyperparams.diffeqsolve_settings)
    m_pred, P_pred = sol[0][-1], sol[1][-1]
    # According to Sarkka's algo 3.24, we only need to return m_pred and P_pred (not P_cross) in continuous-discrete
    return m_pred, P_pred


def _condition_on(m, P, h, R, lamb, w_mean, w_cov, u, y, t):
    """Condition a Gaussian potential on a new observation

    Duplicate of the _condition_on function in discrete case of inference_ukf.py

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        h (Callable): emission function.
        R (D_obs,D_obs): emssion covariance matrix
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        u (D_in,): inputs.
        y (D_obs,): observation.black
        t (): time-instant of conditioning

    Returns:
        ll (float): log-likelihood of observation
        m_cond (D_hid,): filtered mean.
        P_cond (D_hid,D_hid): filtered covariance.

    """
    n = len(m)
    # Form sigma points and propagate
    sigmas_cond = _compute_sigmas(m, P, n, lamb)
    u_s = jnp.array([u] * len(sigmas_cond))
    sigmas_cond_prop = vmap(h, in_axes=(0, None, None))(sigmas_cond, u_s, t)

    # Compute parameters needed to filter
    pred_mean = jnp.tensordot(w_mean, sigmas_cond_prop, axes=1)
    pred_cov = jnp.tensordot(w_cov, _outer(sigmas_cond_prop - pred_mean, sigmas_cond_prop - pred_mean), axes=1) + R
    pred_cross = jnp.tensordot(w_cov, _outer(sigmas_cond - m, sigmas_cond_prop - pred_mean), axes=1)

    # Compute log-likelihood of observation
    ll = MVN(pred_mean, pred_cov).log_prob(y)

    # Compute filtered mean and covariace
    K = psd_solve(pred_cov, pred_cross.T).T  # Filter gain
    m_cond = m + K @ (y - pred_mean)
    P_cond = P - K @ pred_cov @ K.T
    return ll, m_cond, P_cond


def unscented_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    hyperparams: UKFHyperParams = UKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "filtered_means",
        "filtered_covariances",
        "predicted_means",
        "predicted_covariances",
    ],
) -> PosteriorGSSMFiltered:
    """Run a unscented Kalman filter to produce the marginal likelihood and
    filtered state estimates.

    Args:
        params: model parameters.
        emissions: array of observations.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        hyperparams: hyper-parameters.
        inputs: optional array of inputs.
        output_fields: list of fields to include in the output.

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
                (t_emissions[1:, 0], jnp.array([t_emissions[-1, 0] + hyperparams.dt_final]))  # NB: t_{N+1} is simply t_{N}+dt_final
            ),
            t_emissions,
        )
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps)
        t1 = jnp.arange(1, num_timesteps + 1)

    t0_idx = jnp.arange(num_timesteps)

    state_dim = params.dynamics.diffusion_cov.params.shape[0]

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)

    # Only emission function
    h = params.emissions.emission_function.f
    # h = _process_fn(h, inputs)
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        ll, pred_mean, pred_cov = carry
        t0, t1, t0_idx = args

        # Get parameters and inputs for time t0
        u = inputs[t0_idx]
        y = emissions[t0_idx]
        R = params.emissions.emission_cov.f(None, u, t0)

        # Condition on this emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, h, R, lamb, w_mean, w_cov, u, y, t0
        )

        # Update the log likelihood
        ll += log_likelihood

        # Predict the next state, based on UKF predict
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, params, t0, t1, lamb, w_mean, w_cov, W_matrix, u, hyperparams)

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

    # Run the Unscented Kalman Filter
    carry = (0.0, params.initial.mean.f(), params.initial.cov.f())
    (ll, *_), outputs = lax.scan(_step, carry, (t0, t1, t0_idx))
    # for i in range(num_timesteps):
    #     carry, outputs = _step(carry, (t0[i], t1[i], t0_idx[i]))
    # ll, _, _ = carry

    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def unscented_kalman_smoother(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    hyperparams: UKFHyperParams = UKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
) -> PosteriorGSSMSmoothed:
    """Run a unscented Kalman (RTS) smoother.

    #TODO: NOT YET IMPLEMENTED.
    Args:
        params: model parameters.
        emissions: array of observations.
        hyperperams: hyper-parameters.
        inputs: optional inputs.

    Returns:
        nlgssm_posterior: posterior object.

    """

    raise NotImplementedError("Unscented Kalman Smoother not yet implemented in continuous-discrete case, have a look at dynamax for inspiration!")
    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[:, 0], t_emissions)
        t1 = tree_map(
            lambda x: jnp.concatenate(
                (t_emissions[1:, 0], jnp.array([t_emissions[-1, 0] + hyperparams.dt_final]))  # NB: t_{N+1} is simply t_{N}+dt_final
            ),
            t_emissions,
        )
    else:
        num_timesteps = len(emissions)
        t0 = jnp.arange(num_timesteps)
        t1 = jnp.arange(1, num_timesteps + 1)

    t0_idx = jnp.arange(num_timesteps)

    state_dim = params.dynamics.diffusion_cov.shape[0]

    # Run the unscented Kalman filter
    ukf_posterior = unscented_kalman_filter(params, emissions, t_emissions, hyperparams, inputs)
    ll = ukf_posterior.marginal_loglik
    filtered_means = ukf_posterior.filtered_means
    filtered_covs = ukf_posterior.filtered_covariances

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)

    # Only emission functions
    h = params.emissions.emission_function
    h = _process_fn(h, inputs)
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t0, t1, t0_idx, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time t0
        R = _get_params(params.emissions.emission_cov, 2, t0)
        u = inputs[t0_idx]
        y = emissions[t0_idx]

        # Prediction step
        # TODO: Make sure we return all components needed for smoothing!
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, params, t0, t1, lamb, w_mean, w_cov, W_matrix, u)
        m_pred, S_pred, S_cross = _predict(filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, W_matrix, u)
        # TODO: what is G???
        G = psd_solve(S_pred, S_cross.T).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the unscented Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (t0[::-1], t1[::-1], t0_idx[::-1], filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
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

def forecast_unscented_kalman_filter(
    params: ParamsCDNLGSSM,
    init_forecast: tfd.Distribution,
    t_init: Float[Array, "1 1"],
    t_forecast: Float[Array, "num_timesteps 1"],
    hyperparams: UKFHyperParams = UKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=[
        "forecasted_state_means",
        "forecasted_state_covariances",
    ],
) -> GSSMForecast:
    r"""Run an Unscented Kalman filter to forecast states.

    Args:
        params: model parameters.
        init_forecast: initial distribution to forecast with.
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants to forecast
        hyperparams: hyper-parameters of the EKF, related to the approximation order
        inputs: optional array of inputs.
        output_fields: list of fields to return 

    Returns:
        post: forecast object.

    """
    # Figure out timestamps, as vectors to scan over
    # t_forecast is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_forecast is not None:
        num_timesteps = t_forecast.shape[0]
        t0 = tree_map(
            lambda x: jnp.concatenate(
                (t_init, t_forecast[:-1, 0])
            ),
            t_forecast,
        )
        t1 = tree_map(lambda x: x[:,0], t_forecast)
    else:
        raise ValueError("t_forecast must be provided for forecasting")

    # Set-up indexing and inputs
    t0_idx = jnp.arange(num_timesteps)
    inputs = _process_input(inputs, num_timesteps+1)
    
    # Preliminaries
    state_dim = params.dynamics.diffusion_cov.params.shape[0]

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)
    
    def _step(carry, args):
        current_state_mean, current_state_cov = carry
        t0, t1, t0_idx = args
        
        # Predict the next state based on EKF approximations
        pred_state_mean, pred_state_cov = _predict(
            current_state_mean,
            current_state_cov,
            params,
            t0,
            t1,
            lamb,
            w_mean, w_cov, W_matrix,
            inputs[t0_idx],
            hyperparams
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
    
    # Run the Unscented Kalman filter
    _, outputs = lax_scan(
        _step,
        carry,
        (t0, t1, t0_idx),
        debug=DEBUG
    )
    
    # Build the forecast object
    forecast = GSSMForecast(
        **outputs,
    )
    return forecast

def emissions_unscented_kalman_filter(
    params: ParamsCDNLGSSM,
    t_states: Float[Array, "num_timesteps 1"],
    state_means: Float[Array, "num_timesteps state_dim"],
    state_covs: Optional[Float[Array, "num_timesteps state_dim state_dim"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    hyperparams: UKFHyperParams = UKFHyperParams(),
) -> Tuple[
        Float[Array, "num_timesteps emission_dim"], Optional[Float[Array, "num_timesteps emission_dim emission_dim"]]
    ]:
    r"""Compute the emissions corresponding to the UKF linearization of the model.
    
    Args:
        params: model parameters.
        t_states: continuous-time specific time instants of states
        state_means: state means at time instants t_states, always required
        state_covs: state covariances at time instants t_states, optional
            - if None, then we assume that the states are point estimates, and simply push through emission function
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        hyperparams: hyper-parameters of the filter

    Returns:
        emissions_mean: mean of emissions
        emissions_covariance: covariance of emissions, if available
    """
    
    # Figure out timestamps, as vectors to scan over
    # t_states is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_states is not None:
        num_timesteps = t_states.shape[0]
        t0 = tree_map(lambda x: x[:,0], t_states)
    else:
        raise ValueError("t_states must be provided for forecasting")

    # Set-up indexing and inputs
    t0_idx = jnp.arange(num_timesteps)
    inputs = _process_input(inputs, num_timesteps)

    # Preliminaries
    state_dim = params.dynamics.diffusion_cov.params.shape[0]

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)

    # Emission function
    h = params.emissions.emission_function.f
    
    def _step(carry, args):
        state_mean, state_cov, t0, t0_idx = args
        
        # Form sigma points
        sigmas_cond = _compute_sigmas(
            state_mean,
            state_cov,
            state_dim,
            lamb
        )
        
        # Propagate with inputs
        u_s = jnp.array(
            [inputs[t0_idx]] * len(sigmas_cond)
        )
        sigmas_cond_prop = vmap(
            h, in_axes=(0, None, None)
        )(sigmas_cond, u_s, t0)

        # Emission mean, by computing sufficient statistics of sigmas
        emission_mean = jnp.tensordot(
            w_mean,
            sigmas_cond_prop,
            axes=1
        )
        # Emission covariance
        R = params.emissions.emission_cov.f(
            None,
            inputs[t0_idx],
            t0
        )
        emission_cov = jnp.tensordot(
            w_cov,
            _outer(sigmas_cond_prop - emission_mean,
                   sigmas_cond_prop - emission_mean
            ),
            axes=1
        ) + R
        
        # Return carry and output states
        return (state_mean, state_cov), (emission_mean, emission_cov)

    # Initialize the state, based on provided initial distribution's mean and covariance
    carry = (
        state_means[0], state_covs[0]
    )
    # Run the extended Kalman filter
    _, (emissions_mean, emissions_covariance) = lax_scan(
        _step,
        carry,
        (state_means, state_covs, t0, t0_idx),
        debug=DEBUG
    ) # type: ignore

    return emissions_mean, emissions_covariance
