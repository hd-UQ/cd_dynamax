# JAX imports
import jax.numpy as jnp
from jax import lax
from jax import vmap
from jax.tree_util import tree_map

# Typing annotations
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, List

# Distributions, compatible with JAX, from TensorFlow Probability
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
tfd = tfp.distributions
tfb = tfp.bijectors

# Dynamax shared code
from cd_dynamax.dynamax.utils.utils import psd_solve
# To avoid unnecessary redefinitions of code,
# We import those posterior filtering and smoothing equivalent classes that can be reused from dynamax
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

# Our codebase
# CDLGSSM forecasting definition is reused
from ..continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import GSSMForecast
# CDNLGSSM specific param and function definitions
from .cdnlgssm_utils import *
# Diffrax based diff-eq solver
from ..utils.diffrax_utils import diffeqsolve
# Debugging utilities
from ..utils.debug_utils import *
DEBUG = False

#### Helper functions
# Helper functions --- from dynamax 
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_compute_lambda = lambda x, y, z: x**2 * (y + z) - z

#### CD-NLGSSM filtering: Unscented Kalman Filter (UKF)

# Default UKF filtering hyperparameters, as class
# We redefine UKFHyperParams for cd-dynamax, due to dt_final
class UKFHyperParams(NamedTuple):
    """Lightweight container for UKF hyperparameters.

        Default values taken from https://github.com/sbitzer/UKF-exposed
    """
    dt_final: float = 1e-4 # Small dt_final for predicted mean and covariance at the end of sequence 
    alpha: float = jnp.sqrt(3)
    beta: int = 2
    kappa: int = 1
    cov_rescaling: float = 1.0
    diffeqsolve_settings: dict = {}
    state_order: str = "first"
    dt_average: float = 0.1 # Average timestep for discrete state order, if applicable

#  UKF auxiliary functions: sigma point computation
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

#  UKF auxiliary functions: weights computation
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

### CD-NLGSSM filtering key functions: predict and condition_on
# Predict next mean and covariance, under UKF approximations
def _predict(
    m,
    P,  # priors
    params: ParamsCDNLGSSM,
    t0: Float,
    t1: Float,
    lamb,
    w_mean,
    w_cov,
    W_matrix,
    u,
    filter_hyperparams,
    warn: bool = True,
):
    """Predict next mean and covariance using additive UKF

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        params: CD-NLGSSM parameters, containing
            - dynamics RHS drift function
            - diffusion coefficient matrix L
            - Brownian covariance matrix Q
        t0: initial time-instant
        t1: final time-instant
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        W_matrix (2*D_hid+1,2*D_hid+1): matrix of weights defined by combining w_mean and w_cov as in eq. 3.82 of Saarka's Thesis.
        u (D_in,): inputs.
        filter_hyperparams: UKF hyper-parameters.
        warn: whether to issue warnings (e.g., about PSD issues)

    Returns:
        m_pred (D_hid,): predicted mean.
        P_pred (D_hid,D_hid): predicted covariance.

    """

    # Dimensions
    n = len(m)

    # Dynamics drift function
    f = params.dynamics.drift.f

    # Zeroth order UKF approximation
    if filter_hyperparams.state_order in ['zeroth', 'discrete']:
        # According to Saarka's ODE 3.183 in thesis

        # First, we need to compute the sigma points
        X_t = _compute_sigmas(m, P, n, lamb)

        # Then we propagate them through the deterministic part of the dynamics
        # by solving the ODE for each sigma point X_t as initial condition
        # Note that the control is not time-varying across interval [t0, t1]

        # The RHS ODE is f(y, u, t)
        def rhs(t, y, args):
            return f(y, u, t)
        
        # We solve the ODE for each sigma point X_t as initial condition
        def this_solve(x_t):
            return diffeqsolve(
                rhs,
                t0=t0,
                t1=t1,
                y0=x_t,
                **filter_hyperparams.diffeqsolve_settings
            )

        # We vmap each solve over all the sigma points
        sol = vmap(
            this_solve
        )(X_t) # X_t.shape = (2*state_dim+1, state_dim)
        # Extract final state for each sigma point
        X_pred = sol[:, -1, :]  # (2*state_dim+1, state_dim)

        # We then compute the predicted mean and covariance
        # Following UKF formulas
        m_pred = jnp.tensordot(w_mean, X_pred, axes=1)
        P_pred = jnp.tensordot(w_cov, _outer(X_pred - m_pred, X_pred - m_pred), axes=1)

        # Finally, add state noise
        if filter_hyperparams.state_order == 'zeroth':
            # For zeroth order, we use the timestep at each step
            dt = t1 - t0
        else:
            # For discrete state order, we use a user-specified average timestep
            # This maps us into a discrete setting where the deterministic dynamics still map from t0 to t1
            # but the same amount of noise is added after each measurement.
            dt = filter_hyperparams.dt_average

        # Get diffusion parameters at time t0 and input u
        Qc_t = params.dynamics.diffusion_cov.f(None, u, t0)
        L_t = params.dynamics.diffusion_coefficient.f(None, u, t0) * filter_hyperparams.cov_rescaling
        # Compute state noise covariance
        P_pred += dt * L_t @ Qc_t @ L_t.T
        P_pred = psd(P_pred)

    else:
        # Sarkka Thesis's algo 3.24, with weights defined in eq. 3.69;
        # the related weight vector w_m and matrix W are defined in eq 3.81-3.82;
        def rhs_all(t, y, args):
            # Unpack mean and covariance
            m_t, P_t = y

            # Create sigma points X_t
            X_t = _compute_sigmas(m_t, P_t, n, lamb)

            # Get dynamics parameters at time t and input u
            f = params.dynamics.drift.f # TODO: reconsider when we want time-varying dynamics functions
            Qc_t = params.dynamics.diffusion_cov.f(None, u, t)
            L_t = params.dynamics.diffusion_coefficient.f(None, u, t) * filter_hyperparams.cov_rescaling
            
            # Propagate sigma points through dynamics f
            f_X_t = vmap(f, in_axes=(0, None, None))(X_t, u, t)
            # dimensions of f_X_t are (2*state_dim+1, state_dim)

            # Compute RHS for mean updates
            dmdt = f_X_t.T @ w_mean

            # Compute RHS for covariance updates
            # dPdt = f_x W X^T + X W f_x^T + L Qc L^T (Transposes flipped due to shape of f_X_t and X_t)
            foo = f_X_t.T @ W_matrix @ X_t
            dPdt = foo + foo.T + L_t @ Qc_t @ L_t.T

            # Return combined RHS
            return (dmdt, dPdt)

        # Solve ODE 3.183 in Sarkka's thesis
        # Initial condition
        y0 = (m, P)
        # Numerical solve of ODE
        sol = diffeqsolve(
            rhs_all,
            t0=t0,
            t1=t1,
            y0=y0,
            **filter_hyperparams.diffeqsolve_settings
        )
        # Extract final mean and covariance
        m_pred, P_pred = sol[0][-1], psd(sol[1][-1])

    # According to Sarkka's algo 3.24
    # we only need to return m_pred and P_pred (not P_cross) in continuous-discrete
    return m_pred, P_pred

# Condition on a new observation, under UKF approximations
def _condition_on(m, P, h, R, lamb, w_mean, w_cov, u, y, t, warn: bool = True):
    """Condition a Gaussian potential on a new observation,
        using additive UKF approximations.

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
        warn: whether to issue warnings (e.g., about PSD issues)

    Returns:
        ll (float): log-likelihood of observation
        m_cond (D_hid,): filtered mean.
        P_cond (D_hid,D_hid): filtered covariance.

    """
    # Dimensions
    n = len(m)
    
    # Form sigma points and propagate
    sigmas_cond = _compute_sigmas(m, P, n, lamb)
    # Replicate inputs for each sigma point
    u_s = jnp.array([u] * len(sigmas_cond))
    # Propagate sigma points through emission function at time t with input u
    sigmas_cond_prop = vmap(
        h,
        in_axes=(0, None, None)
    )(sigmas_cond, u_s, t)

    # Compute predicted mean, covariance, and cross-covariance
    pred_mean = jnp.tensordot(
        w_mean,
        sigmas_cond_prop,
        axes=1
    )
    pred_cov = psd(
        R + jnp.tensordot(
            w_cov,
            _outer(
                sigmas_cond_prop - pred_mean,
                sigmas_cond_prop - pred_mean
            ),
        axes=1),
        warn=warn,
    )
    pred_cross = jnp.tensordot(
        w_cov,
        _outer(
            sigmas_cond - m,
            sigmas_cond_prop - pred_mean
        ),
        axes=1
    )

    # Compute log-likelihood of observation based on Gaussian distribution,
    # using predicted mean and covariance
    ll = MVN(pred_mean, pred_cov).log_prob(y)

    # UKF gain
    K = psd_solve(pred_cov, pred_cross.T).T
    # Compute UKF filtered mean and covariance
    m_cond = m + K @ (y - pred_mean)
    P_cond = psd(P - K @ pred_cov @ K.T, warn=warn)
    # Return log-likelihood, filtered mean and covariance
    return ll, m_cond, P_cond

# UKF filtering main function
def unscented_kalman_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: UKFHyperParams = UKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "filtered_means",
        "filtered_covariances",
        "predicted_means",
        "predicted_covariances",
    ],
    warn: bool = True,
) -> PosteriorGSSMFiltered:
    r"""Run a unscented Kalman filter
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
        output_fields: list of fields to include in the output.
        warn: whether to issue warnings (e.g., about PSD issues)

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
                (t_emissions[1:, 0], jnp.array([t_emissions[-1, 0] + filter_hyperparams.dt_final]))  # NB: t_{N+1} is simply t_{N}+dt_final
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

    # # UKF Preliminaries
    state_dim = params.dynamics.diffusion_cov.f(None, None, None).shape[0]
    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = filter_hyperparams.alpha, filter_hyperparams.beta, filter_hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)

    # Define one step of UKF filtering
    def _step(carry, args):
        # Unpack the inputs
        ll, pred_mean, pred_cov = carry
        t0, t1, t0_idx = args

        # Get parameters and inputs for time t0
        u = inputs[t0_idx]
        y = emissions[t0_idx]
        R = params.emissions.emission_cov.f(None, u, t0)

        # Condition on this emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(
            pred_mean,
            pred_cov,
            h,
            R,
            lamb,
            w_mean,
            w_cov,
            u,
            y,
            t0,
            warn=warn
        )

        # Update the log likelihood
        ll += log_likelihood

        # Predict the next state, based on UKF predict
        pred_mean, pred_cov = _predict(
            filtered_mean,
            filtered_cov,
            params,
            t0,
            t1,
            lamb,
            w_mean,
            w_cov,
            W_matrix,
            u,
            filter_hyperparams,
            warn=warn
        )

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

        # Return carry and outputs
        return carry, outputs

    # Initialize carry with initial distribution
    carry = (0.0, params.initial.mean.f(), params.initial.cov.f())
    # Run the UKF filter, via lax.scan
    (ll, *_), outputs = lax.scan(
        _step,
        carry,
        (t0, t1, t0_idx)
    )
    # Build and return posterior object
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

# CD-NLGSSM forecast function: Unscented Kalman Filter Forecast
def forecast_unscented_kalman_filter(
    params: ParamsCDNLGSSM,
    init_forecast: tfd.Distribution,
    t_init: Float[Array, "1 1"],
    t_forecast: Float[Array, "num_timesteps 1"],
    filter_hyperparams: UKFHyperParams = UKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=[
        "forecasted_state_means",
        "forecasted_state_covariances",
    ],
    warn: bool = True,
) -> GSSMForecast:
    r"""Run an Unscented Kalman filter to forecast states.

    Args:
        params: CD-NLGSSM parameters, containing
            - dynamics RHS drift function
            - diffusion coefficient matrix L
            - Brownian covariance matrix Q
        init_forecast: initial distribution to forecast with.
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants to forecast
        filter_hyperparams: hyper-parameters of the UKF, related to the approximation order
        inputs: optional array of inputs.
        output_fields: list of fields to return 
        warn: whether to issue warnings (e.g., about PSD issues)

    Returns:
        forecast: forecast object.

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
    
    # UKF Preliminaries
    state_dim = params.dynamics.diffusion_cov.f(None, None, None).shape[0]
    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = filter_hyperparams.alpha, filter_hyperparams.beta, filter_hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)
    
    # Define one step of UKF forecasting
    def _step(carry, args):
        # Unpack the inputs
        current_state_mean, current_state_cov = carry
        t0, t1, t0_idx = args
        
        # Predict the next state based on UKF approximations
        pred_state_mean, pred_state_cov = _predict(
            current_state_mean,
            current_state_cov,
            params,
            t0,
            t1,
            lamb,
            w_mean, w_cov, W_matrix,
            inputs[t0_idx],
            filter_hyperparams,
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

    # Initialize the state,
    # based on provided initial distribution's mean and covariance
    carry = (init_forecast.mean(), init_forecast.covariance())
    
    # Run the Unscented Kalman filter, via lax.scan
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

# CD-NLGSSM emission function: Emissions from UKF approximation
def emissions_unscented_kalman_filter(
    params: ParamsCDNLGSSM,
    t_states: Float[Array, "num_timesteps 1"],
    state_means: Float[Array, "num_timesteps state_dim"],
    state_covs: Optional[Float[Array, "num_timesteps state_dim state_dim"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    filter_hyperparams: UKFHyperParams = UKFHyperParams(),
    warn: bool = True,
) -> Tuple[
        Float[Array, "num_timesteps emission_dim"], Optional[Float[Array, "num_timesteps emission_dim emission_dim"]]
    ]:
    r"""Compute the emissions corresponding to the UKF linearization of the model.
    
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
        warn: whether to issue warnings (e.g., about PSD issues)

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

    # Emission function
    h = params.emissions.emission_function.f
    
    # UKF Preliminaries
    state_dim = params.dynamics.diffusion_cov.f(None, None, None).shape[0]
    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = filter_hyperparams.alpha, filter_hyperparams.beta, filter_hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov, W_matrix = _compute_weights(state_dim, alpha, beta, lamb)

    # Define one step of UKF emissions computation
    def _step(carry, args):
        # Unpack the inputs
        state_mean, state_cov, t0, t0_idx = args
        
        # Form sigma points
        sigmas_cond = _compute_sigmas(
            state_mean,
            state_cov,
            state_dim,
            lamb
        )
        
        # Replicate inputs for each sigma point
        u_s = jnp.array(
            [inputs[t0_idx]] * len(sigmas_cond)
        )
        # Propagate sigma points through emission function at time t0 with input u
        sigmas_cond_prop = vmap(
            h, in_axes=(0, None, None)
        )(sigmas_cond, u_s, t0)

        # Emission mean,
        # by computing sufficient statistics of sigmas
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
        emission_cov = psd(
            R + jnp.tensordot(
                w_cov,
                _outer(
                    sigmas_cond_prop - emission_mean,
                    sigmas_cond_prop - emission_mean
                ),
                axes=1),
            warn=warn,
        )
        
        # Return carry and output states
        return (state_mean, state_cov), (emission_mean, emission_cov)

    # Initialize the state,
    # based on provided initial distribution's mean and covariance
    carry = (
        state_means[0], state_covs[0]
    )
    # Run the Unscented Kalman filter, via lax.scan
    _, (emissions_mean, emissions_covariance) = lax_scan(
        _step,
        carry,
        (state_means, state_covs, t0, t0_idx),
        debug=DEBUG
    ) # type: ignore

    # Return emissions mean and covariance
    return emissions_mean, emissions_covariance
