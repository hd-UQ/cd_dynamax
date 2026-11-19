import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap, jit
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from functools import partial
from jaxtyping import Array, Float, Int
from typing import NamedTuple, Optional
from ..utils.utils import psd_solve
from ..types import PRNGKey

class DiscreteParamsSLDS(NamedTuple):
    initial_distribution: Float[Array, "num_states"]
    transition_matrix : Float[Array, "num_states num_states"]
    proposal_transition_matrix : Float[Array, "num_states num_states"]

class LGParamsSLDS(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $m_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $z_t$ and
    emission $y_t$.
    """
    initial_mean: Float[Array, "num_states state_dim"]
    initial_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_weights: Float[Array, "num_states state_dim state_dim"]
    dynamics_cov: Float[Array, "num_states state_dim state_dim"]
    dynamics_bias: Float[Array, "num_states state_dim"]
    dynamics_input_weights: Float[Array, "num_states num_states input_dim"]
    emission_weights: Float[Array, "num_states emission_dim state_dim"]
    emission_cov: Float[Array, "num_states emission_dim emission_dim"]
    emission_bias: Float[Array, "num_states emission_dim"]
    emission_input_weights: Float[Array, "num_states num_emissions input_dim"]
    initialized: bool = False


class ParamsSLDS(NamedTuple):
    r"""Parameters of a SLDS, which has a discrete component and a continuous component.
    The discrete component is a Markov model for the hidden discrete state $m_t$.
    The linear_gaussian component is a linear Gaussian state space model for the hidden continuous state $z_t$ and
    emission $y_t$.
    """
    discrete: DiscreteParamsSLDS
    linear_gaussian: LGParamsSLDS

    def initialize(self, num_states, state_dim, emission_dim, input_dim=1):
        """
        Initialize the parameters of the SLDS. For each parameter that is the same over all models, the parameter
        is repeated to match the number of states. Also optional paramters are set to zero if not specified.
        """
        params = self.linear_gaussian
        initial_mean = params.initial_mean if params.initial_mean.shape == (num_states, state_dim) else jnp.array([params.initial_mean]*num_states)
        initial_cov = params.initial_cov if params.initial_cov.shape == (num_states, state_dim, state_dim) else jnp.array([params.initial_cov]*num_states)
        dynamics_weights = params.dynamics_weights if params.dynamics_weights.shape == (num_states, state_dim, state_dim) else jnp.array([params.dynamics_weights]*num_states)
        dynamics_cov = params.dynamics_cov if params.dynamics_cov.shape == (num_states, state_dim, state_dim) else jnp.array([params.dynamics_cov]*num_states)
        dynamics_bias = params.dynamics_bias if params.dynamics_bias is not None else jnp.zeros((num_states, state_dim))
        dynamics_bias = dynamics_bias if dynamics_bias.shape == (num_states, state_dim) else jnp.array([dynamics_bias]*num_states)
        dynamics_input_weights = params.dynamics_input_weights if params.dynamics_input_weights is not None else jnp.zeros((num_states, state_dim, input_dim))
        dynamics_input_weights = dynamics_input_weights if dynamics_input_weights.shape == (num_states, state_dim, input_dim) else jnp.array([dynamics_input_weights]*num_states)
        emission_weights = params.emission_weights if params.emission_weights.shape == (num_states, emission_dim, state_dim) else jnp.array([params.emission_weights]*num_states)
        emission_cov = params.emission_cov if params.emission_cov.shape == (num_states, emission_dim, emission_dim) else jnp.array([params.emission_cov]*num_states)
        emission_bias = params.emission_bias if params.emission_bias is not None else jnp.zeros((num_states, emission_dim))
        emission_bias = emission_bias if emission_bias.shape == (num_states, emission_dim) else jnp.array([emission_bias]*num_states)
        emission_input_weights = params.emission_input_weights if params.emission_input_weights is not None else jnp.zeros((num_states, emission_dim, input_dim))
        emission_input_weights = emission_input_weights if emission_input_weights.shape == (num_states, emission_dim, input_dim) else jnp.array([emission_input_weights]*num_states)
        
        return ParamsSLDS(
            discrete=self.discrete,
            linear_gaussian=LGParamsSLDS(
                initial_mean=initial_mean,
                initial_cov=initial_cov,
                dynamics_weights=dynamics_weights,
                dynamics_cov=dynamics_cov,
                dynamics_bias=dynamics_bias,
                dynamics_input_weights=dynamics_input_weights,
                emission_weights=emission_weights,
                emission_cov=emission_cov,
                emission_bias=emission_bias,
                emission_input_weights=emission_input_weights,
                initialized=True
            )
        )


class RBPFiltered(NamedTuple):
    r"""RBPF posterior.
    :param marginal_loglik: approximate marginal log likelihood from the particle-filter normalizers.
    :param weights: weights of the particles.
    :param means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    :param states: integer array of sampled discrete states with shape $(\mathrm{ntime}, \mathrm{num\_particles})$.
    """
    marginal_loglik: Optional[Float[Array, ""]] = None
    weights: Optional[Float[Array, "ntime num_particles"]] = None
    states: Optional[Int[Array, "ntime num_particles"]] = None
    means: Optional[Float[Array, "ntime num_particles state_dim"]] = None
    covariances: Optional[Float[Array, "ntime num_particles state_dim state_dim"]] = None

    def __getitem__(self, key):
        return getattr(self, key) if isinstance(key, str) else tuple.__getitem__(self, key)
   

def resampling(weights, states, means, covariances, key):                                                                  
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    new_states = jnp.take(states, resampled_idx, axis=0)
    filtered_means = jnp.take(means, resampled_idx, axis=0)
    filtered_covs = jnp.take(covariances, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, new_states, filtered_means, filtered_covs, next_key 

@partial(jit, static_argnums=(1,))
def optimal_resampling(weights, N, key):
    """Find the threshold for resampling particles using the optimal resampling algorithm of Fearnhead and Clifford (2003).
    Returns inidices of resampled particles and their weights.
    """
    # sort weights
    M = weights.shape[0]
    sorted_weights = jnp.sort(weights)
    sorted_idx = jnp.argsort(weights)

    # compute threshold p and value L of particles to retain
    lower_diag = jnp.triu(jnp.ones((M, M)), k=0).T
    ps = lax.dynamic_slice(lower_diag, (M-N, 0), (N-1, M)) @ sorted_weights / jnp.arange(1,N)
    ps = jnp.flip(ps)
    bounds = vmap(lambda ind: (sorted_weights[M-ind-1], sorted_weights[M-ind]))(jnp.arange(1, N))
    preds = vmap(lambda y,x,z : jnp.logical_and(y < x, x < z))(bounds[0], ps, bounds[1])
    L = jnp.where(preds, jnp.arange(1, N), 0).sum()
    p = jnp.where(L==0, 1/N, ps[L-1])
    
    # resample
    res_weights = jnp.where(sorted_weights < p, sorted_weights, 0.0)
    res_weights = res_weights / res_weights.sum()
    res_idx = jr.choice(key, M, shape=(M,), replace=True, p=res_weights)
    unsort_res_idx = sorted_idx[res_idx]

    final_idx = jnp.where(sorted_weights < p, unsort_res_idx, sorted_idx)
    final_weights = jnp.where(sorted_weights < p, p, sorted_weights)

    return final_idx[M-N:], final_weights[M-N:] / final_weights[M-N:].sum() 

def _prepare_emission_mask(emissions, emission_mask):
    """Validate and broadcast a fixed-shape emission mask."""
    if emission_mask is None:
        return jnp.ones_like(emissions, dtype=bool)

    emission_mask = jnp.asarray(emission_mask, dtype=bool)
    if emission_mask.shape == (len(emissions),):
        emission_mask = emission_mask[:, None]
    if emission_mask.shape not in ((len(emissions), 1), emissions.shape):
        raise ValueError(
            "emission_mask must have shape (num_timesteps,), "
            "(num_timesteps, 1), or match emissions; "
            f"got {emission_mask.shape}."
        )
    return jnp.broadcast_to(emission_mask, emissions.shape)


def _condition_on(state, mu_pred, Sigma_pred, params, u, y, emission_mask):
    """Update a predicted Gaussian using the observation for this regime."""
    H = params.emission_weights[state]
    d = params.emission_bias[state]
    D = params.emission_input_weights[state]
    R = params.emission_cov[state]

    # Project missing observation coordinates into neutral dummy dimensions.
    # Their observation rows are zero and N(0; 0, 1 / (2 pi)) has log density 0.
    masked_y = jnp.where(emission_mask, y, 0.0)
    M = jnp.diag(emission_mask.astype(H.dtype))
    identity = jnp.eye(len(emission_mask), dtype=H.dtype)
    masked_H = M @ H
    masked_D = M @ D
    masked_d = M @ d
    masked_R = M @ R @ M + (identity - M) / (2.0 * jnp.pi)

    # update
    S = masked_R + masked_H @ Sigma_pred @ masked_H.T
    K = psd_solve(S, masked_H @ Sigma_pred).T
    mu_y = masked_H @ mu_pred + masked_D @ u + masked_d
    ll = MVN(loc=mu_y, covariance_matrix=S).log_prob(masked_y)
    mu_cond = mu_pred + K @ (masked_y - mu_y)
    Sigma_cond = Sigma_pred - K @ S @ K.T
    return ll, mu_cond, Sigma_cond


def _conditional_kalman_step(
    state, mu, Sigma, params, transition_input, observation_input, y, emission_mask
):
    """Predict and update a Gaussian after transitioning to this regime."""
    F = params.dynamics_weights[state]
    b = params.dynamics_bias[state]
    B = params.dynamics_input_weights[state]
    Q = params.dynamics_cov[state]
    mu_pred = F @ mu + B @ transition_input + b
    Sigma_pred = F @ Sigma @ F.T + Q
    return _condition_on(
        state, mu_pred, Sigma_pred, params, observation_input, y, emission_mask
    )


def _condition_initial(params, u, y, emission_mask):
    """Score the first observation and condition the Gaussian for each initial regime."""
    states = jnp.arange(params.initial_mean.shape[0])
    return vmap(_condition_on, in_axes=(0, 0, 0, None, None, None, None))(
        states, params.initial_mean, params.initial_cov, params, u, y, emission_mask
    )


def _resample_if_needed(carry, ess_threshold):
    weights = carry[0]
    ess = 1.0 / jnp.sum(jnp.square(weights))
    return lax.cond(
        ess < ess_threshold * len(weights), resampling, lambda *args: args, *carry
    )


def _scan_transitions(step, initial_carry, initial_loglik, num_timesteps):
    """Collect the initial filtered result and the subsequent transition steps."""
    _, out = lax.scan(step, initial_carry, jnp.arange(1, num_timesteps))
    weights, states, means, covariances, _ = initial_carry
    return RBPFiltered(
        marginal_loglik=initial_loglik + out["marginal_loglik"].sum(),
        weights=jnp.concatenate((weights[None], out["weights"])),
        states=jnp.concatenate((states[None], out["states"])),
        means=jnp.concatenate((means[None], out["means"])),
        covariances=jnp.concatenate((covariances[None], out["covariances"])),
    )


def rbpfilter(
    num_particles: int,
    params: ParamsSLDS,
    emissions:  Float[Array, "ntime emission_dim"],
    key: PRNGKey = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ess_threshold: float = 0.5,
    emission_mask: Optional[Array] = None,
    ):
    '''
    Implementation of the Rao-Blackwellized particle filter, for approximating the 
    filtering distribution of a switching linear dynamical system. The first
    observation updates the initial regimes and their conditional Gaussians.
    For each subsequent observation, the filter samples a regime from the
    transition proposal and runs a Kalman prediction and update. Resampling is
    based on the effective sample size after each observation.

    `inputs[t]` drives the observation at row `t` and, when present,
    the transition from row `t` to row `t + 1`.

    The proposal transition matrix must assign positive probability to every transition
    with positive target probability. Particle weights and marginal likelihood increments
    include the target-to-proposal transition probability ratio.

    emission_mask may have shape (ntime,), (ntime, 1), or match emissions.
    False coordinates are excluded from the Kalman update and likelihood.
    '''

    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 1)) if inputs is None else inputs
    if not params.linear_gaussian.initialized: raise ValueError("ParamsSLDS must be initialized")

    def _step(carry, t):
        r"""
        carry = (weights, prev_states, filtered_means, filtered_covs, key)

        where,
            prev_states : (num_particles, )
            weights : (num_particles, )
            pred_means : (num_particles, state_dim)
            pred_covs : (num_particles, state_dim, state_dim)
        """

        # Unpack carry
        weights, prev_states, filtered_means, filtered_covs, key = carry
        num_states = params.discrete.transition_matrix.shape[0]

        transition_input = inputs[t - 1]
        observation_input = inputs[t]
        y = emissions[t]
        mask = emission_mask[t]

        # Sample discrete states from the proposal
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        new_states = vmap(lambda key, x: jr.choice(key, jnp.arange(num_states), p=params.discrete.proposal_transition_matrix[x]))(keys[1:], prev_states)

        # Run KF step conditional on the sampled states
        lls, filtered_means, filtered_covs = vmap(
            _conditional_kalman_step, in_axes=(0, 0, 0, None, None, None, None, None)
        )(
            new_states,
            filtered_means,
            filtered_covs,
            params.linear_gaussian,
            transition_input,
            observation_input,
            y,
            mask,
        )

        # Correct for sampling from the proposal rather than the target transition.
        log_importance_ratios = (
            jnp.log(params.discrete.transition_matrix[prev_states, new_states])
            - jnp.log(params.discrete.proposal_transition_matrix[prev_states, new_states])
        )
        log_weights = jnp.log(weights) + lls + log_importance_ratios
        marginal_loglik = logsumexp(log_weights)
        weights = jnp.exp(log_weights - marginal_loglik)

        carry = _resample_if_needed(
            (weights, new_states, filtered_means, filtered_covs, next_key),
            ess_threshold,
        )
        weights, new_states, filtered_means, filtered_covs, _ = carry
        outputs = {
            "marginal_loglik": marginal_loglik,
            "weights": weights,
            "states": new_states,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        return carry, outputs

    init_key, next_key = jr.split(key)
    emission_mask = _prepare_emission_mask(emissions, emission_mask)
    lls, means, covs = _condition_initial(
        params.linear_gaussian, inputs[0], emissions[0], emission_mask[0]
    )
    initial_probs = params.discrete.initial_distribution
    states = jr.choice(
        init_key, len(initial_probs), shape=(num_particles,), p=initial_probs
    )
    log_weights = lls[states] - jnp.log(num_particles)
    initial_loglik = logsumexp(log_weights)
    weights = jnp.exp(log_weights - initial_loglik)
    carry = _resample_if_needed(
        (weights, states, means[states], covs[states], next_key), ess_threshold
    )
    return _scan_transitions(_step, carry, initial_loglik, num_timesteps)

def rbpfilter_optimal(
    num_particles: int,
    params: ParamsSLDS,
    emissions:  Float[Array, "ntime emission_dim"],
    key: PRNGKey = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    emission_mask: Optional[Array] = None,
    ):
    '''
    Implementation of the Rao-Blackwellized particle filter with optimal resampling.

    The first observation updates the initial regime mixture exactly, then
    particles are drawn from the posterior regime probabilities. Subsequent
    observations use optimal resampling of the candidate transitions.

    `inputs[t]` drives the observation at row `t` and, when present,
    the transition from row `t` to row `t + 1`.

    emission_mask may have shape (ntime,), (ntime, 1), or match emissions.
    False coordinates are excluded from the Kalman update and likelihood.
    '''

    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 1)) if inputs is None else inputs
    state_dim = params.linear_gaussian.initial_mean.shape[1]

    def _step(carry, t):
        r"""
        carry = (weights, prev_states, filtered_means, filtered_covs, key)

        where,
            prev_states : (num_particles, )
            weights : (num_particles, )
            pred_means : (num_particles, state_dim)
            pred_covs : (num_particles, state_dim, state_dim)
        """
        # Unpack carry
        weights, prev_states, filtered_means, filtered_covs, key = carry
        num_states = params.discrete.transition_matrix.shape[0]

        key, next_key = jr.split(key, 2)

        transition_input = inputs[t - 1]
        observation_input = inputs[t]
        y = emissions[t]
        mask = emission_mask[t]

        # Run KF step conditional on all possible states
        candidate_step = vmap(
            _conditional_kalman_step, in_axes=(0, None, None, None, None, None, None, None)
        )
        lls, filtered_means, filtered_covs = vmap(
            candidate_step, in_axes=(None, 0, 0, None, None, None, None, None)
        )(
            jnp.arange(num_states),
            filtered_means,
            filtered_covs,
            params.linear_gaussian,
            transition_input,
            observation_input,
            y,
            mask,
        )

        # Compute weights
        marginal_loglik = logsumexp(
            jnp.log(weights)[:, None] + jnp.log(params.discrete.transition_matrix[prev_states, :]) + lls
        )
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        trans_weights = params.discrete.transition_matrix[prev_states, :]
        weights = jnp.multiply(jnp.einsum('i,ij->ij', weights, trans_weights), loglik_weights)
        weights /= jnp.sum(weights)

        # Reshape
        weights = weights.reshape((num_particles * num_states,))
        states = jnp.tile(jnp.arange(num_states), num_particles)
        filtered_means = filtered_means.reshape((num_particles * num_states, state_dim))
        filtered_covs = filtered_covs.reshape((num_particles * num_states, state_dim, state_dim))

        # Optimal resampling
        res_idx, res_weights = optimal_resampling(weights, num_particles, key)
        new_states = jnp.take(states, res_idx, axis=0)
        filtered_means = jnp.take(filtered_means, res_idx, axis=0)
        filtered_covs = jnp.take(filtered_covs, res_idx, axis=0)
                                                                                 
        # Build carry and output states
        carry = (res_weights, new_states, filtered_means, filtered_covs, next_key)
        outputs = {
            "marginal_loglik": marginal_loglik,
            "weights": res_weights,
            "states": new_states,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        return carry, outputs

    init_key, next_key = jr.split(key)
    emission_mask = _prepare_emission_mask(emissions, emission_mask)
    lls, means, covs = _condition_initial(
        params.linear_gaussian, inputs[0], emissions[0], emission_mask[0]
    )
    log_weights = jnp.log(params.discrete.initial_distribution) + lls
    initial_loglik = logsumexp(log_weights)
    states = jr.categorical(init_key, log_weights, shape=(num_particles,))
    weights = jnp.full((num_particles,), 1.0 / num_particles, dtype=lls.dtype)
    carry = (weights, states, means[states], covs[states], next_key)
    return _scan_transitions(_step, carry, initial_loglik, num_timesteps)
