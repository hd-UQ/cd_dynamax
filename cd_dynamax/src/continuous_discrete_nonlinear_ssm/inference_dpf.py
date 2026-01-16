from typing import NamedTuple, Union, Tuple

from jaxtyping import Array, Float
from .cdnlssm_utils import ParamsCDNLSSM
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, lax
from jax.scipy.special import logsumexp
from ..utils.diffrax_utils import diffeqsolve


class DPFHyperParams(NamedTuple):
    """Lightweight container for DPF hyperparameters."""

    dt_final: float = 1e-4
    N_particles: int = 100
    ess_threshold_ratio: float = 0.5
    resample_method: str = "stop_gradient"
    softness: float = 0.7
    cov_rescaling: float = 1.0
    state_order: str = "first"
    dt_average: float = 0.1
    diffeqsolve_settings: dict = {}
    return_ess_history: bool = False
    proposal_method: str = (
        "bootstrap"  # Currently, only bootstrap proposals are supported.
    )


def _predict(
    key,
    x,  # particles
    params: ParamsCDNLSSM,
    t0: Float,
    t1: Float,
    u,
    filter_hyperparams,
):
    """Predict evolution of ensemble of particles through the nonlinear stochastic dynamics.

    Args:
        key: Random key.
        x: Particles to predict.
        params: Parameters of the CDNLSSM.
        t0: Initial time.
        t1: Final time.
        u: Inputs.
        filter_hyperparams: Hyperparameters of the filter.

    Returns:
        x_pred: Predicted particles.
    """

    def drift(t, y, args):
        return params.dynamics.drift.f(y, u, t)

    if filter_hyperparams.state_order == "zeroth":
        diffusion = None
    else:

        def diffusion(t, y, args):
            Qc_t = params.dynamics.diffusion_cov.f(None, u, t)
            L_t = (
                params.dynamics.diffusion_coefficient.f(None, u, t)
                * filter_hyperparams.cov_rescaling
            )
            Q_sqrt = jnp.linalg.cholesky(Qc_t)
            combined_diffusion = L_t @ Q_sqrt

            return combined_diffusion

    def my_solve(y0, key0):
        return diffeqsolve(
            key=key0,
            drift=drift,
            diffusion=diffusion,
            t0=t0,
            t1=t1,
            y0=y0,
            **filter_hyperparams.diffeqsolve_settings,
        )

    key_array = jr.split(key, x.shape[0])
    sol = vmap(my_solve, in_axes=0)(x, key_array)  # N_particles x 1 time x D_hid
    x_pred = sol[:, 0, :]  # N_particles x D_hid

    if filter_hyperparams.state_order in ["zeroth", "discrete"]:
        if filter_hyperparams.state_order == "zeroth":
            dt = t1 - t0
        else:
            dt = filter_hyperparams.dt_average

        Qc_t = params.dynamics.diffusion_cov.f(None, u, t0)
        L_t = (
            params.dynamics.diffusion_coefficient.f(None, u, t0)
            * filter_hyperparams.cov_rescaling
        )
        state_noise_cov = dt * L_t @ Qc_t @ L_t.T  # D_hid x D_hid

        key_array = jr.split(key, x.shape[0])
        noise = vmap(
            lambda key: jr.multivariate_normal(
                key=key, mean=jnp.zeros(x.shape[1]), cov=state_noise_cov
            ),
            in_axes=0,
        )(key_array)
        x_pred += noise
    return x_pred


def _normalize_log_weights(log_w):
    log_norm = logsumexp(log_w)
    return log_w - log_norm, log_norm


def _effective_sample_size(log_w):
    return jnp.exp(-logsumexp(2.0 * log_w))


_SUPPORTED_RESAMPLERS = ("multinomial", "soft", "stop_gradient")


def _validate_resample_method(method: str) -> str:
    """Ensure the requested resampling strategy is implemented.

    See :attr:`_SUPPORTED_RESAMPLERS` for supported methods."""
    method_lower = method.lower()
    if method_lower not in _SUPPORTED_RESAMPLERS:
        raise ValueError(
            f"Unsupported resample_method '{method}'. "
            f"Supported methods are {_SUPPORTED_RESAMPLERS}."
        )
    return method_lower


def _multinomial_resample(key, x, log_w):
    """Multinomial resampling.

    This is the classical resampling strategy for particle filters,
    where each particle is resampled with probability proportional to its weight.

    Args:
        key: Random key.
        x: Particles to resample.
        log_w: Log weights of the particles.

    Returns:
        x_resampled: Resampled particles.
        log_w_resampled: Log weights of the resampled particles.
        idx: Indices of the resampled particles.
    """
    probs = jnp.exp(log_w - logsumexp(log_w))
    idx = jr.choice(key, x.shape[0], shape=(x.shape[0],), p=probs, replace=True)
    x_resampled = x[idx]
    log_w_resampled = jnp.full_like(log_w, -jnp.log(x.shape[0]))
    return x_resampled, log_w_resampled, idx


def _stop_gradient_resample(key, x, log_w, *, base_method: str = "multinomial"):
    """Stop-gradient resampling [1].

    Stop-gradient resampling [1] modifies the resampling step such that forward passes are not modified.

    References:
        [1] Scibior A, Wood F (2021). “Differentiable particle filtering without modifying the forward pass.” arXiv:2106.10314

    Args:
        key: Random key.
        x: Particles to resample.
        log_w: Log weights of the particles.
        base_method: Base resampling method to use.

    Returns:
        x_resampled: Resampled particles.
        log_w_resampled: Log weights of the resampled particles.
        idx: Indices of the resampled particles.

    """
    if base_method != "multinomial":
        raise ValueError(
            f"Unsupported base resampler '{base_method}' for stop_gradient resampling."
        )
    x_resampled, new_log_w, idx = _multinomial_resample(key, x, log_w)
    resampled_log_w = log_w[idx]
    log_w_out = new_log_w + resampled_log_w - lax.stop_gradient(resampled_log_w)
    log_w_out, _ = _normalize_log_weights(log_w_out)
    return x_resampled, log_w_out


def _soft_resample(key, x, log_w, softness):
    """Soft resampling [1].

    Soft resampling approximates differentiable resampling by mixing the weights w with a uniform distribution,
        q(k) = softness * w[k] + (1-softness) / n_particles,
    then reweighting via importance weights.

    This strategy generally provides biased gradients, but can still be an efficient approximation.

    References:
        [1] Karkus P, Hsu D, Lee WS (2018). “Particle filter networks with application to visual localization.” In Proc. Conf. Robot Learn., pp. 169–178. PMLR, Zurich, CH.

    Args:
        key: Random key.
        x: Particles to resample.
        log_w: Log weights of the particles.
        softness: Softness parameter.

    Returns:
        x_resampled: Resampled particles.
        log_w_resampled: Log weights of the resampled particles.
    """
    n = x.shape[0]
    log_n = jnp.log(n)
    log_softness = jnp.log(softness)
    log_uniform_component = jnp.log1p(-softness) - log_n
    soft_weight = jnp.logaddexp(log_w + log_softness, log_uniform_component)
    probs = jnp.exp(soft_weight - logsumexp(soft_weight))
    idx = jr.choice(key, n, shape=(n,), p=probs, replace=True)
    x_resampled = x[idx]
    log_w_resampled = log_w[idx] - soft_weight[idx] - log_n
    log_w_resampled = log_w_resampled - logsumexp(log_w_resampled)
    return x_resampled, log_w_resampled


def filter_dpf(
    key,
    params: ParamsCDNLSSM,
    ys: Array,
    us: Array | None = None,
    ts: Array | None = None,
    hyperparams: DPFHyperParams = DPFHyperParams(),
) -> Union[Tuple[Array, Array, Array, float], Tuple[Array, Array, Array, Array, float]]:
    """Differentiable particle filter with configurable resampling.

    A differentiable particle filter (DPF) is a particle filter with the (discrete, non-differentiable) resampling step replaced in some way to allow for gradient-based optimization.
    This implementation supports three different resampling methods:
    - Multinomial resampling (biased)
    - Soft resampling [1] (biased; interpolates between multinomial and uniform resampling)
    - Stop-gradient resampling [2] (unbiased for score estimates)

    Currently, only bootstrap proposals are supported.

    References:
        [1] Karkus P, Hsu D, Lee WS (2018). “Particle filter networks with application to visual localization.” In Proc. Conf. Robot Learn., pp. 169–178. PMLR, Zurich, CH.
        [2] Scibior A, Wood F (2021). “Differentiable particle filtering without modifying the forward pass.” arXiv:2106.10314

    Args:
        key: Random key.
        params: Parameters of the CDNLSSM.
        ys: Emissions.
        us: Inputs.
        ts: Times.
        hyperparams: Hyperparameters of the filter.

    Returns:
        particles: Particles.
        log_weights: Log weights.
        ess_history: (if return_ess_history is True) Effective sample size history.
        log_evidence: Log evidence.
    """
    n_particles = int(hyperparams.N_particles)
    T = ys.shape[0]
    resample_method = _validate_resample_method(hyperparams.resample_method)
    ess_threshold = hyperparams.ess_threshold_ratio * n_particles

    if hyperparams.proposal_method != "bootstrap":
        raise ValueError(
            f"Currently, only bootstrap proposals are supported, but {hyperparams.proposal_method} was provided."
        )

    key_init, key = jr.split(key)
    particles = params.initial.initial_distribution.distribution.sample(
        seed=key_init, sample_shape=(n_particles,)
    )
    log_w = jnp.full((n_particles,), -jnp.log(n_particles))
    log_evidence = jnp.array(0.0)

    t_currs = ts if ts is not None else jnp.arange(T)
    t_prevs = jnp.concatenate([t_currs[:1], t_currs[:-1]], axis=0)

    if us is not None:
        u_currs = us
        u_prevs = jnp.concatenate([us[:1], us[:-1]], axis=0)
    else:
        u_currs = None
        u_prevs = None

    keys = jr.split(key, T)
    idxs = jnp.arange(T)

    def _step(carry, inputs):
        particles, log_w, log_evidence = carry
        key_t, idx, t_curr, t_prev, u_prev, u_curr = inputs
        key_pred, key_resample = jr.split(key_t)

        def _do_predict(p):
            return _predict(
                key_pred,
                p,
                params,
                t0=t_prev,
                t1=t_curr,
                u=u_prev,
                filter_hyperparams=hyperparams,
            )

        particles = lax.cond(idx > 0, _do_predict, lambda p: p, particles)

        log_w = log_w + params.emissions.emission_distribution.log_prob(
            ys[idx], particles, u_curr, t_curr
        )
        log_w, log_norm = _normalize_log_weights(log_w)
        log_evidence = log_evidence + log_norm
        ess = _effective_sample_size(log_w)

        particles_hist = particles
        logw_hist = log_w

        def _resample(args):
            x_in, log_w_in, key_in = args
            if resample_method == "soft":
                return _soft_resample(key_in, x_in, log_w_in, hyperparams.softness)
            if resample_method == "multinomial":
                x_out, log_w_out, _ = _multinomial_resample(key_in, x_in, log_w_in)
                return x_out, log_w_out
            return _stop_gradient_resample(
                key_in, x_in, log_w_in, base_method="multinomial"
            )

        particles, log_w = lax.cond(
            ess < ess_threshold,
            _resample,
            lambda args: (args[0], args[1]),
            (particles, log_w, key_resample),
        )

        return (particles, log_w, log_evidence), (particles_hist, logw_hist, ess)

    (particles, log_w, log_evidence), (particles_hist, logw_hist, ess_hist) = lax.scan(
        _step,
        (particles, log_w, log_evidence),
        (keys, idxs, t_currs, t_prevs, u_prevs, u_currs),
    )
    if hyperparams.return_ess_history:
        return particles_hist, logw_hist, ess_hist, log_evidence
    else:
        return particles_hist, logw_hist, log_evidence
