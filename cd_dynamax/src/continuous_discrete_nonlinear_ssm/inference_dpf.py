from typing import NamedTuple

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
    resample_method: str = "stop_gradient"
    softness: float = 0.7
    cov_rescaling: float = 1.0
    state_order: str = "first"
    dt_average: float = 0.1
    diffeqsolve_settings: dict = {}


def _predict(
    key,
    x,  # particles
    params: ParamsCDNLSSM,
    t0: Float,
    t1: Float,
    u,
    filter_hyperparams,
):
    """Predict evolution of ensemble of particles through the nonlinear stochastic dynamics."""

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


_SUPPORTED_RESAMPLERS = ("multinomial", "soft", "stop_gradient")


def _validate_resample_method(method: str) -> str:
    """Ensure the requested resampling strategy is implemented."""
    method_lower = method.lower()
    if method_lower not in _SUPPORTED_RESAMPLERS:
        raise ValueError(
            f"Unsupported resample_method '{method}'. "
            f"Supported methods are {_SUPPORTED_RESAMPLERS}."
        )
    return method_lower


def _multinomial_resample(key, x, log_w):
    probs = jnp.exp(log_w - logsumexp(log_w))
    idx = jr.choice(key, x.shape[0], shape=(x.shape[0],), p=probs, replace=True)
    x_resampled = x[idx]
    log_w_resampled = jnp.full_like(log_w, -jnp.log(x.shape[0]))
    return x_resampled, log_w_resampled, idx


def _stop_gradient_resample(key, x, log_w, *, base_method: str = "multinomial"):
    """Resample while passing gradients through the chosen particles."""
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
):
    """Differentiable particle filter with configurable resampling (default stop-gradient)."""
    n_particles = int(hyperparams.N_particles)
    T = ys.shape[0]
    resample_method = _validate_resample_method(hyperparams.resample_method)

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

        particles_hist = particles
        logw_hist = log_w

        # TODO: Only resample under criteria, e.g., ESS < threshold.
        if resample_method == "soft":
            particles, log_w = _soft_resample(
                key_resample, particles, log_w, hyperparams.softness
            )
        elif resample_method == "multinomial":
            particles, log_w, _ = _multinomial_resample(key_resample, particles, log_w)
        else:  # stop_gradient
            particles, log_w = _stop_gradient_resample(
                key_resample, particles, log_w, base_method="multinomial"
            )

        return (particles, log_w, log_evidence), (particles_hist, logw_hist)

    (particles, log_w, log_evidence), (particles_hist, logw_hist) = lax.scan(
        _step,
        (particles, log_w, log_evidence),
        (keys, idxs, t_currs, t_prevs, u_prevs, u_currs),
    )

    return particles_hist, logw_hist, log_evidence
