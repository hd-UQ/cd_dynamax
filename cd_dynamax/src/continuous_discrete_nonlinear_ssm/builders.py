import equinox as eqx
import inspect
from jax.scipy.special import gammaln
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .cdnlssm_utils import (
    ParamsCDNLSSM,
    ParamsCDNLSSMEmissions,
    ParamsCDNLSSMInitial,
    ParamsCDNLSSMDynamics,
)

from ..continuous_discrete_nonlinear_gaussian_ssm.builders import (
    wrap_function_or_constant,
    _default,
)


# -------------------------
# Distribution wrappers
# -------------------------
class StaticDistribution(eqx.Module):
    distribution: tfd.Distribution

    def log_prob(self, y, x=None, u=None, t=None):
        return self.distribution.log_prob(y)

    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)


class ConditionalDistribution(eqx.Module):
    distribution_fn: callable

    def _dist(self, x=None, u=None, t=None):
        sig = inspect.signature(self.distribution_fn)
        kwargs = {}
        if "x" in sig.parameters:
            kwargs["x"] = x
        if "u" in sig.parameters:
            kwargs["u"] = u
        if "t" in sig.parameters:
            kwargs["t"] = t
        return self.distribution_fn(**kwargs)

    def log_prob(self, y, x=None, u=None, t=None):
        return self._dist(x, u, t).log_prob(y)

    def sample(self, x=None, u=None, t=None, *args, **kwargs):
        return self._dist(x, u, t).sample(*args, **kwargs)


class GaussianEmission(eqx.Module):
    emission_function: eqx.Module
    emission_cov: eqx.Module

    def log_prob(self, y, x=None, u=None, t=None):
        mean = self.emission_function.f(x, u, t)
        cov = self.emission_cov.f(x, u, t)
        return tfd.MultivariateNormalFullCovariance(mean, cov).log_prob(y)

    def sample(self, x=None, u=None, t=None, *args, **kwargs):
        mean = self.emission_function.f(x, u, t)
        cov = self.emission_cov.f(x, u, t)
        return tfd.MultivariateNormalFullCovariance(mean, cov).sample(*args, **kwargs)


class PoissonEmission(eqx.Module):
    dt: float = eqx.field(static=True)
    bias: jnp.ndarray

    def log_prob(self, y, x=None, u=None, t=None):
        log_rate = x[..., 0] + self.bias + jnp.log(self.dt)
        y0 = jnp.squeeze(jnp.asarray(y, dtype=log_rate.dtype))
        return y0 * log_rate - jnp.exp(log_rate) - gammaln(y0 + 1.0)

    def sample(self, x=None, u=None, t=None, *args, **kwargs):
        log_rate = x[..., 0] + self.bias + jnp.log(self.dt)
        return tfd.Poisson(log_rate=log_rate).sample(*args, **kwargs)


class TransformedDistribution(eqx.Module):
    base_distribution: eqx.Module
    transform: eqx.Module

    def log_prob(self, y, x=None, u=None, t=None):
        return self.base_distribution.log_prob(self.transform.f(y, u, t), x, u, t)

    def sample(self, x=None, u=None, t=None, *args, **kwargs):
        base_sample = self.base_distribution.sample(x, u, t)
        return self.transform.f(base_sample, u, t, *args, **kwargs)


# -------------------------
# Param construction
# -------------------------
def _build_initial_distribution(
    state_dim, initial_distribution, initial_mean, initial_cov
):
    if initial_distribution is not None:
        if hasattr(initial_distribution, "distribution"):
            return initial_distribution
        if callable(initial_distribution):
            return StaticDistribution(distribution=initial_distribution())
        return StaticDistribution(distribution=initial_distribution)

    # Default to a Gaussian if initial_distribution is None
    mean = wrap_function_or_constant(
        _default(initial_mean, jnp.zeros(state_dim)), expected_shape=(state_dim,)
    ).f()
    cov = wrap_function_or_constant(
        _default(initial_cov, jnp.eye(state_dim)), expected_shape=(state_dim, state_dim)
    ).f()
    return StaticDistribution(
        distribution=tfd.MultivariateNormalFullCovariance(mean, cov)
    )


def _build_emission_distribution(
    emission_dim, emission_distribution, emission_function, emission_cov
):
    if emission_distribution is not None:
        if isinstance(emission_distribution, tfd.Distribution):
            return StaticDistribution(distribution=emission_distribution)
        if hasattr(emission_distribution, "log_prob"):
            return emission_distribution
        return ConditionalDistribution(distribution_fn=emission_distribution)

    emission_fn = wrap_function_or_constant(
        _default(emission_function, lambda x, u=None, t=None: x[..., :emission_dim])
    )
    emission_cov_fn = wrap_function_or_constant(
        _default(emission_cov, jnp.eye(emission_dim)),
        expected_shape=(emission_dim, emission_dim),
    )
    return GaussianEmission(
        emission_function=emission_fn,
        emission_cov=emission_cov_fn,
    )


def build_params(
    state_dim,
    emission_dim,
    drift,
    initial_mean=None,
    initial_cov=None,
    emission_function=None,
    emission_cov=None,
    initial_distribution=None,
    emission_distribution=None,
    diffusion_coeff=None,
    diffusion_cov=None,
    approx_order: float = 1.0,
):
    """
    Build parameters for a CDNLSSM model mirroring the CDNLGSSM interface.
    Callable parameters should accept only (x, u, t) or a subset of these.
    """
    return ParamsCDNLSSM(
        initial=ParamsCDNLSSMInitial(
            initial_distribution=_build_initial_distribution(
                state_dim, initial_distribution, initial_mean, initial_cov
            )
        ),
        dynamics=ParamsCDNLSSMDynamics(
            drift=wrap_function_or_constant(drift),
            diffusion_coefficient=wrap_function_or_constant(
                _default(diffusion_coeff, jnp.eye(state_dim)),
                expected_shape=(state_dim, state_dim),
            ),
            diffusion_cov=wrap_function_or_constant(
                _default(diffusion_cov, jnp.eye(state_dim)),
                expected_shape=(state_dim, state_dim),
            ),
            approx_order=approx_order,
        ),
        emissions=ParamsCDNLSSMEmissions(
            emission_distribution=_build_emission_distribution(
                emission_dim, emission_distribution, emission_function, emission_cov
            ),
        ),
    )
