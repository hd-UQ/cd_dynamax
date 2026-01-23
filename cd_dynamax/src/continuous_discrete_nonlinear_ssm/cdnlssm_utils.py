from typing import NamedTuple, Tuple, Union
import abc
import jax.numpy as jnp
import jax.random as jr
from jax import Array as JaxArray
from cd_dynamax.dynamax.parameters import ParameterProperties, ParameterSet
import tensorflow_probability.substrates.jax.distributions as tfd

from ..continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import (
    _get_params,
    LearnableFunction,
    LearnableVector,
    LearnableMatrix,
    ParamsCDNLGSSMDynamics,
)


class LearnableDistribution(NamedTuple):
    distribution: tfd.Distribution
    params: ParameterSet

    @abc.abstractmethod
    def log_prob(self, x, u=None, t=None):
        """Compute the log probability of the distribution"""

    @abc.abstractmethod
    def sample(self, x, u=None, t=None):
        """Sample from the distribution"""

    def get_params(self):
        return _get_params(self)


class LearnableMultivariateNormal(LearnableDistribution):
    def __init__(self, mean: LearnableVector, covariance: LearnableMatrix):
        self.mean = mean
        self.covariance = covariance
        self.params = self.mean.params + self.covariance.params
        self.distribution = tfd.MultivariateNormalFullCovariance(
            self.mean, self.covariance
        )

    def log_prob(self, x, u=None, t=None):
        return self.distribution.log_prob(x)

    def sample(self, x, u=None, t=None):
        return self.distribution.sample(x)


class LearnableTransformedDistribution(LearnableDistribution):
    """Distribution of the transformed variable f(x) where x ~ base_distribution."""

    def __init__(
        self, base_distribution: LearnableDistribution, transform: LearnableFunction
    ):
        self.base_distribution = base_distribution
        self.transform = transform
        self.params = base_distribution.params + transform.params
        self.distribution = base_distribution.distribution

    def log_prob(self, x, u=None, t=None):
        transformed = self.transform.f(x, u, t)
        return self.base_distribution.log_prob(transformed, u, t)

    def sample(self, x, u=None, t=None):
        base_sample = self.base_distribution.sample(x, u, t)
        return self.transform.f(base_sample, u, t)


# Currently, we only support Brownian motion-driven SDEs, so we can reuse the CDNLGSSM dynamics parameters
ParamsCDNLSSMDynamics = ParamsCDNLGSSMDynamics


## CDNLSSM parameter class definitions
# Continuous non-linear dynamics
class ParamsCDNLSSMInitial(NamedTuple):
    initial_distribution: LearnableDistribution


# Discrete non-linear emission parameters
class ParamsCDNLSSMEmissions(NamedTuple):
    emission_distribution: LearnableDistribution


# CDNLGSSM parameters are different to CDLGSSM due to nonlinearities
class ParamsCDNLSSM(NamedTuple):
    r"""Parameters of a continuous-discrete nonlinear SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    The assumed transition and emission distributions are
    $$p(z_0) = p_initial(z_0)$$
    $$p(z_{t_k} | z_{t_{k-1}}, u_{t_k}) = solve_sde(z_{t_{k-1}}, u_{t_k}, t_{k-1}, t_k, f_dynamics, L_dynamics, Q_dynamics)$$
    $$p(y_{t_k} | z_{t_k}) = p_emissions(y_{t_k} | z_{t_k})$$
    """

    initial: ParamsCDNLSSMInitial
    dynamics: ParamsCDNLSSMDynamics
    emissions: ParamsCDNLSSMEmissions


## Only use the values above if the user hasn't specified their own
def default(x, x0):
    return x if x is not None else x0


def create_cdnlssm_params_and_props(
    params: dict,
) -> Tuple[ParamsCDNLSSM, ParameterProperties]:
    """Create CD-NLSSM parameters and properties, based on provided dictionaries."""
    params_and_props = {"params": {}, "props": {}}

    for key in params_and_props.keys():
        params_and_props[key] = ParamsCDNLSSM(
            initial=ParamsCDNLSSMInitial(
                initial_distribution=params["initial_distribution"][key],
            ),
            dynamics=ParamsCDNLSSMDynamics(
                drift=params["dynamics_drift"][key],
                diffusion_coefficient=params["dynamics_diffusion_coefficient"][key],
                diffusion_cov=params["dynamics_diffusion_cov"][key],
                approx_order=params["dynamics_approx_order"][key],
            ),
            emissions=ParamsCDNLSSMEmissions(
                emission_distribution=params["emission_distribution"][key],
            ),
        )

    return params_and_props["params"], params_and_props["props"]


def init_cdnlssm_params(
    default_params,
    init_params=None,
    init_prior=None,
    key=jr.PRNGKey(0),
) -> Tuple[ParamsCDNLSSM, ParamsCDNLSSM]:
    """Create CD-NLSSM parameters and properties using provided prior, init_values or defaults."""
    params = default_params

    for dict_key in init_params.keys():
        params[dict_key] = default(init_params[dict_key], default_params[dict_key])

    if init_prior is not None:
        sampled_params = init_prior.sample(key=key, M=1)
        for dict_name in sampled_params.keys():
            if dict_name not in [
                "initial_distribution",
                "dynamics_drift",
                "dynamics_diffusion_coefficient",
                "dynamics_diffusion_cov",
                "dynamics_approx_order",
                "emission_distribution",
            ]:
                raise ValueError(f"Unknown parameter dictionary name: {dict_name}")

            params_dict = {}
            for param in sampled_params[dict_name].get_params():
                params_dict[param["param_name"]] = param["params"][0]
            params[dict_name]["params"] = sampled_params[dict_name].__class__(
                **params_dict
            )

    return create_cdnlssm_params_and_props(params)


def sample_cdnlssm_params(
    prior,
    M,
    init_params,
    key=jr.PRNGKey(0),
) -> ParamsCDNLSSM:
    """Sample CD-NLSSM parameters from the provided prior."""
    params = init_params

    for dict_key in init_params.keys():
        if isinstance(init_params[dict_key], dict):
            if isinstance(init_params[dict_key]["params"], (float, int)):
                params[dict_key]["params"] = jnp.broadcast_to(
                    init_params[dict_key]["params"],
                    (M, 1),
                )
            elif isinstance(init_params[dict_key]["params"], (jnp.ndarray, JaxArray)):
                params[dict_key]["params"] = jnp.broadcast_to(
                    init_params[dict_key]["params"],
                    (M,) + init_params[dict_key]["params"].shape,
                )
            else:
                if hasattr(init_params[dict_key]["params"], "get_params"):
                    params_dict = {}
                    for param in init_params[dict_key]["params"].get_params():
                        params_dict[param["param_name"]] = jnp.broadcast_to(
                            param["params"],
                            (M,) + param["params"].shape,
                        )
                    params[dict_key]["params"] = init_params[dict_key][
                        "params"
                    ].__class__(**params_dict)
                else:
                    raise NotImplementedError(
                        "Non-learnable functions with no get_params function are not supported."
                    )

    sampled_params = prior.sample(key=key, M=M)
    for dict_name in sampled_params.keys():
        if dict_name not in [
            "initial_distribution",
            "dynamics_drift",
            "dynamics_diffusion_coefficient",
            "dynamics_diffusion_cov",
            "dynamics_approx_order",
            "emission_distribution",
        ]:
            raise ValueError(f"Unknown parameter dictionary name: {dict_name}")

        params[dict_name]["params"] = sampled_params[dict_name]

    return create_cdnlssm_params_and_props(params)


def update_params(params, updates: dict):
    """Returns a copy of `params` with all updates applied."""

    def set_nested_attr(obj, attr_path: str, value):
        attrs = attr_path.split(".")
        if len(attrs) == 1:
            return obj._replace(**{attrs[0]: value})
        first, rest = attrs[0], ".".join(attrs[1:])
        nested_obj = getattr(obj, first)
        updated_nested = set_nested_attr(nested_obj, rest, value)
        return obj._replace(**{first: updated_nested})

    updated = params
    for path, val in updates.items():
        updated = set_nested_attr(updated, path, val)
    return updated
