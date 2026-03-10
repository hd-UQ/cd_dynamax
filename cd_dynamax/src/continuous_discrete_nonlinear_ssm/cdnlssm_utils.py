# JAX imports
import jax.numpy as jnp
import jax.random as jr

# Type annotations 
from typing import NamedTuple, Tuple, Union
from jax import Array as JaxArray
from jaxtyping import Array, Float

# For cd-dynamax's abstract classes
import abc

# Imports from dynamax
from cd_dynamax.dynamax.parameters import ParameterProperties, ParameterSet

# Imports from cdnlgssm, including learnable functions and parameters, to avoid redefining them here (as the dynamics are the same, just with different distributions)
from ..continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import (
    _get_params,
    LearnableFunction,
    LearnableMatrix,
    ParamsCDNLGSSMDynamics,
)

# TensorFlow Probability imports for distributions
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)
from jax.scipy.special import gammaln

### CD-NLSSM learnable functions, now defined as distributions (as Gaussianity is not assumed anymore)
# Definition of an abstract learnable distribution
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

# Multivariate normal distribution with learnable mean and covariance
class StaticGaussianDistribution(NamedTuple):
    mean: Union[Float[Array, " dim"], ParameterProperties]
    cov: Union[Float[Array, " dim dim"], ParameterProperties]

    def log_prob(self, y):
        return MVN(self.mean, self.cov).log_prob(y)

    def sample(self, *args, **kwargs):
        return MVN(self.mean, self.cov).sample(*args, **kwargs)

# Multivariate normal distribution with learnable mean function and covariance matrix
# These can depend on the state, input and time
class LearnableGaussianEmission(NamedTuple):
    emission_function: Union[LearnableFunction, ParameterProperties]
    emission_covariance: Union[LearnableMatrix, ParameterProperties]

    def log_prob(self, y, x=None, u=None, t=None):
        mean = self.emission_function.f(x, u, t)
        cov = self.emission_covariance.f(x, u, t)
        return MVN(mean, cov).log_prob(y)

    def sample(self, x=None, u=None, t=None, *args, **kwargs):
        mean = self.emission_function.f(x, u, t)
        cov = self.emission_covariance.f(x, u, t)
        return MVN(mean, cov).sample(*args, **kwargs)
    
# Poisson emission distribution for count data, with learnable bias and time bin size
# These can depend on the state, input and time
class LearnablePoissonEmission(NamedTuple):
    r"""Learnable Poisson emission distribution for count data.

    The Poisson distribution is defined by the following probability mass function:
        $P(Y=y) = \frac{\lambda^y e^{-\lambda}}{y!}$
    where $\lambda$ is the rate parameter of the distribution.
    """
    # Learnable parameters
    dt: Union[Float[Array, "1"], ParameterProperties]
    bias: Union[Float[Array, " dim"], ParameterProperties]
    
    def log_prob(self, y, x=None, u=None, t=None):
        log_rate = x[..., 0] + self.bias + jnp.log(self.dt)
        y0 = jnp.squeeze(jnp.asarray(y, dtype=log_rate.dtype))
        return y0 * log_rate - jnp.exp(log_rate) - gammaln(y0 + 1.0)

    def sample(self, x=None, u=None, t=None, *args, **kwargs):
        log_rate = x + self.bias + jnp.log(self.dt)
        return tfd.Poisson(log_rate=log_rate).sample(*args, **kwargs)
    
    def get_params(self):
        return _get_params(self.dt, self.bias)

### CDNLSSM parameter class definitions
# Continuous non-linear dynamics
class ParamsCDNLSSMInitial(NamedTuple):
    initial_distribution: LearnableDistribution

# Currently, we only support Brownian motion-driven SDEs,
# so we can reuse the CDNLGSSM dynamics parameters
ParamsCDNLSSMDynamics = ParamsCDNLGSSMDynamics

# Discrete non-linear emission parameters
class ParamsCDNLSSMEmissions(NamedTuple):
    emission_distribution: LearnableDistribution


### CDNLSSM parameter class definitions
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

### Create CD-NLSSM parameters and properties, based on provided dictionaries
def create_cdnlssm_params_and_props(
    params: dict,
) -> Tuple[ParamsCDNLSSM, ParameterProperties]:
    """Create CD-NLSSM parameters and properties, based on provided dictionaries.
    
    Args:
        :param params: dictionary of parameters

    Returns:
        :return: Tuple of parameters and properties objects
    """
    ## Create nested dictionary of params
    params_and_props = {"params": {}, "props": {}}

    # Iterate over params and properties
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


# Initialize CD-NLGSSM parameters and properties
# based on the provided prior, init_values or defaults
def init_cdnlssm_params(
    default_params,
    init_params=None,
    init_prior=None,
    key=jr.PRNGKey(0),
) -> Tuple[ParamsCDNLSSM, ParamsCDNLSSM]:
    """Create CD-NLSSM parameters and properties using provided prior, init_values or defaults.
    
    Args:
        default_params: dictionary of default parameters: we at least need some default values
        init_params: dictionary of all parameters
        init_prior: prior distribution for the initialization. Defaults to None.
        key: random key for sampling. Defaults to jr.PRNGKey(0).

    Returns:
        Tuple of CD-NLSSM parameters and properties objects
    """
    # First, make sure we have all the necessary default parameters
    params = default_params

    # Replace defaults with provided initialization as needed
    for dict_key in init_params.keys():
        params[dict_key] = default(init_params[dict_key], default_params[dict_key])

    # If init_prior is provided, sample from the prior
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

    # Create and return CD-NLGSSM parameter and properties objects
    return create_cdnlssm_params_and_props(params)


# Sample CD-NLSSM parameters,
# based on the provided prior and init_values
def sample_cdnlssm_params(
    prior,
    M,
    init_params,
    key=jr.PRNGKey(0),
) -> ParamsCDNLSSM:
    """Sample CD-NLSSM parameters from the provided prior.
    Args:
        :param prior: prior distribution for the initialization.
        :param M: number of samples to draw
        :param init_params: dictionary of all parameters
        :param key: random key for sampling from the prior. Defaults to jr.PRNGKey(0).

    Returns:
        :return: Tuple of CD-NLSSM parameters and properties objects
    """
    # First, make sure we have all the necessary parameters
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

    # Draw parameters from the provided prior
    sampled_params = prior.sample(key=key, M=M)
    # And replace the provided params with the sampled ones
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

# Utility function to update parameters of a CD-NLSSM
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
