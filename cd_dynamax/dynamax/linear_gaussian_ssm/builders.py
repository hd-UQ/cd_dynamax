# build_params_linear.py

import jax.numpy as jnp
from typing import Optional, Callable, Union

# Dynamax-style param containers you already use
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSMInitial,
    ParamsLGSSMEmissions,
    ParamsLGSSMDynamics,
    ParamsLGSSM
)

ArrayLike = jnp.ndarray
MaybeArray = Optional[ArrayLike]
MaybeCallable = Optional[Callable[[float], ArrayLike]]
ParamSpec = Union[ArrayLike, Callable[[float], ArrayLike], None]


def build_params(
    *,
    # Required model sizes
    state_dim: int,
    emission_dim: int,
    input_dim: int = 0, # no inputs is default assumption

    has_dynamics_bias: bool = False,
    has_emissions_bias: bool = False,

    # Dynamics: Ax + Bu + b + N(0,Q)
    dynamics_weights: ParamSpec = None,
    dynamics_input_weights: ParamSpec = None,
    dynamics_bias: ParamSpec = None,
    dynamics_cov: ParamSpec = None,

    # Emissions: y_t ~ N(H(t) x_t + D(t) u_t + d(t), R(t))
    emission_weights: ParamSpec = None,
    emission_input_weights: ParamSpec = None,
    emission_bias: ParamSpec = None,
    emission_cov: ParamSpec = None,

    # Initial distribution z_0 ~ N(x0_mean, x0_cov)
    x0_mean: Optional[ArrayLike] = None,
    x0_cov: Optional[ArrayLike] = None,

) -> ParamsLGSSM:
    """
    Build a Continuous-Discrete *Linear* Gaussian SSM parameter set that plugs
    straight into your cdlgssm_* (filter/smoother/forecast/…).

    All parameters may be:
      • a constant JAX array (recommended for time-invariant systems), or
      • a callable of the form `fn(t) -> array` (recommended for time-varying systems).

    Arguments
    ---------
    state_dim, emission_dim : int
        Sizes of the latent state and observation.

    Initial distribution
    --------------------
    x0_mean : (D,), default zeros
    x0_cov  : (D,D), default I

    Returns
    -------
    ParamsLGSSM
        A parameter container compatible with your cdlgssm_* functions.

    """


    # Arbitrary default values, for demo purposes.
    _x0_mean = jnp.zeros(state_dim)
    _x0_cov = jnp.eye(state_dim)
    _dynamics_weights = 0.99 * jnp.eye(state_dim)
    _dynamics_input_weights = jnp.zeros((state_dim, input_dim))
    _dynamics_bias = jnp.zeros((state_dim,)) if has_dynamics_bias else None
    _dynamics_cov = 0.1 * jnp.eye(state_dim)
    _emission_weights = jnp.eye(emission_dim, state_dim)
    _emission_input_weights = jnp.zeros((emission_dim, input_dim))
    _emission_bias = jnp.zeros((emission_dim,)) if has_emissions_bias else None
    _emission_cov = 0.1 * jnp.eye(emission_dim)

    # Only use the values above if the user hasn't specified their own
    default = lambda x, x0: x if x is not None else x0

    # Create nested dictionary of params
    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=default(x0_mean, _x0_mean),
            cov=default(x0_cov, _x0_cov)),
        dynamics=ParamsLGSSMDynamics(
            weights=default(dynamics_weights, _dynamics_weights),
            bias=default(dynamics_bias, _dynamics_bias),
            input_weights=default(dynamics_input_weights, _dynamics_input_weights),
            cov=default(dynamics_cov, _dynamics_cov)),
        emissions=ParamsLGSSMEmissions(
            weights=default(emission_weights, _emission_weights),
            bias=default(emission_bias, _emission_bias),
            input_weights=default(emission_input_weights, _emission_input_weights),
            cov=default(emission_cov, _emission_cov))
        )

    return params
