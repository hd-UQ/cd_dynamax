# build_params_linear.py

import jax.numpy as jnp
from typing import Optional, Callable, Union

# Dynamax-style param containers you already use
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions
from .cdlgssm_utils import ParamsCDLGSSM, ParamsCDLGSSMDynamics

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

    # Dynamics: dx = F(t) x dt + B(t) u dt + b(t) dt + L(t) dW,  Qc(t) = cov(dW)
    dynamics_drift_weights: ParamSpec = None,
    dynamics_input_weights: ParamSpec = None,
    dynamics_bias: ParamSpec = None,
    diffusion_coeff: ParamSpec = None,
    diffusion_cov: ParamSpec = None,

    # Emissions: y_t ~ N(H(t) x_t + D(t) u_t + d(t), R(t))
    emission_weights: ParamSpec = None,
    emission_input_weights: ParamSpec = None,
    emission_bias: ParamSpec = None,
    emission_cov: ParamSpec = None,

    # Initial distribution z_0 ~ N(x0_mean, x0_cov)
    x0_mean: Optional[ArrayLike] = None,
    x0_cov: Optional[ArrayLike] = None,

) -> ParamsCDLGSSM:
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
    ParamsCDLGSSM
        A parameter container compatible with your cdlgssm_* functions.

    """

    # ---------- Defaults (constants), if user didn’t pass anything ----------
    # Initials
    x0_mean = jnp.zeros(state_dim) if x0_mean is None else x0_mean
    x0_cov = jnp.eye(state_dim) if x0_cov is None else x0_cov

    # Dynamics
    dynamics_drift_weights = (-0.1 * jnp.eye(state_dim)) if dynamics_drift_weights is None else dynamics_drift_weights # state-to-state
    dynamics_input_weights = jnp.zeros((state_dim, input_dim)) if dynamics_input_weights is None else dynamics_input_weights # input-to-state
    dynamics_bias = jnp.zeros(state_dim) if dynamics_bias is None else dynamics_bias # state bias

    diffusion_coeff = jnp.eye(state_dim) if diffusion_coeff is None else diffusion_coeff
    diffusion_cov = jnp.eye(state_dim) if diffusion_cov is None else diffusion_cov

    # Emissions
    emission_weights = jnp.eye(emission_dim, state_dim) if emission_weights is None else emission_weights
    emission_bias = jnp.zeros(emission_dim) if emission_bias is None else emission_bias # observation bias
    emission_input_weights = jnp.zeros((emission_dim, input_dim)) if emission_input_weights is None else emission_input_weights #Input-to-emissions weights
    emission_cov = jnp.eye(emission_dim) if emission_cov is None else emission_cov # observation cov

    # ---------- Package in the structures your KF expects ----------
    initial = ParamsLGSSMInitial(mean=x0_mean, cov=x0_cov)
    dynamics = ParamsCDLGSSMDynamics(
        weights=dynamics_drift_weights,
        bias=dynamics_bias,
        input_weights=dynamics_input_weights,
        diffusion_coefficient=diffusion_coeff,
        diffusion_cov=diffusion_cov,
    )
    emissions = ParamsLGSSMEmissions(
        weights=emission_weights,
        bias=emission_bias,
        input_weights=emission_input_weights,
        cov=emission_cov,
    )

    return ParamsCDLGSSM(initial=initial, dynamics=dynamics, emissions=emissions)
