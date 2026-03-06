# This file contains utility functions for simulation of cd dynamax models

# Imports
import jax.numpy as jnp
import jax.random as jr

from itertools import count

# Typing imports
from typing import Tuple, Optional
from jaxtyping import Float, Array

from jax.tree_util import tree_map

# For distributional forecasting, import MVN
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

# Our own custom src codebase
# continuous-discrete nonlinear Gaussian SSM codebase
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm import (
    cdlgssm_filter,
    cdlgssm_forecast,
    ParamsCDLGSSM,
    KFHyperParams
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    cdnlgssm_filter,
    cdnlgssm_forecast,
    ParamsCDNLGSSM,
    EKFHyperParams,
    UKFHyperParams,
    EnKFHyperParams,
)

from cd_dynamax.src.continuous_discrete_nonlinear_ssm import (
    cdnlssm_filter,
    cdnlssm_forecast,
    ParamsCDNLSSM,
    DPFHyperParams,
)


def tree_to_dict(tree):
    """Convert a JAX tree to a dictionary."""
    return tree_map(lambda x: x if x is not None else None, tree._asdict())


def make_key_sequence(seed: int):
    """Returns an infinite sequence of PRNG keys based on the given seed."""
    return map(jr.PRNGKey, count(start=seed, step=1))


### Simulation Utilities
# Function to generate irregular measurement time-points
# uniformly sampled from a time domain $[0,T_total]$
def generate_irregular_t_emissions(
    T_total: Float,
    num_timesteps: int,
    T_filter: Optional[Float] = None,
    key=jr.PRNGKey(0),
) -> Tuple[Array, Array, Array, int, int, int]:
    """
    Generate random time points for measurements, filtering and forecasting.
    sampled uniformly from a time domain $[0,T_total]$
    where user can specificy filtering and forecasting intervals

    Args:
    - T_total: Float, total time length
    - num_timesteps: int, total number of time points
    - T_filter: Optional[Float], end of filtering time
    - key: jr.PRNGKey, random key

    Returns:
    - t_emissions: Array, time points for measurements
    - t_filter: Array, time points for filtering
    - t_forecast: Array, time points for forecasting
    - num_timesteps: int, total number of time points
    - num_timesteps_filter: int, number of time points for filtering
    - num_timesteps_forecast: int, number of time points for forecasting
    """

    # This procedure produces times sampled uniformly from [0, T_total].
    u = jr.uniform(key, (num_timesteps,), minval=0, maxval=1)
    s = jnp.cumsum(u)  # Convert them into sorted cumulative sum
    # Normalize to [0, T_total] interval
    t_emissions = s / s[-1] * T_total

    # drop duplicates, and format as column vector
    t_emissions = jnp.unique(t_emissions)[:, None]

    # If interested, separate filtering and forecasting time points
    if T_filter is not None:
        t_filter = t_emissions[t_emissions <= T_filter, None]
        t_forecast = t_emissions[t_emissions > T_filter, None]
    else:
        t_filter = t_emissions
        t_forecast = None

    # Count number of time points
    num_timesteps = len(t_emissions)
    num_timesteps_filter = len(t_filter)
    num_timesteps_forecast = len(t_forecast) if t_forecast is not None else 0

    # Return time points and counts
    return (
        t_emissions,
        t_filter,
        t_forecast,
        num_timesteps,
        num_timesteps_filter,
        num_timesteps_forecast,
    )


# Function to filter and forecast, based on model with given parameters
def filter_and_forecast(
    model_params,
    filter_hyperparams,
    t_emissions,
    emissions,
    T0=50,
    T_filter_end=70,
    T_forecast_end=100,
    key=0,
    filter_spec="model"
):
    # Create a sequence of keys
    keys = make_key_sequence(key)

    # Figure out the time points for filtering
    start_idx_filter = jnp.where(t_emissions >= T0)[0][0]
    stop_idx_filter = jnp.where(t_emissions >= T_filter_end)[0][0]

    # Figure out the time points for forecasting
    start_idx_forecast = jnp.where(t_emissions >= T_filter_end)[0][0]
    stop_idx_forecast = jnp.where(t_emissions >= T_forecast_end)[0][0]

    assert start_idx_filter < stop_idx_filter, "Filtering time points are invalid."
    assert start_idx_forecast < stop_idx_forecast, (
        "Forecasting time points are invalid."
    )

    if filter_spec == "model":
        # Check whether model_params are linear or nonlinear, based on class type
        # To decide what filtering to use
        if isinstance(model_params, ParamsCDLGSSM):
            # Linear case
            filtering_function = cdlgssm_filter
            forecasting_function = cdlgssm_forecast
            extra_args_filter = {}
            extra_args_forecast = {}
        elif isinstance(model_params, ParamsCDNLGSSM):
            # Nonlinear Gaussian case
            filtering_function = cdnlgssm_filter
            forecasting_function = cdnlgssm_forecast
            extra_args_filter = {"key": next(keys)}
            extra_args_forecast = {"key": next(keys)}
        elif isinstance(model_params, ParamsCDNLSSM):
            # Nonlinear case
            filtering_function = cdnlssm_filter
            forecasting_function = cdnlssm_forecast
            extra_args_filter = {"key": next(keys)}
            extra_args_forecast = {"key": next(keys)}
    
    elif filter_spec == "filter":
        # Check filter_hyperarams type, based on class type
        # To decide what filtering to use
        if isinstance(filter_hyperparams, KFHyperParams):
            # Linear case
            filtering_function = cdlgssm_filter
            forecasting_function = cdlgssm_forecast
            extra_args_filter = {}
            extra_args_forecast = {}
        # EKF, UKF or EnKF
        elif isinstance(filter_hyperparams, EKFHyperParams) or isinstance(filter_hyperparams, UKFHyperParams) or isinstance(filter_hyperparams, EnKFHyperParams):
            # Nonlinear Gaussian case
            filtering_function = cdnlgssm_filter
            forecasting_function = cdnlgssm_forecast
            extra_args_filter = {"key": next(keys)}
            extra_args_forecast = {"key": next(keys)}
        elif isinstance(filter_hyperparams, DPFHyperParams):
            # Nonlinear case
            filtering_function = cdnlssm_filter
            forecasting_function = cdnlssm_forecast
            extra_args_filter = {"key": next(keys)}
            extra_args_forecast = {"key": next(keys)}

    # Run filter on filtering time points
    filtered = filtering_function(
        params=model_params,
        emissions=emissions[start_idx_filter:stop_idx_filter],
        t_emissions=t_emissions[start_idx_filter:stop_idx_filter],
        filter_hyperparams=filter_hyperparams,
        **extra_args_filter,
    )

    # Initialize forecast with last filtered state
    init_time = t_emissions[stop_idx_filter - 1]
    if filter_spec == "model":
        if isinstance(model_params, (ParamsCDLGSSM, ParamsCDNLGSSM)):
            init_forecast = MVN(
                filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :]
            )
        elif isinstance(model_params, ParamsCDNLSSM):
            init_forecast = filtered.particles[-1, ...]
    elif filter_spec == 'filter':
        # Non-Gaussian filters, use empirical distribution of particles as initial condition for forecasting
        if isinstance(filter_hyperparams, DPFHyperParams):
            init_forecast = filtered.particles[-1, ...]
        else:
            init_forecast = MVN(
                filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :]
            )

    # Run forecast on forecasting time points
    forecasted = forecasting_function(
        params=model_params,
        init_forecast=init_forecast,
        t_init=init_time,
        t_forecast=t_emissions[start_idx_forecast:stop_idx_forecast],
        filter_hyperparams=filter_hyperparams,
        **extra_args_forecast,
    )

    if isinstance(filter_hyperparams, DPFHyperParams):
        # For DPF, we compute mean and covariance of forecasted particles for evaluation purposes
        # TODO: shall we weight them by the particle weights? For now we just compute the unweighted empirical mean and covariance
        # Make a copy of the forecasted object, and add mean and covariance to it
        particles = forecasted # shape num_timesteps_forecast \times M \times state_dim
        forecasted_means = forecasted.mean(axis=1)
        centered = forecasted - forecasted_means[:, None, :]
        forecasted_covariances = jnp.einsum('tmi, tmj -> tij', centered, centered) / (particles.shape[1] - 1)
        # CDLGSSM forecasting definition
        from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import GSSMForecast
        forecasted = GSSMForecast(
            forecasted_state_means=forecasted_means,
            forecasted_state_covariances=forecasted_covariances,
            forecasted_state_path=particles
        )

    return (
        filtered,
        forecasted,
        start_idx_filter,
        stop_idx_filter,
        start_idx_forecast,
        stop_idx_forecast,
    )
