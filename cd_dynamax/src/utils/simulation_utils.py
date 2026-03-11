# This file contains utility functions for simulation of cd dynamax models

# Imports
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

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
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
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
    dpf_moments
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

# cd-dynamax function to filter, based on model with given parameters
def cddynamax_filter(
    model_params,
    filter_hyperparams,
    t_emissions,
    emissions,
    start_idx_filter,
    stop_idx_filter,
    key=jr.PRNGKey(0),
    filter_spec="model",
    warn=True
):
    if filter_spec == "model":
        # Check whether model_params are linear or nonlinear, based on class type
        # To decide what filtering to use
        if isinstance(model_params, ParamsCDLGSSM):
            # Linear case
            filtering_function = cdlgssm_filter
            extra_args_filter = {}
        elif isinstance(model_params, ParamsCDNLGSSM):
            # Nonlinear Gaussian case
            filtering_function = cdnlgssm_filter
            extra_args_filter = {"key": key}
        elif isinstance(model_params, ParamsCDNLSSM):
            # Nonlinear case
            filtering_function = cdnlssm_filter
            extra_args_filter = {"key": key}
    
    elif filter_spec == "filter":
        # Check filter_hyperarams type, based on class type
        # To decide what filtering to use
        if isinstance(filter_hyperparams, KFHyperParams):
            # Linear case
            filtering_function = cdlgssm_filter
            extra_args_filter = {}
        # EKF, UKF or EnKF
        elif isinstance(filter_hyperparams, (EKFHyperParams, UKFHyperParams, EnKFHyperParams)):
            # Nonlinear Gaussian case
            filtering_function = cdnlgssm_filter
            extra_args_filter = {"key": key}
        elif isinstance(filter_hyperparams, DPFHyperParams):
            # Nonlinear case
            filtering_function = cdnlssm_filter
            extra_args_filter = {"key": key}

    # Run filter on filtering time points
    filtered = filtering_function(
        params=model_params,
        emissions=emissions[start_idx_filter:stop_idx_filter],
        t_emissions=t_emissions[start_idx_filter:stop_idx_filter],
        filter_hyperparams=filter_hyperparams,
        **extra_args_filter,
        warn=warn,
    )

    return filtered

# cd-dynamax function to forecast, based on model with given parameters
def cddynamax_forecast(
    model_params,
    filter_hyperparams,
    init_forecast, # This can be a distribution or a fixed state depending on the model and filter type
    t_init,
    t_forecast,
    key,
    filter_spec="model",
    particle_weights=None,
    warn=True
):

    if filter_spec == "model":
        # Check whether model_params are linear or nonlinear, based on class type
        # To decide what filtering to use
        if isinstance(model_params, ParamsCDLGSSM):
            # Linear case
            forecasting_function = cdlgssm_forecast
            extra_args_forecast = {}
        elif isinstance(model_params, ParamsCDNLGSSM):
            # Nonlinear Gaussian case
            forecasting_function = cdnlgssm_forecast
            extra_args_forecast = {"key": key}
        elif isinstance(model_params, ParamsCDNLSSM):
            # Nonlinear case
            forecasting_function = cdnlssm_forecast
            extra_args_forecast = {"key": key}
    
    elif filter_spec == "filter":
        # Check filter_hyperarams type, based on class type
        # To decide what filtering to use
        if isinstance(filter_hyperparams, KFHyperParams):
            # Linear case
            forecasting_function = cdlgssm_forecast
            extra_args_forecast = {}
        # EKF, UKF or EnKF
        elif isinstance(filter_hyperparams, (EKFHyperParams, UKFHyperParams, EnKFHyperParams)):
            # Nonlinear Gaussian case
            forecasting_function = cdnlgssm_forecast
            extra_args_forecast = {"key": key}
        elif isinstance(filter_hyperparams, DPFHyperParams):
            # Nonlinear case
            forecasting_function = cdnlssm_forecast
            extra_args_forecast = {"key": key}

    # Run forecast on forecasting time points
    forecasted = forecasting_function(
        params=model_params,
        init_forecast=init_forecast,
        t_init=t_init,
        t_forecast=t_forecast,
        filter_hyperparams=filter_hyperparams,
        **extra_args_forecast,
        warn=warn
    )

    # For DPF, we compute mean and covariance of forecasted particles for evaluation purposes
    if isinstance(filter_hyperparams, DPFHyperParams):
        # Make a copy of the forecasted object, and add mean and covariance to it
        particles = forecasted # shape num_timesteps_forecast \times M \times state_dim
        
        # If particle weights are not provided, use uniform weights
        if particle_weights is None:
            particle_weights = jnp.ones(particles.shape[1]) / particles.shape[1]
        
        # Weight the particles by the particle weights from the last filtering step, and compute weighted mean and covariance
        # first particle axis is time, second axis is particles, third axis is state dimension
        forecasted_means, forecasted_covariances = vmap(
            dpf_moments,
            in_axes=(0, None)
        )(
            particles,
            particle_weights
        )

        # CDLGSSM forecasting definition
        from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import GSSMForecast
        forecasted = GSSMForecast(
            forecasted_state_means=forecasted_means,
            forecasted_state_covariances=forecasted_covariances,
            forecasted_state_path=particles
        )

    return forecasted
    
# cd-dynamax function to compute emissions, based on model with given parameters, and latent states
def cddynamax_emissions(
        model,
        model_params,
        t_emissions_filter,
        filtered_state,
        t_emissions_forecast=None,
        forecasted_state=None,
        filtering_inputs=None,
        forecasting_inputs=None,
        key=jr.PRNGKey(0),
        warn=True
    ):

    # TODO: check emissions_covariance computations
    # Emission generation function based on model type
    if isinstance(model, ContDiscreteLinearGaussianSSM):
        from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.inference import (
            cdlgssm_emissions,
        )

        cddynamax_emissions_f = cdlgssm_emissions
    elif isinstance(model, ContDiscreteNonlinearGaussianSSM):
        from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import (
            cdnlgssm_emissions,
        )

        cddynamax_emissions_f = cdnlgssm_emissions
    elif isinstance(model, ContDiscreteNonlinearSSM):
        from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
            cdnlssm_emissions,
        )

        cddynamax_emissions_f = cdnlssm_emissions
    else:
        raise ValueError(
            "Model type not supported for emissions generation in filter-then-forecast."
        )
    
    # For Gaussian-based emissions, we can compute the emissions means and covariances directly from the filtered and forecasted states.
    if isinstance(model, (ContDiscreteLinearGaussianSSM, ContDiscreteNonlinearGaussianSSM)):
        # Generate emissions means/covs from filtered states
        f_emissions_mean, f_emissions_cov = cddynamax_emissions_f(
            params=model_params,
            t_states=t_emissions_filter,
            state_means=filtered_state.filtered_means,
            state_covs=filtered_state.filtered_covariances,
            inputs=filtering_inputs,
            key=key,
            warn=warn
        )

        # Dictionary with results
        filtered_emissions={
            'mean': f_emissions_mean,
            'cov': f_emissions_cov,
        }

        if t_emissions_forecast is not None:
            # Generate emissions means/covs from forecasted states
            fc_emissions_mean, fc_emissions_cov = cddynamax_emissions_f(
                params=model_params,
                t_states=t_emissions_forecast,
                state_means=forecasted_state.forecasted_state_means,
                state_covs=forecasted_state.forecasted_state_covariances,
                inputs=forecasting_inputs,
                key=key,
                warn=warn
            )

            # Dictionary with results
            forecasted_emissions={
                'mean': fc_emissions_mean,
                'cov': fc_emissions_cov,
            }


    elif isinstance(model, ContDiscreteNonlinearSSM):
        # For the more general nonlinear case, we can only compute the emissions by pushing the filtered and forecasted particles through the model's emission function.
        f_emissions = cddynamax_emissions_f(
            params=model_params,
            t_states=t_emissions_filter,
            states=filtered_state.particles,
            inputs=filtering_inputs,
            key=key,
            warn=warn
        )

        # We compute mean and covariance of emission particles for evaluation purposes
        # If particle weights are not provided, use uniform weights
        if filtered_state.log_weights is None:
            particle_weights = jnp.ones(filtered_state.particles.shape[1]) / filtered_state.particles.shape[1]
        else:
            particle_weights = jnp.exp(filtered_state.log_weights[-1,...]) # shape (num_particles,)
        
        # Weight the particles by the particle weights from the last filtering step, and compute weighted mean and covariance
        # first particle axis is time, second axis is particles, third axis is state dimension
        f_emissions_mean, f_emissions_cov = vmap(
            dpf_moments,
            in_axes=(0, None)
        )(
            f_emissions,
            particle_weights
        )

        # Dictionary with results
        filtered_emissions={
            'path': f_emissions,
            'mean': f_emissions_mean,
            'cov': f_emissions_cov,
        }
        
        if t_emissions_forecast is not None:
            fc_emissions = cddynamax_emissions_f(
                params=model_params,
                t_states=t_emissions_forecast,
                states=forecasted_state.forecasted_state_path, # We expect particles to be saved here
                inputs=forecasting_inputs,
                key=key,
                warn=warn
            )

            # Weight the particles by the particle weights from the last filtering step, and compute weighted mean and covariance
            # first particle axis is time, second axis is particles, third axis is state dimension
            fc_emissions_mean, fc_emissions_cov = vmap(
                dpf_moments,
                in_axes=(0, None)
            )(
                fc_emissions,
                particle_weights
            )
            
            # Dictionary with results
            forecasted_emissions={
                'path': fc_emissions,
                'mean': fc_emissions_mean,
                'cov': fc_emissions_cov,
            }
    
    # Return
    return filtered_emissions, forecasted_emissions
        
    
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
    filter_spec="model",
    warn=True
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

    # Filter
    filtered = cddynamax_filter(
        model_params=model_params,
        filter_hyperparams=filter_hyperparams,
        t_emissions=t_emissions,
        emissions=emissions,
        start_idx_filter=start_idx_filter,
        stop_idx_filter=stop_idx_filter,
        key=next(keys),
        filter_spec=filter_spec,
        warn=warn
    )    

    # Initialize forecast with last filtered state
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

    # Forecast
    forecasted = cddynamax_forecast(
        model_params=model_params,
        filter_hyperparams=filter_hyperparams,
        init_forecast=init_forecast,
        t_init=t_emissions[stop_idx_filter - 1],
        t_forecast=t_emissions[start_idx_forecast:stop_idx_forecast],
        key=next(keys),
        filter_spec=filter_spec,
        warn=warn
    )

    return (
        filtered,
        forecasted,
        start_idx_filter,
        stop_idx_filter,
        start_idx_forecast,
        stop_idx_forecast,
    )
