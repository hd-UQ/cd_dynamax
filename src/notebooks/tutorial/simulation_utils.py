# This file contains utility functions for simulating data and evaluating the performance of the CD-NLGSSM model.

# Imports
from typing import Tuple, Optional
from jaxtyping import Float, Array
import jax.numpy as jnp
import jax.random as jr

# For distributional forecasting, import MVN
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

# Our own custom src codebase
# continuous-discrete nonlinear Gaussian SSM codebase
from continuous_discrete_nonlinear_gaussian_ssm import cdnlgssm_filter, cdnlgssm_forecast

### Simulation Utilities
# Function to generate irregular measurement time-points
# uniformly sampled from a time domain $[0,T_total]$
def generate_irregular_t_emissions(
        T_total: Float,
        num_timesteps: int,
        T_filter: Optional[Float] = None,
        key=jr.PRNGKey(0)
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
    
    # Generate random num_timesteps time points
    t_emissions = jnp.array(
        sorted(
            jr.uniform(
                key,
                (num_timesteps, 1),
                minval=0,
                maxval=T_total
            )
        )
    )
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
    return t_emissions, t_filter, t_forecast, num_timesteps, num_timesteps_filter, num_timesteps_forecast

# Function to filter and forecast, based on model with given parameters
def filter_and_forecast(
    model_params,
    filter_hyperparams,
    t_emissions,
    emissions,
    T0=50,
    T_filter_end=70,
    T_forecast_end=100,
    ):

    # Figure out the time points for filtering
    start_idx_filter = jnp.where(t_emissions >= T0)[0][0]
    stop_idx_filter = jnp.where(t_emissions >= T_filter_end)[0][0]

    # Figure out the time points for forecasting
    start_idx_forecast = jnp.where(t_emissions >= T_filter_end)[0][0]
    stop_idx_forecast = jnp.where(t_emissions >= T_forecast_end)[0][0]

    # Run filter on filtering time points
    filtered = cdnlgssm_filter(
        params=model_params,
        emissions=emissions[start_idx_filter:stop_idx_filter],
        t_emissions=t_emissions[start_idx_filter:stop_idx_filter],
        hyperparams=filter_hyperparams,
    )

    # Initialize forecast with last filtered state
    init_time = t_emissions[stop_idx_filter]
    init_forecast = MVN(filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :])

    forecasted = cdnlgssm_forecast(
        params=model_params,
        init_forecast=init_forecast,
        t_init=init_time,
        t_forecast=t_emissions[start_idx_forecast:stop_idx_forecast],
        hyperparams=filter_hyperparams,
    )

    return filtered, forecasted, start_idx_filter, stop_idx_filter, start_idx_forecast, stop_idx_forecast

### Evaluation
# Compute RMSE
def compute_rmse(y, y_est):
    return jnp.sqrt(jnp.sum((y - y_est) ** 2) / len(y))

# Compute RMSE of estimate and print comparison with
# standard deviation of measurement noise
def compute_and_print_rmse_comparison(y, y_est, R, est_type=""):
    rmse_est = compute_rmse(y, y_est)
    print(f'{f"The RMSE of the {est_type} estimate is":<40}: {rmse_est:.2f}')
    print(f'{"The std of measurement noise is":<40}: {jnp.sqrt(R):.2f}')