from typing import NamedTuple, Tuple, Optional, Union
from jaxtyping import Array, Float, PyTree
from jax.tree_util import tree_map
import jax.random as jr
import jax.numpy as jnp
from dynamax.parameters import ParameterProperties, ParameterSet

#### Parameter definitions
# To avoid unnecessary redefinitions of code,
# We import those that can be reused from LGSSM first
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions

# Continuous dynamic distributions are different than discrete ones, we define them here
class ParamsCDLGSSMDynamics(NamedTuple):
    r"""Parameters of the dynamics distribution

    $$p(z_{t+1} \mid z_t, u_t) = \mathcal{N}(z_{t+1} \mid A z_t + B u_t + b, Q)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: dynamics weights $F$ -> used to compute A based on ODE
    :param bias: dynamics bias $b$
    :param input_weights: dynamics input weights $B$
    :param cov: dynamics covariance $Q$

    """
    weights: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"]]
    
    bias: Union[ParameterProperties,
        Float[Array, "state_dim"],
        Float[Array, "ntime state_dim"]]
    
    input_weights: Union[ParameterProperties,
        Float[Array, "state_dim input_dim"],
        Float[Array, "ntime state_dim input_dim"]]
    
    diffusion_coefficient: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"]
    ]
    diffusion_cov: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"],
        Float[Array, "state_dim_triu"]]

# CDLGSSM parameters are different to LGSSM due to different dynamics
class ParamsCDLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    """
    initial: ParamsLGSSMInitial
    dynamics: ParamsCDLGSSMDynamics
    emissions: ParamsLGSSMEmissions

class GSSMForecast(NamedTuple):
    r"""Object definition used when forecasting.

    # If we forecast Gaussian distributions, based on filtering methods
    :param forecasted_state_means: array of forecasted state means $\mathbb{E}[z_{t+1:t+t_f} \mid y_{1:t}, u_{1:t}, u_{t+1:t+f}]$
    :param filtered_covariances: array of forecasted state covariances $\mathrm{Cov}[z_{t+1:t+t_f} \mid y_{1:t}, u_{1:t}, u_{t+1:t+f}]$
    :param forecasted_emission_means: array of forecasted emission means $\mathbb{E}[y_{t+1:t+t_f} \mid y_{1:t}, u_{1:t}, u_{t+1:t+f}]$
    :param forecasted_emission_covariances: array of forecasted emission covariances $\mathrm{Cov}[y_{t+1:t+t_f} \mid y_{1:t}, u_{1:t}, u_{t+1:t+f}]$

    # If we forecast paths, based on solving the SDE
    :param forecasted_state_path: array of forecasted state path $z_{t+1:t+t_f}$ 
    :param forecasted_emission_path: array of forecasted emission path $y_{t+1:t+t_f}$
    """

    # If we forecast Gaussian distributions, based on filtering methods
    forecasted_state_means: Optional[Float[Array, "ntime state_dim"]] = None
    forecasted_state_covariances: Optional[Float[Array, "ntime state_dim"]] = None
    forecasted_emission_means: Optional[Float[Array, "ntime state_dim"]] = None
    forecasted_emission_covariances: Optional[Float[Array, "ntime state_dim"]] = None

    # If we forecast paths, based on solving the SDE
    forecasted_state_path: Optional[Float[Array, "ntime state_dim"]] = None

# Some auxiliary functions for parameter handling
## Only use the values above if the user hasn't specified their own
default = lambda x, x0: x if x is not None else x0

## Create CD-LGSSM parameters and properties, based on provided dictionaries
def create_cdlgssm_params_and_props(
        params: dict
    ) -> Tuple[ParamsCDLGSSM, ParameterProperties]:
    r"""Create CD-LGSSM parameters and properties, based on provided dictionaries

    Args:
        :param params: dictionary of parameters

    Returns:
        :return: Tuple of parameters and properties objects
    """

    ## Create nested dictionary of params
    params_and_props = {"params": {}, "props": {}}

    # Iterate over params and properties
    for key in params_and_props.keys():
        params_and_props[key] = ParamsCDLGSSM(
            initial=ParamsLGSSMInitial(
                mean=params["initial_mean"][key],
                cov=params["initial_cov"][key]
            ),
            dynamics=ParamsCDLGSSMDynamics(
                    weights=params["dynamics_weights"][key],
                    input_weights=params["dynamics_input_weights"][key],
                    bias=params["dynamics_bias"][key],
                    diffusion_coefficient=params["dynamics_diffusion_coefficient"][key],
                    diffusion_cov=params["dynamics_diffusion_cov"][key],
                ),
            emissions=ParamsLGSSMEmissions(
                weights=params["emission_weights"][key],
                input_weights=params["emission_input_weights"][key],
                bias=params["emission_bias"][key],
                cov=params["emission_cov"][key],
            )
        )

    return params_and_props["params"], params_and_props["props"]

# Create CD-LGSSM parameters and properties, based on the provided prior, init_values or defaults
def init_cdlgssm_params(
        default_params,
        init_params = None,
        init_prior = None,
        key = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
    r"""Create CD-LGSSM parameters and properties, based on sampling from the provided prior, init_values or defaults

    Args:
        :param default_params: dictionary of default parameters: we at least need some default values
        :param init_params: dictionary of all parameters
        :param init_prior: prior distribution for the initialization. Defaults to None.
        :param key: random key for sampling from the prior. Defaults to jr.PRNGKey(0).

    Returns:
        :return: Tuple of CD-LGSSM parameters and properties objects
    """
    
    # First, make sure we have all the necessary default parameters
    params = default_params

    # Replace defaults with provided initialization as needed
    for dict_key in params.keys():
        params[dict_key] = default(
            init_params[dict_key],
            default_params[dict_key]
        )
    
    # If init_prior is provided, sample from the prior
    if init_prior is not None:
        # Draw a single parameter from the prior
        sampled_params = init_prior.sample(
            key=key,
            M = 1
        )
        for dict_name in sampled_params.keys():
            if dict_name not in ['initial_mean', 'initial_cov', 'dynamics_weights', 'dynamics_input_weights', 'dynamics_bias', 'dynamics_diffusion_coefficient', 'dynamics_diffusion_cov', 'emission_weights', 'emission_input_weights', 'emission_bias', 'emission_cov']:
                raise ValueError(f"Unknown parameter dictionary name: {dict_name}")
            
            # Replace the provided params with the sampled ones only if they are None
            if params[dict_name]["params"] is None:
                print('Initializing {} with sampled parameters'.format(dict_name))
                # Because we only draw 1 sample, do not keep an extra first dimension of parameters
                params[dict_name]["params"] = sampled_params[dict_name][0]
            else:
                print('Ignoring sampled parameters for {}, keeping user initialized values'.format(dict_name))

    # Create and return CD-LGSSM parameter and properties objects
    return create_cdlgssm_params_and_props(params)

# Sample CD-LGSSM parameters, based on the provided prior and init_values
def sample_cdlgssm_params(
        prior,
        M,
        init_params,
        key = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
    r"""Sample CD-LGSSM parameters from the provided prior, with init_params used for non-sampled parameters

    Args:
        :param prior: prior distribution for the initialization.
        :param M: number of samples to draw
        :param init_params: dictionary of all parameters
        :param key: random key for sampling from the prior. Defaults to jr.PRNGKey(0).

    Returns:
        :return: Tuple of CD-LGSSM parameters and properties objects
    """
    
    # First, make sure we have all the necessary parameters
    params = init_params

    # Making sure we broadcast actual "params" to the number of samples
    for dict_key in init_params.keys():
        params[dict_key]["params"] = jnp.broadcast_to(
            init_params[dict_key]["params"],
            (M,) + init_params[dict_key]["params"].shape
        )
    
    # Draw parameters from the provided prior
    sampled_params = prior.sample(
        key=key,
        M = M
    )
    
    # And replace the provided params with the sampled ones
    for dict_name in sampled_params.keys():
        if dict_name not in ['initial_mean', 'initial_cov', 'dynamics_weights', 'dynamics_input_weights', 'dynamics_bias', 'dynamics_diffusion_coefficient', 'dynamics_diffusion_cov', 'emission_weights', 'emission_input_weights', 'emission_bias', 'emission_cov']:
            raise ValueError(f"Unknown parameter dictionary name: {dict_name}")
        
        # Replace
        params[dict_name]["params"] = sampled_params[dict_name]
    
    # Create and return CD-LGSSM parameter and properties objects
    return create_cdlgssm_params_and_props(params)

# Auxiliary functions for parameter handling
_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)
def make_cdlgssm_params(initial_mean,
                      initial_cov,
                      dynamics_weights,
                      dynamics_diffusion_coeff,
                      dynamics_diffusion_cov,
                      emissions_weights,
                      emissions_cov,
                      dynamics_bias=None,
                      dynamics_input_weights=None,
                      emissions_bias=None,
                      emissions_input_weights=None):
    """Helper function to construct a ParamsCDLGSSM object from arguments."""
    state_dim = len(initial_mean)
    emission_dim = emissions_cov.shape[-1]
    input_dim = max(dynamics_input_weights.shape[-1] if dynamics_input_weights is not None else 0,
                    emissions_input_weights.shape[-1] if emissions_input_weights is not None else 0)

    params = ParamsCDLGSSM(
        initial=ParamsLGSSMInitial(
            mean=initial_mean,
            cov=initial_cov
        ),
        dynamics=ParamsCDLGSSMDynamics(
            weights=dynamics_weights,
            bias=_zeros_if_none(dynamics_bias,state_dim),
            input_weights=_zeros_if_none(dynamics_input_weights, (state_dim, input_dim)),
            diffusion_coefficient=dynamics_diffusion_coeff,
            diffusion_cov=dynamics_diffusion_cov,
        ),
        emissions=ParamsLGSSMEmissions(
            weights=emissions_weights,
            bias=_zeros_if_none(emissions_bias, emission_dim),
            input_weights=_zeros_if_none(emissions_input_weights, (emission_dim, input_dim)),
            cov=emissions_cov
        )
    )
    return params