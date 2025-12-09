# JAX imports
import jax.random as jr
import jax.numpy as jnp

# Type annotations
from typing import NamedTuple, Tuple, Optional, Union
from jaxtyping import Array, Float

# Imports from dynamax
from cd_dynamax.dynamax.parameters import ParameterProperties

#### Parameter definitions
# To avoid unnecessary redefinitions of code,
# We import those that can be reused from dynamax first
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions

### CD-LGSSM parameter class definitions
# Continuous-discrete linear Gaussian dynamics
class ParamsCDLGSSMDynamics(NamedTuple):
    r"""Parameters of the CD-LGSSM state dynamics
            The tuple doubles as a container for the ParameterProperties.
    
    We assume a model of the form
        $dz_t = F_t z_t dt + L_t dB_t$

    The resulting transition distribution is
        $p(z_{t1}} | z_{t0}, u_t1) = N(z_{t1} | A z_{t0} + B u_{t1} + b, P)$
    where A, B, b, P are computed based on the SDE defined by F_t, L_t and Q.

    :param weights: dynamics weights $F_t$
    :param bias: dynamics bias $b$
    :param input_weights: dynamics input weights $B$
    :param diffusion_coefficient: dynamics diffusion coefficient $L_t$
    :param diffusion_cov: dynamics covariance $Q$

    """
    # F_t: parameters and properties
    weights: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"]]
    
    # b: parameters and properties
    bias: Union[ParameterProperties,
        Float[Array, "state_dim"],
        Float[Array, "ntime state_dim"]]
    
    # B: parameters and properties
    input_weights: Union[ParameterProperties,
        Float[Array, "state_dim input_dim"],
        Float[Array, "ntime state_dim input_dim"]]
    
    # L_t: parameters and properties
    diffusion_coefficient: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"]
    ]
    # Q: parameters and properties
    diffusion_cov: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"],
        Float[Array, "state_dim_triu"]]

# Set of CD-LGSSM parameters
class ParamsCDLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian CD-LGSSM.

    :param initial: initial distribution parameters, same as in LGSSM
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    """
    initial: ParamsLGSSMInitial
    dynamics: ParamsCDLGSSMDynamics
    emissions: ParamsLGSSMEmissions

# Object definition used when forecasting with a CD-LGSSM (also useful for CD-NLGSSM)
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
    # State means and covariances
    forecasted_state_means: Optional[Float[Array, "ntime state_dim"]] = None
    forecasted_state_covariances: Optional[Float[Array, "ntime state_dim"]] = None
    # Emission means and covariances
    forecasted_emission_means: Optional[Float[Array, "ntime state_dim"]] = None
    forecasted_emission_covariances: Optional[Float[Array, "ntime state_dim"]] = None

    # If we forecast paths, based on solving the SDE
    # State and emission paths
    forecasted_state_path: Optional[Float[Array, "ntime state_dim"]] = None
    forecasted_emission_path: Optional[Float[Array, "ntime state_dim"]] = None

### Auxiliary functions for parameter handling
default = lambda x, x0: x if x is not None else x0
_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)

### Create CD-LGSSM parameters and properties, based on provided dictionaries
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

# Initialize CD-LGSSM parameters and properties
# based on the provided prior, init_values or defaults
def init_cdlgssm_params(
        default_params,
        init_params = None,
        init_prior = None,
        key = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
    r"""Initialize CD-LGSSM parameters and properties,
        based on sampling from the provided prior, init_values or defaults

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
    for dict_key in init_params.keys():
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

# Sample CD-LGSSM parameters,
# based on the provided prior and init_values
def sample_cdlgssm_params(
        prior,
        M,
        init_params,
        key = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
    r"""Sample CD-LGSSM parameters from the provided prior,
        with init_params used for non-sampled parameters

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
        if dict_name not in [
                'initial_mean',
                'initial_cov',
                'dynamics_weights',
                'dynamics_input_weights',
                'dynamics_bias',
                'dynamics_diffusion_coefficient',
                'dynamics_diffusion_cov',
                'emission_weights',
                'emission_input_weights',
                'emission_bias',
                'emission_cov'
            ]:
            raise ValueError(f"Unknown parameter dictionary name: {dict_name}")
        
        # Replace the provided params with the sampled ones
        params[dict_name]["params"] = sampled_params[dict_name]
    
    # Create and return CD-LGSSM parameter and properties objects
    return create_cdlgssm_params_and_props(params)

