from logging import config
from typing import NamedTuple, Tuple, Optional, Union
from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map
import jax.random as jr
import jax.numpy as jnp
from dynamax.parameters import ParameterProperties, ParameterSet
import abc
from configparser import ConfigParser


def _get_params(self):
    ''' A function to return the parameters of the learnable function
        This is used for parameter initialization and sampling

        Returns:
            list of parameter dictionaries, each with keys
                "param_name"
                "param_shape" (the shape of the parameter)
                "params" (the actual parameters)
    '''

    params = []
    for field in self._fields:
        value = getattr(self, field)
        shape = getattr(value, "shape", ())  # use empty tuple if no .shape
        params.append({
            "param_name": field,
            "param_shape": shape,
            "params": value
        })

    return params

### CD-NLGSSM learnable functions
# Learnable function abstract definition
class LearnableFunction(NamedTuple):
    ''' All Learnable functions should have
        params propertie
        a definiton of a function that takes as input x, u and t
    '''
    # Parameters as properties of the class
    params: ParameterSet

    '''
    def __init__(
        self,
        params,
    ):
        self.params = params
    '''
    # A function definition
    @abc.abstractmethod
    def f(self, x, u=None, t=None):
        ''' A function to be defined by specific classes
            With inputs
            x: state
            u: inputs
            t: time
        '''

    def get_params(self):
        return _get_params(self)

# Simple learnable function examples
class LearnableVector(NamedTuple):
    params: Union[Float[Array, "dim"], ParameterProperties]

    def f(self, x=None, u=None, t=None):
        return self.params
    
    def get_params(self):
        return _get_params(self)

class LearnableMatrix(NamedTuple):
    params: Union[Float[Array, "row_dim col_dim"], ParameterProperties]

    def f(self, x=None, u=None, t=None):
        return self.params
    
    def get_params(self):
        return _get_params(self)

# Typically use this with learnable scale and a fixed (e.g. Identity) matrix
class LearnableScaledMatrix(NamedTuple):
    scale: Union[Float[Array, "1"], ParameterProperties]
    matrix: Union[Float[Array, "row_dim col_dim"], ParameterProperties]

    def f(self, x=None, u=None, t=None):
        return self.scale * self.matrix

    def get_params(self):
        return _get_params(self)

# Typically use this for learnable diagonal matrix
class LearnableDiagonalMatrix(NamedTuple):
    diag_params: Union[Float[Array, "row_dim"], ParameterProperties]

    def f(self, x=None, u=None, t=None):
        return jnp.diag(self.diag_params)

    def get_params(self):
        return _get_params(self)

class LearnableLinear(NamedTuple):
    '''Linear function with learnable parameters
            weights: weights of the linear function
            bias: bias of the linear function

            f(x) = weights @ x + bias
    '''
    weights: Union[Float[Array, "output_dim input_dim"], ParameterProperties]
    bias: Union[Float[Array, "output_dim"], ParameterProperties]

    def f(self, x, u=None, t=None):
        return self.weights @ x + self.bias
    
    def get_params(self):
        return _get_params(self)

# More examples, specific to data_driven or physics_driven models are provided separately

#### Parameter definitions
# To avoid unnecessary redefinitions of code,
# We import parameters that can be reused from LGSSM first
# And define the rest later
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial

## CDNLGSSM parameter class definitions
# Continuous non-linear Gaussian dynamics
class ParamsCDNLGSSMDynamics(NamedTuple):
    r"""Parameters of the state dynamics of a CDNLGSSM model.

    This model does not obey an SDE as in Sarkaa's equation (3.151):
        the solution to 3.151 is not necessarily a Gaussian Process
            (note there are cases where that is indeed the case)

    We instead assume an approximation to the model of zero-th, first or second order

    The resulting transition and emission distributions are
    $$p(z_1) = N(z_1 | m, S)$$
    $$p(z_t | z_{t-1}, u_t) = N(z_t | z_t, P_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    The tuple doubles as a container for the ParameterProperties.

    :param drift_function: $f$
    :param diffusion_coefficient: $L$
    :param diffusion_cov: $Q$
    :param dynamics_approx: 'zeroth', 'first' or 'second'

    """
    '''
    # the deterministic drift $f$ of the nonlinear RHS of the state
    drift_function: Union[FnStateToState, FnStateAndInputToState]
    # the coefficient matrix L of the state's diffusion process
    diffusion_coefficient: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], ParameterProperties]
    # The covariance matrix Q of the state noise process
    diffusion_cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]
    '''
    
    # These are all learnable functions to be initialized
    drift: LearnableFunction
    diffusion_coefficient: LearnableFunction
    diffusion_cov: LearnableFunction
    
    # Dynamics SDE approximation order, defined as a Float
    approx_order: Union[Float, ParameterProperties]

'''
# Continuous non-linear dynamic parameters
class ParamsCDNLSSMDynamics(NamedTuple):
    r"""Parameters of the state dynamics of a CDNLGSSM model.

    This model does obey the SDE as in Sarkaa's equation (3.151):
        the solution to 3.151 is not necessarily a Gaussian Process
            (note there are cases where that is indeed the case)

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    The tuple doubles as a container for the ParameterProperties.

    :param drift_function: $f$
    :param diffusion_coefficient: $L$
    :param diffusion_cov: $Q$

    """
    # the deterministic drift $f$ of the nonlinear RHS of the state
    drift_function: Union[FnStateToState, FnStateAndInputToState]
    # the coefficient matrix L of the state's diffusion process
    diffusion_coefficient: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], ParameterProperties]
    # The covariance matrix Q of the state noise process
    diffusion_cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]
'''

# Discrete non-linear emission parameters
# TODO: function definitions within parameter classes breaks fit_sgd: where should they be placed?
class ParamsCDNLGSSMEmissions(NamedTuple):
    r"""Parameters of the state dynamics

    $$p(z_{t+1} \mid z_t, u_t) = \mathcal{N}(z_{t+1} \mid A z_t + B u_t + b, Q)$$

    The tuple doubles as a container for the ParameterProperties.

    :param drift_function: $f$
    :param diffusion_coefficient: $L$
    :param diffusion_cov: $Q$
    :param dynamics_approx: 'zeroth', 'first' or 'second'

    """
    # These are all learnable functions to be initialized
    emission_function: LearnableFunction
    emission_cov: LearnableFunction
    
    '''
    # Emission distribution h
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    # The covariance matrix R of the observation noise process
    emission_cov: Union[Float[Array, "emission_dim emission_dim"], ParameterProperties]
    '''

# CDNLGSSM parameters are different to CDLGSSM due to nonlinearities
class ParamsCDNLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    The assumed transition and emission distributions are
    $$p(z_1) = N(z_1 | m, S)$$
    $$p(z_t | z_{t-1}, u_t) = N(z_t | m_t, P_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$

    """
    initial: ParamsLGSSMInitial
    dynamics: ParamsCDNLGSSMDynamics
    emissions: ParamsCDNLGSSMEmissions 

'''
# CDNLSSM parameters are different to CDNLGSSM due to non-gaussian transitions
class ParamsCDNLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    The assumed transition and emission distributions are
    $$p(z_1) = N(z_1 | m, S)$$
    
    """
    initial: ParamsLGSSMInitial
    dynamics: ParamsCDNLSSMDynamics
    emissions: ParamsCDNLGSSMEmissions 
'''

# Some auxiliary functions for parameter handling
## Only use the values above if the user hasn't specified their own
default = lambda x, x0: x if x is not None else x0

## Create CD-NLGSSM parameters and properties, based on provided dictionaries
def create_cdnlgssm_params_and_props(
        params: dict
    ) -> Tuple[ParamsCDNLGSSM, ParameterProperties]:
    r"""Create CD-LGSSM parameters and properties, based on provided dictionaries

    Args:
        :param params: dictionary of parameters

    Returns:
        :return: Tuple of parameters and properties objects
    """
    ## Create nested dictionary of params
    params_and_props = {"params": {}, "props": {}}

    for key in params_and_props.keys():
        params_and_props[key] = ParamsCDNLGSSM(
            initial=ParamsLGSSMInitial(
                mean=params["initial_mean"][key],
                cov=params["initial_cov"][key]
            ),
            dynamics=ParamsCDNLGSSMDynamics(
                drift=params["dynamics_drift"][key],
                diffusion_coefficient=params["dynamics_diffusion_coefficient"][key],
                diffusion_cov=params["dynamics_diffusion_cov"][key],
                approx_order=params["dynamics_approx_order"][key],
            ),
            emissions=ParamsCDNLGSSMEmissions(
                emission_function=params["emission_function"][key],
                emission_cov=params["emission_cov"][key],
            )
        )

    return params_and_props["params"], params_and_props["props"]

# Create CD-NLGSSM parameters and properties, based on the provided prior, init_values or defaults
def init_cdnlgssm_params(
        default_params,
        init_params = None,
        init_prior = None,
        key = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDNLGSSM, ParamsCDNLGSSM]:
    r"""Create CD-NLGSSM parameters and properties, based on the provided prior, init_values or defaults

    Args:
        :param default_params: dictionary of default parameters: we at least need some default values
        :param init_params: dictionary of all parameters
        :param init_prior: prior distribution for the initialization. Defaults to None.
        :param key: random key for sampling. Defaults to jr.PRNGKey(0).

    Returns:
        :return: dictionary of parameters and properties
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
            if dict_name not in ['initial_mean', 'initial_cov', 'dynamics_drift', 'dynamics_diffusion_coefficient', 'dynamics_diffusion_cov', 'dynamics_approx_order', 'emission_function', 'emission_cov']:
                raise ValueError(f"Unknown parameter dictionary name: {dict_name}")
            
            # Replace the provided params with the sampled ones
            print('Initializing {} with sampled parameters'.format(dict_name))
            # Note that for CD-NLGSSM we have learnable functions, so reinstantiate them
            
            # Because we sample only one set of parameters, we need to remove the extra dimension
            # We create a new dictionary for the parameters,
            # with key=param['param_name']
            # value = param['params'][0] (the first and only sample)
            params_dict = {}
            for param in sampled_params[dict_name].get_params(): params_dict[param['param_name']] = param['params'][0]
            # And create a new learnable function with the sampled parameters
            params[dict_name]["params"] = sampled_params[dict_name].__class__(
                **params_dict
            )

    # Create and return CD-NLGSSM parameter and properties objects
    return create_cdnlgssm_params_and_props(params)


# Sample CD-NLGSSM parameters, based on the provided prior and init_values
# TODO: revise this as above, to match the new structure, use get_params()
def sample_cdnlgssm_params(
        prior,
        M,
        init_params,
        key = jr.PRNGKey(0),
    ) -> ParamsCDNLGSSM:
    r"""Sample CD-NLGSSM parameters from the provided prior, with init_params used for non-sampled parameters

    Args:
        :param prior: prior distribution for the initialization.
        :param M: number of samples to draw
        :param init_params: dictionary of all parameters
        :param key: random key for sampling from the prior. Defaults to jr.PRNGKey(0).

    Returns:
        :return: ParamsCDNLGSSM object
    """

    # First, make sure we have all the necessary parameters
    params = init_params
    
    # Making sure we broadcast actual "params" to the number of samples
    for dict_key in init_params.keys():
        # Consider only parameters defined as dictionaries (with keys "params" and "props")
        if isinstance(init_params[dict_key], dict):
            # If the init_params[dict_key] has a "params" as a scalar, we need to broadcast it
            if isinstance(init_params[dict_key]["params"], (float, int)):
                params[dict_key]["params"] = jnp.broadcast_to(
                    init_params[dict_key]["params"],
                    (M,1)
                )
            # If the init_params[dict_key] has a "params" as an array, we can broadcast it
            elif isinstance(
                init_params[dict_key]["params"],
                (jnp.ndarray, Array)
            ):
                params[dict_key]["params"] = jnp.broadcast_to(
                    init_params[dict_key]["params"],
                    (M,) + init_params[dict_key]["params"].shape
                )
            else:
                # Note that for CD-NLGSSM we have learnable functions, so we reinstantiate them

                # Only proceed if the object has a get_params() method
                if hasattr(init_params[dict_key]["params"], "get_params"):
                    # Recover initial parameters, and broadcast their values according to the number of samples M
                    params_dict = {}
                    for param in init_params[dict_key]["params"].get_params():
                        params_dict[param['param_name']] = jnp.broadcast_to(
                                param['params'],
                                (M,) + param['params'].shape
                            )
                    # Create a new learnable function with the broadcasted parameters    
                    params[dict_key]["params"] = init_params[dict_key]["params"].__class__(
                            **params_dict
                        )
                else:
                    # If not a learnable function, skip broadcasting or handle as needed
                    raise NotImplementedError("Non-learnable functions with no get_params function are not supported.")

    # Draw parameters from the provided prior
    sampled_params = prior.sample(
        key=key,
        M = M
    )
    # And replace the provided params with the sampled ones
    for dict_name in sampled_params.keys():
        if dict_name not in ['initial_mean', 'initial_cov', 'dynamics_drift', 'dynamics_diffusion_coefficient', 'dynamics_diffusion_cov', 'dynamics_approx_order', 'emission_function', 'emission_cov']:
            raise ValueError(f"Unknown parameter dictionary name: {dict_name}")
        
        # Replace. Note that for CD-NLGSSM we have learnable functions
        params[dict_name]["params"] = sampled_params[dict_name]

    # Create and return CD-NLGSSM parameter and properties objects
    return create_cdnlgssm_params_and_props(params)


def update_params(params, updates: dict):
    """
    Returns a copy of `params` with all updates applied.
    
    updates: dict with keys like "initial.mean.params" or "dynamics.drift.sigma"
    
    Example usage:
        updates = {
            "initial.mean.params": jnp.ones(3),  # Vector of ones
            "dynamics.drift.sigma": 11.3  # Scalar value
        }
        new_params = update_params(params, updates)
    """

    def set_nested_attr(obj, attr_path: str, value):
        """
        Recursively returns a copy of `obj` with the nested attribute replaced by `value`.
        Works for NamedTuples and Eqx Modules too.
        
        attr_path: e.g., "dynamics.drift.sigma"
        """
        attrs = attr_path.split(".")
        if len(attrs) == 1:
            # Base case: replace the final attribute
            return obj._replace(**{attrs[0]: value})
        else:
            first, rest = attrs[0], ".".join(attrs[1:])
            nested_obj = getattr(obj, first)
            updated_nested = set_nested_attr(nested_obj, rest, value)
            return obj._replace(**{first: updated_nested})

    updated = params
    for path, val in updates.items():
        updated = set_nested_attr(updated, path, val)
    return updated

