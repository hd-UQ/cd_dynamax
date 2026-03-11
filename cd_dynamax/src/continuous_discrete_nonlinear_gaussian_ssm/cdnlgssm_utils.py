# JAX imports
import jax.random as jr
import jax.numpy as jnp

# Type annotations
from typing import NamedTuple, Tuple, Union
from jaxtyping import Float
from jax import Array

# For cd-dynamax's abstract SSM class and method
from abc import abstractmethod

# Imports from dynamax
from cd_dynamax.dynamax.parameters import ParameterProperties, ParameterSet

#### Parameter definitions
# To avoid unnecessary redefinitions of code,
# We import those that can be reused from dynamax first
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial

### Auxiliary functions for learnable functions
# Auxiliary functions for parameter handling
default = lambda x, x0: x if x is not None else x0


# Function to get parameters of a learnable function
def _get_params(self):
    r"""A function to return the parameters of the learnable function of a CD-NLGSSM

    This is used for parameter initialization and sampling

    Returns:
        list of parameter dictionaries, each with keys
            "param_name"
            "param_shape" (the shape of the parameter)
            "params" (the actual parameters)
    """

    # Initialize list of parameters
    params = []
    # Loop over all fields of the NamedTuple
    for field in self._fields:
        # Get the value of the field
        value = getattr(self, field)
        # Get the shape of the value
        shape = getattr(value, "shape", ())  # use empty tuple if no .shape
        # Append to the list
        params.append({"param_name": field, "param_shape": shape, "params": value})

    # Return the list of parameters
    return params


### CD-NLGSSM learnable functions
# Definition of an abstract learnable function
class LearnableFunction(NamedTuple):
    r"""A definiton of a function that takes as input x, u and t

    All Learnable functions should have params properties
    """

    # Parameters as properties of the class
    params: ParameterSet

    # Abstract function definition
    @abstractmethod
    def f(self, x, u=None, t=None):
        """A function to be defined by each specific class
        With inputs
            x: state
            u: inputs
            t: time
        """

    # Method to get parameters
    def get_params(self):
        return _get_params(self)


# Simple learnable vector
class LearnableVector(NamedTuple):
    r"""A learnable vector
    i.e., f(x,u,t) = params = vector

    """

    # Parameters and properties
    params: Union[Float[Array, " dim"], ParameterProperties]

    # Function definition
    def f(self, x=None, u=None, t=None):
        return self.params

    # Method to get parameters
    def get_params(self):
        return _get_params(self)


# Simple learnable matrix
class LearnableMatrix(NamedTuple):
    r"""A learnable matrix
    i.e., f(x,u,t) = params = matrix

    """

    # Parameters and properties
    params: Union[Float[Array, "row_dim col_dim"], ParameterProperties]

    # Function definition
    def f(self, x=None, u=None, t=None):
        return self.params

    # Method to get parameters
    def get_params(self):
        return _get_params(self)


# Learnable matrix: scale and a fixed (e.g. Identity) matrix
class LearnableScaledMatrix(NamedTuple):
    r"""A learnable scaled matrix
    i.e., f(x,u,t) = scale * matrix

    """

    # Parameters and properties
    scale: Union[Float[Array, "1"], ParameterProperties]
    matrix: Union[Float[Array, "row_dim col_dim"], ParameterProperties]

    # Function definition
    def f(self, x=None, u=None, t=None):
        return self.scale * self.matrix

    # Method to get parameters
    def get_params(self):
        return _get_params(self)


# Learnable diagonal matrix
class LearnableDiagonalMatrix(NamedTuple):
    r"""A learnable diagonal matrix
    i.e., f(x,u,t) = diag_params = diagonal matrix

    """

    # Parameters and properties
    diag_params: Union[Float[Array, " row_dim"], ParameterProperties]

    # Function definition
    def f(self, x=None, u=None, t=None):
        return jnp.diag(self.diag_params)

    # Method to get parameters
    def get_params(self):
        return _get_params(self)


# A learnable linear function, based on weights and bias
class LearnableLinear(NamedTuple):
    r"""A linear function with learnable parameters
        i.e., f(x,u,t) = weights @ x + bias

    weights: weights of the linear function
    bias: bias of the linear function

    """

    # Parameters and properties
    weights: Union[Float[Array, "output_dim input_dim"], ParameterProperties]
    bias: Union[Float[Array, " output_dim"], ParameterProperties]

    # Function definition
    def f(self, x, u=None, t=None):
        if len(x.shape) == 1:
            out = self.weights @ x + self.bias
        else:
            # If x has shape (batch_size, input_dim), we need to do a batch matrix multiplication
            out = jnp.einsum("oi,bi->bo", self.weights, x) + self.bias
        return out

    # Method to get parameters
    def get_params(self):
        return _get_params(self)


# More learnable function examples
# specific to data_driven or physics_driven models are provided separately in
# ../utils/data_driven_models.py
# ../utils/physics_based_models.py


### CDNLGSSM parameter class definitions
# Continuous-discrete non-linear Gaussian dynamics
class ParamsCDNLGSSMDynamics(NamedTuple):
    r"""Parameters of the CD-NLGSSM state dynamics
        The tuple doubles as a container for the ParameterProperties.

    We assume a model of the form
        $dz_t = f(z_t, u_t) dt + L_t dB_t$

    The resulting transition distribution is
        $p(z_{t1}| z_{t0}, u_{t1}) = N(z_{t1} | m(z_{t0}, u_{t1}), P)$
    where the mean m(z_{t0}, u_{t1}) and covariance Q are computed
        based on numerically solving the SDE defined by f, L_t and Q.

    For the solution of the SDE, we use an approximation of order defined by dynamics_approx:

    This model does not obey the solution to the SDE as in Särkkä's equation (3.151):
        the true solution is not necessarily a Gaussian Process
            (note there are cases where that is indeed the case)
        we here approximate the solution at each time step with a Gaussian distribution.

    Args:
        drift_function: Drift $f$.
        diffusion_coefficient: Diffusion coefficient $L_t$.
        diffusion_cov: Diffusion covariance $Q$.
        dynamics_approx: One of 'zeroth', 'first', or 'second'.
    """

    # CD-NLGSSM dynamics are all defined in terms of learnable functions to be initialized
    # Drift function f
    drift: LearnableFunction
    # Diffusion coefficient L_t
    diffusion_coefficient: LearnableFunction
    # Diffusion covariance Q
    diffusion_cov: LearnableFunction

    # Dynamics SDE approximation order, defined as a Float
    approx_order: Union[Float, ParameterProperties]


# CD-NLGSSM emission parameters
class ParamsCDNLGSSMEmissions(NamedTuple):
    r"""Parameters of the CD-NLGSSM emission model.
            The tuple doubles as a container for the ParameterProperties.

    We assume a Gaussian observation model
        $p(y_k | z_k) = N(y_k | h(z_k), R)$
    where h is the emission function and R the observation noise covariance.

    Args:
        emission_function: Emission function h.
        emission_cov: Observation noise covariance R.
    """

    # These are all learnable functions to be initialized
    emission_function: LearnableFunction
    emission_cov: LearnableFunction


# Set of CD-NLGSSM parameters
class ParamsCDNLGSSM(NamedTuple):
    r"""Parameters of a nonlinear Gaussian CD-NLGSSM.

    Args:
        initial: Initial distribution parameters, same as in LGSSM.
        dynamics: Dynamics distribution parameters.
        emissions: Emission distribution parameters, same as in LGSSM.
    """

    initial: ParamsLGSSMInitial
    dynamics: ParamsCDNLGSSMDynamics
    emissions: ParamsCDNLGSSMEmissions


### Create CD-NLGSSM parameters and properties, based on provided dictionaries
def create_cdnlgssm_params_and_props(
    params: dict,
) -> Tuple[ParamsCDNLGSSM, ParameterProperties]:
    r"""Create CD-NLGSSM parameters and properties, based on provided dictionaries.

    Args:
        params: Dictionary of parameters.

    Returns:
        Tuple of parameters and properties objects.
    """
    ## Create nested dictionary of params
    params_and_props = {"params": {}, "props": {}}

    # Iterate over params and properties
    for key in params_and_props.keys():
        params_and_props[key] = ParamsCDNLGSSM(
            initial=ParamsLGSSMInitial(
                mean=params["initial_mean"][key], cov=params["initial_cov"][key]
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
            ),
        )

    return params_and_props["params"], params_and_props["props"]


# Initialize CD-NLGSSM parameters and properties
# based on the provided prior, init_values or defaults
def init_cdnlgssm_params(
    default_params,
    init_params=None,
    init_prior=None,
    key=jr.PRNGKey(0),
) -> Tuple[ParamsCDNLGSSM, ParamsCDNLGSSM]:
    r"""Initialize CD-NLGSSM parameters and properties from prior, init_values, or defaults.

    Args:
        default_params: Dictionary of default parameters; at least some default values are required.
        init_params: Dictionary of all parameters.
        init_prior: Prior distribution for the initialization. Defaults to None.
        key: Random key for sampling. Defaults to jr.PRNGKey(0).

    Returns:
        Tuple of CD-NLGSSM parameters and properties objects.
    """
    # First, make sure we have all the necessary default parameters
    params = default_params

    # Replace defaults with provided initialization as needed
    for dict_key in init_params.keys():
        params[dict_key] = default(init_params[dict_key], default_params[dict_key])

    # If init_prior is provided, sample from the prior
    if init_prior is not None:
        # Draw a single parameter from the prior
        sampled_params = init_prior.sample(key=key, M=1)
        for dict_name in sampled_params.keys():
            if dict_name not in [
                "initial_mean",
                "initial_cov",
                "dynamics_drift",
                "dynamics_diffusion_coefficient",
                "dynamics_diffusion_cov",
                "dynamics_approx_order",
                "emission_function",
                "emission_cov",
            ]:
                raise ValueError(f"Unknown parameter dictionary name: {dict_name}")

            # Replace the provided params with the sampled ones
            print("Initializing {} with sampled parameters".format(dict_name))
            # Note that for CD-NLGSSM we have learnable functions, so we reinstantiate them

            # Because we sample only one set of parameters, we need to remove the extra dimension
            # We create a new dictionary for the parameters,
            # with key=param['param_name'] and value = param['params'][0] (the first and only sample)
            params_dict = {}
            for param in sampled_params[dict_name].get_params():
                params_dict[param["param_name"]] = param["params"][0]
            # And create a new learnable function with the sampled parameters
            params[dict_name]["params"] = sampled_params[dict_name].__class__(
                **params_dict
            )

    # Create and return CD-NLGSSM parameter and properties objects
    return create_cdnlgssm_params_and_props(params)


# Sample CD-NLGSSM parameters,
# based on the provided prior and init_values
def sample_cdnlgssm_params(
    prior,
    M,
    init_params,
    key=jr.PRNGKey(0),
) -> Tuple[ParamsCDNLGSSM, ParamsCDNLGSSM]:
    r"""Sample CD-NLGSSM parameters from the provided prior; init_params used for non-sampled parameters.

    Args:
        prior: Prior distribution for the initialization.
        M: Number of samples to draw.
        init_params: Dictionary of all parameters.
        key: Random key for sampling from the prior. Defaults to jr.PRNGKey(0).

    Returns:
        Tuple of CD-NLGSSM parameters and properties objects.
    """

    # First, make sure we have all the necessary parameters
    params = init_params

    # Making sure we broadcast actual "params" to the number of samples
    for dict_key in init_params.keys():
        # Consider only parameters defined as dictionaries
        #   with keys "params" and "props"
        if isinstance(init_params[dict_key], dict):
            # If the init_params[dict_key] has a "params" as a scalar
            # we need to broadcast it
            if isinstance(init_params[dict_key]["params"], (float, int)):
                params[dict_key]["params"] = jnp.broadcast_to(
                    init_params[dict_key]["params"], (M, 1)
                )
            # If the init_params[dict_key] has a "params" as an array
            # we can broadcast it
            elif isinstance(init_params[dict_key]["params"], (jnp.ndarray, Array)):
                params[dict_key]["params"] = jnp.broadcast_to(
                    init_params[dict_key]["params"],
                    (M,) + init_params[dict_key]["params"].shape,
                )
            else:
                # Note that for CD-NLGSSM we have learnable functions, so we reinstantiate them

                # Only proceed if the object has a get_params() method
                if hasattr(init_params[dict_key]["params"], "get_params"):
                    # Recover initial parameters
                    # and broadcast their values according to the number of samples M
                    params_dict = {}
                    for param in init_params[dict_key]["params"].get_params():
                        params_dict[param["param_name"]] = jnp.broadcast_to(
                            param["params"], (M,) + param["params"].shape
                        )
                    # Create a new learnable function with the broadcasted parameters
                    params[dict_key]["params"] = init_params[dict_key][
                        "params"
                    ].__class__(**params_dict)
                else:
                    # If not a learnable function, skip broadcasting or handle as needed
                    raise NotImplementedError(
                        "Non-learnable functions with no get_params function are not supported."
                    )

    # Draw parameters from the provided prior
    sampled_params = prior.sample(key=key, M=M)
    # And replace the provided params with the sampled ones
    for dict_name in sampled_params.keys():
        if dict_name not in [
            "initial_mean",
            "initial_cov",
            "dynamics_drift",
            "dynamics_diffusion_coefficient",
            "dynamics_diffusion_cov",
            "dynamics_approx_order",
            "emission_function",
            "emission_cov",
        ]:
            raise ValueError(f"Unknown parameter dictionary name: {dict_name}")

        # Replace the provided params with the sampled ones
        params[dict_name]["params"] = sampled_params[dict_name]

    # Create and return CD-NLGSSM parameter and properties objects
    return create_cdnlgssm_params_and_props(params)


# Utility function to update parameters of a CD-NLGSSM
def update_params(params, updates: dict):
    r"""Update parameters of a CD-NLGSSM.
    Returns a copy of `params` with all updates applied.

    updates: dict with keys like "initial.mean.params" or "dynamics.drift.sigma"

    Example usage:
        updates = {
            "initial.mean.params": jnp.ones(3),  # Vector of ones
            "dynamics.drift.sigma": 11.3  # Scalar value
        }
        new_params = update_params(params, updates)
    """

    # Recursive function to set nested attributes
    def set_nested_attr(obj, attr_path: str, value):
        r"""
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

    # Copy params
    updated = params
    # Apply each update
    for path, val in updates.items():
        updated = set_nested_attr(updated, path, val)
    # Return updated parameters
    return updated
