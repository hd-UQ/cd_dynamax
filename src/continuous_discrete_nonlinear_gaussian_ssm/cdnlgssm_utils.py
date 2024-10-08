from typing import NamedTuple, Tuple, Optional, Union
from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp
from dynamax.parameters import ParameterProperties, ParameterSet
import abc

# To avoid unnecessary redefinitions of code,
# We import parameters that can be reused from LGSSM first
# And define the rest later
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial

# TODO: do we need @dataclass(frozen=True)?
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

class LearnableVector(NamedTuple):
    params: Union[Float[Array, "dim"], ParameterProperties]

    def f(self, x=None, u=None, t=None):
        return self.params

class LearnableMatrix(NamedTuple):
    params: Union[Float[Array, "row_dim col_dim"], ParameterProperties]

    def f(self, x=None, u=None, t=None):
        return self.params

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

class LearnableLorenz63(NamedTuple):
    '''Lorenz63 model with learnable parameters
            sigma: sigma parameter
            rho: rho parameter
            beta: beta parameter

            f(x) = sigma * (y - x)\n
            f(y) = x * (rho - z) - y\n
            f(z) = x * y - beta * z\n
    '''
    sigma: Union[Float, ParameterProperties]
    rho: Union[Float, ParameterProperties]
    beta: Union[Float, ParameterProperties]

    def f(self, x, u=None, t=None):

        return jnp.array([
            self.sigma * (x[1] - x[0]),
            x[0] * (self.rho - x[2]) - x[1],
            x[0] * x[1] - self.beta * x[2]
        ])


# Continuous non-linear Gaussian dynamics
# TODO: function definitions within parameter classes breaks fit_sgd: where should they be placed?
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
    :param drift_parameters: parameters $\theta$ of the drift_function
    :param diffusion_coefficient: $L$
    :param diffusion_cov: $Q$
    :param dynamics_approx: 'zeroth', 'first' or 'second'

    """
    '''
    # the deterministic drift $f$ of the nonlinear RHS of the state
    drift_function: Union[FnStateToState, FnStateAndInputToState]
    # TODO: How to define learnable parameters for emission function?
    #drift_parameters: Union[Float[Array], ParameterProperties] 
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
    :param drift_parameters: parameters $\theta$ of the drift_function
    :param diffusion_coefficient: $L$
    :param diffusion_cov: $Q$

    """
    # the deterministic drift $f$ of the nonlinear RHS of the state
    drift_function: Union[FnStateToState, FnStateAndInputToState]
    # TODO: How to define learnable parameters for dynamics drift function?
    #drift_parameters: Union[Float[Array], ParameterProperties] 
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
    :param drift_parameters: parameters $\theta$ of the drift_function
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
    # TODO: How to define learnable parameters for emission function?
    # emission_parameters: Union[Float[Array], ParameterProperties] 
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
