import pdb
from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit
from jax import jacfwd, jacrev
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree

import jax.debug as jdb

from cdssm_utils import diffeqsolve

from typing import NamedTuple, Tuple, Optional, Union, Callable
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd

# from continuous_discrete_nonlinear_gaussian_ssm.inference import ParamsCDNLGSSM

from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.utils.bijectors import RealToPSDBijector

tfd = tfp.distributions
tfb = tfp.bijectors

# Our codebase
from ssm_temissions import SSM

FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateToStateByState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim state_dim"]]
FnStateToStateByStateByState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim state_dim state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToStateByState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim state_dim"]]
FnStateAndInputToStateByStateByState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim state_dim state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]

def _get_params(x, dim, t):
    # TODO: This function is defined in many places...should be a utils function   
    if callable(x):
        try:
            return x(t)
        except:
            return partial(x,t=t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x

class ParamsCDNLGSSM(NamedTuple):
    """Parameters for a CDNLGSSM model.

    This model does not obey an SDE as in Sarkaa's equation (3.151):
        the solution to 3.151 is not necessarily a Gaussian Process
            (note there are cases where that is indeed the case)

    We instead assume a model of the form
    $$ dz=f(z,u_t,t)dt  $$
    $$ dP=L(t) Q_c L(t) $$ or $$ dP = F_t @ P + P @ F.T + L(t) Q_c_t @ L_t.T $$

    The resulting transition and emission distributions are
    $$p(z_1) = N(z_1 | m, S)$$
    $$p(z_t | z_{t-1}, u_t) = N(z_t | z_t, P_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param initial_mean: $m$
    :param initial_covariance: $S$
    :param dynamics_function: $f$
    :param dynamics_coefficients: $L$
    :param dynamics_covariance: $Q$
    :param dynamics_approx: 'zeroth' or 'first'
    :param emissions_function: $h$
    :param emissions_covariance: $R$

    """

    # TODO: Do we want to break this up like ParamsCDLGSSM?:
    #   - Initial
    #   - Dynamics
    #   - Emission

    # Initial state distribution
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    # f is the deterministic, nonlinear RHS of the state's mean evolution
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_function_jacobian: Union[FnStateToStateByState, FnStateAndInputToStateByState]
    dynamics_function_hessian: Union[FnStateToStateByStateByState, FnStateAndInputToStateByStateByState]
    # L is the diffusion coefficient matrix of the state's covariance process
    dynamics_coefficients: Float[Array, "state_dim state_dim"]
    # Q is the covariance of the state noise process
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    # Dynamics SDE approximation type
    dynamics_approx: str
    # Emission distribution h
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]


class ParamsCDNLSSM(NamedTuple):
    """Parameters for a CDNLSSM model.

    A continuous-discrete nonlinear model, with a state driven by an SDE
        as in Sarkka's Equation 3.151

    Note that such process is not necessarily a Gaussian Process
        although there are certain cases where that is indeed the case

    The resulting transition and emission distributions are
    $$p(z_1) = N(z_1 | m, S)$$
    $$p(z_t | z_{t-1}, u_t) $$ as given by the Fokker-Planck equation
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param initial_mean: $m$
    :param initial_covariance: $S$
    :param dynamics_function: $f$
    :param dynamics_coefficients: $L$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$

    """

    # TODO: Do we want to break this up like ParamsCDLGSSM?:
    #   - InitialStateParams
    #   - DynamicsParams
    #   - EmissionParams

    # Initial state distribution
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    # f as in Sarkka's Equation 3.151
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_function_jacobian: Union[FnStateToStateByState, FnStateAndInputToStateByState]
    dynamics_function_hessian: Union[FnStateToStateByStateByState, FnStateAndInputToStateByStateByState]
    # L in Sarkka's Equation 3.151
    dynamics_coefficients: Float[Array, "state_dim state_dim"]
    # Q in Sarkka's Equation 3.151
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    # Emission distribution h
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]


# CDNLGSSM push-forward is model-specific
def compute_pushforward(
    x0: Float[Array, "state_dim"],
    P0: Float[Array, "state_dim state_dim"],
    params: ParamsCDNLGSSM,
    t0: Float,
    t1: Float,
    inputs: Optional[Float[Array, "input_dim"]] = None,
) -> Tuple[Float[Array, "state_dim state_dim"], Float[Array, "state_dim state_dim"]]:

    # Initialize
    y0 = (x0, P0)
    def rhs_all(t, y, args):
        x, P = y
        
        # possibly time-dependent functions
        # TODO: figure out how to use get_params functionality for time-varying functions
        #f_t = _get_params(params.dynamics_function, 2, t)
        f_t = params.dynamics_function
        Qc_t = _get_params(params.dynamics_covariance, 2, t)
        L_t = _get_params(params.dynamics_coefficients, 2, t)

       
        # Different SDE approximations
        if params.dynamics_approx=='zeroth':
            # Mean evolution
            # TODO: figure out how to vectorize via vmap
            dxdt = f_t(x, inputs)            
            # Covariance evolution
            dPdt = L_t @ Qc_t @ L_t.T
        
        # following Sarkka thesis eq. 3.153
        elif params.dynamics_approx=='first':
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t=params.dynamics_function_jacobian(x,inputs)
            
            # Mean evolution
            # TODO: figure out how to vectorize via vmap
            dxdt = f_t(x, inputs)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        
        # follow Sarkka thesis eq. 3.155
        elif params.dynamics_approx=='second':
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t=params.dynamics_function_jacobian(x,inputs)
            # Evaluate the Hessian of the dynamics function at x and inputs
            H_t=params.dynamics_function_hessian(x,inputs)
        
            # Mean evolution
            # TODO: figure out how to vectorize via vmap
            dxdt = f_t(x, inputs) + 0.5*jnp.trace(H_t @ P)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        else:
            raise ValueError('params.dynamics_approx = {} not implemented yet'.format(params.dynamics_approx))

        return (dxdt, dPdt)
    
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0)
    x, P = sol[0][-1], sol[1][-1]
        
    return x, P

class ContDiscreteNonlinearGaussianSSM(SSM):
    """
    Continuous Discrete Nonlinear Gaussian State Space Model.

    We instead assume a model of the form
    $$ dz=f(z,u_t,t)dt  $$
    $$ dP=L(t) Q_c L(t) $$ or $$ dP = F_t @ P + P @ F.T + L(t) Q_c_t @ L_t.T $$
    
    The resulting transition and emission distributions are
    $$p(z_1) = N(z_1 | m, S)$$
    $$p(z_t | z_{t-1}, u_t) = N(z_t | z_t, P_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$

    where the model parameters are

    * $z_t$ = hidden variables of size `state_dim`,
    * $y_t$ = observed variables of size `emission_dim`
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).
    * $f$ = dynamics deterministic function (RHS), used to compute transition function
    * $L$ = dynamics coefficient multiplying brownian motion 
    * $Q$ = dynamics brownian motion's covariance (system) noise
    * $h$ = emission (observation) function
    * $R$ = covariance matrix for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state


    These parameters of the model are stored in a separate object of type :class:`ParamsCDNLGSSM`.
    """

    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int = 0
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    # TODO: why no need to define initialize()?
    def initialize(
        self,
        key: Float[Array, "key"],
        dynamics_function: Optional[Union[FnStateToState, FnStateAndInputToState]] = None,
        dynamics_diffusion_coefficient: Optional[Float[Array, "state_dim state_dim"]] = None,
        dynamics_diffusion_covariance: Optional[Float[Array, "state_dim state_dim"]] = None,
        dynamics_approx: Optional[str] = 'zeroth',
        emission_function: Optional[Union[FnStateToEmission, FnStateAndInputToEmission]] = None,
        emission_covariance: Optional[Float[Array, "emission_dim emission_dim"]] = None
    ) -> Tuple[ParamsCDNLGSSM, PyTree]:

        if dynamics_function is None:
            dynamics_function = lambda z, u: -z
        
        # TODO: double check these compute jacobian and hessian with respect to states only
        # Compute jacobian and Hesisan of dynamics function (computed here once, for computational benefits)
        dynamics_function_jacobian = jacfwd(dynamics_function)
        # Based on these recommendationshttps://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev
        dynamics_function_hessian = jacfwd(jacrev(dynamics_function))
        
        if dynamics_diffusion_coefficient is None:
            dynamics_diffusion_coefficient = 0.1 * jnp.eye(self.state_dim)
        if dynamics_diffusion_covariance is None:
            dynamics_diffusion_covariance = 0.1 * jnp.eye(self.state_dim)
        if emission_function is None:
            emission_function = lambda z, u: z
        if emission_covariance is None:
            emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        params = ParamsCDNLGSSM(
            initial_mean = jnp.zeros(self.state_dim),
            initial_covariance = jnp.eye(self.state_dim),
            dynamics_function=dynamics_function,
            dynamics_function_jacobian=dynamics_function_jacobian,
            dynamics_function_hessian=dynamics_function_hessian,
            dynamics_coefficients=dynamics_diffusion_coefficient,
            dynamics_covariance=dynamics_diffusion_covariance,
            dynamics_approx=dynamics_approx,
            emission_function=emission_function,
            emission_covariance=emission_covariance,
        )

        props = ParamsCDNLGSSM(
            initial_mean=ParameterProperties(),
            initial_covariance=ParameterProperties(constrainer=RealToPSDBijector()),
            dynamics_function=ParameterProperties(),
            dynamics_function_jacobian=ParameterProperties(),
            dynamics_function_hessian=ParameterProperties(),
            dynamics_coefficients=ParameterProperties(constrainer=RealToPSDBijector()),
            dynamics_covariance=ParameterProperties(constrainer=RealToPSDBijector()),
            dynamics_approx=dynamics_approx,
            emission_function=ParameterProperties(),
            emission_covariance=ParameterProperties(constrainer=RealToPSDBijector()),
        )

        return params, props

    def initial_distribution(
        self,
        params: ParamsCDNLGSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(
        self,
        params: ParamsCDNLGSSM,
        state: Float[Array, "state_dim"],
        t0: Optional[Float] = None,
        t1: Optional[Float] = None,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        # Push-forward with assumed CDNLGSSM
        mean, covariance = compute_pushforward(
            x0 = state,
            P0 = jnp.zeros((state.shape[-1], state.shape[-1])), # TODO: check that last dimension is always state-dimension, even when vectorized
            params=params,
            t0=t0, t1=t1,
            inputs=inputs,
        )
        # TODO: for CDNLSSM we can not return a specific distribution,
        # unless we solve the Fokker-Planck equation for the model SDE
        # However, we should be able to sample from it!
        
        return MVN(mean,covariance)

    def emission_distribution(
        self,
        params: ParamsCDNLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
     ) -> tfd.Distribution:
        h = params.emission_function
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params.emission_covariance)
