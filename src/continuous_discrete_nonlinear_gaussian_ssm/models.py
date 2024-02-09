from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union, Callable
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd

tfd = tfp.distributions
tfb = tfp.bijectors

# Our codebase
from ssm_temissions import SSM

FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]

# CDNLGSSM push-forward is model-specific
def compute_pushforward(
    x0: Float[Array, "state_dim"]],
    P0: Float[Array, "state_dim state_dim"]],
    params: ParamsCDNLGSSM,
    t0: Float,
    t1: Float,
    inputs: Optional[Float[Array, "input_dim"]] = None,
) -> Tuple[Float[Array, "state_dim state_dim"], Float[Array, "state_dim state_dim"]]:

    y0 = (x0, P0)

    def rhs_all(t, y, args):
        x, P = y

        # possibly time-dependent functions
        f_t = _get_params(params.dynamics_function, 2, t)
        Qc_t = _get_params(params.dynamics_covariance, 2, t)
        L_t = _get_params(params.dynamics_coefficients, 2, t)

        # Mean evolution
        dxdt = vmap(f_t)(x, inputs)
        # Covariance evolution
        if params.dynamics_covariance_order=='zeroth':
            dPdt = L_t @ Qc_t @ L_t.T
        elif params.dynamics_covariance_order=='zeroth':
            raise ValueError('params.dynamics_covariance_order = {} not implemented yet'.format(params.dynamics_covariance_order)
            # TODO: compute F_t=Jacobian of f
            dPdt = F_t @ P + P @ F.T + L_t @ Qc_t @ L_t.T
        else:
            raise ValueError('params.dynamics_covariance_order = {} not implemented yet'.format(params.dynamics_covariance_order)

        return (dxdt, dPdt)
    
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0)
    x, P = sol[0][-1], sol[1][-1]
        
    return x, P

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
    :param emissions_function: $h$
    :param emissions_covariance: $R$

    """
    # Initial state distribution
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    # f is the deterministic, nonlinear RHS of the state's mean evolution
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    # L is the diffusion coefficient matrix of the state's covariance process
    dynamics_coefficients: Float[Array, "state_dim state_dim"]
    # Q is the covariance of the state noise process
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    # Covariance evolution type
    # TODO: check this works
    dynamics_covariance_order: String
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
    # Initial state distribution
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    # f as in Sarkka's Equation 3.151
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    # L in Sarkka's Equation 3.151
    dynamics_coefficients: Float[Array, "state_dim state_dim"]
    # Q in Sarkka's Equation 3.151
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    # Emission distribution h
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]

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
        t0: Optional[Float]=None,
        t1: Optional[Float]=None,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        # Push-forward with assumed CDNLGSSM
        mean,covariance = compute_pushforward(
            state,
            jnp.eye(state_dim), # Assume initial identity covariance
            params,
            t0, t1,
            inputs
        )
        # TODO: for CDNLSSM we can not return an specific distribution,
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
