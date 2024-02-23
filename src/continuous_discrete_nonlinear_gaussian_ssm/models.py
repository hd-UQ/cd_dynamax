import pdb
from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit
from jax import jacfwd, jacrev
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree

import jax.debug as jdb

from cdssm_utils import diffeqsolve

from typing import NamedTuple, Tuple, Optional, Union, List
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd

# Dynamax shared code
from dynamax.types import Scalar
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

tfd = tfp.distributions
tfb = tfp.bijectors

# Our codebase
from ssm_temissions import SSM
# CDNLGSSM param and function definition
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import *
from continuous_discrete_nonlinear_gaussian_ssm.inference_ekf import EKFHyperParams, iterated_extended_kalman_filter
from continuous_discrete_nonlinear_gaussian_ssm.inference_enkf import EnKFHyperParams, ensemble_kalman_filter
from continuous_discrete_nonlinear_gaussian_ssm.inference_ukf import UKFHyperParams, unscented_kalman_filter

# TODO: This function is defined in many places... unclear whether we need to redefine, or move to utils   
def _get_params(x, dim, t):
    if callable(x):
        try:
            return x(t)
        except:
            return partial(x,t=t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x
   
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
        
        # TODO: possibly time- and parameter-dependent functions
        f=params.dynamics.drift.f

        # Get time-varying parameters
        Qc_t = params.dynamics.diffusion_cov.f(None,inputs,t)
        L_t = params.dynamics.diffusion_coefficient.f(None,inputs,t)
        #Qc_t = _get_params(params.dynamics.diffusion_cov, 2, t0)
        #L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t0)

        # Different SDE approximations
        if params.dynamics.approx_order==0.:
            # Mean evolution
            dxdt = f(x, inputs, t)
            # Covariance evolution
            dPdt = L_t @ Qc_t @ L_t.T
        
        # following Sarkka thesis eq. 3.153
        elif params.dynamics.approx_order==1.:
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t=jacfwd(f)(x, inputs, t)
        
            # Mean evolution
            dxdt = f(x, inputs, t)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        
        # follow Sarkka thesis eq. 3.155
        elif params.dynamics.approx_order==2.:
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t=jacfwd(f)(x, inputs, t)
            # Evaluate the Hessian of the dynamics function at x and inputs
            # Based on these recommendationshttps://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev
            H_t=jacfwd(jacrev(f))(x, inputs, t)
        
            # Mean evolution
            dxdt = f(x, inputs, t) + 0.5*jnp.trace(H_t @ P)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
        else:
            raise ValueError('params.dynamics.approx_order = {} not implemented yet'.format(params.dynamics.approx_order))

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
        initial_mean: Optional[Float[Array, "state_dim"]]=None,
        initial_cov: Optional[Float[Array, "state_dim state_dim"]] = None,
        dynamics_drift: Optional[LearnableFunction] = None,
        dynamics_diffusion_coefficient: Optional[LearnableFunction] = None,
        dynamics_diffusion_cov: Optional[LearnableFunction] = None,
        dynamics_approx_order: Optional[float] = 2.,
        emission_function: Optional[LearnableFunction] = None,
        #emission_parameters = None,
        emission_cov: Optional[Float[Array, "emission_dim emission_dim"]] = None
    ) -> Tuple[ParamsCDNLGSSM, PyTree]:

        # Arbitrary default values, for demo purposes.
        # Initial
        _initial_mean = jnp.zeros(self.state_dim)
        _initial_cov = jnp.eye(self.state_dim)
        # Dynamics
        _dynamics_drift = LinearLearnableFunction(
                            params=-0.1*jnp.eye(self.state_dim)
                        )
        #_dynamics_drift_parameters = -1.
        _dynamics_diffusion_coefficient = ConstantLearnableFunction(
                                            0.1 * jnp.eye(self.state_dim)
                                        )
        _dynamics_diffusion_cov = ConstantLearnableFunction(
                                    0.1 * jnp.eye(self.state_dim)
                                )
        _dynamics_approx_order = 2.
        # Emission
        _emission_function = LinearLearnableFunction(
                            params=jr.normal(key, (self.emission_dim, self.state_dim))
                        )
        #_emission_parameters = 1.
        _emission_cov = ConstantLearnableFunction(
                            0.1 * jnp.eye(self.emission_dim)
                        )

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # Create nested dictionary of params
        params = ParamsCDNLGSSM(
            initial=ParamsLGSSMInitial(
                mean=default(initial_mean, _initial_mean),
                cov=default(initial_cov, _initial_cov)
                ),
            dynamics=ParamsCDNLGSSMDynamics(
                drift=default(dynamics_drift, _dynamics_drift),
                diffusion_coefficient=default(dynamics_diffusion_coefficient, _dynamics_diffusion_coefficient),
                diffusion_cov=default(dynamics_diffusion_cov, _dynamics_diffusion_cov),
                approx_order=default(dynamics_approx_order, _dynamics_approx_order)
                ),
            emissions=ParamsCDNLGSSMEmissions(
                emission_function=default(emission_function, _emission_function),
                emission_cov=default(emission_cov, _emission_cov)
                )
            )
        
        # The keys of param_props must match those of params!
        # TODO: can we have learnablefunctions
        props = ParamsCDNLGSSM(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())
                ),
            dynamics=ParamsCDNLGSSMDynamics(
                drift=LinearLearnableFunction(
                        params=ParameterProperties()
                    ),
                diffusion_coefficient=ConstantLearnableFunction(
                        params=ParameterProperties()
                    ),
                diffusion_cov=ConstantLearnableFunction(
                        params=ParameterProperties(constrainer=RealToPSDBijector())
                    ),
                approx_order=ParameterProperties(trainable=False)
                ),
            emissions=ParamsCDNLGSSMEmissions(
                emission_function=LinearLearnableFunction(
                        params=ParameterProperties()
                ),
                emission_cov=ConstantLearnableFunction(
                        params=ParameterProperties(constrainer=RealToPSDBijector())
                    ),
                )
            )
        return params, props
    
    def initial_distribution(
        self,
        params: ParamsCDNLGSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean, params.initial.cov)

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
        # TODO: change the emission distribution function to be time-dependent
        mean = params.emissions.emission_function.f(state, inputs, t=None)
        R = params.emissions.emission_cov.f(state, inputs, t=None)
        return MVN(mean, R)

    def marginal_log_prob(
        self,
        params: ParamsCDNLGSSM,
        filter_hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]],
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ) -> Scalar:
        filtered_posterior = self.cdnlgssm_filter(
            params=params,
            emissions=emissions,
            filter_hyperparams=filter_hyperparams,
            t_emissions=t_emissions,
            inputs=inputs
        )
        return filtered_posterior.marginal_loglik
        
    def cdnlgssm_filter(
        self,
        params: ParamsCDNLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        filter_hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]]=EKFHyperParams(),
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        num_iter: Optional[int] = 1,
        output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
    ) -> PosteriorGSSMFiltered:
        r"""Run an continuous-discrete nonlinear filter to produce the
            marginal likelihood and filtered state estimates.
            
            Depending on the hyperparameter class provided, it can execute EKF, UKF or EnKF
        
        Args:
            params: model parameters.
            emissions: observation sequence.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
            num_iter: number of linearizations around posterior for update step (default 1).
            hyperparams: hyper-parameters of the filter
            inputs: optional array of inputs.
            output_fields: list of fields to return in posterior object.
                These can take the values "filtered_means", "filtered_covariances",
                "predicted_means", "predicted_covariances", and "marginal_loglik".

        Returns:
            post: posterior object.

        """
        # TODO: this can be condensed, by incorporating num_iter into hyperparams of EKF
        # TODO: use and leverage output_fields to have more or less granular returned posterior object
        if isinstance(filter_hyperparams, EKFHyperParams):
            filtered_posterior=iterated_extended_kalman_filter(
                params = params,
                emissions = emissions,
                t_emissions = t_emissions,
                hyperparams = filter_hyperparams,
                inputs = inputs,
                num_iter = num_iter,
            )
        elif isinstance(filter_hyperparams, EnKFHyperParams):
            filtered_posterior=ensemble_kalman_filter(
                params = params,
                emissions = emissions,
                t_emissions = t_emissions,
                hyperparams = filter_hyperparams,
                inputs = inputs,
            )
        elif isinstance(filter_hyperparams, UKFHyperParams):
            filtered_posterior=unscented_kalman_filter(
                params = params,
                emissions = emissions,
                t_emissions = t_emissions,
                hyperparams = filter_hyperparams,
                inputs = inputs,
            )
        
        return filtered_posterior
