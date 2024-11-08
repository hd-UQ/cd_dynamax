import pdb
from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import lax
from jax import jacfwd, jacrev
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree

import jax.debug as jdb

from typing import NamedTuple, Tuple, Optional, Union, List
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd

# Dynamax shared code
from dynamax.types import PRNGKey, Scalar
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

tfd = tfp.distributions
tfb = tfp.bijectors

# Our codebase
from ssm_temissions import SSM
# CDNLGSSM param and function definition
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import *
from continuous_discrete_nonlinear_gaussian_ssm.inference_ekf import EKFHyperParams, iterated_extended_kalman_filter, iterated_extended_kalman_smoother, forecast_extended_kalman_filter, emissions_extended_kalman_filter
from continuous_discrete_nonlinear_gaussian_ssm.inference_enkf import EnKFHyperParams, ensemble_kalman_filter, forecast_ensemble_kalman_filter, emissions_ensemble_kalman_filter
from continuous_discrete_nonlinear_gaussian_ssm.inference_ukf import UKFHyperParams, unscented_kalman_filter, forecast_unscented_kalman_filter, emissions_unscented_kalman_filter
# Diffrax based diff-eq solver
from utils.diffrax_utils import diffeqsolve
from utils.debug_utils import lax_scan
DEBUG = False

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

_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x

# CDNLGSSM push-forward is model-specific
def compute_pushforward(
    x0: Float[Array, "state_dim"],
    P0: Float[Array, "state_dim state_dim"],
    params: ParamsCDNLGSSM,
    t0: Float,
    t1: Float,
    inputs: Optional[Float[Array, "input_dim"]] = None,
    diffeqsolve_settings: dict = {},
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
        # Qc_t = _get_params(params.dynamics.diffusion_cov, 2, t0)
        # L_t = _get_params(params.dynamics.diffusion_coefficient, 2, t0)

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

    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0, **diffeqsolve_settings)
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
        input_dim: int = 0,
        diffeqsolve_settings: dict = {},
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0
        self._diffeqsolve_settings = diffeqsolve_settings

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    @property
    def diffeqsolve_settings(self):
        return self._diffeqsolve_settings

    # This is a revised initialize, consistent across cd-dynamax, based on dicts
    def initialize(
        self,
        key: Optional[Float[Array, "key"]] = jr.PRNGKey(0),
        initial_mean: dict = None,
        initial_cov: dict = None,
        dynamics_drift: dict = None,
        dynamics_diffusion_coefficient: dict = None,
        dynamics_diffusion_cov: dict = None,
        dynamics_approx_order: Optional[float] = 2.,
        emission_function: dict = None,
        emission_cov: dict = None,
    ) -> Tuple[ParamsCDNLGSSM, PyTree]:

        ### Arbitrary default values, for demo purposes
        # Default is to have NOTHING learnable.
        ## Initial
        _initial_mean = {
            "params": jnp.zeros(self.state_dim),
            "props": ParameterProperties(trainable=False)
        }

        _initial_cov = {
            "params": jnp.eye(self.state_dim),
            "props": ParameterProperties(
                        trainable=False,
                        constrainer=RealToPSDBijector()
                    )
        }

        ## Dynamics
        _dynamics_drift = {
            "params": LearnableLinear(
                weights=-0.1*jnp.eye(self.state_dim),
                bias=jnp.zeros(self.state_dim)
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(trainable=False),
                bias=ParameterProperties(trainable=False)
            )
        }

        _dynamics_diffusion_coefficient = {
            "params": LearnableMatrix(
                    params=0.1*jnp.eye(self.state_dim)
                ),
            "props": LearnableMatrix(
                params=ParameterProperties(trainable=False)
                )
        }

        _dynamics_diffusion_cov = {
            "params": LearnableMatrix(
                    params=0.1*jnp.eye(self.state_dim)
                ),
            "props": LearnableMatrix(
                    params=ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
                )
        }

        _dynamics_approx_order =  2.

        ## Emission
        _emission_function = {
            "params": LearnableLinear(
                weights=jr.normal(key, (self.emission_dim, self.state_dim)),
                bias=jnp.zeros(self.emission_dim)
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(trainable=False),
                bias=ParameterProperties(trainable=False)
            ),
        }

        _emission_cov = {
            "params": LearnableMatrix(
                    params=0.1*jnp.eye(self.emission_dim)
                ),
            "props": LearnableMatrix(
                    params=ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
                )
            }

        ## Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # replace defaults as needed
        initial_mean = default(initial_mean, _initial_mean)
        initial_cov = default(initial_cov, _initial_cov)
        dynamics_drift = default(dynamics_drift, _dynamics_drift)
        dynamics_diffusion_coefficient = default(dynamics_diffusion_coefficient, _dynamics_diffusion_coefficient)
        dynamics_diffusion_cov = default(dynamics_diffusion_cov, _dynamics_diffusion_cov)
        dynamics_approx_order = {
            "params": default(dynamics_approx_order, _dynamics_approx_order),
            "props": ParameterProperties(trainable=False), # never trainable, no constraints to apply.
        }
        emission_function = default(emission_function, _emission_function)
        emission_cov = default(emission_cov, _emission_cov)

        ## Create nested dictionary of params
        params_dict = {"params": {}, "props": {}}
        for key in params_dict.keys():
            params_dict[key] = ParamsCDNLGSSM(
                initial=ParamsLGSSMInitial(
                    mean=initial_mean[key],
                    cov=initial_cov[key]
                ),
                dynamics=ParamsCDNLGSSMDynamics(
                    drift=dynamics_drift[key],
                    diffusion_coefficient=dynamics_diffusion_coefficient[key],
                    diffusion_cov=dynamics_diffusion_cov[key],
                    approx_order=dynamics_approx_order[key],
                ),
                emissions=ParamsCDNLGSSMEmissions(
                    emission_function=emission_function[key],
                    emission_cov=emission_cov[key],
                )
            )

        return params_dict["params"], params_dict["props"]

    def initial_distribution(
        self,
        params: ParamsCDNLGSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean.f(), params.initial.cov.f())

    def transition_distribution(
        self,
        params: ParamsCDNLGSSM,
        state: Float[Array, "state_dim"],
        t0: Optional[Float] = None,
        t1: Optional[Float] = None,
        inputs: Optional[Float[Array, "input_dim"]] = None,
    ) -> tfd.Distribution:
        # Push-forward with assumed CDNLGSSM
        mean, covariance = compute_pushforward(
            x0 = state,
            P0 = jnp.zeros((state.shape[-1], state.shape[-1])), # TODO: check that last dimension is always state-dimension, even when vectorized
            params=params,
            t0=t0, t1=t1,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
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
    
    def sample_dist(
        self,
        params: ParamsCDNLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
                Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample from the joint distribution to produce state and emission trajectories.
        
        Args:
            params: model parameters
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
            inputs: optional array of inputs.
        
        Returns:
            latent states and emissions
        
        """
        print('Sampling from continuous-discrete non-linear Gaussian SSM distributions')
        return cdnlgssm_joint_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
        )
    
    def sample_path(
        self,
        params: ParamsCDNLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
                Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample from a forward path to produce state and emission trajectories.

        Args:
            params: model parameters
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
            inputs: optional array of inputs.

        Returns:
            latent states and emissions

        """
        print('Sampling from continuous-discrete non-linear Gaussian SSM path')
        return cdnlgssm_path_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
        )

    def marginal_log_prob(
        self,
        params: ParamsCDNLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]]=EKFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ) -> Scalar:
        filtered_posterior = cdnlgssm_filter(
            params=params,
            emissions=emissions,
            t_emissions=t_emissions,
            hyperparams=filter_hyperparams,
            inputs=inputs
        )
        return filtered_posterior.marginal_loglik

def cdnlgssm_joint_sample(
    params: ParamsCDNLGSSM,
    key: PRNGKey,
    num_timesteps: int,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
    diffeqsolve_settings={}
)-> Tuple[Float[Array, "num_timesteps state_dim"],
          Float[Array, "num_timesteps emission_dim"]]:
    r"""Sample from the joint distribution of a CD-NLGSSM to produce state and emission trajectories.
    
    Args:
        params: model parameters
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional array of inputs.

    Returns:
        latent states and emissions

    """
    def _sample_initial(key, params, inputs):
        key1, key2 = jr.split(key)

        initial_state = MVN(
            params.initial.mean.f(),
            params.initial.cov.f()
        ).sample(seed=key1)

        # Sample from emission
        u0 = tree_map(lambda x: x[0], inputs)
        emission_mean = params.emissions.emission_function.f(
            initial_state,
            u0,
            t=0
        )
        emission_cov = params.emissions.emission_cov.f(
            initial_state,
            u0,
            t=0
        )
        initial_emission = MVN(
            emission_mean,
            emission_cov
        ).sample(seed=key2)

        return initial_state, initial_emission

    def _step(prev_state, args):
        key, t0, t1, inpt = args
        key1, key2 = jr.split(key, 2)

        # Push-forward with assumed CDNLGSSM
        mean, covariance = compute_pushforward(
            x0 = prev_state,
            P0 = jnp.zeros((prev_state.shape[-1], prev_state.shape[-1])), # TODO: check that last dimension is always state-dimension, even when vectorized
            params=params,
            t0=t0, t1=t1,
            inputs=inpt,
            diffeqsolve_settings=diffeqsolve_settings
        )
        # Sample from transition 
        state = MVN(mean, covariance).sample(seed=key1)
        
        # Sample from emission
        emission_mean = params.emissions.emission_function.f(
            state,
            inpt,
            t=t1
        )
        emission_cov = params.emissions.emission_cov.f(
            state,
            inpt,
            t=t1
        )
        emission = MVN(
            emission_mean,
            emission_cov
        ).sample(seed=key2)

        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)
    
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Sample the remaining emissions and states
    next_keys = jr.split(key2, num_timesteps - 1)

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
        t1 = tree_map(lambda x: x[1:,0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps-1)
        t1 = jnp.arange(1,num_timesteps)
    
    next_inputs = tree_map(lambda x: x[1:], inputs)
    # Sample the remaining emissions and states via scan
    _, (next_states, next_emissions) = lax.scan(
        _step,
        initial_state,
        (next_keys, t0, t1, next_inputs)
    )

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions

def cdnlgssm_path_sample(
        params: ParamsCDNLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
        diffeqsolve_settings={}
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
                Float[Array, "num_timesteps emission_dim"]]:
    r"""Sample from a forward path of the CD-NLGSSM to produce state and emission trajectories.
    
    Args:
        params: model parameters
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        inputs: optional array of inputs.

    Returns:
        latent states and emissions

    """
    def _sample_initial(key, params, inputs):
        key1, key2 = jr.split(key)

        initial_state = MVN(
            params.initial.mean.f(),
            params.initial.cov.f()
        ).sample(seed=key1)

        # Sample from emission
        u0 = tree_map(lambda x: x[0], inputs)
        emission_mean = params.emissions.emission_function.f(
            initial_state,
            u0,
            t=0
        )
        emission_cov = params.emissions.emission_cov.f(
            initial_state,
            u0,
            t=0
        )
        initial_emission = MVN(
            emission_mean,
            emission_cov
        ).sample(seed=key2)

        return initial_state, initial_emission
    
    def _step(prev_state, args):
        key, t0, t1, inpt = args
        key1, key2 = jr.split(key, 2)

        # SDE definition as per the CD-NLGSSM
        def drift(t, y, args):
            return params.dynamics.drift.f(y, inpt, t)

        def diffusion(t, y, args):
            Qc_t = params.dynamics.diffusion_cov.f(None, inpt, t)
            L_t = params.dynamics.diffusion_coefficient.f(None, inpt, t)
            Q_sqrt = jnp.linalg.cholesky(Qc_t)
            combined_diffusion = L_t @ Q_sqrt
            return combined_diffusion

        # solve the SDE
        state = diffeqsolve(
            key=key1,
            drift=drift,
            diffusion=diffusion,
            t0=t0,
            t1=t1,
            y0=prev_state,
            **diffeqsolve_settings
        )[0]
        
        # Sample from emission
        emission_mean = params.emissions.emission_function.f(
            state,
            inpt,
            t=t1
        )
        emission_cov = params.emissions.emission_cov.f(
            state,
            inpt,
            t=t1
        )
        emission = MVN(
            emission_mean,
            emission_cov
        ).sample(seed=key2)

        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)
    
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Sample the remaining emissions and states
    next_keys = jr.split(key2, num_timesteps - 1)

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
        t1 = tree_map(lambda x: x[1:,0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps-1)
        t1 = jnp.arange(1,num_timesteps)
    
    next_inputs = tree_map(lambda x: x[1:], inputs)
    # Sample the remaining emissions and states via scan
    _, (next_states, next_emissions) = lax.scan(
         _step,
         initial_state,
         (next_keys, t0, t1, next_inputs)
    )
    '''
    _, (next_states, next_emissions) = lax_scan(
        _step,
        initial_state,
        (next_keys, t0, t1, next_inputs),
        debug=True
    )
    '''

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions

def cdnlgssm_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]]=EKFHyperParams(),
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
        hyperparams: hyper-parameters of the filter
        inputs: optional array of inputs.
        num_iter: number of linearizations around posterior for update step (default 1).
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    # TODO: this can be condensed, by incorporating num_iter into hyperparams of EKF
    # TODO: use and leverage output_fields to have more or less granular returned posterior object
    if isinstance(hyperparams, EKFHyperParams):
        filtered_posterior=iterated_extended_kalman_filter(
            params = params,
            emissions = emissions,
            t_emissions = t_emissions,
            hyperparams = hyperparams,
            inputs = inputs,
            num_iter = num_iter,
            output_fields=output_fields
        )
    elif isinstance(hyperparams, EnKFHyperParams):
        filtered_posterior=ensemble_kalman_filter(
            params = params,
            emissions = emissions,
            t_emissions = t_emissions,
            hyperparams = hyperparams,
            inputs = inputs,
            output_fields=output_fields
        )
    elif isinstance(hyperparams, UKFHyperParams):
        filtered_posterior=unscented_kalman_filter(
            params = params,
            emissions = emissions,
            t_emissions = t_emissions,
            hyperparams = hyperparams,
            inputs = inputs,
            output_fields=output_fields
        )
    
    return filtered_posterior

def cdnlgssm_smoother(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
    hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]]=EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    num_iter: Optional[int] = 1,
) -> PosteriorGSSMFiltered:
    r"""Run an continuous-discrete nonlinear smoother to produce the
        marginal likelihood and smoothed state estimates.
        
        Depending on the hyperparameter class provided, it can execute EKF, UKF or EnKF
    
    Args:
        params: model parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
        hyperparams: hyper-parameters of the smoother to use
        inputs: optional array of inputs.
        num_iter: optinal, number of linearizations around posterior for update step (default 1).
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "smoothed_means", "smoothed_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    # TODO: this can be condensed, by incorporating num_iter into hyperparams of EKF
    # TODO: use and leverage output_fields to have more or less granular returned posterior object
    if isinstance(hyperparams, EKFHyperParams):
        smoothed_posterior=iterated_extended_kalman_smoother(
            params = params,
            emissions = emissions,
            t_emissions = t_emissions,
            hyperparams = hyperparams,
            inputs = inputs,
            num_iter = num_iter,
        )
    elif isinstance(hyperparams, EnKFHyperParams):
        raise ValueError('EnKS not implemented yet')
    elif isinstance(hyperparams, UKFHyperParams):
        raise ValueError('UKS not implemented yet')
    
    return smoothed_posterior

# TODO: replicate this for linear models
def cdnlgssm_forecast(
    params: ParamsCDNLGSSM,
    init_forecast: Union[tfd.Distribution, Float[Array, "state_dim 1"]],
    t_init: Float[Array, "1 1"],
    t_forecast: Optional[Float[Array, "num_timesteps 1"]]=None,
    hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]]=EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=[
        "forecasted_state_means",
        "forecasted_state_covariances",
    ],
    key: Optional[Float[Array, "key"]] = jr.PRNGKey(0),
    diffeqsolve_settings: dict = {},
) -> GSSMForecast:
    r"""Run an continuous-discrete nonlinear model to produce the forecasted state estimates.
        
        Depending on the hyperparameter class provided, it can execute EKF, UKF or EnKF
    
    Args:
        params: model parameters.
        init_forecast: initial condition to start forecasting with:
            - if init_forecast is a distribution, then we forecast such distribution based on different filtering methods
            - if init_forecast is a point estimate of state, then we forecast a forward path starting at that state
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants of observations: if not None, it is an array 
        hyperparams: hyper-parameters of the filter
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        output_fields: list of fields to return in posterior object.
            These can take the values
                If we forecast Gaussian distributions, based on filtering methods
                    "forecasted_state_means",
                    "forecasted_state_covariances",
                If we forecast paths, based on solving the SDE
                    "forecasted_state_path",
                    "forecasted_emission_path".

    Returns:
        post: forecasted object.

    """
    # Check whether init_forecast is a distribution or a point estimate
    if isinstance(init_forecast, tfd.Distribution):
        # Forecasting a distribution, based on different filters
        if isinstance(hyperparams, EKFHyperParams):
            forecast=forecast_extended_kalman_filter(
                params = params,
                init_forecast = init_forecast,
                t_init = t_init,
                t_forecast = t_forecast,
                hyperparams = hyperparams,
                inputs = inputs,
                output_fields=output_fields
            )
        elif isinstance(hyperparams, EnKFHyperParams):
            forecast=forecast_ensemble_kalman_filter(
                params = params,
                init_forecast = init_forecast,
                t_init = t_init,
                t_forecast = t_forecast,
                hyperparams = hyperparams,
                inputs = inputs,
                output_fields=output_fields
            )
        elif isinstance(hyperparams, UKFHyperParams):
            forecast=forecast_unscented_kalman_filter(
                params = params,
                init_forecast = init_forecast,
                t_init = t_init,
                t_forecast = t_forecast,
                hyperparams = hyperparams,
                inputs = inputs,
                output_fields=output_fields
            )
    else:
        # Forecasting point estimates, based on pushing forward the model
        
        # Figure out timestamps, as vectors to scan over
        # t_forecast is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps \times 0
        if t_forecast is not None:
            num_timesteps = t_forecast.shape[0]
            t0 = tree_map(
                lambda x: jnp.concatenate(
                    (t_init, t_forecast[:-1, 0])
                ),
                t_forecast,
            )
            t1 = tree_map(lambda x: x[:,0], t_forecast)
        else:
            raise ValueError("t_forecast must be provided for forecasting")

        # Set-up indexing and inputs
        t0_idx = jnp.arange(num_timesteps)
        inputs = _process_input(inputs, num_timesteps+1)

        # Define the function to scan over
        def _step(prev_state, args):
            key, t0, t1, t0_idx = args

            # Split the key
            key1, key2 = jr.split(key, 2)

            # Define the drift and diffusion functions
            def drift(t, y, args):
                return params.dynamics.drift.f(
                    y,
                    inputs[t0_idx],
                    t
                )
            def diffusion(t, y, args):
                Qc_t = params.dynamics.diffusion_cov.f(
                    None,
                    inputs[t0_idx],
                    t
                )
                Q_sqrt = jnp.linalg.cholesky(Qc_t)
                L_t = params.dynamics.diffusion_coefficient.f(
                    None,
                    inputs[t0_idx],
                    t
                )
                combined_diffusion = L_t @ Q_sqrt
                return combined_diffusion
            
            # Solve the SDE to compute the next state
            state = diffeqsolve(
                key=key2,
                drift=drift,
                diffusion=diffusion,
                t0=t0, t1=t1,
                y0=prev_state,
                **diffeqsolve_settings
            )[0]
            
            # Sample the emission, at t1
            emission = MVN(
                params.emissions.emission_function.f(
                    state,
                    inputs[t0_idx+1],
                    t=t1
                ),
                params.emissions.emission_cov.f(
                    state,
                    inputs[t0_idx+1],
                    t=t1
                )
            ).sample(seed=key1)
            
            # Return the state and emission
            return state, (state, emission)

        # Split the key
        next_keys = jr.split(key, num_timesteps)

        # Forecast states and emissions, over time, via scan       
        _, (next_states, next_emissions) = lax_scan(
            _step,
            init_forecast,
            (next_keys, t0, t1, t0_idx),
            debug=DEBUG
        ) # type: ignore
        
        # Build the forecast object
        forecast = GSSMForecast(
            forecasted_state_path=next_states,
            forecasted_emission_path=next_emissions
        )
    
    return forecast

# TODO: replicate this for linear models
def cdnlgssm_emissions(
    params: ParamsCDNLGSSM,
    t_states: Float[Array, "num_timesteps 1"],
    state_means: Float[Array, "num_timesteps state_dim"],
    state_covs: Optional[Float[Array, "num_timesteps state_dim state_dim"]]=None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    hyperparams: Optional[Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]]=None,
    key: Optional[Float[Array, "key"]] = jr.PRNGKey(0),
) -> Tuple[
        Float[Array, "num_timesteps emission_dim"], Optional[Float[Array, "num_timesteps emission_dim emission_dim"]]
    ]:
    r"""Compute the emissions corresponding to
        - a continuous-discrete nonlinear model, as specified by params
        - a filter method for a continuous-discrete nonlinear model
            Depending on the hyperparameter class provided, it can execute EKF, UKF or EnKF
    
    Args:
        params: model parameters.
        t_states: continuous-time specific time instants of states
        state_means: state means at time instants t_states, always required
        state_covs: state covariances at time instants t_states, optional
            - if None, then we assume that the states are point estimates, and simply push through emission function
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        hyperparams: hyper-parameters of the filter, optional
        key: random key for sampling

    Returns:
        emissions_mean: mean of emissions
        emissions_covariance: covariance of emissions, if available
    """

    # Check whether we are using model or a filter
    if hyperparams is not None:
        # Emissions, based on different filters
        if isinstance(hyperparams, EKFHyperParams):
            emissions_mean, emissions_covariance=emissions_extended_kalman_filter(
                params = params,
                t_states = t_states,
                state_means = state_means,
                state_covs = state_covs,
                inputs = inputs,
                hyperparams = hyperparams,
            )
        elif isinstance(hyperparams, EnKFHyperParams):
            emissions_mean, emissions_covariance=emissions_ensemble_kalman_filter(
                params = params,
                t_states = t_states,
                state_means = state_means,
                state_covs = state_covs,
                inputs = inputs,
                hyperparams = hyperparams,
                key=key,
            )
        elif isinstance(hyperparams, UKFHyperParams):
            emissions_mean, emissions_covariance=emissions_unscented_kalman_filter(
                params = params,
                t_states = t_states,
                state_means = state_means,
                state_covs = state_covs,
                inputs = inputs,
                hyperparams = hyperparams,
            )
    else:
        # Emissions of point estimates, based on pushing the state through the model emission function
        
        # Figure out timestamps, as vectors to scan over
        # t_states is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps \times 0
        if t_states is not None:
            num_timesteps = t_states.shape[0]
            t0 = tree_map(lambda x: x[:,0], t_states)
        else:
            raise ValueError("t_states must be provided for forecasting")

        # Set-up indexing and inputs
        t0_idx = jnp.arange(num_timesteps)
        inputs = _process_input(inputs, num_timesteps)

        # Define the function to scan over
        def _step(state, args):
            this_state, t0, t0_idx = args

            # Push the state through the emission function
            emission_mean=params.emissions.emission_function.f(
                    this_state,
                    inputs[t0_idx],
                    t=t0
                )
            # Emission covariance, as determined by the model
            emission_cov=params.emissions.emission_cov.f(
                this_state,
                inputs[t0_idx],
                t=t0
            )
            
            # Return the state and emission
            return this_state, (emission_mean, emission_cov)

        # Compute emissions, over time, via scan       
        _, (emissions_mean, emissions_covariance) = lax_scan(
            _step,
            state_means[0],
            (state_means, t0, t0_idx),
            debug=DEBUG
        ) # type: ignore

    # Return the emissions
    return emissions_mean, emissions_covariance
