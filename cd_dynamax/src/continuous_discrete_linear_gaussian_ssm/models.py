# JAX imports
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

# Type annotations
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

# Distributions, compatible with JAX, from TensorFlow Probability
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
tfd = tfp.distributions
tfb = tfp.bijectors

# Imports from dynamax
from cd_dynamax.dynamax.types import PRNGKey, Scalar
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector
from cd_dynamax.dynamax.utils.utils import psd_solve

# Imports from the cd-dynamax codebase
from ..ssm_temissions import SSM, Prior
# To avoid unnecessary redefinitions of code,
# We import posterior classes that can be reused from dynamax LGSSM first
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
# Param definition
from .cdlgssm_utils import ParamsCDLGSSM, init_cdlgssm_params, sample_cdlgssm_params
# Filtering functions
from .inference import KFHyperParams
from .inference import cdlgssm_filter, cdlgssm_smoother, cdlgssm_forecast
# Unclear why we define this here, but not in models
from .inference import cdlgssm_joint_sample, cdlgssm_path_sample, cdlgssm_posterior_sample
from .inference import compute_pushforward
# Debug utilities
from ..utils.debug_utils import psd
DEBUG = False # By default, debugging is off, e.g., no extra checks in lax_scan

class SuffStatsCDLGSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statistics for CDLGSSM parameter estimation."""
    pass

# CD-LGSSM model definition
class ContDiscreteLinearGaussianSSM(SSM):
    r"""
    Definition of a Continuous-Discrete Linear Gaussian State Space Model.

    The CD-LGSSM model is defined in the following way, according to equation (3.134) in Sarkka (2019):
    
    $$p(z_0) = \mathcal{N}(z_0 \mid m, S)$$
    $$dz = F_t z_t dt + B_t u_t dt + b_t dt + L_t d\beta_t$$
        with $\beta_t$ a standard Brownian motion, implying
        
        $$p(z_{t_k} \mid z_{t_{k-1}}, u_t) = \mathcal{N}(z_{t_k} \mid A_{t_k} z_{t_{k-1}} + B_{t_k} u_{t_k} + b_{t_k}, Q_{t_k})$$
            where $A_{t_k}$ and $Q_{t_k}$ are computed as the solution to the SDE above over the interval $[t_{t_{k-1}}, t_{t_k}]$,
    and emissions defined as

    $$p(y_{t_k} \mid z_{t_k}) = \mathcal{N}(y_{t_k} \mid H_{t_k} z_{t_k} + D_{t_k} u_{t_k} + d_{t_k}, R_{t_k})$$

    where

    * $z_{t_k}$ is a latent state of size `state_dim`,
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state
    
    * $F_t$ = is the linear dynamics of the state, as in the SDE definition above
    * $A_t$ = are the dynamics (transition) matrix of the state ---computed as the pushforward of the continuous dynamics---
            A_t is the solution to the ODE in eq (3.135) in Sarkka (2019)
    * $L$ = diffusion coefficient of the dynamics SDE, used to compute Q_t
    * $Q$ = diffusion covariance matrix of dynamics (system) ---brownian motion--- noise, computed as the solution to the ODE in eq (3.135) in Sarkka (2019)
    * $u_t$ is an input of size `input_dim` (defaults to 0)
    * $B$ = (optional) input-to-state weight matrix
    * $b$ = (optional) state bias vector
    
    * $y_{t_k}$ is an observed variable (emissions) of size `emission_dim`
    * $H$ = emission (observation) matrix
    * $R$ = covariance function for emission (observation) noise
    * $D$ = (optional) input-to-emission weight matrix
    * $d$ = (optional) emission bias vector

    The parameters of the model are stored in a :class:`ParamsCDLGSSM`.
    You can create the parameters manually, or by calling :meth:`initialize`.

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term $b$. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term $d$. Defaults to True.
    :param diffeqsolve_settings: settings to pass to the differential equation solver when computing the
        pushforward of the continuous-time dynamics. Defaults to {}.

    """

    # Default constructor
    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0,
        has_dynamics_bias: bool=True,
        has_emissions_bias: bool=True,
        diffeqsolve_settings: dict={},
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias
        self._diffeqsolve_settings = diffeqsolve_settings
        # By default, we have no prior
        self.prior = None

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    @property
    def diffeqsolve_settings(self):
        return self._diffeqsolve_settings
        
    # SSM methods
    # Define default set of CD-LGSSM parameters,
    # with all learnable parameters set to False
    def _default_cdlgssm_params(self) -> dict:
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
        # Just a matrix with -0.1 in the diagonal, for the drift
        _dynamics_weights = {
            "params": -0.1 * jnp.eye(self.state_dim),
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_input_weights = {
            "params": jnp.zeros((self.state_dim, self.input_dim)),
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_bias = {
            "params": jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None,
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_diffusion_coefficient = {
            "params": 0.1 * jnp.eye(self.state_dim),
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_diffusion_cov = {
            "params": 0.1 * jnp.eye(self.state_dim),
            "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        }
        
        ## Emission
        # Randomly drawn weights
        key = jr.PRNGKey(0)
        _emission_weights = {
            "params": jr.normal(key, (self.emission_dim, self.state_dim)),
            "props": ParameterProperties(trainable=False)
        }
        _emission_input_weights = {
            "params": jnp.zeros((self.emission_dim, self.input_dim)),
            "props": ParameterProperties(trainable=False)
        }
        _emission_bias = {
            "params": jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None,
            "props": ParameterProperties(trainable=False)
        }
        _emission_cov = {
            "params": 0.1 * jnp.eye(self.emission_dim),
            "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        }

        # Return the default parameters as a dictionary
        return {
            'initial_mean': _initial_mean,
            'initial_cov': _initial_cov,
            'dynamics_weights': _dynamics_weights,
            'dynamics_input_weights': _dynamics_input_weights,
            'dynamics_bias': _dynamics_bias,
            'dynamics_diffusion_coefficient': _dynamics_diffusion_coefficient,
            'dynamics_diffusion_cov': _dynamics_diffusion_cov,
            'emission_weights': _emission_weights,
            'emission_input_weights': _emission_input_weights,
            'emission_bias': _emission_bias,
            'emission_cov': _emission_cov,
        }

    # CD-LGSSM initialize method, consistent across cd-dynamax, based on dicts
    def initialize(
        self,
        key: PRNGKey =jr.PRNGKey(0),
        init_prior: Prior = None,
        initial_mean: dict = None,
        initial_cov: dict = None,
        dynamics_weights: dict = None,
        dynamics_bias: dict = None,
        dynamics_input_weights: dict = None,
        dynamics_diffusion_coefficient: dict = None,
        dynamics_diffusion_cov: dict = None,
        emission_weights: dict = None,
        emission_bias: dict = None,
        emission_input_weights: dict = None,
        emission_cov: dict = None,
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
        r"""Initialize CD-LGSSM parameters, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            init_prior: prior distribution for the initialization. Defaults to None.
            initial_mean: parameter $m$. Defaults to None.
            initial_cov: parameter $S$. Defaults to None.
            dynamics_weights: parameter $F$. Defaults to None.
            dynamics_bias: parameter $b$. Defaults to None.
            dynamics_input_weights: parameter $B$. Defaults to None.
            dynamics_diffusion_coefficient: parameter $L$. Defaults to None.
            dynamics_diffusion_cov: parameter $Q$. Defaults to None.
            emission_weights: parameter $H$. Defaults to None.
            emission_bias: parameter $d$. Defaults to None.
            emission_input_weights: parameter $D$. Defaults to None.
            emission_cov: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsCDLGSSM, ParamsCDLGSSM]: parameters and their properties.
        """

        # Create CD-NLGSSM parameters and properties,
        # based on the provided samples, init_values or defaults
        params_dict_values, params_dict_props = init_cdlgssm_params(
            default_params = self._default_cdlgssm_params(),
            init_params = {
                'initial_mean': initial_mean,
                'initial_cov': initial_cov,
                'dynamics_weights': dynamics_weights,
                'dynamics_input_weights': dynamics_input_weights,
                'dynamics_bias': dynamics_bias,
                'dynamics_diffusion_coefficient': dynamics_diffusion_coefficient,
                'dynamics_diffusion_cov': dynamics_diffusion_cov,
                'emission_weights': emission_weights,
                'emission_input_weights': emission_input_weights,
                'emission_bias': emission_bias,
                'emission_cov': emission_cov,
            },
            init_prior = init_prior,
        )

        # If provided, initialize prior for future use
        self.prior = init_prior

        # Return the parameters and properties
        return params_dict_values, params_dict_props   

    # SSM distribution methods
    def initial_distribution(
        self,
        params: ParamsCDLGSSM,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        r"""Initial distribution.
        Args:
            params: CD-LGSSM model parameters.
            inputs: optional initial inputs (not used in state initialization).
        Returns:
            initial Gaussian state distribution.
        """
        # Gaussian initial distribution
        return MVN(params.initial.mean, params.initial.cov)

    # Transition distribution: defined as the pushforward
    # of the continuous state dynamics and discrete inputs (controls) 
    def transition_distribution(
        self,
        params: ParamsCDLGSSM,
        state: Float[Array, "state_dim"],
        t0: Optional[Float]=None,
        t1: Optional[Float]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        r"""CD-LGSSM Transition distribution,
            defined as the pushforward of the continuous-time, linear dynamics and discrete inputs (controls).
        
        Args:
            params: CD-LGSSM model parameters.
            state: current state.
            t0: initial time.
            t1: final time.
            inputs: optional inputs.

        Returns:
            Gaussian, state transition distribution.
        """
        # Process inputs
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        
        # Compute pushforward map:
        # A maps the state from t0 to t1
        # Q is the covariance at t1
        A, Q = compute_pushforward(params, t0, t1, diffeqsolve_settings=self.diffeqsolve_settings)
        # Pushforward the state from t0 to t1, then add controls at t1 
        mean = A @ state + params.dynamics.input_weights @ inputs
        if self.has_dynamics_bias:
            mean += params.dynamics.bias
        
        # Return the corresponding Gaussian transition distribution
        return MVN(mean, Q)
        
    # Emission distribution
    def emission_distribution(
        self,
        params: ParamsCDLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        r"""CD-LGSSM Emission distribution.
        Args:
            params: CD-LGSSM model parameters.
            state: current state.
            inputs: optional inputs.
        
        Returns:
            Gaussian, state-conditional emission distribution
        """
        # Process inputs
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        # Compute emission mean
        mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias
        # Return the corresponding Gaussian emission distribution
        return MVN(mean, params.emissions.cov)

    # Sampling methods
    # Sampling from the prior
    def sample_prior(
        self,
        prior: Prior,
        M: int,
        init_params: Optional[ParamsCDLGSSM]=None,   
        key: Optional[PRNGKey]=jr.PRNGKey(0),
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
        r"""Sample from the prior distribution over CD-LGSSM model parameters.

        Args:
            :param prior: prior distribution
            :param M: number of samples to draw
            :param init_params: dictionary of parameters to use as initialization
                if not provided, default parameters are used
            :param key: random number generator

        Returns:
            :return: Tuple with sampled CD-LGSSM parameters and properties objects
        """
        if init_params is None:
            # Initialize with default parameters
            init_params=self._default_cdlgssm_params()

        # Sample from the prior
        return sample_cdlgssm_params(
            prior=prior,
            M=M,
            init_params=init_params,
            key = key,
        )
    
    # Sampling from the joint distribution of states and emissions, using the CD-LGSSM distributions
    def sample_dist(
        self,
        params: ParamsCDLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample from the joint distribution of the CD-LGSSM
            to produce states and emission trajectories.

        Args:
            params: CD-LGSSM model parameters
            key: random number generator key
            num_timesteps: number of time steps to sample
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
            inputs: optional array of inputs.
        Returns:
            latent states and observed emissions
        """
        
        print('CD-LGSSM Sampling from continuous-discrete linear Gaussian SSM distributions')
        return cdlgssm_joint_sample(
            params,
            key,
            num_timesteps,
            t_emissions,
            inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
        )
    
    # Sampling from the path distribution of states and emissions, using the SDE solver path
    def sample_path(
        self,
        params: ParamsCDLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
                Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample from a forward path of the CD-LGSSM to produce state and emission trajectories.

        Args:
            params: model parameters
            key: random number generator key
            num_timesteps: number of time steps to sample
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
            inputs: optional array of inputs.

        Returns:
            latent states and observed emissions

        """
        print('CD-LGSSM Sampling from continuous-discrete linear Gaussian SSM path')
        return cdlgssm_path_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
        )
    
    # Inference methods
    def marginal_log_prob(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=KFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Scalar:
        r"""Compute the marginal log likelihood of a sequence of emissions under the CD-LGSSM model.

        Args:
            params: CD-LGSSM model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            filter_hyperparams: hyperparameters for the Kalman filter.
            inputs: optional sequence of inputs.
            key: random number generator.
        Returns:
            Marginal log likelihood of the emissions, $\log p(y_{1:T})$.
        """
        # Run CD-Kalman filter to compute marginal log likelihood
        filtered_posterior = cdlgssm_filter(params, emissions, t_emissions, filter_hyperparams, inputs)
        return filtered_posterior.marginal_loglik

    # A high-level, user-friendly filtering interface (with default settings)
    def filter(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=KFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        warn: bool = True,
    ) -> PosteriorGSSMFiltered:
        r"""Run the CD-Kalman filter to compute the filtered posterior distribution over states.
        Args:
            params: CD-LGSSM model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            filter_hyperparams: hyperparameters for the Kalman filter.
            inputs: optional sequence of inputs.
            warn: whether to warn about numerical issues.
        Returns:
            filtered posterior distribution over states.
        """

        # Directly run the CD-Kalman filter
        return cdlgssm_filter(params, emissions, t_emissions, filter_hyperparams, inputs, warn=warn)

    # High-level, user-friendly interface combining filtering and forecasting steps
    def filter_and_forecast(
        self,
        params,
        emissions_filter,
        t_emissions_filter,
        t_emissions_forecast,
        inputs_filter=None,
        inputs_forecast=None,
        warn: bool = True,
    ):
        r"""Run the CD-Kalman filter to compute the filtered posterior distributions over states,
        and then run forecasting from the last filtered state.
        
        Args:
            params: model parameters.
            emissions_filter: sequence of observations for filtering.
            t_emissions_filter: continuous-time specific time instants of observations for filtering: if not None, it is an array
            t_emissions_forecast: continuous-time specific time instants for forecasting: if not None, it is an array
            inputs_filter: optional sequence of inputs for filtering.
            inputs_forecast: optional sequence of inputs for forecasting.
            warn: whether to warn about numerical issues.
        
        Returns:
            filtered and forecasted posterior distributions over states.
        """

        # Run filter on filtering time points
        filtered = self.filter(
            params=params,
            emissions=emissions_filter,
            t_emissions=t_emissions_filter,
            inputs=inputs_filter,
            warn=warn,
        )

        # Initialize forecast with last filtered state
        init_time = t_emissions_filter[-1]
        init_forecast = MVN(filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :])

        # Run forecast on forecasting time points
        forecasted = cdlgssm_forecast(
            params=params,
            init_forecast=init_forecast,
            t_init=init_time,
            t_forecast=t_emissions_forecast,
            inputs=inputs_forecast,
            warn=warn,
        )

        # Return both filtered and forecasted posteriors
        return filtered, forecasted

    # Smoothing method
    def smoother(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=KFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMSmoothed:
        r"""Compute the smoothing distribution over states using the CD-Kalman smoother.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            filter_hyperparams: hyperparameters for the Kalman filter.
            inputs: optional sequence of inputs.

        Returns:
            smoothed posterior distributions over states.
        """

        return cdlgssm_smoother(params, emissions, t_emissions, filter_hyperparams, inputs)

    # Sampling from the posterior distribution of states given emissions
    def posterior_sample(
        self,
        key: PRNGKey,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Float[Array, "ntime state_dim"]:
        r"""Sample from the posterior distribution over states given emissions using the CD-Kalman filter/smoother.
        Args:
            key: random number generator.
            params: model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            inputs: optional sequence of inputs.
        Returns:
            sampled latent states.
        """
        
        return cdlgssm_posterior_sample(key, params, emissions, t_emissions, inputs)

    # Posterior predictive distribution for emissions
    def posterior_predictive(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=KFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            filter_hyperparams: hyperparameters for the Kalman filter.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        # Run CD-Kalman smoother to compute smoothed states
        posterior = cdlgssm_smoother(params, emissions, t_emissions, filter_hyperparams, inputs)
        # Compute posterior predictive for emissions
        H = params.emissions.weights
        b = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + b
        smoothed_emissions_cov = psd(H @ posterior.smoothed_covariances @ H.T + R)
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        
        # Return posterior predictive means and standard deviations
        return smoothed_emissions, smoothed_emissions_std

    # Expectation-maximization (EM) code
    def e_step(
        self,
        params: ParamsCDLGSSM,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        t_emissions: Optional[Union[Float[Array, "num_timesteps 1"],
                        Float[Array, "num_batches num_timesteps 1"]]]=None,
        filter_hyperparams: Optional[KFHyperParams]=None,
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
    ) -> Tuple[SuffStatsCDLGSSM, Scalar]:

        raise NotImplementedError("EM E-step for CD-LGSSM is not yet implemented.")

    def initialize_m_step_state(
            self,
            params: ParamsCDLGSSM,
            props: ParamsCDLGSSM
    ) -> Any:
        raise NotImplementedError("EM M-step for CD-LGSSM is not yet implemented.")

    def m_step(
        self,
        params: ParamsCDLGSSM,
        props: ParamsCDLGSSM,
        batch_stats: SuffStatsCDLGSSM,
        m_step_state: Any
    ) -> Tuple[ParamsCDLGSSM, Any]:
                
        raise NotImplementedError("EM M-step for CD-LGSSM is not yet implemented.")

