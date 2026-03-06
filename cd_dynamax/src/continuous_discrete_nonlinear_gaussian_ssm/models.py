# JAX imports
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jacfwd, jacrev
from jax.tree_util import tree_map

# Type annotations
from jaxtyping import Array, Float
from typing import Tuple, Optional, Union, List

# Distributions, compatible with JAX, from TensorFlow Probability
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)
import tensorflow_probability.substrates.jax.distributions as tfd

# Imports from dynamax
from cd_dynamax.dynamax.types import PRNGKey, Scalar
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered

# Imports from the cd-dynamax codebase
from ..ssm_temissions import SSM, Prior

# CDLGSSM forecasting definition
from ..continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import GSSMForecast

# CDNLGSSM param and function definition
from .cdnlgssm_utils import (
    ParamsCDNLGSSM,
    LearnableVector,
    LearnableMatrix,
    LearnableLinear,
    init_cdnlgssm_params,
    sample_cdnlgssm_params,
)

# CDNLGSSM filtering functions, based on different Kalman filter variants
from .inference_ekf import (
    EKFHyperParams,
    iterated_extended_kalman_filter,
    iterated_extended_kalman_smoother,
    forecast_extended_kalman_filter,
    emissions_extended_kalman_filter,
)
from .inference_enkf import (
    EnKFHyperParams,
    ensemble_kalman_filter,
    forecast_ensemble_kalman_filter,
    emissions_ensemble_kalman_filter,
)
from .inference_ukf import (
    UKFHyperParams,
    unscented_kalman_filter,
    forecast_unscented_kalman_filter,
    emissions_unscented_kalman_filter,
)

# Diffrax based diff-eq solver
from ..utils.diffrax_utils import diffeqsolve

# Debugging utilities
from ..utils.debug_utils import psd, lax_scan

tfb = tfp.bijectors

DEBUG = False  # By default, debugging is off, e.g., no extra checks in lax_scan

# Auxiliary function to process inputs ---from dynamax
_process_input = lambda x, y: jnp.zeros((y, 1)) if x is None else x

# CD-NLGSSM push-forward: this is a model-specific push-forward function
# computed based on the assumed continuous-discrete nonlinear Gaussian SSM dynamics
# using different SDE approximations (zeroth, first, second order) solved numerically via diffrax
def compute_pushforward(
    x0: Float[Array, " state_dim"],
    P0: Float[Array, "state_dim state_dim"],
    params: ParamsCDNLGSSM,
    t0: Float,
    t1: Float,
    inputs: Optional[Float[Array, " input_dim"]] = None,
    diffeqsolve_settings: dict = {},
) -> Tuple[Float[Array, "state_dim state_dim"], Float[Array, "state_dim state_dim"]]:
    r"""Compute the push-forward of the sufficient statistics of a Gaussian distribution through the CDNLGSSM dynamics.

    Args:
        x0: initial mean
        P0: initial covariance
        params: model parameters
        t0: initial time
        t1: final time
        inputs: optional inputs
        diffeqsolve_settings: settings for the diffrax diffeqsolve function

    Returns:
        mean and covariance of the push-forward distribution
    """

    # Initialize
    y0 = (x0, P0)

    # Define the right-hand side of the SDEs for mean and covariance
    def rhs_all(t, y, args):
        x, P = y

        # TODO: possibly time- and parameter-dependent functions
        f = params.dynamics.drift.f

        # Get time-varying parameters
        Qc_t = params.dynamics.diffusion_cov.f(None, inputs, t)
        L_t = params.dynamics.diffusion_coefficient.f(None, inputs, t)

        # Different SDE approximations to the dynamics
        # Zeroth-order (no gradient information),
        def dynamics_order0():
            # Mean evolution
            dxdt = f(x, inputs, t)
            # Covariance evolution
            dPdt = L_t @ Qc_t @ L_t.T
            return (dxdt, dPdt)

        # First order (using Jacobian of the dynamics)
        def dynamics_order1():
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t = jacfwd(f)(x, inputs, t)

            # Mean evolution
            dxdt = f(x, inputs, t)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
            return (dxdt, dPdt)

        # Second order (using Jacobian and Hessian of the dynamics)
        def dynamics_order2():
            # Evaluate the jacobian of the dynamics function at x and inputs
            F_t = jacfwd(f)(x, inputs, t)
            # Evaluate the Hessian of the dynamics function at x and inputs
            # Based on these recommendations: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev
            H_t = jacfwd(jacrev(f))(x, inputs, t)

            # Mean evolution
            dxdt = f(x, inputs, t) + 0.5 * jnp.trace(H_t @ P)
            # Covariance evolution
            dPdt = F_t @ P + P @ F_t.T + L_t @ Qc_t @ L_t.T
            return (dxdt, dPdt)

        # Use lax.switch for conditional dynamic dispatch
        return lax.switch(
            jnp.squeeze(params.dynamics.approx_order).astype(int),
            [dynamics_order0, dynamics_order1, dynamics_order2],
        )

    # Solve the SDE as specified by rhs_all
    sol = diffeqsolve(rhs_all, t0=t0, t1=t1, y0=y0, **diffeqsolve_settings)
    # Extract final mean and covariance, ensure PSD covariance
    x, P = sol[0][-1], psd(sol[1][-1])
    # Return mean and covariance
    return x, P


# CD-NLGSSM model definition
class ContDiscreteNonlinearGaussianSSM(SSM):
    """
    Definition of a Continuous-Discrete Nonlinear Gaussian State Space Model.

    We assume a model of the form
    $$ dz=f(z,u_t,t)dt  $$
    $$ dP=L(t) Q_c L(t) $$ or $$ dP = F_t @ P + P @ F.T + L(t) Q_c_t @ L_t.T $$

    The resulting transition and emission distributions are
    $$p(z_0) = N(z_0 | m, S)$$
    $$p(z_{t_k} | z_{t_{k-1}}, u_t) = N(z_{t_k} | z_{t_{k-1}}, P_{t_k})$$
    $$p(y_{t_k} | z_{t_k}) = N(y_{t_k} | h(z_{t_k}, u_{t_k}), R_{t_k})$$

    where the model parameters are

    * $z_{t_k}$ is a latent state of size `state_dim`,
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state

    * $f$ = dynamics deterministic function (RHS), used to compute transition function
    * $L$ = dynamics coefficient multiplying brownian motion
    * $Q$ = dynamics brownian motion's covariance (system) noise
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).

    * $y_{t_k}$ is an observed variable (emissions) of size `emission_dim`
    * $h$ = emission (observation) function
    * $R$ = covariance matrix for emission (observation) noise

    These parameters of the model are stored in a separate object of type :class:`ParamsCDNLGSSM`.
    """

    # Default constructor
    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int = 0,
        diffeqsolve_settings: dict = {},
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
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
    # Define default set of CD-NLGSSM parameters,
    # with all learnable parameters set to False
    def _default_cdnlgssm_params(self) -> dict:
        ## Initial
        _initial_mean = {
            "params": LearnableVector(params=jnp.zeros(self.state_dim)),
            "props": LearnableVector(params=ParameterProperties(trainable=False)),
        }
        _initial_cov = {
            "params": LearnableMatrix(params=jnp.eye(self.state_dim)),
            "props": LearnableMatrix(
                params=ParameterProperties(
                    trainable=False, constrainer=RealToPSDBijector()
                )
            ),
        }

        ## Dynamics
        # Just a matrix with -0.1 in the diagonal, for the drift
        _dynamics_drift = {
            "params": LearnableLinear(
                weights=-0.1 * jnp.eye(self.state_dim), bias=jnp.zeros(self.state_dim)
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(trainable=False),
                bias=ParameterProperties(trainable=False),
            ),
        }

        _dynamics_diffusion_coefficient = {
            "params": LearnableMatrix(params=0.1 * jnp.eye(self.state_dim)),
            "props": LearnableMatrix(params=ParameterProperties(trainable=False)),
        }

        _dynamics_diffusion_cov = {
            "params": LearnableMatrix(params=jnp.eye(self.state_dim)),
            "props": LearnableMatrix(
                params=ParameterProperties(
                    trainable=False, constrainer=RealToPSDBijector()
                )
            ),
        }

        # Dynamics approximation: second order
        _dynamics_approx_order = {
            "params": 2.0,
            "props": ParameterProperties(
                trainable=False
            ),  # never trainable, no constraints to apply.
        }

        ## Emission
        # Linear emission function: with identity weights and zero bias
        _emission_function = {
            "params": LearnableLinear(
                weights=jnp.eye(self.emission_dim, self.state_dim),
                bias=jnp.zeros(self.emission_dim),
            ),
            "props": LearnableLinear(
                weights=ParameterProperties(trainable=False),
                bias=ParameterProperties(trainable=False),
            ),
        }

        _emission_cov = {
            "params": LearnableMatrix(params=jnp.eye(self.emission_dim)),
            "props": LearnableMatrix(
                params=ParameterProperties(
                    trainable=False, constrainer=RealToPSDBijector()
                )
            ),
        }

        # Return the default parameters as a dictionary
        return {
            "initial_mean": _initial_mean,
            "initial_cov": _initial_cov,
            "dynamics_drift": _dynamics_drift,
            "dynamics_diffusion_coefficient": _dynamics_diffusion_coefficient,
            "dynamics_diffusion_cov": _dynamics_diffusion_cov,
            "dynamics_approx_order": _dynamics_approx_order,
            "emission_function": _emission_function,
            "emission_cov": _emission_cov,
        }

    # CD-NLGSSM initialize, consistent across cd-dynamax, based on dicts
    def initialize(
        self,
        key: Optional[Float[Array, " key"]] = jr.PRNGKey(0),
        init_prior: Prior = None,
        initial_mean: dict = None,
        initial_cov: dict = None,
        dynamics_drift: dict = None,
        dynamics_diffusion_coefficient: dict = None,
        dynamics_diffusion_cov: dict = None,
        dynamics_approx_order: Optional[float] = 2.0,
        emission_function: dict = None,
        emission_cov: dict = None,
    ) -> Tuple[ParamsCDNLGSSM, ParamsCDNLGSSM]:
        r"""Initialize CD-NLGSSM parameters, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            init_prior: prior distribution for the initialization. Defaults to None.
            initial_mean: parameter $m$. Defaults to None.
            initial_cov: parameter $S$. Defaults to None.
            dynamics_drift: The drift function of the latent dynamics. Defaults to None.
            dynamics_diffusion_coefficient: parameter $L$. Defaults to None.
            dynamics_diffusion_cov: parameter $Q$. Defaults to None.
            dynamics_approx_order: order of the approximation to the dynamics. Defaults to 2.
            emission_function: The emission function. Defaults to None.
            emission_cov: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsCDNLGSSM, ParamsCDNLGSSM]: parameters and their properties.
        """

        # Create CD-NLGSSM parameters and properties,
        # based on the provided prior, init_values or defaults
        params_dict_values, params_dict_props = init_cdnlgssm_params(
            default_params=self._default_cdnlgssm_params(),
            init_params={
                "initial_mean": initial_mean,
                "initial_cov": initial_cov,
                "dynamics_drift": dynamics_drift,
                "dynamics_diffusion_coefficient": dynamics_diffusion_coefficient,
                "dynamics_diffusion_cov": dynamics_diffusion_cov,
                "dynamics_approx_order": {
                    "params": dynamics_approx_order,
                    "props": ParameterProperties(
                        trainable=False
                    ),  # never trainable, no constraints to apply.
                },
                "emission_function": emission_function,
                "emission_cov": emission_cov,
            },
            init_prior=init_prior,
        )

        # If provided, initialize prior for future use
        self.prior = init_prior

        # Return the parameters and properties
        return params_dict_values, params_dict_props

    # SSM distribution methods
    def initial_distribution(
        self,
        params: ParamsCDNLGSSM,
        inputs: Optional[Float[Array, " input_dim"]] = None,
    ) -> tfd.Distribution:
        r"""Initial distribution.

        Args:
            params: CD-NLGSSM model parameters.
            inputs: optional initial inputs (not used in state initialization).

        Returns:
            initial Gaussian state distribution.
        """
        # Gaussian initial distribution
        return MVN(params.initial.mean.f(), params.initial.cov.f())

    # Transition distribution: defined as the pushforward
    # of the continuous state dynamics and discrete inputs (controls)
    def transition_distribution(
        self,
        params: ParamsCDNLGSSM,
        state: Float[Array, " state_dim"],
        t0: Optional[Float] = None,
        t1: Optional[Float] = None,
        inputs: Optional[Float[Array, " input_dim"]] = None,
    ) -> tfd.Distribution:
        r"""CD-NLGSSM Transition distribution,
            defined as the pushforward of the continuous-time, nonlinear dynamics and discrete inputs (controls).

        Args:
            params: CD-NLGSSM model parameters.
            state: current state.
            t0: initial time.
            t1: final time.
            inputs: optional inputs.

        Returns:
            Gaussian, state transition distribution.

            # NOTE: for general CD-NLSSMs, we can not return a specific distribution,
            # unless we solve the Fokker-Planck equation for the model SDE
            # Here, we are dealing with CD-NLGSSM: we return a Gaussian distribution.
        """
        # Push-forward with assumed CD-NLGSSM
        mean, covariance = compute_pushforward(
            x0=state,
            P0=jnp.zeros(
                (state.shape[-1], state.shape[-1])
            ),  # TODO: check that last dimension is always state-dimension, even when vectorized
            params=params,
            t0=t0,
            t1=t1,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings,
        )

        # Return the Gaussian transition distribution
        return MVN(mean, covariance)

    # Emission distribution
    def emission_distribution(
        self,
        params: ParamsCDNLGSSM,
        state: Float[Array, " state_dim"],
        inputs: Optional[Float[Array, " input_dim"]] = None,
    ) -> tfd.Distribution:
        r"""CD-NLGSSM Emission distribution.
        Args:
            params: CD-NLGSSM model parameters.
            state: current state.
            inputs: optional inputs.

        Returns:
            Gaussian, state-conditional emission distribution
        """
        # TODO: change the emission distribution function to be time-dependent
        mean = params.emissions.emission_function.f(state, inputs, t=None)
        R = params.emissions.emission_cov.f(state, inputs, t=None)
        # Return the corresponding Gaussian emission distribution
        return MVN(mean, R)

    # Sampling methods
    # Sampling from the prior
    def sample_prior(
        self,
        prior: Prior,
        M: int,
        init_params: Optional[ParamsCDNLGSSM] = None,
        key: Optional[PRNGKey] = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDNLGSSM, ParamsCDNLGSSM]:
        r"""Sample from the prior distribution over CD-NLGSSM model parameters.

        Args:
            :param prior: prior distribution
            :param M: number of samples to draw
            :param init_params: dictionary of parameters to use as initialization
                if not provided, default parameters are used
            :param key: random number generator

        Returns:
            :return: Tuple with sampled CD-NLGSSM parameters and properties objects
        """
        if init_params is None:
            # Initialize with default parameters
            init_params = self._default_cdnlgssm_params()

        # Sample from the prior
        return sample_cdnlgssm_params(
            prior=prior,
            M=M,
            init_params=init_params,
            key=key,
        )

    # Sampling from the joint distribution of states and emissions, using the CD-NLGSSM distributions
    def sample_dist(
        self,
        params: ParamsCDNLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[
        Float[Array, "num_timesteps state_dim"],
        Float[Array, "num_timesteps emission_dim"],
    ]:
        r"""Sample from the joint distribution of the CD-NLGSSM
            to produce state and emission trajectories.

        Args:
            params: CD-NLGSSM model parameters
            key: random number generator key
            num_timesteps: number of time steps to sample
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            inputs: optional array of inputs.

        Returns:
            latent states and observed emissions

        """
        print(
            "CD-NLGSSM Sampling from continuous-discrete non-linear Gaussian SSM distributions"
        )
        return cdnlgssm_joint_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings,
        )

    # Sampling from the path distribution of states and emissions, using the SDE solver path
    def sample_path(
        self,
        params: ParamsCDNLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[
        Float[Array, "num_timesteps state_dim"],
        Float[Array, "num_timesteps emission_dim"],
    ]:
        r"""Sample from a forward path of the CD-NLGSSM to produce state and emission trajectories.

        Args:
            params: model parameters
            key: random number generator key
            num_timesteps: number of time steps to sample
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            inputs: optional array of inputs.

        Returns:
            latent states and observed emissions

        """
        print(
            "CD-NLGSSM Sampling from continuous-discrete non-linear Gaussian SSM path"
        )
        return cdnlgssm_path_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings,
        )

    # Inference methods
    def marginal_log_prob(
        self,
        params: ParamsCDNLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]] = None,
        filter_hyperparams: Optional[
            Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]
        ] = EKFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
        warn: bool = True,
    ) -> Scalar:
        r"""Compute the marginal log-likelihood of a sequence of emissions under the CD-NLGSSM model,
        Args:
            params: CD-NLGSSM model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            filter_hyperparams: Hyperparameters for the filtering algorithm.
            inputs: Optional input sequence.
            key: Random number generator key.
        Returns:
            Marginal log-likelihood of the emissions, $\log p(y_{1:T})$.
        """

        # Run a CD-filter, as specified via filter_hyperparams, to compute marginal log likelihood
        filtered_posterior = cdnlgssm_filter(
            params=params,
            emissions=emissions,
            t_emissions=t_emissions,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs,
            key=key,
            warn=warn,
        )
        return filtered_posterior.marginal_loglik

    # A high-level, user-friendly filtering interface (with default settings)
    def filter(
        self,
        params,
        emissions,
        t_emissions=None,
        inputs=None,
        filter_type: str = "EnKF",
        filter_state_order: str = "first",
        filter_emission_order: str = "first",
        filter_num_iter: int = 1,
        filter_state_cov_rescaling: float = 1.0,
        filter_dt_average: float = 0.1,
        enkf_N_particles: int = 25,
        enkf_inflation_delta: float = 0.0,
        diffeqsolve_max_steps: int = 100,
        diffeqsolve_dt0: float = 1e-2,
        output_fields=None,
        key=jr.PRNGKey(0),
        diffeqsolve_kwargs: Optional[dict] = {},
        extra_filter_kwargs: Optional[dict] = {},
        warn: bool = True,
    ):
        """A high-level filtering interface, to compute the filtered posterior distribution over states.

        Args:
            params: CD-NLGSSM model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            inputs: Optional input sequence.
            filter_type: Which filter to run ("EKF", "EnKF", "UKF").
            filter_state_order: Order of Taylor expansion for dynamics used in the filter.
            filter_emission_order: Order of Taylor expansion for emissions used in the EKF filter only.
            filter_num_iter: Number of iterations for iterated filters (EKF only).
            filter_state_cov_rescaling: Rescale state covariance by this factor after each update (inflation delta is better for accurate likelihoods)
            filter_dt_average: [Only for state_order="Discrete"] Average step size to determine constant state noise cov in filter.
            enkf_N_particles: Number of particles (for EnKF only).
            enkf_inflation_delta: EnKF covariance inflation (ignored by EKF/UKF).
            diffeqsolve_max_steps: Max steps for ODE solver between observations.
            diffeqsolve_dt0: Initial step size for ODE/SDE solver (default is fixed step size).
            output_fields: Which fields to return from the filter.
            key: Random number generator key.
            diffeqsolve_kwargs: Extra kwargs for the ODE solver
                (e.g., {"solver": diffrax.Heun(), "dt0": 1e-2}).
            filter_kwargs: Extra kwargs specific to the chosen filter
                (e.g., {"emission_order": "zeroth"} for EKF).
            warn: whether to issue warnings (e.g., about PSD issues)
        """

        # Build filter hyperparameters, using the provided settings
        filter_hyperparams = build_filter_hyperparams(
            filter_type=filter_type,
            filter_state_order=filter_state_order,
            filter_emission_order=filter_emission_order,
            filter_state_cov_rescaling=filter_state_cov_rescaling,
            filter_dt_average=filter_dt_average,
            enkf_N_particles=enkf_N_particles,
            enkf_inflation_delta=enkf_inflation_delta,
            diffeqsolve_max_steps=diffeqsolve_max_steps,
            diffeqsolve_dt0=diffeqsolve_dt0,
            diffeqsolve_kwargs=diffeqsolve_kwargs,
            extra_filter_kwargs=extra_filter_kwargs,
        )

        # Run the filter
        return cdnlgssm_filter(
            params=params,
            emissions=emissions,
            t_emissions=t_emissions,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs,
            num_iter=filter_num_iter,
            output_fields=output_fields,
            key=key,
            warn=warn,
        )

    # High-level, user-friendly interface combining filtering and forecasting steps
    def filter_and_forecast(
        self,
        params,
        emissions_filter,
        t_emissions_filter,
        t_emissions_forecast,
        inputs_filter=None,
        inputs_forecast=None,
        filter_type: str = "EnKF",
        filter_state_order: str = "first",
        filter_emission_order: str = "first",
        filter_num_iter: int = 1,
        filter_state_cov_rescaling: float = 1.0,
        filter_dt_average: float = 0.1,
        enkf_N_particles: int = 25,
        enkf_inflation_delta: float = 0.0,
        diffeqsolve_max_steps: int = 100,
        diffeqsolve_dt0: float = 1e-2,
        key=jr.PRNGKey(0),
        diffeqsolve_kwargs: Optional[dict] = {},
        extra_filter_kwargs: Optional[dict] = {},
        warn: bool = True,
    ):
        r"""A high-level interface combining filtering and forecasting steps.
            It computes the filtered posterior distributions over states,
            and then runs forecasting from the last filtered state.

        Args:
            params: CD-NLGSSM model parameters.
            emissions_filter: sequence of observations for filtering.
            t_emissions_filter: continuous-time specific time instants of observations for filtering: if not None, it is an array
            t_emissions_forecast: continuous-time specific time instants of observations for forecasting: if not None, it is an array
            inputs_filter: Optional input sequence for filtering.
            inputs_forecast: Optional input sequence for forecasting.
            filter_type: Which filter to run ("EKF", "EnKF", "UKF").
            filter_state_order: Order of Taylor expansion for dynamics used in the filter.
            filter_emission_order: Order of Taylor expansion for emissions used in the EKF filter only.
            filter_num_iter: Number of iterations for iterated filters (EKF only).
            filter_state_cov_rescaling: Rescale state covariance by this factor after each update (inflation delta is better for accurate likelihoods)
            filter_dt_average: [Only for state_order="Discrete"] Average step size to determine constant state noise cov in filter.
            enkf_N_particles: Number of particles (for EnKF only).
            enkf_inflation_delta: EnKF covariance inflation (ignored by EKF/UKF).
            diffeqsolve_max_steps: Max steps for ODE solver between observations.
            diffeqsolve_dt0: Initial step size for ODE/SDE solver (default is fixed step size).
            key: Random number generator key.
            diffeqsolve_kwargs: Extra kwargs for the ODE solver
                (e.g., {"solver": diffrax.Heun(), "dt0": 1e-2}).
            filter_kwargs: Extra kwargs specific to the chosen filter
                (e.g., {"emission_order": "zeroth"} for EKF).
            warn: whether to issue warnings (e.g., about PSD issues)

        Returns:
            filtered: filtering posterior over states
            forecasted: forecasting distribution over future states
        """

        # Split key for filtering and forecasting steps
        key_filter, key_forecast = jr.split(key)

        # Build filter hyperparameters, using the provided settings
        filter_hyperparams = build_filter_hyperparams(
            filter_type=filter_type,
            filter_state_order=filter_state_order,
            filter_emission_order=filter_emission_order,
            filter_state_cov_rescaling=filter_state_cov_rescaling,
            filter_dt_average=filter_dt_average,
            enkf_N_particles=enkf_N_particles,
            enkf_inflation_delta=enkf_inflation_delta,
            diffeqsolve_max_steps=diffeqsolve_max_steps,
            diffeqsolve_dt0=diffeqsolve_dt0,
            diffeqsolve_kwargs=diffeqsolve_kwargs,
            extra_filter_kwargs=extra_filter_kwargs,
        )

        # Run filter on filtering time points
        filtered = cdnlgssm_filter(
            params=params,
            emissions=emissions_filter,
            t_emissions=t_emissions_filter,
            inputs=inputs_filter,
            filter_hyperparams=filter_hyperparams,
            num_iter=filter_num_iter,
            key=key_filter,
            warn=warn,
        )

        # Initialize forecast with last filtered state
        init_time = t_emissions_filter[-1]
        init_forecast = MVN(
            filtered.filtered_means[-1, :], filtered.filtered_covariances[-1, :]
        )

        # Run forecast on forecasting time points
        forecasted = cdnlgssm_forecast(
            params=params,
            init_forecast=init_forecast,
            t_init=init_time,
            t_forecast=t_emissions_forecast,
            inputs=inputs_forecast,
            filter_hyperparams=filter_hyperparams,
            key=key_forecast,
            warn=warn,
        )

        # Return both filtered and forecasted posteriors
        return filtered, forecasted

    # High-level, user-friendly smoothing interface
    def smoother(self, *args, **kwargs):
        return cdnlgssm_smoother(*args, **kwargs)


#### CD-NLGSSM interface functions, outside the class definition, for convenience
# CD-NLGSSM sampling function, based on the model distributions
def cdnlgssm_joint_sample(
    params: ParamsCDNLGSSM,
    key: PRNGKey,
    num_timesteps: int,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    diffeqsolve_settings={},
) -> Tuple[
    Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]
]:
    r"""Sample from the joint distribution of a CD-NLGSSM
        to produce state and emission trajectories.

    Args:
        params: CD-NLGSSM model parameters
        key: random number generator key
        num_timesteps: number of time steps to sample
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        inputs: optional array of inputs.
        diffeqsolve_settings: settings for the SDE solver.

    Returns:
        latent states and observed emissions

    """

    # Function to sample initial state and emission
    def _sample_initial(key, params, inputs):
        # Split key for initial state and emission sampling
        key1, key2 = jr.split(key)

        # Sample initial state
        initial_state = MVN(params.initial.mean.f(), params.initial.cov.f()).sample(
            seed=key1
        )

        # Sample from emission
        u0 = tree_map(lambda x: x[0], inputs)
        # Compute emission mean and covariance at initial time
        emission_mean = params.emissions.emission_function.f(initial_state, u0, t=0)
        emission_cov = params.emissions.emission_cov.f(initial_state, u0, t=0)
        # Draw initial emission
        initial_emission = MVN(emission_mean, emission_cov).sample(seed=key2)

        return initial_state, initial_emission

    # Function to sample next state and emission given previous state
    def _step(prev_state, args):
        # Unpack arguments
        key, t0, t1, inpt = args
        # Split key for state and emission sampling
        key1, key2 = jr.split(key, 2)

        # Push-forward with assumed CDNLGSSM
        mean, covariance = compute_pushforward(
            x0=prev_state,
            P0=jnp.zeros(
                (prev_state.shape[-1], prev_state.shape[-1])
            ),  # TODO: check that last dimension is always state-dimension, even when vectorized
            params=params,
            t0=t0,
            t1=t1,
            inputs=inpt,
            diffeqsolve_settings=diffeqsolve_settings,
        )

        # Sample from transition
        state = MVN(mean, covariance).sample(seed=key1)

        # Sample from emission
        # Compute emission mean and covariance at time t1
        emission_mean = params.emissions.emission_function.f(state, inpt, t=t1)
        emission_cov = params.emissions.emission_cov.f(state, inpt, t=t1)
        # Draw emission at time t1
        emission = MVN(emission_mean, emission_cov).sample(seed=key2)

        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)

    # Sample the initial state and emission
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Set keys for the remaining time steps
    next_keys = jr.split(key2, num_timesteps - 1)

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
        t1 = tree_map(lambda x: x[1:, 0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps - 1)
        t1 = jnp.arange(1, num_timesteps)

    # Get inputs for remaining time steps
    next_inputs = tree_map(lambda x: x[1:], inputs)

    # Sample the remaining emissions and states via scan
    _, (next_states, next_emissions) = lax.scan(
        _step, initial_state, (next_keys, t0, t1, next_inputs)
    )

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions


# CDNLGSSM path sampling function, based on the model distributions and SDE solver
def cdnlgssm_path_sample(
    params: ParamsCDNLGSSM,
    key: PRNGKey,
    num_timesteps: int,
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    diffeqsolve_settings={},
) -> Tuple[
    Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]
]:
    r"""Sample from a forward path of the CD-NLGSSM
        to produce state and emission trajectories.

    Args:
        params: CD-NLGSSM parameters
        key: random number generator key
        num_timesteps: number of time steps to sample
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        inputs: optional array of inputs.
        diffeqsolve_settings: settings for the SDE solver.

    Returns:
        latent states and observed emissions

    """

    # Function to sample initial state and emission
    def _sample_initial(key, params, inputs):
        # Split key for initial state and emission sampling
        key1, key2 = jr.split(key)

        # Sample initial state
        initial_state = MVN(params.initial.mean.f(), params.initial.cov.f()).sample(
            seed=key1
        )

        # Sample from emission
        u0 = tree_map(lambda x: x[0], inputs)
        # Compute emission mean and covariance at initial time
        emission_mean = params.emissions.emission_function.f(initial_state, u0, t=0)
        emission_cov = params.emissions.emission_cov.f(initial_state, u0, t=0)
        # Draw initial emission
        initial_emission = MVN(emission_mean, emission_cov).sample(seed=key2)

        return initial_state, initial_emission

    # Function to path-sample next state and emission given previous state
    def _step(prev_state, args):
        # Unpack arguments
        key, t0, t1, inpt = args
        # Split key for state and emission sampling
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

        # Numerically solve the SDE, from t0 to t1
        state = diffeqsolve(
            key=key1,
            drift=drift,
            diffusion=diffusion,
            t0=t0,
            t1=t1,
            y0=prev_state,
            **diffeqsolve_settings,
        )[0]

        # Sample from emission, conditional on the new state
        # Compute emission mean and covariance at time t1
        emission_mean = params.emissions.emission_function.f(state, inpt, t=t1)
        emission_cov = params.emissions.emission_cov.f(state, inpt, t=t1)
        # Draw emission at time t1
        emission = MVN(emission_mean, emission_cov).sample(seed=key2)

        return state, (state, emission)

    # Split key for initial state and emission sampling
    key1, key2 = jr.split(key)

    # Sample the initial state and emission
    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Set keys for the remaining time steps
    next_keys = jr.split(key2, num_timesteps - 1)

    # Figure out timestamps, as vectors to scan over
    # t_emissions is of shape num_timesteps \times 1
    # t0 and t1 are num_timesteps \times 0
    if t_emissions is not None:
        num_timesteps = t_emissions.shape[0]
        t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
        t1 = tree_map(lambda x: x[1:, 0], t_emissions)
    else:
        t0 = jnp.arange(num_timesteps - 1)
        t1 = jnp.arange(1, num_timesteps)

    # Get inputs for remaining time steps
    next_inputs = tree_map(lambda x: x[1:], inputs)

    # Sample the remaining emissions and states via scan
    _, (next_states, next_emissions) = lax.scan(
        _step, initial_state, (next_keys, t0, t1, next_inputs)
    )

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions


# CDNLGSSM filtering function, dependent on the chosen filter type
def cdnlgssm_filter(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[
        Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]
    ] = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    num_iter: Optional[int] = 1,
    output_fields: Optional[List[str]] = None,
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> PosteriorGSSMFiltered:
    r"""Run an continuous-discrete nonlinear filter
        to produce the marginal likelihood and filtered state estimates.

        Depending on the hyperparameter class provided, it can execute EKF, UKF or EnKF

    Args:
        params: CD-NLGSSM parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyper-parameters of the filter
        inputs: optional array of inputs.
        num_iter: number of linearizations around posterior for update step (default 1).
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".
        key: random key (e.g., for EnKF).
        warn: whether to issue warnings during filtering.

    Returns:
        post: filtered posterior object.

    """
    # Double-check filter_hyperparams is not None
    if filter_hyperparams is None:
        filter_hyperparams = EKFHyperParams()

    # Common arguments for different filters
    common_args = {
        "params": params,
        "emissions": emissions,
        "t_emissions": t_emissions,
        "filter_hyperparams": filter_hyperparams,
        "inputs": inputs,
        "warn": warn,
    }

    # Include output_fields if provided
    if output_fields is not None:
        # Use output_fields to have more or less granular returned posterior object
        common_args["output_fields"] = output_fields

    # Run the appropriate filter based on filter_hyperparams type
    if isinstance(filter_hyperparams, EKFHyperParams):
        filtered_posterior = iterated_extended_kalman_filter(**common_args, num_iter=num_iter)
    elif isinstance(filter_hyperparams, EnKFHyperParams):
        filtered_posterior = ensemble_kalman_filter(**common_args, key=key)
    elif isinstance(filter_hyperparams, UKFHyperParams):
        filtered_posterior = unscented_kalman_filter(**common_args)

    # Return the filtered posterior
    return filtered_posterior


# CDNLGSSM smoothing function, dependent on the chosen smoother type
def cdnlgssm_smoother(
    params: ParamsCDNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[
        Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]
    ] = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    num_iter: Optional[int] = 1,
    output_fields: Optional[List[str]] = [
        "filtered_means",
        "filtered_covariances",
        "smoothed_means",
        "smoothed_covariances",
        "marginal_loglik",
    ],
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> PosteriorGSSMFiltered:
    r"""Run an continuous-discrete nonlinear smoother
        to produce the marginal likelihood and smoothed state estimates.

        Depending on the hyperparameter class provided, it can execute different smoothers

    Args:
        params: CD-NLGSSM parameters.
        emissions: observation sequence.
        t_emissions: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyper-parameters of the smoother to use
        inputs: optional array of inputs.
        num_iter: optinal, number of linearizations around posterior for update step (default 1).
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "smoothed_means", "smoothed_covariances", and "marginal_loglik".
        key: random key (e.g., for EnKS).
        warn: whether to issue warnings during smoothing (e.g., PSD issues).

    Returns:
        post: posterior object.

    """
    # Run the appropriate smoother based on filter_hyperparams type
    if isinstance(filter_hyperparams, EKFHyperParams):
        smoothed_posterior = iterated_extended_kalman_smoother(
            params=params,
            emissions=emissions,
            t_emissions=t_emissions,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs,
            num_iter=num_iter,
            warn=warn,
        )
    elif isinstance(filter_hyperparams, EnKFHyperParams):
        raise ValueError("EnKS not implemented yet")
    elif isinstance(filter_hyperparams, UKFHyperParams):
        raise ValueError("UKS not implemented yet")

    # Return the smoothed posterior
    return smoothed_posterior


# CDNLGSSM forecasting function, dependent on the chosen filter type
def cdnlgssm_forecast(
    params: ParamsCDNLGSSM,
    init_forecast: Union[tfd.Distribution, Float[Array, "state_dim 1"]],
    t_init: Float[Array, "1 1"],
    t_forecast: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[
        Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]
    ] = EKFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]] = [
        "forecasted_state_means",
        "forecasted_state_covariances",
    ],
    key: PRNGKey = jr.PRNGKey(0),
    diffeqsolve_settings: dict = {},
    warn: bool = True,
) -> GSSMForecast:
    r"""Run an continuous-discrete nonlinear model
        to produce the forecasted state estimates.

        Depending on the hyperparameter class provided,
            it can execute EKF, UKF or EnKF

    Args:
        params: CD-NLGSSM parameters.
        init_forecast: initial condition to start forecasting with:
            - if init_forecast is a distribution, then we forecast such distribution based on different filtering methods
            - if init_forecast is a point estimate of state, then we forecast a forward path starting at that state
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyper-parameters of the filter
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        output_fields: list of fields to return in posterior object.
            These can take the values
                If we forecast Gaussian distributions, based on filtering methods
                    "forecasted_state_means",
                    "forecasted_state_covariances",
                If we forecast paths, based on solving the SDE
                    "forecasted_state_path".
        key: random key (e.g., for Ensemble Kalman).
        diffeqsolve_settings: settings for the SDE solver
        warn: whether to issue warnings during forecasting (e.g., PSD issues).

    Returns:
        post: forecasted object.

    """
    # Double-check filter_hyperparams is not None
    if filter_hyperparams is None:
        filter_hyperparams = EKFHyperParams()

    # Common arguments for different methods
    common_args = {
        "params": params,
        "init_forecast": init_forecast,
        "t_init": t_init,
        "t_forecast": t_forecast,
        "filter_hyperparams": filter_hyperparams,
        "inputs": inputs,
        "output_fields": output_fields,
        "warn": warn,
    }

    # Check whether init_forecast is a distribution or a point estimate
    if isinstance(init_forecast, tfd.Distribution):
        # Forecasting a distribution, based on different filters
        if isinstance(filter_hyperparams, EKFHyperParams):
            forecast = forecast_extended_kalman_filter(**common_args)
        elif isinstance(filter_hyperparams, EnKFHyperParams):
            forecast = forecast_ensemble_kalman_filter(**common_args, key=key)
        elif isinstance(filter_hyperparams, UKFHyperParams):
            forecast = forecast_unscented_kalman_filter(**common_args)
    else:
        # Forecasting point estimates, based on pushing forward the model

        # Figure out timestamps, as vectors to scan over
        # t_forecast is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps \times 0
        if t_forecast is not None:
            num_timesteps = t_forecast.shape[0]
            t0 = tree_map(
                lambda x: jnp.concatenate((t_init, t_forecast[:-1, 0])),
                t_forecast,
            )
            t1 = tree_map(lambda x: x[:, 0], t_forecast)
        else:
            raise ValueError("t_forecast must be provided for forecasting")

        # Set-up indexing and inputs
        t0_idx = jnp.arange(num_timesteps)
        inputs = _process_input(inputs, num_timesteps + 1)

        # Define the function to scan over
        def _step(prev_state, args):
            # Unpack arguments
            key, t0, t1, t0_idx = args

            # Define the drift and diffusion functions
            def drift(t, y, args):
                return params.dynamics.drift.f(y, inputs[t0_idx], t)

            def diffusion(t, y, args):
                Qc_t = params.dynamics.diffusion_cov.f(None, inputs[t0_idx], t)
                Q_sqrt = jnp.linalg.cholesky(Qc_t)
                L_t = params.dynamics.diffusion_coefficient.f(None, inputs[t0_idx], t)
                combined_diffusion = L_t @ Q_sqrt
                return combined_diffusion

            # Solve the SDE numerically, from t0 to t1
            state = diffeqsolve(
                key=key,
                drift=drift,
                diffusion=diffusion,
                t0=t0,
                t1=t1,
                y0=prev_state,
                **diffeqsolve_settings,
            )[0]

            # Return the state
            return state, (state)

        # Split keys for each time step
        next_keys = jr.split(key, num_timesteps)

        # Forecast states over time, via scan
        _, (next_states) = lax_scan(
            _step, init_forecast, (next_keys, t0, t1, t0_idx), debug=DEBUG
        )  # type: ignore

        # Build the forecast object
        forecast = GSSMForecast(forecasted_state_path=next_states)

    # Return the forecasted object
    return forecast


# CDNLGSSM emissions function, dependent on the chosen filter type
def cdnlgssm_emissions(
    params: ParamsCDNLGSSM,
    t_states: Float[Array, "num_timesteps 1"],
    state_means: Float[Array, "num_timesteps state_dim"],
    state_covs: Optional[Float[Array, "num_timesteps state_dim state_dim"]] = None,
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    filter_hyperparams: Optional[
        Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]
    ] = None,
    key: PRNGKey = jr.PRNGKey(0),
) -> Tuple[
    Float[Array, "num_timesteps emission_dim"],
    Float[Array, "num_timesteps emission_dim emission_dim"],
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
        filter_hyperparams: hyper-parameters of the filter, optional
        key: random key for sampling

    Returns:
        emissions_mean: mean of emissions
        emissions_covariance: covariance of emissions
    """

    # Check whether we are using model or a filter
    if filter_hyperparams is not None:
        # Emissions, based on different filters
        if isinstance(filter_hyperparams, EKFHyperParams):
            emissions_mean, emissions_covariance = emissions_extended_kalman_filter(
                params=params,
                t_states=t_states,
                state_means=state_means,
                state_covs=state_covs,
                inputs=inputs,
                filter_hyperparams=filter_hyperparams,
            )
        elif isinstance(filter_hyperparams, EnKFHyperParams):
            emissions_mean, emissions_covariance = emissions_ensemble_kalman_filter(
                params=params,
                t_states=t_states,
                state_means=state_means,
                state_covs=state_covs,
                inputs=inputs,
                filter_hyperparams=filter_hyperparams,
                key=key,
            )
        elif isinstance(filter_hyperparams, UKFHyperParams):
            emissions_mean, emissions_covariance = emissions_unscented_kalman_filter(
                params=params,
                t_states=t_states,
                state_means=state_means,
                state_covs=state_covs,
                inputs=inputs,
                filter_hyperparams=filter_hyperparams,
            )
    else:
        # Emissions, based on pushing the state through the model emission function

        # Figure out timestamps, as vectors to scan over
        # t_states is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps \times 0
        if t_states is not None:
            num_timesteps = t_states.shape[0]
            t0 = tree_map(lambda x: x[:, 0], t_states)
        else:
            raise ValueError("t_states must be provided for forecasting")

        # Set-up indexing and inputs
        t0_idx = jnp.arange(num_timesteps)
        inputs = _process_input(inputs, num_timesteps)

        # Define the function to scan over
        def _step(state, args):
            # Unpack arguments
            this_state, t0, t0_idx = args

            # Push the state through the emission function
            emission_mean = params.emissions.emission_function.f(
                this_state, inputs[t0_idx], t=t0
            )
            # Emission covariance, as determined by the model
            emission_cov = params.emissions.emission_cov.f(
                this_state, inputs[t0_idx], t=t0
            )

            # Return the state and emission's mean and covariance
            return this_state, (emission_mean, emission_cov)

        # Compute emissions, over time, via scan
        _, (emissions_mean, emissions_covariance) = lax_scan(
            _step, state_means[0], (state_means, t0, t0_idx), debug=DEBUG
        )  # type: ignore

    # Return the emission mean and covariance
    return emissions_mean, emissions_covariance


# Helper function to build filter hyperparameters ---useful in high-level interfaces
def build_filter_hyperparams(
    filter_type: str = "EnKF",
    filter_state_order: str = "first",
    filter_emission_order: str = "first",
    filter_state_cov_rescaling: float = 1.0,
    filter_dt_average: float = 0.1,
    enkf_N_particles: int = 25,
    enkf_inflation_delta: float = 0.0,
    diffeqsolve_max_steps: int = 100,
    diffeqsolve_dt0: float = 1e-2,
    diffeqsolve_kwargs: Optional[dict] = {},
    extra_filter_kwargs: Optional[dict] = {},
) -> Union[EKFHyperParams, EnKFHyperParams, UKFHyperParams]:
    # Prepare diffeqsolve settings
    diffeqsolve_settings = {
        "max_steps": diffeqsolve_max_steps,
        "dt0": diffeqsolve_dt0,
        **diffeqsolve_kwargs,
    }

    # Prepare filtering settings
    common_filter_args = {
        "state_order": filter_state_order,
        "diffeqsolve_settings": diffeqsolve_settings,
        "cov_rescaling": filter_state_cov_rescaling,
        "dt_average": filter_dt_average,
    }

    if filter_type == "EKF":
        filter_hyperparams = EKFHyperParams(
            emission_order=filter_emission_order,
            **common_filter_args,
            **extra_filter_kwargs,
        )
    elif filter_type == "EnKF":
        filter_hyperparams = EnKFHyperParams(
            N_particles=enkf_N_particles,
            inflation_delta=enkf_inflation_delta,
            **common_filter_args,
            **extra_filter_kwargs,
        )
    elif filter_type == "UKF":
        filter_hyperparams = UKFHyperParams(
            **common_filter_args,
            **extra_filter_kwargs,
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filter_hyperparams
