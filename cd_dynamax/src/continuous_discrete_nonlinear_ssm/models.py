# JAX imports
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax
from jax.tree_util import tree_map

# Type annotations
from jaxtyping import Array, Float, PRNGKeyArray
from typing import List, NamedTuple, Optional, Tuple

# Distributions, compatible with JAX, from TensorFlow Probability
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

# Imports from dynamax
from cd_dynamax.dynamax.types import PRNGKey, Scalar
from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector

# Imports from the cd-dynamax codebase
from ..ssm_temissions import SSM, Prior

# CDNLSSM filtering
from .inference_dpf import (
    DPFHyperParams,
    build_dpf_hyperparams,
    diff_particle_filter,
    dpf_moments,
)

# CDNLSSM pushforward, which currently just calls the CDNLGSSM pushforward
# as only Brownian motion-driven SDEs are supported
from ..continuous_discrete_nonlinear_gaussian_ssm.models import (
    compute_pushforward as compute_pushforward_cdnlgssm,
)

# Imports from cdnlgssm, including learnable functions and parameters
from ..continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import (
    LearnableVector,
    LearnableMatrix,
)

# Learnable distributions and several utilities
from .cdnlssm_utils import (
    ParamsCDNLSSM,
    LearnableGaussianEmission,
    init_cdnlssm_params,
    sample_cdnlssm_params,
)

# Diffrax based diff-eq solver
from ..utils.diffrax_utils import diffeqsolve

# Debugging utilities
from ..utils.debug_utils import lax_scan

DEBUG = False  # By default, debugging is off, e.g., no extra checks in lax_scan

# Auxiliary function to process inputs ---from dynamax
_process_input = lambda x, y: jnp.zeros((y, 1)) if x is None else x


# CD-NLSSM push-forward: compute the pushforward of particles through the CD-NLSSM dynamics.
def compute_pushforward(
    x0: Array,
    P0: Array,
    params: ParamsCDNLSSM,
    t0: Float,
    t1: Float,
    inputs: Optional[Array] = None,
    diffeqsolve_settings: Optional[dict] = None,
) -> Tuple[Array, Array]:
    """Compute the pushforward of particles through the CD-NLSSM dynamics.

    Currently, as only Brownian motion-driven SDEs are supported, this simply calls the CDNLGSSM pushforward.

    Returns:
        Tuple[Array, Array]: Mean and covariance of the pushforward.
    """
    return compute_pushforward_cdnlgssm(
        x0=x0,
        P0=P0,
        params=params,
        t0=t0,
        t1=t1,
        inputs=inputs,
        diffeqsolve_settings=diffeqsolve_settings,
    )


# CD-NLSSM posterior definition
class PosteriorCDNLSSMFiltered(NamedTuple):
    filtered_means: Array
    filtered_covariances: Array
    particles: Array
    log_weights: Array
    marginal_loglik: float


# CD-NLSSM model definition
class ContDiscreteNonlinearSSM(SSM):
    r"""Continuous-discrete nonlinear SSM with generic (possibly non-Gaussian) initial and emission distributions.

    We assume a model of the form

    $$ dz=f(z,u_t,t)dt + L(z, u, t) db(t)$$

    with diffusion covariance $Q_c$.

    We allow for arbitrary initial and emission distributions,
    $$p(z_0) = p(z_0)$$
    $$p(y_{t_k} | z_{t_k}) = p(y_{t_k} | z_{t_k})$$

    where the model parameters are
    * $z_t$ = hidden variables of size `state_dim`,

    * $f$ = dynamics deterministic function (RHS), used to compute transition function
    * $L$ = dynamics coefficient multiplying brownian motion
    * $Q$ = dynamics brownian motion's covariance (system) noise
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).

    * $y_t$ = observed variables of size `emission_dim`
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
        self.input_dim = input_dim
        self._diffeqsolve_settings = diffeqsolve_settings
        self.prior: Optional[Prior] = None

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
    # Define default set of CD-NLSSM parameters,
    # with all learnable parameters set to False
    def _default_cdnlssm_params(self) -> dict:
        zero_state = jnp.zeros(self.state_dim)
        eye_state = jnp.eye(self.state_dim)
        zero_emission = jnp.zeros(self.emission_dim)
        eye_emission = jnp.eye(self.emission_dim)

        initial_distribution = {
            "params": MVN(zero_state, eye_state),
            "props": ParameterProperties(trainable=False),
        }

        emission_distribution = {
            "params": LearnableGaussianEmission(
                emission_function=LearnableVector(params=zero_emission),
                emission_cov=LearnableMatrix(params=eye_emission),
            ),
            "props": ParameterProperties(
                trainable=False, constrainer=RealToPSDBijector()
            ),
        }

        dynamics_drift = {
            "params": LearnableVector(params=jnp.zeros(self.state_dim)),
            "props": ParameterProperties(trainable=False),
        }

        dynamics_diffusion_coefficient = {
            "params": LearnableMatrix(params=eye_state),
            "props": ParameterProperties(trainable=False),
        }

        dynamics_diffusion_cov = {
            "params": LearnableMatrix(params=eye_state),
            "props": ParameterProperties(
                trainable=False, constrainer=RealToPSDBijector()
            ),
        }

        dynamics_approx_order = {
            "params": 1.0,
            "props": ParameterProperties(trainable=False),
        }

        return {
            "initial_distribution": initial_distribution,
            "dynamics_drift": dynamics_drift,
            "dynamics_diffusion_coefficient": dynamics_diffusion_coefficient,
            "dynamics_diffusion_cov": dynamics_diffusion_cov,
            "dynamics_approx_order": dynamics_approx_order,
            "emission_distribution": emission_distribution,
        }

    # CD-NLSSM initialize, consistent across cd-dynamax, based on dicts
    def initialize(
        self,
        key: Float[Array, "2"] = jr.PRNGKey(0),
        init_prior: Prior = None,
        initial_distribution: dict = None,
        dynamics_drift: dict = None,
        dynamics_diffusion_coefficient: dict = None,
        dynamics_diffusion_cov: dict = None,
        dynamics_approx_order: Optional[float] = 1.0,
        emission_distribution: dict = None,
    ) -> Tuple[ParamsCDNLSSM, ParamsCDNLSSM]:
        """Initialize the model parameters.

        Args:
            key: Random key.
            init_prior: Prior distribution.
            initial_distribution: Initial distribution.
            dynamics_drift: Dynamics drift.
            dynamics_diffusion_coefficient: Dynamics diffusion coefficient.
            dynamics_diffusion_cov: Dynamics diffusion covariance.
            dynamics_approx_order: Dynamics approximation order.
            emission_distribution: Emission distribution.

        rns:
            Tuple[ParamsCDNLSSM, ParamsCDNLSSM]: Parameters and their properties.
        """
        params_values, params_props = init_cdnlssm_params(
            default_params=self._default_cdnlssm_params(),
            init_params={
                "initial_distribution": initial_distribution,
                "dynamics_drift": dynamics_drift,
                "dynamics_diffusion_coefficient": dynamics_diffusion_coefficient,
                "dynamics_diffusion_cov": dynamics_diffusion_cov,
                "dynamics_approx_order": {
                    "params": dynamics_approx_order,
                    "props": ParameterProperties(trainable=False),
                },
                "emission_distribution": emission_distribution,
            },
            init_prior=init_prior,
            key=key,
        )
        self.prior = init_prior
        return params_values, params_props

    # SSM distribution methods
    def initial_distribution(self, params: ParamsCDNLSSM):
        return params.initial.initial_distribution.distribution

    def emission_distribution(self, params: ParamsCDNLSSM, state, inputs=None, t=None):
        return params.emissions.emission_distribution.distribution

    def transition_distribution(
        self, params: ParamsCDNLSSM, state, t0=None, t1=None, inputs=None
    ):
        # Particle filter path does not expose a closed-form transition; users should sample via filtering.
        raise NotImplementedError(
            "CD-NLSSM transition distribution is not available in closed form."
        )

    # Sampling methods
    # Sampling from the prior
    def sample_prior(
        self,
        prior: Prior,
        M: int,
        init_params: Optional[ParamsCDNLSSM] = None,
        key: Float[Array, "2"] = jr.PRNGKey(0),
    ) -> Tuple[ParamsCDNLSSM, ParamsCDNLSSM]:
        """Sample from the prior distribution over CD-NLGSSM model parameters.

        :param prior: prior distribution.
        :param M: number of samples to draw.
        :param init_params: dictionary of parameters to use as initialization
            if not provided, default parameters are used
        :param key: random number generator key

        Returns:
            Tuple[ParamsCDNLSSM, ParamsCDNLSSM]: Parameters and their properties.
        """
        if init_params is None:
            init_params = self._default_cdnlssm_params()
        return sample_cdnlssm_params(
            prior=prior,
            M=M,
            init_params=init_params,
            key=key,
        )

    # Sampling from the joint distribution of states and emissions, using the CD-NLSSM distributions
    def sample_dist(
        self,
        params: ParamsCDNLSSM,
        key: PRNGKeyArray,
        num_timesteps: int,
        t_emissions: Optional[Array] = None,
        inputs: Optional[Array] = None,
    ):
        """Sample from the joint distribution to produce state and emission trajectories.

        Args:
            params: Parameters of the CDNLSSM.
            key: Random key.
            num_timesteps: Number of timesteps.
            t_emissions: Time instants of observations.
            inputs: Inputs.

        Returns:
            Tuple[Array, Array]: States and emissions.
        """
        return cdnlssm_joint_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings,
        )

    # Sampling from the path distribution of states and emissions, using the SDE solver pathwise sampling method
    def sample_path(
        self,
        params: ParamsCDNLSSM,
        key: PRNGKeyArray,
        num_timesteps: int,
        t_emissions: Optional[Array] = None,
        inputs: Optional[Array] = None,
    ):
        """Sample states and emissions by integrating the SDE and drawing from the emission distribution.

        Args:
            params: Parameters of the CDNLSSM.
            key: Random key.
            num_timesteps: Number of timesteps.
            t_emissions: Time instants of observations.
            inputs: Inputs.

        Returns:
            Tuple[Array, Array]: States and emissions.
        """
        # Splitting keys like this is necessary for consistency with the CDNLGSSM path sampler.
        key0, key_loop = jr.split(key)
        key_state0, key_emit0 = jr.split(key0, 2)

        # Time grid
        if t_emissions is not None:
            # Keep singleton-time inputs indexable (e.g. shape (1, 1) -> (1,))
            # so downstream DPF logic can safely do t_currs[:1], t_currs[:-1].
            ts = jnp.atleast_1d(jnp.squeeze(t_emissions))
        else:
            ts = jnp.arange(num_timesteps)

        # Inputs aligned with intervals (use previous input for [t0, t1])
        if inputs is not None:
            u_prev = inputs[:-1]
            u0 = inputs[0]
        else:
            u_prev = None
            u0 = None

        # Initial state and emission
        # iurteaga: removed .distribution
        init_state = params.initial.initial_distribution.sample(seed=key_state0)

        init_emission = params.emissions.emission_distribution.sample(
            x=init_state, u=u0, t=ts[0], seed=key_emit0
        )

        if num_timesteps == 1:
            return init_state[None, ...], init_emission[None, ...]

        keys_scan = jr.split(key_loop, num_timesteps - 1)
        t0 = ts[:-1]
        t1 = ts[1:]

        def _step(state_prev, args):
            key_t, t0_t, t1_t, u_prev_t = args
            key_drift, key_emit = jr.split(key_t)

            def drift(t, y, _):
                return params.dynamics.drift.f(y, u_prev_t, t)

            def diffusion(t, y, _):
                Qc_t = params.dynamics.diffusion_cov.f(y, u_prev_t, t)
                L_t = params.dynamics.diffusion_coefficient.f(y, u_prev_t, t)
                Q_sqrt = jnp.linalg.cholesky(Qc_t)
                return L_t @ Q_sqrt

            state = diffeqsolve(
                key=key_drift,
                drift=drift,
                diffusion=diffusion,
                t0=t0_t,
                t1=t1_t,
                y0=state_prev,
                **self.diffeqsolve_settings,
            )[0]

            emission = params.emissions.emission_distribution.sample(
                x=state, u=u_prev_t, t=t1_t, seed=key_emit
            )

            return state, (state, emission)

        _, (next_states, next_emissions) = lax.scan(
            _step,
            init_state,
            (keys_scan, t0, t1, u_prev),
        )

        states = jnp.concatenate([init_state[None, ...], next_states], axis=0)
        emissions = jnp.concatenate([init_emission[None, ...], next_emissions], axis=0)
        return states, emissions

    # Inference methods
    def marginal_log_prob(
        self,
        params: ParamsCDNLSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]] = None,
        filter_hyperparams: Optional[DPFHyperParams] = DPFHyperParams(),
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
        warn: bool = True,
    ) -> Scalar:
        r"""Compute the marginal log-likelihood of a sequence of emissions under the CD-NLSSM model,
        Args:
            params: CD-NLSSM model parameters.
            emissions: sequence of observations.
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array
            filter_hyperparams: Hyperparameters for the Differentiable Particle Filter (DPF) algorithm.
            inputs: Optional input sequence.
            key: Random number generator key.
        Returns:
            Marginal log-likelihood of the emissions, $\log p(y_{1:T})$.
        """

        # Run a CD-Differentiable Particle Filter (DPF), as specified via filter_hyperparams, to compute marginal log likelihood
        filtered_posterior = cdnlssm_filter(
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
        params: ParamsCDNLSSM,
        emissions: Array,
        t_emissions: Optional[Array] = None,
        inputs: Optional[Array] = None,
        filter_type: str = "DPF",
        filter_state_order: str = "first",
        filter_state_cov_rescaling: float = 1.0,
        filter_dt_average: float = 0.1,
        N_particles: int = 1_000,
        diffeqsolve_max_steps: int = 100,
        diffeqsolve_dt0: float = 1e-2,
        output_fields=None,
        key: PRNGKeyArray = jr.PRNGKey(0),
        diffeqsolve_kwargs: Optional[dict] = None,
        extra_filter_kwargs: Optional[dict] = None,
        warn: bool = True,
    ):
        """Filters a CD-NLSSM; by default, this runs a bootstrap differentiable particle filter (DPF).

        Depending on the filter_type, certain arguments are ignored.

        Args:
            params: Parameters of the CDNLSSM.
            emissions: Emission sequence.
            t_emissions: Time instants of observations.
            inputs: Inputs.
            filter_state_order: Order of Taylor expansion for dynamics used in the filter.
            filter_emission_order: Order of Taylor expansion for emissions used in the filter.
            filter_num_iter: Number of iterations for iterated filters.
            filter_state_cov_rescaling: Rescale state covariance by this factor after each update (inflation delta is better for accurate likelihoods)
            filter_dt_average: [Only for state_order="Discrete"] Average step size to determine constant state noise cov in filter.
            N_particles: Number of particles (for DPF only).
            diffeqsolve_max_steps: Max steps for ODE solver between observations.
            diffeqsolve_dt0: Initial step size for ODE/SDE solver (default is fixed step size).
            output_fields: Which fields to return from the filter.
                Defaults to `None`. This argument is currently ignored by
                `cdnlssm_filter`. Always returned:
                `"filtered_means"`
                `"filtered_covariances"`
                `"particles"`
                `"log_weights"`
                `"marginal_loglik"`
            key: Random key.
            diffeqsolve_kwargs: Extra kwargs for the ODE solver
                (e.g., {"solver": diffrax.Heun(), "dt0": 1e-2}).
            filter_kwargs: Extra kwargs specific to the chosen filter
                (e.g., {"emission_order": "zeroth"} for EKF).
            warn: whether to issue warnings (e.g., about PSD issues)

        Returns:
            PosteriorCDNLSSMFiltered: Posterior distribution of the CDNLSSM.
        """
        filter_hyperparams = build_dpf_hyperparams(
            filter_state_order=filter_state_order,
            filter_state_cov_rescaling=filter_state_cov_rescaling,
            filter_dt_average=filter_dt_average,
            N_particles=N_particles,
            diffeqsolve_dt0=diffeqsolve_dt0,
            diffeqsolve_max_steps=diffeqsolve_max_steps,
            diffeqsolve_kwargs=diffeqsolve_kwargs,
            extra_filter_kwargs=extra_filter_kwargs,
        )
        return cdnlssm_filter(
            params=params,
            emissions=emissions,
            t_emissions=t_emissions,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs,
            output_fields=output_fields,
            key=key,
            warn=warn,
        )

    # High-level, user-friendly interface combining filtering and forecasting steps
    def filter_and_forecast(
        self,
        params: ParamsCDNLSSM,
        emissions_filter: Array,
        t_emissions_filter: Array,
        t_emissions_forecast: Array,
        inputs_filter: Optional[Array] = None,
        inputs_forecast: Optional[Array] = None,
        filter_type: str = "DPF",
        filter_state_order: str = "first",
        filter_state_cov_rescaling: float = 1.0,
        filter_dt_average: float = 0.1,
        N_particles: int = 1_000,
        diffeqsolve_max_steps: int = 100,
        diffeqsolve_dt0: float = 1e-2,
        output_fields=None,
        key: PRNGKeyArray = jr.PRNGKey(0),
        diffeqsolve_kwargs: Optional[dict] = None,
        extra_filter_kwargs: Optional[dict] = None,
        warn: bool = True,
    ):
        """Filters a CD-NLSSM; by default, this runs a bootstrap differentiable particle filter (DPF).

        Depending on the filter_type, certain arguments are ignored.

        Args:
            params: Parameters of the CDNLSSM.
            emissions_filter: Emission sequence for filtering.
            t_emissions_filter: Time instants of observations for filtering.
            t_emissions_forecast: Time instants for forecasting.
            inputs_filter: Inputs for filtering.
            inputs_forecast: Inputs for forecasting.
            filter_state_order: Order of Taylor expansion for dynamics used in the filter.
            filter_state_cov_rescaling: Rescale state covariance by this factor after each update (inflation delta is better for accurate likelihoods)
            filter_dt_average: [Only for state_order="Discrete"] Average step size to determine constant state noise cov in filter.
            N_particles: Number of particles (for DPF only).
            diffeqsolve_max_steps: Max steps for ODE solver between observations.
            diffeqsolve_dt0: Initial step size for ODE/SDE solver (default is fixed step size).
            output_fields: Which fields to return from the filter.
                Defaults to `None`. This argument is currently ignored by
                `cdnlssm_filter`, which always returns
                `"filtered_means"`, `"filtered_covariances"`, `"particles"`,
                `"log_weights"`, and `"marginal_loglik"`.
            key: Random key.
            diffeqsolve_kwargs: Extra kwargs for the ODE solver
                (e.g., {"solver": diffrax.Heun(), "dt0": 1e-2}).
            filter_kwargs: Extra kwargs specific to the chosen filter
                (e.g., {"emission_order": "zeroth"} for EKF).
            warn: whether to issue warnings (e.g., about PSD issues)

        Returns:
            filtered_posterior: PosteriorCDNLSSMFiltered, posterior distribution of the CDNLSSM.
            forecasted: Float[Array, "num_timesteps state_dim M"], forecasted states over time.
        """

        # Split key for filtering and forecasting steps
        key_filter, key_forecast = jr.split(key)
        filter_hyperparams = build_dpf_hyperparams(
            filter_state_order=filter_state_order,
            filter_state_cov_rescaling=filter_state_cov_rescaling,
            filter_dt_average=filter_dt_average,
            N_particles=N_particles,
            diffeqsolve_dt0=diffeqsolve_dt0,
            diffeqsolve_max_steps=diffeqsolve_max_steps,
            diffeqsolve_kwargs=diffeqsolve_kwargs,
            extra_filter_kwargs=extra_filter_kwargs,
        )

        # Run filter on filtering time-points
        filtered = cdnlssm_filter(
            params=params,
            emissions=emissions_filter,
            t_emissions=t_emissions_filter,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs_filter,
            output_fields=output_fields,
            key=key,
            warn=warn,
        )

        # Initialize forecast with last filtered state's particles
        init_time = t_emissions_filter[-1]
        init_forecast = filtered.particles[-1, ...]  # shape M \times state_dim

        # Run forecast on forecasting time-points, using the initialized particles as initial condition for the forecast
        forecasted = cdnlssm_forecast(
            params=params,
            init_forecast=init_forecast,
            t_init=init_time,
            t_forecast=t_emissions_forecast,
            filter_hyperparams=filter_hyperparams,
            inputs=inputs_forecast,
            key=key_forecast,
            diffeqsolve_settings=filter_hyperparams.diffeqsolve_settings,
            warn=warn,
        )

        return filtered, forecasted


def cdnlssm_joint_sample(
    params: ParamsCDNLSSM,
    key: PRNGKeyArray,
    num_timesteps: int,
    t_emissions: Optional[Array] = None,
    inputs: Optional[Array] = None,
    diffeqsolve_settings: Optional[dict] = None,
):
    """Sample states and emissions jointly by integrating the SDE and drawing emissions."""
    diffeqsolve_settings = diffeqsolve_settings or {}

    key0, key_loop = jr.split(key)
    key_state0, key_emit0 = jr.split(key0, 2)

    ts = (
        jnp.squeeze(t_emissions)
        if t_emissions is not None
        else jnp.arange(num_timesteps)
    )

    if inputs is not None:
        u_prev = inputs[:-1]
        u0 = inputs[0]
    else:
        u_prev = None
        u0 = None

    init_state = params.initial.initial_distribution.sample(seed=key_state0)
    init_emission = params.emissions.emission_distribution.sample(
        x=init_state, u=u0, t=ts[0], seed=key_emit0
    )

    if num_timesteps == 1:
        return init_state[None, ...], init_emission[None, ...]

    keys_scan = jr.split(key_loop, num_timesteps - 1)
    t0 = ts[:-1]
    t1 = ts[1:]
    state_dim = init_state.shape[-1]
    zero_cov = jnp.zeros((state_dim, state_dim))

    def _step(state_prev, args):
        key_t, t0_t, t1_t, u_prev_t = args
        key_drift, key_emit = jr.split(key_t)

        mean, covariance = compute_pushforward(
            x0=state_prev,
            P0=zero_cov,
            params=params,
            t0=t0_t,
            t1=t1_t,
            inputs=u_prev_t,
            diffeqsolve_settings=diffeqsolve_settings,
        )
        state = MVN(mean, covariance).sample(seed=key_drift)

        emission = params.emissions.emission_distribution.sample(
            x=state, u=u_prev_t, t=t1_t, seed=key_emit
        )
        return state, (state, emission)

    _, (next_states, next_emissions) = lax.scan(
        _step,
        init_state,
        (keys_scan, t0, t1, u_prev),
    )

    states = jnp.concatenate([init_state[None, ...], next_states], axis=0)
    emissions = jnp.concatenate([init_emission[None, ...], next_emissions], axis=0)
    return states, emissions


# CDNLSSM filtering function: DPF with configurable hyperparameters
def cdnlssm_filter(
    params: ParamsCDNLSSM,
    emissions: Array,
    t_emissions: Optional[Array] = None,
    filter_hyperparams: Optional[DPFHyperParams] = None,
    inputs: Optional[Array] = None,
    output_fields: Optional[List[str]] = None,
    key: PRNGKeyArray = jr.PRNGKey(0),
    warn: bool = True,
):
    """Run particle filtering (configurable DPF) for a CD-NLSSM and return particles, log-weights, and log-evidence.

    Args:
        params: Parameters of the CDNLSSM.
        emissions: Emission sequence.
        t_emissions: Time instants of observations.
        filter_hyperparams: Hyperparameters of the filter.
        inputs: Inputs.
        output_fields: Fields to return.
            Defaults to `None`. This argument is currently ignored; the
            returned posterior always includes:
            `"filtered_means"`
            `"filtered_covariances"`
            `"particles"`
            `"log_weights"`
            `"marginal_loglik"`
        key: Random key.
        warn: Whether to warn.

    Returns:
        PosteriorCDNLSSMFiltered: Posterior distribution of the CDNLSSM.
    """
    if filter_hyperparams is None:
        filter_hyperparams = DPFHyperParams()

    if t_emissions is None:
        ts = jnp.arange(emissions.shape[0])
    else:
        ts = jnp.squeeze(t_emissions)

    particles, log_weights, log_evidence = diff_particle_filter(
        key=key,
        params=params,
        ys=emissions,
        us=inputs,
        ts=ts,
        hyperparams=filter_hyperparams,
    )

    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)

    filtered_means, filtered_covariances = vmap(dpf_moments)(particles, weights)

    return PosteriorCDNLSSMFiltered(
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        particles=particles,
        log_weights=log_weights,
        marginal_loglik=log_evidence,
    )


# CDNLSSM forecasting function
def cdnlssm_forecast(
    params: ParamsCDNLSSM,
    init_forecast: Float[Array, "state_dim M"],
    t_init: Float[Array, "1 1"],
    t_forecast: Optional[Float[Array, "num_timesteps 1"]] = None,
    filter_hyperparams: Optional[DPFHyperParams] = DPFHyperParams(),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    key: PRNGKey = jr.PRNGKey(0),
    diffeqsolve_settings: dict = {},
    warn: bool = True,
) -> Float[Array, "num_timesteps state_dim M"]:
    """Run an continuous-discrete nonlinear model
        to produce the forecasted state estimates.

        It supports two modes of forecasting:
        1) Forecasting through nonlinear distributions, based on DPF: in this case, the initial condition of the forecast is a distribution (e.g., the filtering distribution at the last observation), and we forecast the evolution of such distribution based on DPF with no resampling.
        2) Forecasting paths, based on solving the SDE: in this case, the initial condition of the forecast is a point estimate of state, and we

    Args:
        params: CD-NLSSM parameters.
        init_forecast: initial condition to start forecasting with, which we push forward starting at that state
        t_init: time-instant of the initial condition of forecast
        t_forecast: continuous-time specific time instants of observations: if not None, it is an array
        filter_hyperparams: hyper-parameters of the filter
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        key: random key (e.g., for sampling).
        diffeqsolve_settings: settings for the SDE solver
        warn: whether to issue warnings during forecasting (e.g., PSD issues).

    Returns:
        post: forecasted states over time of shape num_timesteps state_dim M.

    """

    # Point-estimate forecasting, based on pushing forward the initial condition through the model dynamics, via numerical SDE solving
    def _cdnlssm_forecast(
        this_init_forecast,
    ) -> Float[Array, "num_timesteps state_dim"]:
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
        # Avoid shadowing the outer-scope `inputs` captured by this closure.
        forecast_inputs = _process_input(inputs, num_timesteps + 1)

        # Define the function to scan over
        def _step(prev_state, args):
            # Unpack arguments
            key, t0, t1, t0_idx = args

            # Define the drift and diffusion functions
            def drift(t, y, args):
                return params.dynamics.drift.f(y, forecast_inputs[t0_idx], t)

            def diffusion(t, y, args):
                Qc_t = params.dynamics.diffusion_cov.f(y, forecast_inputs[t0_idx], t)
                Q_sqrt = jnp.linalg.cholesky(Qc_t)
                L_t = params.dynamics.diffusion_coefficient.f(
                    y, forecast_inputs[t0_idx], t
                )
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
            _step, this_init_forecast, (next_keys, t0, t1, t0_idx), debug=DEBUG
        )  # type: ignore

        # Return the forecasted object
        return next_states

    # Vmap or not, depending on whether we have multiple particles in the initial condition of the forecast
    if init_forecast.ndim == 1:
        return _cdnlssm_forecast(init_forecast)
    else:
        # vmap over the initial conditions of the forecast, to produce a forecast for each particle in the initial condition
        # input axis is 0, as init_forecast is of shape M \times state_dim,
        # and output axis is 1, as we want to keep the particle axis in the output, so we get num_timesteps \times M \times state_dim
        return vmap(_cdnlssm_forecast, in_axes=0, out_axes=1)(init_forecast)


# CDNLSSM emissions function
def cdnlssm_emissions(
    params: ParamsCDNLSSM,
    t_states: Float[Array, "num_timesteps 1"],
    states: Float[Array, "num_timesteps state_dim M"],
    inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    filter_hyperparams: Optional[DPFHyperParams] = DPFHyperParams(),
    key: PRNGKey = jr.PRNGKey(0),
    warn: bool = True,
) -> Float[Array, "num_timesteps emission_dim M"]:
    r"""Compute the emissions corresponding to
        - a continuous-discrete nonlinear model, as specified by params

    Args:
        params: model parameters.
        t_states: continuous-time specific time instants of states
        states: states at time instants t_states, always required
            it may handle multiple particles, in which case states is of shape num_timesteps \times state_dim \times M
        inputs: optional array of inputs, of shape (1 + num_timesteps) \times input_dim
            - The extra input is needed for the initial emission, i.e., it should be at time t_init
        filter_hyperparams: hyper-parameters of the filter, optional, actually ignored
        key: random key for sampling

    Returns:
        emissions: emissions at time instants t_states, of shape num_timesteps \times emission_dim \times M
    """

    # Point-estimate emissions
    def _cdnlssm_emissions(this_states) -> Float[Array, "num_timesteps state_dim"]:
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
        emission_inputs = _process_input(inputs, num_timesteps)
        key_emissions = jr.split(key, num_timesteps)

        # Define the function to scan over
        def _step(state, args):
            # Unpack arguments
            this_state, this_input, t0, t0_idx, this_key = args

            # Push the state through the emission distribution to get observation samples
            emissions = params.emissions.emission_distribution.sample(
                x=this_state, u=this_input, t=t0, seed=this_key
            )

            # Return the state and the emissions'
            return this_state, (emissions)

        # Compute emissions, over time, via scan
        _, (emissions) = lax_scan(
            _step,
            this_states[0],
            (this_states, emission_inputs, t0, t0_idx, key_emissions),
            debug=DEBUG,
        )  # type: ignore

        # Return the emissions
        return emissions

    # vmap over particles, if we have multiple particles in the states
    if states.ndim == 3:
        # States is of shape num_timesteps \times M \times state_dim, so we vmap over axis 1 (particles)
        return vmap(_cdnlssm_emissions, in_axes=1, out_axes=1)(states)
    else:
        return _cdnlssm_emissions(states)
