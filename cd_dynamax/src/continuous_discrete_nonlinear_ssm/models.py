from typing import List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, jacfwd, jacrev
from jaxtyping import Array, Float, PRNGKeyArray

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
)

from cd_dynamax.dynamax.parameters import ParameterProperties
from cd_dynamax.dynamax.utils.bijectors import RealToPSDBijector

from ..ssm_temissions import SSM, Prior
from .builders import build_params
from .inference_dpf import DPFHyperParams, filter_dpf
from ..utils.diffrax_utils import diffeqsolve
from .cdnlssm_utils import (
    ParamsCDNLSSM,
    LearnableVector,
    LearnableMatrix,
    LearnableMultivariateNormal,
    init_cdnlssm_params,
    sample_cdnlssm_params,
)
from ..continuous_discrete_nonlinear_gaussian_ssm.models import (
    compute_pushforward as compute_pushforward_cdnlgssm,
)


class PosteriorCDNLSSMFiltered(NamedTuple):
    filtered_means: Array
    filtered_covariances: Array
    particles: Array
    log_weights: Array
    marginal_loglik: float


class ContDiscreteNonlinearSSM(SSM):
    r"""Continuous-discrete nonlinear SSM with generic (possibly non-Gaussian) initial and emission distributions.

    We assume a model of the form
    $$ dz=f(z,u_t,t)dt  $$
    $$ dP=L(t) Q_c L(t) $$ or $$ dP = F_t @ P + P @ F.T + L(t) Q_c_t @ L_t.T $$

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

    def build_params(self, *args, **kwargs):
        return build_params(
            state_dim=self.state_dim,
            emission_dim=self.emission_dim,
            *args,
            **kwargs,
        )

    def _default_cdnlssm_params(self) -> dict:
        zero_state = jnp.zeros(self.state_dim)
        eye_state = jnp.eye(self.state_dim)
        zero_emission = jnp.zeros(self.emission_dim)
        eye_emission = jnp.eye(self.emission_dim)

        initial_distribution = {
            "params": LearnableMultivariateNormal(
                mean=LearnableVector(params=zero_state),
                covariance=LearnableMatrix(params=eye_state),
            ),
            "props": ParameterProperties(trainable=False),
        }

        emission_distribution = {
            "params": LearnableMultivariateNormal(
                mean=LearnableVector(params=zero_emission),
                covariance=LearnableMatrix(params=eye_emission),
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

        Returns:
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
            ts = jnp.squeeze(t_emissions)
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
        init_state = params.initial.initial_distribution.distribution.sample(
            seed=key_state0
        )

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
                Qc_t = params.dynamics.diffusion_cov.f(None, u_prev_t, t)
                L_t = params.dynamics.diffusion_coefficient.f(None, u_prev_t, t)
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

    def filter(
        self,
        params: ParamsCDNLSSM,
        emissions: Array,
        t_emissions: Optional[Array] = None,
        inputs: Optional[Array] = None,
        filter_type: str = "DPF",
        filter_state_order: str = "first",
        filter_emission_order: str = "first",
        filter_num_iter: int = 1,
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

    init_state = params.initial.initial_distribution.distribution.sample(
        seed=key_state0
    )
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


def build_dpf_hyperparams(
    filter_state_order: str = "first",
    filter_state_cov_rescaling: float = 1.0,
    filter_dt_average: float = 0.1,
    N_particles: int = 100,
    diffeqsolve_dt0: float = 1e-2,
    diffeqsolve_max_steps: int = 100,
    diffeqsolve_kwargs: Optional[dict] = None,
    extra_filter_kwargs: Optional[dict] = None,
) -> DPFHyperParams:
    defaults = DPFHyperParams()
    extra_kwargs = extra_filter_kwargs or {}
    diffeqsolve_settings = {
        "dt0": diffeqsolve_dt0,
        "max_steps": diffeqsolve_max_steps,
    }
    if diffeqsolve_kwargs:
        diffeqsolve_settings.update(diffeqsolve_kwargs)

    return DPFHyperParams(
        dt_final=extra_kwargs.get("dt_final", defaults.dt_final),
        N_particles=extra_kwargs.get("N_particles", N_particles),
        resample_method=extra_kwargs.get("resample_method", defaults.resample_method),
        softness=extra_kwargs.get("softness", defaults.softness),
        cov_rescaling=extra_kwargs.get("cov_rescaling", filter_state_cov_rescaling),
        state_order=filter_state_order,
        dt_average=extra_kwargs.get("dt_average", filter_dt_average),
        diffeqsolve_settings=extra_kwargs.get(
            "diffeqsolve_settings", diffeqsolve_settings
        ),
    )


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

    particles, log_weights, log_evidence = filter_dpf(
        key=key,
        params=params,
        ys=emissions,
        us=inputs,
        ts=ts,
        hyperparams=filter_hyperparams,
    )

    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)

    def _moment(p, w):
        mean = jnp.sum(w[:, None] * p, axis=0)
        centered = p - mean
        cov = jnp.einsum("n,ni,nj->ij", w, centered, centered)
        return mean, cov

    filtered_means, filtered_covariances = vmap(_moment)(particles, weights)

    return PosteriorCDNLSSMFiltered(
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        particles=particles,
        log_weights=log_weights,
        marginal_loglik=log_evidence,
    )
