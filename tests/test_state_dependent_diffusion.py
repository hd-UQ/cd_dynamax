import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple

from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    ContDiscreteNonlinearGaussianSSM,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_ekf import (
    EKFHyperParams,
    extended_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm import ContDiscreteNonlinearSSM
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.inference_dpf import (
    DPFHyperParams,
    diff_particle_filter,
)


class StateDependentDiagonalDiffusion(NamedTuple):
    scale: float = 0.1

    def f(self, x, u=None, t=None):
        if x is None:
            raise ValueError("state-dependent diffusion requires a state argument")
        x = jnp.atleast_1d(x)
        return jnp.diag(1.0 + self.scale * jnp.square(x))


def test_cdnlgssm_sample_path_supports_state_dependent_diffusion():
    model = ContDiscreteNonlinearGaussianSSM(state_dim=1, emission_dim=1)
    params, _ = model.initialize()
    params = params._replace(
        dynamics=params.dynamics._replace(
            diffusion_coefficient=StateDependentDiagonalDiffusion()
        )
    )

    states, emissions = model.sample_path(params, key=jr.PRNGKey(0), num_timesteps=3)

    assert states.shape == (3, 1)
    assert emissions.shape == (3, 1)
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all(jnp.isfinite(emissions))


def test_extended_kalman_filter_supports_state_dependent_diffusion():
    model = ContDiscreteNonlinearGaussianSSM(state_dim=1, emission_dim=1)
    params, _ = model.initialize()
    params = params._replace(
        dynamics=params.dynamics._replace(
            diffusion_coefficient=StateDependentDiagonalDiffusion()
        )
    )

    emissions = jnp.zeros((2, 1))
    t_emissions = jnp.array([[0.0], [0.1]])

    posterior = extended_kalman_filter(
        params,
        emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EKFHyperParams(state_order="first"),
    )

    assert posterior.filtered_means.shape == (2, 1)
    assert posterior.predicted_covariances.shape == (2, 1, 1)
    assert jnp.all(jnp.isfinite(posterior.filtered_means))
    assert jnp.all(jnp.isfinite(posterior.predicted_covariances))


def test_diff_particle_filter_supports_state_dependent_diffusion():
    model = ContDiscreteNonlinearSSM(state_dim=1, emission_dim=1)
    params, _ = model.initialize()
    params = params._replace(
        dynamics=params.dynamics._replace(
            diffusion_coefficient=StateDependentDiagonalDiffusion()
        )
    )

    emissions = jnp.zeros((2, 1))
    ts = jnp.array([0.0, 0.1])

    particles, log_weights, log_evidence = diff_particle_filter(
        jr.PRNGKey(0),
        params,
        emissions,
        ts=ts,
        hyperparams=DPFHyperParams(N_particles=8),
    )

    assert particles.shape == (2, 8, 1)
    assert log_weights.shape == (2, 8)
    assert jnp.isfinite(log_evidence)
