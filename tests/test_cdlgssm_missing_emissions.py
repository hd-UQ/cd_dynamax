import jax.numpy as jnp
import numpy as np
import pytest

from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    KFHyperParams,
    cdlgssm_filter,
)
from cd_dynamax.dynamax.parameters import ParameterProperties


def _fixed(value):
    return {
        "params": value,
        "props": ParameterProperties(trainable=False),
    }


def _make_model(state_dim=2, emission_dim=2):
    model = ContDiscreteLinearGaussianSSM(
        state_dim=state_dim,
        emission_dim=emission_dim,
        has_dynamics_bias=False,
        has_emissions_bias=False,
    )
    params, _ = model.initialize()
    return model, params


def test_timestep_mask_matches_dropping_missing_emissions():
    model, params = _make_model()
    t_emissions = jnp.array([[0.0], [0.1], [0.25], [0.4], [0.7]])
    emissions = jnp.array(
        [
            [0.2, -0.1],
            [jnp.nan, jnp.nan],
            [0.4, 0.3],
            [jnp.nan, jnp.nan],
            [-0.2, 0.5],
        ]
    )
    emission_mask = jnp.array([True, False, True, False, True])
    observed = np.asarray(emission_mask)
    hyperparams = KFHyperParams(diffeqsolve_settings={"dt0": 0.01})

    masked_filter = model.filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=hyperparams,
        emission_mask=emission_mask[:, None],
        warn=False,
    )
    reference_filter = model.filter(
        params=params,
        emissions=emissions[observed],
        t_emissions=t_emissions[observed],
        filter_hyperparams=hyperparams,
        warn=False,
    )

    np.testing.assert_allclose(
        masked_filter.filtered_means[observed],
        reference_filter.filtered_means,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        masked_filter.filtered_covariances[observed],
        reference_filter.filtered_covariances,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        masked_filter.marginal_loglik,
        reference_filter.marginal_loglik,
        rtol=1e-6,
        atol=1e-7,
    )

    masked_smoother = model.smoother(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=hyperparams,
        emission_mask=emission_mask,
        warn=False,
    )
    reference_smoother = model.smoother(
        params=params,
        emissions=emissions[observed],
        t_emissions=t_emissions[observed],
        filter_hyperparams=hyperparams,
        warn=False,
    )

    np.testing.assert_allclose(
        masked_smoother.smoothed_means[observed],
        reference_smoother.smoothed_means,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        masked_smoother.smoothed_covariances[observed],
        reference_smoother.smoothed_covariances,
        rtol=1e-6,
        atol=1e-7,
    )


def test_partial_emission_mask_matches_scalar_gaussian_update():
    model = ContDiscreteLinearGaussianSSM(
        state_dim=2,
        emission_dim=2,
        has_dynamics_bias=False,
        has_emissions_bias=False,
    )
    initial_mean = jnp.array([0.1, -0.2])
    initial_cov = jnp.array([[1.2, 0.25], [0.25, 0.8]])
    emission_weights = jnp.array([[1.0, 0.4], [-0.3, 0.8]])
    emission_cov = jnp.array([[0.3, 0.1], [0.1, 0.5]])
    params, _ = model.initialize(
        initial_mean=_fixed(initial_mean),
        initial_cov=_fixed(initial_cov),
        emission_weights=_fixed(emission_weights),
        emission_cov=_fixed(emission_cov),
    )

    observed_value = 0.7
    posterior = cdlgssm_filter(
        params=params,
        emissions=jnp.array([[observed_value, jnp.nan]]),
        t_emissions=jnp.array([[0.0]]),
        emission_mask=jnp.array([[True, False]]),
        warn=False,
    )

    H_observed = emission_weights[:1]
    predicted_observation = H_observed @ initial_mean
    innovation_cov = H_observed @ initial_cov @ H_observed.T + emission_cov[:1, :1]
    gain = jnp.linalg.solve(innovation_cov, H_observed @ initial_cov).T
    expected_mean = initial_mean + gain[:, 0] * (
        observed_value - predicted_observation[0]
    )
    expected_cov = initial_cov - gain @ innovation_cov @ gain.T
    expected_loglik = -0.5 * (
        jnp.log(2.0 * jnp.pi * innovation_cov[0, 0])
        + (observed_value - predicted_observation[0]) ** 2 / innovation_cov[0, 0]
    )

    np.testing.assert_allclose(
        posterior.filtered_means[0], expected_mean, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        posterior.filtered_covariances[0], expected_cov, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        posterior.marginal_loglik, expected_loglik, rtol=1e-6, atol=1e-7
    )


def test_all_missing_emissions_leave_initial_distribution_unchanged():
    _, params = _make_model(state_dim=2, emission_dim=1)
    posterior = cdlgssm_filter(
        params=params,
        emissions=jnp.full((1, 1), jnp.nan),
        t_emissions=jnp.array([[0.0]]),
        emission_mask=jnp.array([False]),
        warn=False,
    )

    np.testing.assert_allclose(posterior.filtered_means[0], params.initial.mean)
    np.testing.assert_allclose(posterior.filtered_covariances[0], params.initial.cov)
    np.testing.assert_allclose(posterior.marginal_loglik, 0.0, atol=1e-7)


def test_invalid_emission_mask_shape_raises():
    _, params = _make_model(state_dim=1, emission_dim=2)

    with pytest.raises(ValueError, match="emission_mask must have shape"):
        cdlgssm_filter(
            params=params,
            emissions=jnp.zeros((3, 2)),
            t_emissions=jnp.arange(3.0)[:, None],
            emission_mask=jnp.ones((2, 2), dtype=bool),
            warn=False,
        )
