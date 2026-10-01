import jax.numpy as jnp
import jax.random as jr
import pytest

from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    _conditional_kalman_step,
    rbpfilter,
    rbpfilter_optimal,
)


def _linear_params(emission_cov=None):
    emission_cov = (
        jnp.array([[0.5, 0.22], [0.22, 0.4]]) if emission_cov is None else emission_cov
    )
    return LGParamsSLDS(
        initial_mean=jnp.array([[0.1, -0.2]]),
        initial_cov=jnp.array([0.3 * jnp.eye(2)]),
        dynamics_weights=jnp.array([[[0.8, 0.1], [-0.1, 0.9]]]),
        dynamics_cov=jnp.array([0.05 * jnp.eye(2)]),
        dynamics_bias=jnp.array([[0.03, -0.02]]),
        dynamics_input_weights=jnp.zeros((1, 2, 1)),
        emission_weights=jnp.array([[[1.0, 0.4], [-0.3, 0.8]]]),
        emission_cov=jnp.array([emission_cov]),
        emission_bias=jnp.array([[0.2, -0.1]]),
        emission_input_weights=jnp.zeros((1, 2, 1)),
        initialized=True,
    )


def _slds_params():
    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.ones(1),
            transition_matrix=jnp.ones((1, 1)),
            proposal_transition_matrix=jnp.ones((1, 1)),
        ),
        linear_gaussian=_linear_params(),
    )


def test_correlated_emission_covariance_matches_observed_subspace():
    """Masking must marginalize missing coordinates, not condition on them."""
    params = _linear_params()
    mu = jnp.array([0.4, -0.5])
    covariance = jnp.array([[0.7, 0.1], [0.1, 0.6]])
    y = jnp.array([0.35, jnp.nan])
    u = jnp.zeros(1)

    masked = _conditional_kalman_step(
        0, mu, covariance, params, u, y, jnp.array([True, False])
    )

    observed_only_params = params._replace(
        emission_weights=params.emission_weights[:, :1, :],
        emission_cov=params.emission_cov[:, :1, :1],
        emission_bias=params.emission_bias[:, :1],
        emission_input_weights=params.emission_input_weights[:, :1, :],
    )
    observed_only = _conditional_kalman_step(
        0,
        mu,
        covariance,
        observed_only_params,
        u,
        y[:1],
        jnp.array([True]),
    )

    for actual, expected in zip(masked, observed_only):
        assert jnp.allclose(actual, expected, atol=1e-6)


def test_fully_missing_emission_is_neutral():
    params = _linear_params()
    mu = jnp.array([0.4, -0.5])
    covariance = jnp.array([[0.7, 0.1], [0.1, 0.6]])

    ll, filtered_mean, filtered_cov = _conditional_kalman_step(
        0,
        mu,
        covariance,
        params,
        jnp.zeros(1),
        jnp.array([jnp.nan, jnp.nan]),
        jnp.array([False, False]),
    )

    F = params.dynamics_weights[0]
    expected_mean = F @ mu + params.dynamics_bias[0]
    expected_cov = F @ covariance @ F.T + params.dynamics_cov[0]
    assert jnp.allclose(ll, 0.0, atol=1e-6)
    assert jnp.allclose(filtered_mean, expected_mean, atol=1e-6)
    assert jnp.allclose(filtered_cov, expected_cov, atol=1e-6)


@pytest.mark.parametrize("filter_fn", [rbpfilter, rbpfilter_optimal])
def test_rbpf_accepts_coordinatewise_nan_emissions(filter_fn):
    emissions = jnp.array([[0.2, -0.1], [0.4, jnp.nan], [jnp.nan, jnp.nan], [0.1, 0.3]])
    mask = ~jnp.isnan(emissions)
    posterior = filter_fn(
        16,
        _slds_params(),
        emissions,
        jr.PRNGKey(0),
        emission_mask=mask,
    )

    assert jnp.isfinite(posterior.marginal_loglik)
    assert jnp.isfinite(posterior.weights).all()
    assert jnp.isfinite(posterior.means).all()
    assert jnp.isfinite(posterior.covariances).all()


@pytest.mark.parametrize("filter_fn", [rbpfilter, rbpfilter_optimal])
def test_all_ones_mask_matches_unmasked_rbpf(filter_fn):
    emissions = jnp.array([[0.2, -0.1], [0.4, 0.5], [0.1, 0.3]])
    args = (16, _slds_params(), emissions, jr.PRNGKey(2))
    unmasked = filter_fn(*args)
    masked = filter_fn(*args, emission_mask=jnp.ones_like(emissions, dtype=bool))

    for actual, expected in zip(masked, unmasked):
        if actual is not None:
            assert jnp.allclose(actual, expected)


def test_invalid_emission_mask_shape_is_rejected():
    emissions = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="emission_mask must have shape"):
        rbpfilter(
            4,
            _slds_params(),
            emissions,
            jr.PRNGKey(0),
            emission_mask=jnp.ones((2, 2), dtype=bool),
        )
