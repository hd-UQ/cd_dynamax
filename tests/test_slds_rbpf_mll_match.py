import jax.numpy as jnp
import jax.random as jr
import pytest

from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_filter
from cd_dynamax.dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    rbpfilter,
    rbpfilter_optimal,
)


NUM_STATES = 3
STATE_DIM = 2
EMISSION_DIM = 2
NUM_TIMESTEPS = 8
NUM_PARTICLES = 1000
MLL_ATOL = 0.05
MEAN_ATOL = 0.1


@pytest.fixture
def seed():
    return 0


def init_lgssm_model():
    """Initialize a small stable LGSSM for the degenerate SLDS comparison."""
    model = LinearGaussianSSM(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)
    params, _ = model.initialize(
        initial_mean=jnp.array([0.2, -0.1]),
        initial_covariance=0.5 * jnp.eye(STATE_DIM),
        dynamics_weights=jnp.array([[0.75, 0.1], [-0.05, 0.8]]),
        dynamics_bias=jnp.zeros(STATE_DIM),
        dynamics_covariance=0.05 * jnp.eye(STATE_DIM),
        emission_weights=jnp.array([[1.0, 0.3], [-0.2, 0.7]]),
        emission_bias=jnp.zeros(EMISSION_DIM),
        emission_covariance=0.1 * jnp.eye(EMISSION_DIM),
    )
    return model, params


def make_degenerate_slds_params(lgssm_params):
    """Duplicate one LGSSM across modes so the discrete state has no effect."""
    transition_matrix = jnp.ones((NUM_STATES, NUM_STATES)) / NUM_STATES

    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.ones(NUM_STATES) / NUM_STATES,
            transition_matrix=transition_matrix,
            proposal_transition_matrix=transition_matrix,
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=jnp.tile(lgssm_params.initial.mean[None, :], (NUM_STATES, 1)),
            initial_cov=jnp.tile(
                lgssm_params.initial.cov[None, :, :], (NUM_STATES, 1, 1)
            ),
            dynamics_weights=jnp.tile(
                lgssm_params.dynamics.weights[None, :, :], (NUM_STATES, 1, 1)
            ),
            dynamics_cov=jnp.tile(
                lgssm_params.dynamics.cov[None, :, :], (NUM_STATES, 1, 1)
            ),
            dynamics_bias=jnp.tile(
                lgssm_params.dynamics.bias[None, :], (NUM_STATES, 1)
            ),
            dynamics_input_weights=jnp.zeros((NUM_STATES, STATE_DIM, 1)),
            emission_weights=jnp.tile(
                lgssm_params.emissions.weights[None, :, :], (NUM_STATES, 1, 1)
            ),
            emission_cov=jnp.tile(
                lgssm_params.emissions.cov[None, :, :], (NUM_STATES, 1, 1)
            ),
            emission_bias=jnp.tile(
                lgssm_params.emissions.bias[None, :], (NUM_STATES, 1)
            ),
            emission_input_weights=jnp.zeros((NUM_STATES, EMISSION_DIM, 1)),
            initialized=True,
        ),
    )


def make_rbpf_implied_lgssm_params(lgssm_model, lgssm_params):
    """Return the LGSSM prior implied by the current RBPF initialization.

    The RBPF initializes each particle by sampling an initial mean from
    N(m, S) while retaining covariance S. Its first Kalman step then predicts
    through the dynamics before conditioning on y[0]."""
    initial_mean = (
        lgssm_params.dynamics.weights @ lgssm_params.initial.mean
        + lgssm_params.dynamics.bias
    )
    initial_cov = (
        2
        * lgssm_params.dynamics.weights
        @ lgssm_params.initial.cov
        @ lgssm_params.dynamics.weights.T
        + lgssm_params.dynamics.cov
    )

    params, _ = lgssm_model.initialize(
        initial_mean=initial_mean,
        initial_covariance=initial_cov,
        dynamics_weights=lgssm_params.dynamics.weights,
        dynamics_bias=lgssm_params.dynamics.bias,
        dynamics_covariance=lgssm_params.dynamics.cov,
        emission_weights=lgssm_params.emissions.weights,
        emission_bias=lgssm_params.emissions.bias,
        emission_covariance=lgssm_params.emissions.cov,
    )
    return params


def test_degenerate_slds_rbpf_mll_matches_lgssm(seed):
    lgssm_model, lgssm_params = init_lgssm_model()
    _, emissions = lgssm_model.sample(lgssm_params, jr.PRNGKey(seed), NUM_TIMESTEPS)

    slds_params = make_degenerate_slds_params(lgssm_params)
    kf_params = make_rbpf_implied_lgssm_params(lgssm_model, lgssm_params)
    kf_post = lgssm_filter(kf_params, emissions)

    rbpf_post = rbpfilter(NUM_PARTICLES, slds_params, emissions, jr.PRNGKey(seed + 1))
    optimal_post = rbpfilter_optimal(
        NUM_PARTICLES, slds_params, emissions, jr.PRNGKey(seed + 2)
    )

    rbpf_mean = jnp.einsum("tp,tpm->tm", rbpf_post.weights, rbpf_post.means)
    optimal_mean = jnp.einsum("tp,tpm->tm", optimal_post.weights, optimal_post.means)

    print(
        "KF MLL:",
        kf_post.marginal_loglik,
        "RBPF MLL:",
        rbpf_post.marginal_loglik,
        "optimal RBPF MLL:",
        optimal_post.marginal_loglik,
    )

    assert jnp.isfinite(rbpf_post.marginal_loglik)
    assert jnp.isfinite(optimal_post.marginal_loglik)
    assert jnp.allclose(
        rbpf_post.marginal_loglik, kf_post.marginal_loglik, atol=MLL_ATOL
    )
    assert jnp.allclose(
        optimal_post.marginal_loglik, kf_post.marginal_loglik, atol=MLL_ATOL
    )
    assert jnp.allclose(rbpf_mean, kf_post.filtered_means, atol=MEAN_ATOL)
    assert jnp.allclose(optimal_mean, kf_post.filtered_means, atol=MEAN_ATOL)
