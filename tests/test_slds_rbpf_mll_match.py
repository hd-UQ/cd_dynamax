import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import norm

from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_filter
from cd_dynamax.dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    rbpfilter,
    rbpfilter_optimal,
)


STATE_DIM = 2
EMISSION_DIM = 2
NUM_TIMESTEPS = 8
# Optimal resampling indexes a threshold array of length N - 1.
NUM_PARTICLES = 2
ATOL = 2e-6


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


def make_one_regime_slds_params(lgssm_params):
    """Add a regime axis to the LGSSM parameters."""
    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.ones(1),
            transition_matrix=jnp.ones((1, 1)),
            proposal_transition_matrix=jnp.ones((1, 1)),
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=lgssm_params.initial.mean[None],
            initial_cov=lgssm_params.initial.cov[None],
            dynamics_weights=lgssm_params.dynamics.weights[None],
            dynamics_cov=lgssm_params.dynamics.cov[None],
            dynamics_bias=lgssm_params.dynamics.bias[None],
            dynamics_input_weights=jnp.zeros((1, STATE_DIM, 1)),
            emission_weights=lgssm_params.emissions.weights[None],
            emission_cov=lgssm_params.emissions.cov[None],
            emission_bias=lgssm_params.emissions.bias[None],
            emission_input_weights=jnp.zeros((1, EMISSION_DIM, 1)),
            initialized=True,
        ),
    )


@pytest.mark.parametrize("filter_fn", [rbpfilter, rbpfilter_optimal])
def test_one_regime_slds_matches_lgssm(filter_fn):
    lgssm_model, lgssm_params = init_lgssm_model()
    _, emissions = lgssm_model.sample(lgssm_params, jr.PRNGKey(0), NUM_TIMESTEPS)
    slds_params = make_one_regime_slds_params(lgssm_params)
    kf_post = lgssm_filter(lgssm_params, emissions)
    posterior = filter_fn(NUM_PARTICLES, slds_params, emissions, jr.PRNGKey(1))

    np.testing.assert_allclose(
        posterior.marginal_loglik, kf_post.marginal_loglik, atol=ATOL, rtol=0
    )
    expected_means = jnp.broadcast_to(
        kf_post.filtered_means[:, None], posterior.means.shape
    )
    expected_covs = jnp.broadcast_to(
        kf_post.filtered_covariances[:, None], posterior.covariances.shape
    )
    np.testing.assert_allclose(posterior.means, expected_means, atol=ATOL, rtol=0)
    np.testing.assert_allclose(posterior.covariances, expected_covs, atol=ATOL, rtol=0)


@pytest.fixture
def switching_params():
    transition = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.array([1.0, 0.0]),
            transition_matrix=transition,
            proposal_transition_matrix=transition,
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=jnp.array([[1.0], [-2.0]]),
            initial_cov=jnp.full((2, 1, 1), 2.0),
            dynamics_weights=jnp.ones((2, 1, 1)),
            dynamics_cov=jnp.ones((2, 1, 1)),
            dynamics_bias=jnp.zeros((2, 1)),
            dynamics_input_weights=jnp.full((2, 1, 1), 5.0),
            emission_weights=jnp.ones((2, 1, 1)),
            emission_cov=jnp.ones((2, 1, 1)),
            emission_bias=jnp.zeros((2, 1)),
            emission_input_weights=jnp.ones((2, 1, 1)),
            initialized=True,
        ),
    )


@pytest.mark.parametrize("filter_fn", [rbpfilter, rbpfilter_optimal])
@pytest.mark.parametrize("observed", [True, False], ids=["observed", "masked"])
def test_first_row_observes_the_initial_regime(filter_fn, observed, switching_params):
    posterior = filter_fn(
        NUM_PARTICLES,
        switching_params,
        jnp.array([[4.0 if observed else jnp.nan]]),
        jr.PRNGKey(2),
        inputs=jnp.array([[2.0]]),
        emission_mask=jnp.array([observed]),
    )

    # Regime 0 starts with x ~ N(1, 2), so y | u=2 ~ N(3, 3).
    expected_loglik = norm.logpdf(4.0, loc=3.0, scale=np.sqrt(3.0)) if observed else 0.0
    np.testing.assert_array_equal(posterior.states, np.zeros((1, NUM_PARTICLES)))
    np.testing.assert_allclose(posterior.marginal_loglik, expected_loglik, atol=ATOL)
    np.testing.assert_allclose(
        posterior.means, 5.0 / 3.0 if observed else 1.0, atol=ATOL
    )
    np.testing.assert_allclose(
        posterior.covariances, 2.0 / 3.0 if observed else 2.0, atol=ATOL
    )


def test_optimal_rbpf_scores_initial_mixture_before_sampling(switching_params):
    regime_probs = jnp.array([0.25, 0.75])
    params = switching_params._replace(
        discrete=switching_params.discrete._replace(initial_distribution=regime_probs)
    )
    posterior = rbpfilter_optimal(
        NUM_PARTICLES, params, jnp.array([[0.5]]), jr.PRNGKey(2)
    )

    densities = norm.pdf(0.5, loc=[1.0, -2.0], scale=np.sqrt(3.0))
    expected_loglik = np.log(regime_probs @ densities)
    np.testing.assert_allclose(posterior.marginal_loglik, expected_loglik, atol=ATOL)
