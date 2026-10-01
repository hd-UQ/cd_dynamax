"""Proposal invariance for the SLDS Rao-Blackwellized particle filter."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    rbpfilter,
)


NUM_PARTICLES = 20_000
EMISSIONS = np.array([[-0.5], [0.2], [1.0], [-0.2]])
# Absolute Monte Carlo tolerances, fixed before correcting the implementation.
REFERENCE_ATOL = np.array([0.06, 0.04, 0.03])  # log likelihood, mean, mode probability
PROPOSAL_ATOL = np.array([0.08, 0.06, 0.04])


def make_switching_params():
    """Zero dynamics remove dependence on the sampled initial continuous means."""
    transition = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.array([0.7, 0.3]),
            transition_matrix=transition,
            proposal_transition_matrix=transition,
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=jnp.zeros((2, 1)),
            initial_cov=jnp.ones((2, 1, 1)),
            dynamics_weights=jnp.zeros((2, 1, 1)),
            dynamics_cov=jnp.full((2, 1, 1), 0.5),
            dynamics_bias=jnp.array([[-1.0], [1.0]]),
            dynamics_input_weights=jnp.zeros((2, 1, 1)),
            emission_weights=jnp.ones((2, 1, 1)),
            emission_cov=jnp.ones((2, 1, 1)),
            emission_bias=jnp.zeros((2, 1)),
            emission_input_weights=jnp.zeros((2, 1, 1)),
            initialized=True,
        ),
    )


def exact_filter(params, emissions):
    """An HMM forward recursion with scalar Gaussian conditional state means.

    Since F=0, z_t | s_t is N(b_s, Q_s) independently of the previous z.
    Like rbpfilter, transition the initial mode distribution before observing y[0].
    This reference does not use the particle filter's Kalman or weighting helpers.
    """
    transition = np.asarray(params.discrete.transition_matrix)
    mode_probs = np.asarray(params.discrete.initial_distribution)
    means = np.asarray(params.linear_gaussian.dynamics_bias)[:, 0]
    process_var = np.asarray(params.linear_gaussian.dynamics_cov)[:, 0, 0]
    observation_var = np.asarray(params.linear_gaussian.emission_cov)[:, 0, 0]
    predictive_var = process_var + observation_var
    gain = process_var / predictive_var
    marginal_loglik = 0.0
    filtered_means, filtered_mode_probs = [], []
    for y in emissions[:, 0]:
        log_weights = np.log(mode_probs @ transition) + norm.logpdf(
            y, loc=means, scale=np.sqrt(predictive_var)
        )
        increment = logsumexp(log_weights)
        mode_probs = np.exp(log_weights - increment)
        conditional_means = means + gain * (y - means)
        marginal_loglik += increment
        filtered_means.append(mode_probs @ conditional_means)
        filtered_mode_probs.append(mode_probs)
    return (
        np.asarray(marginal_loglik),
        np.asarray(filtered_means),
        np.asarray(filtered_mode_probs),
    )


def summarize(posterior):
    weights = np.asarray(posterior.weights)
    states = np.asarray(posterior.states)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=2e-6, rtol=0)
    assert np.all(np.isfinite(weights))
    return (
        np.asarray(posterior.marginal_loglik),
        np.einsum("tp,tp->t", weights, np.asarray(posterior.means)[:, :, 0]),
        np.stack(
            [np.sum(weights * (states == state), axis=1) for state in range(2)],
            axis=1,
        ),
    )


def max_errors(actual, expected):
    return np.array([np.max(np.abs(a - b)) for a, b in zip(actual, expected)])


@pytest.mark.parametrize(
    "ess_threshold", [0.0, 1.0], ids=["no_resampling", "resampling"]
)
def test_rbpf_is_invariant_to_proposal(ess_threshold):
    params = make_switching_params()
    expected = exact_filter(params, EMISSIONS)
    proposals = {
        "bootstrap": params.discrete.transition_matrix,
        "uniform": jnp.full((2, 2), 0.5),
    }
    results, reference_errors = {}, []
    for name, proposal in proposals.items():
        proposal_params = params._replace(
            discrete=params.discrete._replace(proposal_transition_matrix=proposal)
        )
        posterior = rbpfilter(
            NUM_PARTICLES,
            proposal_params,
            jnp.asarray(EMISSIONS),
            jr.PRNGKey(17),
            ess_threshold=ess_threshold,
        )
        results[name] = summarize(posterior)
        reference_errors.append(max_errors(results[name], expected))
        print(
            f"{ess_threshold=}, {name}: MLL={results[name][0]:.6f}, "
            f"exact MLL={expected[0]:.6f}, "
            f"max errors [MLL, mean, mode]={reference_errors[-1]}"
        )
        if ess_threshold == 1.0:
            np.testing.assert_allclose(
                posterior.weights, 1.0 / NUM_PARTICLES, atol=1e-10, rtol=0
            )

    proposal_errors = max_errors(results["bootstrap"], results["uniform"])
    print(f"{ess_threshold=}, between-proposal errors={proposal_errors}")
    # Test all outputs together so a failure reports both likelihood and state bias.
    np.testing.assert_allclose(
        np.asarray(reference_errors) / REFERENCE_ATOL,
        0.0,
        atol=1.0,
        rtol=0,
        err_msg="Rows: bootstrap/uniform; columns: MLL/mean/mode error in tolerances",
    )
    np.testing.assert_allclose(proposal_errors / PROPOSAL_ATOL, 0.0, atol=1.0, rtol=0)


def test_rbpf_zero_target_transitions_receive_zero_weight():
    params = make_switching_params()
    params = params._replace(
        discrete=params.discrete._replace(
            initial_distribution=jnp.array([1.0, 0.0]),
            transition_matrix=jnp.eye(2),
            proposal_transition_matrix=jnp.full((2, 2), 0.5),
        )
    )
    posterior = rbpfilter(
        256, params, jnp.asarray(EMISSIONS[:2]), jr.PRNGKey(7), ess_threshold=0.0
    )
    states = np.asarray(posterior.states)
    weights = np.asarray(posterior.weights)
    # Once a path leaves mode 0 it has zero target probability, even if it returns.
    impossible_paths = np.logical_or.accumulate(states != 0, axis=0)
    assert np.any(impossible_paths)
    assert np.any(~impossible_paths)
    np.testing.assert_array_equal(weights[impossible_paths], 0.0)
    assert np.all(np.isfinite(weights))
    assert np.isfinite(posterior.marginal_loglik)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=2e-6, rtol=0)
