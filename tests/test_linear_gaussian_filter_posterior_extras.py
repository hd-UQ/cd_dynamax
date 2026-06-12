import jax.numpy as jnp

from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    KFHyperParams,
    cdlgssm_filter,
)


def _make_test_problem():
    model = ContDiscreteLinearGaussianSSM(state_dim=1, emission_dim=1)
    params, _ = model.initialize()
    emissions = jnp.zeros((3, 1))
    t_emissions = jnp.array([[0.0], [0.1], [0.2]])
    return model, params, emissions, t_emissions


def test_cdlgssm_filter_posterior_extras_match_predictive_state_moments():
    _, params, emissions, t_emissions = _make_test_problem()
    H = params.emissions.weights
    D = (
        params.emissions.input_weights
        if params.emissions.input_weights is not None
        else jnp.zeros((H.shape[0], 0))
    )
    d = (
        params.emissions.bias
        if params.emissions.bias is not None
        else jnp.zeros((H.shape[0],))
    )
    u = jnp.zeros((D.shape[1],))
    R = params.emissions.cov

    posterior = cdlgssm_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=KFHyperParams(),
        output_fields=[
            "predicted_means",
            "predicted_covariances",
            "posterior_extras",
        ],
        warn=False,
    )

    extras = posterior.posterior_extras
    assert extras is not None
    assert extras["y_pred_mean"].shape == (3, 1)
    assert extras["y_pred_cov"].shape == (3, 1, 1)
    assert extras["y_obs_pred_mean"].shape == (3, 1)
    assert extras["y_obs_pred_cov"].shape == (3, 1, 1)
    assert jnp.all(jnp.isfinite(extras["y_pred_mean"]))
    assert jnp.all(jnp.isfinite(extras["y_pred_cov"]))
    assert jnp.all(jnp.isfinite(extras["y_obs_pred_mean"]))
    assert jnp.all(jnp.isfinite(extras["y_obs_pred_cov"]))

    assert jnp.allclose(extras["y_pred_mean"][0], H @ params.initial.mean + D @ u + d)
    assert jnp.allclose(extras["y_pred_cov"][0], H @ params.initial.cov @ H.T)
    assert jnp.allclose(extras["y_obs_pred_mean"], extras["y_pred_mean"])
    assert jnp.allclose(extras["y_obs_pred_cov"], extras["y_pred_cov"] + R)
    assert jnp.allclose(
        extras["y_pred_mean"][1:], posterior.predicted_means[:-1] @ H.T + D @ u + d
    )
    assert jnp.allclose(
        extras["y_pred_cov"][1:],
        jnp.einsum("ij,tjk,kl->til", H, posterior.predicted_covariances[:-1], H.T),
    )


def test_linear_model_filter_can_return_only_posterior_extras():
    model, params, emissions, t_emissions = _make_test_problem()

    posterior = model.filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=KFHyperParams(),
        output_fields=["posterior_extras"],
        warn=False,
    )

    assert posterior.posterior_extras is not None
    assert posterior.filtered_means is None
    assert posterior.filtered_covariances is None
    assert posterior.predicted_means is None
    assert posterior.predicted_covariances is None
    assert jnp.isfinite(posterior.marginal_loglik)
