import jax.numpy as jnp
import jax.random as jr
import pytest

from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    ContDiscreteNonlinearGaussianSSM,
    EKFHyperParams,
    EnKFHyperParams,
    UKFHyperParams,
    cdnlgssm_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_enkf import (
    ensemble_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_ukf import (
    unscented_kalman_filter,
)


def _make_test_problem():
    model = ContDiscreteNonlinearGaussianSSM(state_dim=1, emission_dim=1)
    params, _ = model.initialize()
    emissions = jnp.zeros((3, 1))
    t_emissions = jnp.array([[0.0], [0.1], [0.2]])
    return params, emissions, t_emissions


def _ensemble_covariance(ensemble):
    centered = ensemble - jnp.mean(ensemble, axis=1, keepdims=True)
    return jnp.einsum("tni,tnj->tij", centered, centered) / (ensemble.shape[1] - 1)


@pytest.mark.parametrize(
    "filter_hyperparams",
    [
        EKFHyperParams(state_order="first"),
        UKFHyperParams(state_order="first"),
        EnKFHyperParams(N_particles=8, perturb_measurements=False),
    ],
)
def test_cdnlgssm_filter_can_return_only_posterior_extras(filter_hyperparams):
    params, emissions, t_emissions = _make_test_problem()

    posterior = cdnlgssm_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=filter_hyperparams,
        output_fields=["posterior_extras"],
        key=jr.PRNGKey(0),
        warn=False,
    )

    assert posterior.posterior_extras is not None
    assert posterior.filtered_means is None
    assert posterior.filtered_covariances is None
    assert posterior.predicted_means is None
    assert posterior.predicted_covariances is None
    assert jnp.isfinite(posterior.marginal_loglik)


@pytest.mark.parametrize(
    ("filter_fn", "filter_hyperparams"),
    [
        (extended_kalman_filter, EKFHyperParams(state_order="first")),
        (unscented_kalman_filter, UKFHyperParams(state_order="first")),
    ],
)
def test_ekf_and_ukf_posterior_extras_match_predictive_state_moments(
    filter_fn, filter_hyperparams
):
    params, emissions, t_emissions = _make_test_problem()

    posterior = filter_fn(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=filter_hyperparams,
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
    assert jnp.all(jnp.isfinite(extras["y_pred_mean"]))
    assert jnp.all(jnp.isfinite(extras["y_pred_cov"]))

    assert jnp.allclose(extras["y_pred_mean"][0], params.initial.mean.f())
    assert jnp.allclose(extras["y_pred_cov"][0], params.initial.cov.f())
    assert jnp.allclose(extras["y_pred_mean"][1:], posterior.predicted_means[:-1])
    assert jnp.allclose(extras["y_pred_cov"][1:], posterior.predicted_covariances[:-1])


def test_enkf_posterior_extras_include_predictive_emission_ensemble():
    params, emissions, t_emissions = _make_test_problem()
    n_particles = 8

    posterior = ensemble_kalman_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EnKFHyperParams(
            N_particles=n_particles, perturb_measurements=False
        ),
        output_fields=[
            "predicted_means",
            "predicted_covariances",
            "posterior_extras",
        ],
        key=jr.PRNGKey(0),
        warn=False,
    )

    extras = posterior.posterior_extras
    assert extras is not None
    assert extras["y_ens_pred"].shape == (3, n_particles, 1)
    assert extras["x_ens_predicted"].shape == (3, n_particles, 1)
    assert jnp.all(jnp.isfinite(extras["y_ens_pred"]))

    expected_mean = jnp.mean(extras["y_ens_pred"], axis=1)
    expected_cov = _ensemble_covariance(extras["y_ens_pred"])
    assert jnp.allclose(expected_mean[1:], posterior.predicted_means[:-1])
    assert jnp.allclose(expected_cov[1:], posterior.predicted_covariances[:-1])

    assert jnp.allclose(extras["y_ens_pred"][1:], extras["x_ens_predicted"][:-1])
