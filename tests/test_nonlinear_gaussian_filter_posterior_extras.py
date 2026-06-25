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


@pytest.mark.parametrize(
    ("filter_hyperparams", "output_field"),
    [
        (EKFHyperParams(state_order="first"), "y_pred_mean"),
        (UKFHyperParams(state_order="first"), "y_obs_pred_cov"),
        (
            EnKFHyperParams(N_particles=8, perturb_measurements=False),
            "y_obs_pred_cov",
        ),
        (
            EnKFHyperParams(N_particles=8, perturb_measurements=False),
            "predicted_ensembles",
        ),
    ],
)
def test_cdnlgssm_filter_can_return_only_requested_predictive_field(
    filter_hyperparams, output_field
):
    params, emissions, t_emissions = _make_test_problem()

    posterior = cdnlgssm_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=filter_hyperparams,
        output_fields=[output_field],
        key=jr.PRNGKey(0),
        warn=False,
    )

    assert getattr(posterior, output_field) is not None
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
def test_ekf_and_ukf_predictive_observation_fields_match_predictive_state_moments(
    filter_fn, filter_hyperparams
):
    params, emissions, t_emissions = _make_test_problem()
    R = params.emissions.emission_cov.f(None, None, None)

    posterior = filter_fn(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=filter_hyperparams,
        output_fields=[
            "predicted_means",
            "predicted_covariances",
            "y_pred_mean",
            "y_pred_cov",
            "y_obs_pred_mean",
            "y_obs_pred_cov",
        ],
        warn=False,
    )

    assert posterior.y_pred_mean.shape == (3, 1)
    assert posterior.y_pred_cov.shape == (3, 1, 1)
    assert posterior.y_obs_pred_mean.shape == (3, 1)
    assert posterior.y_obs_pred_cov.shape == (3, 1, 1)
    assert jnp.all(jnp.isfinite(posterior.y_pred_mean))
    assert jnp.all(jnp.isfinite(posterior.y_pred_cov))
    assert jnp.all(jnp.isfinite(posterior.y_obs_pred_mean))
    assert jnp.all(jnp.isfinite(posterior.y_obs_pred_cov))

    assert jnp.allclose(posterior.y_pred_mean[0], params.initial.mean.f())
    assert jnp.allclose(posterior.y_pred_cov[0], params.initial.cov.f())
    assert jnp.allclose(posterior.y_obs_pred_mean, posterior.y_pred_mean)
    assert jnp.allclose(posterior.y_obs_pred_cov, posterior.y_pred_cov + R)
    assert jnp.allclose(posterior.y_pred_mean[1:], posterior.predicted_means[:-1])
    assert jnp.allclose(posterior.y_pred_cov[1:], posterior.predicted_covariances[:-1])


def test_enkf_predictive_observation_ensembles_are_top_level_outputs():
    params, emissions, t_emissions = _make_test_problem()
    n_particles = 64

    posterior = ensemble_kalman_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EnKFHyperParams(
            N_particles=n_particles, perturb_measurements=False
        ),
        output_fields=[
            "filtered_ensembles",
            "predicted_ensembles",
            "predicted_means",
            "predicted_covariances",
            "y_ens_pred",
            "y_obs_ens_pred",
            "posterior_extras",
        ],
        key=jr.PRNGKey(0),
        warn=False,
    )

    assert posterior.filtered_ensembles.shape == (3, n_particles, 1)
    assert posterior.predicted_ensembles.shape == (3, n_particles, 1)
    assert posterior.y_ens_pred.shape == (3, n_particles, 1)
    assert posterior.y_obs_ens_pred.shape == (3, n_particles, 1)
    assert posterior.posterior_extras is not None
    assert "x_ens_filtered" not in posterior.posterior_extras
    assert "x_ens_predicted" not in posterior.posterior_extras
    assert posterior.posterior_extras["S"].shape == (3, 1, 1)
    assert jnp.all(jnp.isfinite(posterior.filtered_ensembles))
    assert jnp.all(jnp.isfinite(posterior.predicted_ensembles))
    assert jnp.all(jnp.isfinite(posterior.y_ens_pred))
    assert jnp.all(jnp.isfinite(posterior.y_obs_ens_pred))
    assert jnp.all(jnp.isfinite(posterior.posterior_extras["S"]))
    assert jnp.any(jnp.abs(posterior.y_obs_ens_pred - posterior.y_ens_pred) > 1e-6)


def test_enkf_predictive_observation_moments_are_top_level_outputs():
    params, emissions, t_emissions = _make_test_problem()
    R = params.emissions.emission_cov.f(None, None, None)

    posterior = ensemble_kalman_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EnKFHyperParams(N_particles=64, perturb_measurements=False),
        output_fields=[
            "y_pred_mean",
            "y_pred_cov",
            "y_obs_pred_mean",
            "y_obs_pred_cov",
        ],
        key=jr.PRNGKey(0),
        warn=False,
    )

    assert posterior.y_pred_mean.shape == (3, 1)
    assert posterior.y_pred_cov.shape == (3, 1, 1)
    assert posterior.y_obs_pred_mean.shape == (3, 1)
    assert posterior.y_obs_pred_cov.shape == (3, 1, 1)
    assert jnp.all(jnp.isfinite(posterior.y_pred_mean))
    assert jnp.all(jnp.isfinite(posterior.y_pred_cov))
    assert jnp.all(jnp.isfinite(posterior.y_obs_pred_mean))
    assert jnp.all(jnp.isfinite(posterior.y_obs_pred_cov))
    assert jnp.allclose(posterior.y_obs_pred_mean, posterior.y_pred_mean)
    assert jnp.allclose(posterior.y_obs_pred_cov, posterior.y_pred_cov + R)


def test_enkf_posterior_extras_are_not_returned_by_default():
    params, emissions, t_emissions = _make_test_problem()

    posterior = ensemble_kalman_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EnKFHyperParams(N_particles=16, perturb_measurements=False),
        key=jr.PRNGKey(0),
        warn=False,
    )

    assert posterior.posterior_extras is None
    assert posterior.filtered_means is not None
    assert posterior.predicted_means is not None
    assert jnp.isfinite(posterior.marginal_loglik)


@pytest.mark.parametrize(
    ("filter_fn", "filter_hyperparams"),
    [
        (extended_kalman_filter, EKFHyperParams(state_order="first")),
        (unscented_kalman_filter, UKFHyperParams(state_order="first")),
        (
            ensemble_kalman_filter,
            EnKFHyperParams(N_particles=32, perturb_measurements=False),
        ),
    ],
)
def test_requested_marginal_loglik_is_scalar_total(filter_fn, filter_hyperparams):
    params, emissions, t_emissions = _make_test_problem()

    default_kwargs = dict(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=filter_hyperparams,
        warn=False,
    )
    requested_kwargs = dict(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=filter_hyperparams,
        output_fields=["marginal_loglik", "filtered_means"],
        warn=False,
    )
    if filter_fn is ensemble_kalman_filter:
        default_kwargs["key"] = jr.PRNGKey(0)
        requested_kwargs["key"] = jr.PRNGKey(0)

    posterior_default = filter_fn(**default_kwargs)
    posterior_requested = filter_fn(**requested_kwargs)

    assert jnp.shape(posterior_requested.marginal_loglik) == ()
    assert jnp.ndim(posterior_requested.marginal_loglik) == 0
    assert jnp.allclose(
        posterior_requested.marginal_loglik, posterior_default.marginal_loglik
    )
    assert posterior_requested.filtered_means.shape == (3, 1)


def test_cdnlgssm_filter_requested_marginal_loglik_is_scalar_total():
    params, emissions, t_emissions = _make_test_problem()

    posterior_default = cdnlgssm_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EKFHyperParams(state_order="first"),
        key=jr.PRNGKey(0),
        warn=False,
    )
    posterior_requested = cdnlgssm_filter(
        params=params,
        emissions=emissions,
        t_emissions=t_emissions,
        filter_hyperparams=EKFHyperParams(state_order="first"),
        output_fields=["marginal_loglik", "filtered_means"],
        key=jr.PRNGKey(0),
        warn=False,
    )

    assert jnp.shape(posterior_requested.marginal_loglik) == ()
    assert jnp.ndim(posterior_requested.marginal_loglik) == 0
    assert jnp.allclose(
        posterior_requested.marginal_loglik, posterior_default.marginal_loglik
    )
    assert posterior_requested.filtered_means.shape == (3, 1)


@pytest.mark.parametrize(
    ("filter_fn", "filter_hyperparams", "output_field", "filter_name"),
    [
        (
            extended_kalman_filter,
            EKFHyperParams(state_order="first"),
            "y_ens_pred",
            "extended_kalman_filter",
        ),
        (
            unscented_kalman_filter,
            UKFHyperParams(state_order="first"),
            "y_obs_ens_pred",
            "unscented_kalman_filter",
        ),
    ],
)
def test_filters_reject_unsupported_predictive_output_fields(
    filter_fn, filter_hyperparams, output_field, filter_name
):
    params, emissions, t_emissions = _make_test_problem()

    with pytest.raises(ValueError, match=filter_name):
        kwargs = dict(
            params=params,
            emissions=emissions,
            t_emissions=t_emissions,
            filter_hyperparams=filter_hyperparams,
            output_fields=[output_field],
            warn=False,
        )
        if filter_name == "ensemble_kalman_filter":
            kwargs["key"] = jr.PRNGKey(0)
        filter_fn(**kwargs)
