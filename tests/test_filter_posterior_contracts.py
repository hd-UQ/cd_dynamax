from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as jr
import pytest

from cd_dynamax import ContDiscreteLinearGaussianSSM, KFHyperParams, cdlgssm_filter
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    FILTERED_POSTERIOR_FIELD_NAMES,
    PosteriorGSSMFiltered,
)
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.inference import (
    CDLGSSM_FILTER_OUTPUT_FIELDS,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    ContDiscreteNonlinearGaussianSSM,
    EKFHyperParams,
    EnKFHyperParams,
    UKFHyperParams,
    cdnlgssm_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_ekf import (
    EKF_FILTER_OUTPUT_FIELDS,
    extended_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_enkf import (
    ENKF_FILTER_OUTPUT_FIELDS,
    ensemble_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.inference_ukf import (
    UKF_FILTER_OUTPUT_FIELDS,
    unscented_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm import (
    ContDiscreteNonlinearSSM,
    DPFHyperParams,
    cdnlssm_filter,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
    PosteriorCDNLSSMFiltered,
)
from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
    RBPFiltered,
    rbpfilter,
    rbpfilter_optimal,
)


NTIME = 3
STATE_DIM = 1
EMISSION_DIM = 1
ENKF_PARTICLES = 16
DPF_PARTICLES = 8
RBPF_PARTICLES = 16
SLDS_NUM_STATES = 2
TEST_EMISSIONS = jnp.zeros((NTIME, EMISSION_DIM))
TEST_T_EMISSIONS = jnp.array([[0.0], [0.1], [0.2]])

BASE_GSSM_FIELD_SHAPES = {
    "filtered_means": (NTIME, STATE_DIM),
    "filtered_covariances": (NTIME, STATE_DIM, STATE_DIM),
    "predicted_means": (NTIME, STATE_DIM),
    "predicted_covariances": (NTIME, STATE_DIM, STATE_DIM),
    "y_pred_mean": (NTIME, EMISSION_DIM),
    "y_pred_cov": (NTIME, EMISSION_DIM, EMISSION_DIM),
    "y_obs_pred_mean": (NTIME, EMISSION_DIM),
    "y_obs_pred_cov": (NTIME, EMISSION_DIM, EMISSION_DIM),
}

ENKF_POSTERIOR_EXTRAS_SHAPES = {
    "loglik_step": (NTIME,),
    "S": (NTIME, EMISSION_DIM, EMISSION_DIM),
    "K": (NTIME, STATE_DIM, EMISSION_DIM),
    "innovation": (NTIME, EMISSION_DIM),
    "nis": (NTIME,),
    "min_eig_S": (NTIME,),
    "cond_S": (NTIME,),
    "cond_K": (NTIME,),
}

ALL_GSSM_FILTER_OUTPUT_FIELDS = {
    "cdlgssm_filter": CDLGSSM_FILTER_OUTPUT_FIELDS,
    "extended_kalman_filter": EKF_FILTER_OUTPUT_FIELDS,
    "unscented_kalman_filter": UKF_FILTER_OUTPUT_FIELDS,
    "ensemble_kalman_filter": ENKF_FILTER_OUTPUT_FIELDS,
}

NONLINEAR_FILTER_SPECS = {
    "EKF": {
        "output_fields": EKF_FILTER_OUTPUT_FIELDS,
        "hyperparams": EKFHyperParams(state_order="first"),
        "ensemble_size": None,
        "supports_posterior_extras": False,
        "model_filter_kwargs": {},
    },
    "UKF": {
        "output_fields": UKF_FILTER_OUTPUT_FIELDS,
        "hyperparams": UKFHyperParams(state_order="first"),
        "ensemble_size": None,
        "supports_posterior_extras": False,
        "model_filter_kwargs": {},
    },
    "EnKF": {
        "output_fields": ENKF_FILTER_OUTPUT_FIELDS,
        "hyperparams": EnKFHyperParams(
            N_particles=ENKF_PARTICLES, perturb_measurements=False
        ),
        "ensemble_size": ENKF_PARTICLES,
        "supports_posterior_extras": True,
        "model_filter_kwargs": {
            "enkf_N_particles": ENKF_PARTICLES,
            "extra_filter_kwargs": {"perturb_measurements": False},
        },
    },
}


@dataclass(frozen=True)
class GSSMContractCase:
    case_id: str
    runner: Callable[[list[str]], PosteriorGSSMFiltered]
    output_fields: tuple[str, ...]
    ensemble_size: Optional[int] = None
    supports_posterior_extras: bool = False


@dataclass(frozen=True)
class DPFContractCase:
    case_id: str
    runner: Callable[[], PosteriorCDNLSSMFiltered]


def _initialize_model(model_cls):
    model = model_cls(state_dim=STATE_DIM, emission_dim=EMISSION_DIM)
    params, _ = model.initialize()
    return model, params


def _assert_scalar_total_loglik(value):
    assert jnp.shape(value) == ()
    assert jnp.ndim(value) == 0
    assert jnp.isfinite(value)


def _assert_enkf_posterior_extras_contract(extras):
    assert isinstance(extras, dict)

    for field, shape in ENKF_POSTERIOR_EXTRAS_SHAPES.items():
        assert extras[field].shape == shape, field


def _assert_posterior_cdnlssm_contract(posterior):
    assert isinstance(posterior, PosteriorCDNLSSMFiltered)
    assert posterior.filtered_means.shape == (NTIME, STATE_DIM)
    assert posterior.filtered_covariances.shape == (NTIME, STATE_DIM, STATE_DIM)
    assert posterior.particles.shape == (NTIME, DPF_PARTICLES, STATE_DIM)
    assert posterior.log_weights.shape == (NTIME, DPF_PARTICLES)
    _assert_scalar_total_loglik(posterior.marginal_loglik)


def _assert_rbpfiltered_contract(posterior):
    assert isinstance(posterior, RBPFiltered)
    _assert_scalar_total_loglik(posterior.marginal_loglik)
    assert posterior.weights.shape == (NTIME, RBPF_PARTICLES)
    assert posterior.states.shape == (NTIME, RBPF_PARTICLES)
    assert posterior.means.shape == (NTIME, RBPF_PARTICLES, STATE_DIM)
    assert posterior.covariances.shape == (NTIME, RBPF_PARTICLES, STATE_DIM, STATE_DIM)
    assert posterior["means"] is posterior.means


def _expected_gssm_field_shapes(ensemble_size=None):
    expected_shapes = dict(BASE_GSSM_FIELD_SHAPES)

    if ensemble_size is not None:
        expected_shapes.update(
            {
                "filtered_ensembles": (NTIME, ensemble_size, STATE_DIM),
                "predicted_ensembles": (NTIME, ensemble_size, STATE_DIM),
                "y_ens_pred": (NTIME, ensemble_size, EMISSION_DIM),
                "y_obs_ens_pred": (NTIME, ensemble_size, EMISSION_DIM),
            }
        )

    return expected_shapes


def _assert_requested_fields_are_supported(
    requested_fields, *, expected_shapes, supports_posterior_extras
):
    supported_fields = {"marginal_loglik", *expected_shapes}

    if supports_posterior_extras:
        supported_fields.add("posterior_extras")

    unsupported_fields = sorted(set(requested_fields) - supported_fields)
    assert not unsupported_fields, (
        f"Unsupported fields for this posterior contract: {unsupported_fields}. "
        f"Supported fields are {sorted(supported_fields)}."
    )


def _assert_posterior_gssm_optional_fields(
    posterior,
    requested_fields,
    *,
    expected_shapes,
):
    for field in FILTERED_POSTERIOR_FIELD_NAMES:
        if field in {"marginal_loglik", "posterior_extras"}:
            continue

        value = getattr(posterior, field)
        if field in requested_fields:
            assert value is not None, field
            assert value.shape == expected_shapes[field], field
        else:
            assert value is None, field


def _assert_posterior_gssm_contract(
    posterior,
    requested_fields,
    *,
    ensemble_size=None,
    supports_posterior_extras=False,
):
    assert isinstance(posterior, PosteriorGSSMFiltered)

    expected_shapes = _expected_gssm_field_shapes(ensemble_size)
    _assert_requested_fields_are_supported(
        requested_fields,
        expected_shapes=expected_shapes,
        supports_posterior_extras=supports_posterior_extras,
    )
    _assert_scalar_total_loglik(posterior.marginal_loglik)
    _assert_posterior_gssm_optional_fields(
        posterior,
        requested_fields,
        expected_shapes=expected_shapes,
    )

    if "posterior_extras" in requested_fields:
        _assert_enkf_posterior_extras_contract(posterior.posterior_extras)
    else:
        assert posterior.posterior_extras is None


def _make_slds_params():
    transition_matrix = jnp.ones((SLDS_NUM_STATES, SLDS_NUM_STATES)) / SLDS_NUM_STATES
    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.ones(SLDS_NUM_STATES) / SLDS_NUM_STATES,
            transition_matrix=transition_matrix,
            proposal_transition_matrix=transition_matrix,
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=jnp.zeros((SLDS_NUM_STATES, STATE_DIM)),
            initial_cov=jnp.tile(
                jnp.eye(STATE_DIM)[None, :, :], (SLDS_NUM_STATES, 1, 1)
            ),
            dynamics_weights=jnp.tile(
                (0.9 * jnp.eye(STATE_DIM))[None, :, :], (SLDS_NUM_STATES, 1, 1)
            ),
            dynamics_cov=jnp.tile(
                (0.1 * jnp.eye(STATE_DIM))[None, :, :], (SLDS_NUM_STATES, 1, 1)
            ),
            dynamics_bias=jnp.zeros((SLDS_NUM_STATES, STATE_DIM)),
            dynamics_input_weights=jnp.zeros((SLDS_NUM_STATES, STATE_DIM, 1)),
            emission_weights=jnp.ones((SLDS_NUM_STATES, EMISSION_DIM, STATE_DIM)),
            emission_cov=jnp.tile(
                (0.1 * jnp.eye(EMISSION_DIM))[None, :, :], (SLDS_NUM_STATES, 1, 1)
            ),
            emission_bias=jnp.zeros((SLDS_NUM_STATES, EMISSION_DIM)),
            emission_input_weights=jnp.zeros((SLDS_NUM_STATES, EMISSION_DIM, 1)),
            initialized=True,
        ),
    )


def _run_cdlgssm_filter(requested_fields):
    _, params = _initialize_model(ContDiscreteLinearGaussianSSM)
    return cdlgssm_filter(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        filter_hyperparams=KFHyperParams(),
        output_fields=requested_fields,
        warn=False,
    )


def _run_linear_model_filter(requested_fields):
    model, params = _initialize_model(ContDiscreteLinearGaussianSSM)
    return model.filter(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        filter_hyperparams=KFHyperParams(),
        output_fields=requested_fields,
        warn=False,
    )


def _run_direct_nonlinear_filter(filter_type, requested_fields):
    _, params = _initialize_model(ContDiscreteNonlinearGaussianSSM)
    filter_spec = NONLINEAR_FILTER_SPECS[filter_type]
    filter_fn = {
        "EKF": extended_kalman_filter,
        "UKF": unscented_kalman_filter,
        "EnKF": ensemble_kalman_filter,
    }[filter_type]
    call_kwargs = dict(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        filter_hyperparams=filter_spec["hyperparams"],
        output_fields=requested_fields,
        warn=False,
    )

    if filter_type == "EnKF":
        call_kwargs["key"] = jr.PRNGKey(0)

    return filter_fn(**call_kwargs)


def _run_cdnlgssm_filter(filter_type, requested_fields):
    _, params = _initialize_model(ContDiscreteNonlinearGaussianSSM)
    return cdnlgssm_filter(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        filter_hyperparams=NONLINEAR_FILTER_SPECS[filter_type]["hyperparams"],
        output_fields=requested_fields,
        key=jr.PRNGKey(0),
        warn=False,
    )


def _run_nonlinear_model_filter(filter_type, requested_fields):
    model, params = _initialize_model(ContDiscreteNonlinearGaussianSSM)
    filter_spec = NONLINEAR_FILTER_SPECS[filter_type]
    return model.filter(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        filter_type=filter_type,
        output_fields=requested_fields,
        key=jr.PRNGKey(0),
        warn=False,
        **filter_spec["model_filter_kwargs"],
    )


def _run_cdnlssm_filter():
    _, params = _initialize_model(ContDiscreteNonlinearSSM)
    return cdnlssm_filter(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        filter_hyperparams=DPFHyperParams(N_particles=DPF_PARTICLES),
        key=jr.PRNGKey(0),
        warn=False,
    )


def _run_cdnlssm_model_filter():
    model, params = _initialize_model(ContDiscreteNonlinearSSM)
    return model.filter(
        params=params,
        emissions=TEST_EMISSIONS,
        t_emissions=TEST_T_EMISSIONS,
        N_particles=DPF_PARTICLES,
        output_fields=["filtered_means"],
        key=jr.PRNGKey(0),
        warn=False,
    )


GSSM_CONTRACT_CASES = [
    GSSMContractCase(
        case_id="cdlgssm_filter",
        runner=_run_cdlgssm_filter,
        output_fields=CDLGSSM_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="linear_model_filter",
        runner=_run_linear_model_filter,
        output_fields=CDLGSSM_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="extended_kalman_filter",
        runner=partial(_run_direct_nonlinear_filter, "EKF"),
        output_fields=EKF_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="unscented_kalman_filter",
        runner=partial(_run_direct_nonlinear_filter, "UKF"),
        output_fields=UKF_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="ensemble_kalman_filter",
        runner=partial(_run_direct_nonlinear_filter, "EnKF"),
        output_fields=ENKF_FILTER_OUTPUT_FIELDS,
        ensemble_size=ENKF_PARTICLES,
        supports_posterior_extras=True,
    ),
    GSSMContractCase(
        case_id="cdnlgssm_filter_ekf",
        runner=partial(_run_cdnlgssm_filter, "EKF"),
        output_fields=EKF_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="cdnlgssm_filter_ukf",
        runner=partial(_run_cdnlgssm_filter, "UKF"),
        output_fields=UKF_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="cdnlgssm_filter_enkf",
        runner=partial(_run_cdnlgssm_filter, "EnKF"),
        output_fields=ENKF_FILTER_OUTPUT_FIELDS,
        ensemble_size=ENKF_PARTICLES,
        supports_posterior_extras=True,
    ),
    GSSMContractCase(
        case_id="nonlinear_model_filter_ekf",
        runner=partial(_run_nonlinear_model_filter, "EKF"),
        output_fields=EKF_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="nonlinear_model_filter_ukf",
        runner=partial(_run_nonlinear_model_filter, "UKF"),
        output_fields=UKF_FILTER_OUTPUT_FIELDS,
    ),
    GSSMContractCase(
        case_id="nonlinear_model_filter_enkf",
        runner=partial(_run_nonlinear_model_filter, "EnKF"),
        output_fields=ENKF_FILTER_OUTPUT_FIELDS,
        ensemble_size=ENKF_PARTICLES,
        supports_posterior_extras=True,
    ),
]

DPF_CONTRACT_CASES = [
    DPFContractCase(
        case_id="cdnlssm_filter",
        runner=_run_cdnlssm_filter,
    ),
    DPFContractCase(
        case_id="cdnlssm_model_filter_ignores_output_fields",
        runner=_run_cdnlssm_model_filter,
    ),
]


@pytest.mark.parametrize(
    "case",
    GSSM_CONTRACT_CASES,
    ids=[case.case_id for case in GSSM_CONTRACT_CASES],
)
def test_gssm_filter_output_fields_match_posterior_contract(case):
    requested_fields = list(case.output_fields)
    posterior = case.runner(requested_fields)

    _assert_posterior_gssm_contract(
        posterior,
        requested_fields,
        ensemble_size=case.ensemble_size,
        supports_posterior_extras=case.supports_posterior_extras,
    )


def test_filter_output_field_constants_reference_known_posterior_fields():
    known_fields = set(FILTERED_POSTERIOR_FIELD_NAMES)

    for filter_name, output_fields in ALL_GSSM_FILTER_OUTPUT_FIELDS.items():
        unknown_fields = sorted(set(output_fields) - known_fields)
        assert not unknown_fields, (
            f"{filter_name} advertises unknown posterior fields: {unknown_fields}. "
            f"Known fields are {sorted(known_fields)}."
        )


@pytest.mark.parametrize(
    "case",
    DPF_CONTRACT_CASES,
    ids=[case.case_id for case in DPF_CONTRACT_CASES],
)
def test_particle_filter_posterior_contract(case):
    posterior = case.runner()
    _assert_posterior_cdnlssm_contract(posterior)


def test_slds_rbpf_posterior_contract():
    params = _make_slds_params()
    posterior = rbpfilter(RBPF_PARTICLES, params, TEST_EMISSIONS, jr.PRNGKey(0))
    _assert_rbpfiltered_contract(posterior)


def test_slds_optimal_rbpf_posterior_contract():
    params = _make_slds_params()
    posterior = rbpfilter_optimal(RBPF_PARTICLES, params, TEST_EMISSIONS, jr.PRNGKey(0))
    _assert_rbpfiltered_contract(posterior)
