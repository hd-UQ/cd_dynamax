# Basic jax imports
import jax.random as jr
import jax.numpy as jnp

# cd-dynamax models
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.models import (
    ContDiscreteLinearGaussianSSM,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import (
    ContDiscreteNonlinearGaussianSSM,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
    ContDiscreteNonlinearSSM,
)
from cd_dynamax.dynamax.slds import SLDS
from cd_dynamax.dynamax.slds.inference import (
    DiscreteParamsSLDS,
    LGParamsSLDS,
    ParamsSLDS,
)

# Useful auxiliary function to sample from model with num_timesteps and t_emissions
# and check states and emissions match
from cd_dynamax.src.utils.test_utils import compare


def check_cddynamax_model_sample_match(model, params, key, T):
    # Sample a path, using num_timesteps and check shapes
    states, emissions = model.sample_path(params, key=key, num_timesteps=T)
    assert states.shape == (T, model.state_dim)
    assert emissions.shape == (T, model.emission_dim)

    # Sample a path, using equally spaced t_emissions, and check shapes
    t_emissions = jnp.arange(T)[:, None]
    states_t, emissions_t = model.sample_path(
        params, key=key, num_timesteps=T, t_emissions=t_emissions
    )
    assert states_t.shape == (T, model.state_dim)
    assert emissions_t.shape == (T, model.emission_dim)

    # Check that the states and emissions are the same for the two sampling methods
    compare(states, states_t)
    compare(emissions, emissions_t)


# CD-LGSSM basic tests
def test_cdlgssm_basic_init():
    """Test that a model can be created and has correct dimensions."""
    model = ContDiscreteLinearGaussianSSM(state_dim=3, emission_dim=2)
    assert model.state_dim == 3
    assert model.emission_dim == 2
    assert model.inputs_shape is None


def test_cdlgssm_initialize_defaults():
    """Check that default initialization returns params/props tuples."""
    model = ContDiscreteLinearGaussianSSM(state_dim=3, emission_dim=2)
    params, props = model.initialize()

    # CD-LGSSM parameters
    # Imports
    from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import (
        ParamsCDLGSSM,
        ParamsCDLGSSMDynamics,
    )
    from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
        ParamsLGSSMInitial,
        ParamsLGSSMEmissions,
    )

    # Checks
    assert isinstance(params, ParamsCDLGSSM)
    # Initial should be a ParamsLGSSMInitial
    assert hasattr(params, "initial")
    assert isinstance(params.initial, ParamsLGSSMInitial)
    # Dynamics should be a ParamsCDLGSSMDynamics
    assert hasattr(params, "dynamics")
    assert isinstance(params.dynamics, ParamsCDLGSSMDynamics)
    # Emissions should be a ParamsLGSSMEmissions
    assert hasattr(params, "emissions")
    assert isinstance(params.emissions, ParamsLGSSMEmissions)
    assert props is not None


def test_cdlgssm_sample_path():
    """Run a short forward sample to check integration with diffrax."""
    model = ContDiscreteLinearGaussianSSM(state_dim=3, emission_dim=2)
    params, _ = model.initialize()

    # Test model sampling
    check_cddynamax_model_sample_match(
        model=model, params=params, key=jr.PRNGKey(0), T=10
    )


# CD-NLGSSM basic tests
def test_cdnlgssm_basic_init():
    """Test that a model can be created and has correct dimensions."""
    model = ContDiscreteNonlinearGaussianSSM(state_dim=3, emission_dim=2)
    assert model.state_dim == 3
    assert model.emission_dim == 2
    assert model.inputs_shape is None


def test_cdnlgssm_initialize_defaults():
    """Check that default initialization returns params/props tuples."""
    model = ContDiscreteNonlinearGaussianSSM(state_dim=3, emission_dim=2)
    params, props = model.initialize()

    # CD-NLGSSM parameters
    # Imports
    from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import (
        ParamsCDNLGSSM,
        ParamsCDNLGSSMDynamics,
        ParamsCDNLGSSMEmissions,
    )
    from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
        ParamsLGSSMInitial,
    )

    # Checks
    assert isinstance(params, ParamsCDNLGSSM)
    # Initial should be a ParamsLGSSMInitial
    assert hasattr(params, "initial")
    assert isinstance(params.initial, ParamsLGSSMInitial)
    # Dynamics should be a ParamsCDNLGSSMDynamics
    assert hasattr(params, "dynamics")
    assert isinstance(params.dynamics, ParamsCDNLGSSMDynamics)
    # Emissions should be a ParamsCDNLGSSMEmissions
    assert hasattr(params, "emissions")
    assert isinstance(params.emissions, ParamsCDNLGSSMEmissions)
    assert props is not None


def test_cdnlgssm_sample_path():
    """Run a short forward sample to check integration with diffrax."""
    model = ContDiscreteNonlinearGaussianSSM(state_dim=3, emission_dim=2)
    params, _ = model.initialize()

    # Test model sampling
    check_cddynamax_model_sample_match(
        model=model, params=params, key=jr.PRNGKey(0), T=10
    )


# CD-NLSSM basic tests
def test_cdnlssm_basic_init():
    """Test that a model can be created and has correct dimensions."""
    model = ContDiscreteNonlinearSSM(state_dim=3, emission_dim=2)
    assert model.state_dim == 3
    assert model.emission_dim == 2
    assert model.inputs_shape is None


def test_cdnlssm_initialize_defaults():
    """Check that default initialization returns params/props tuples."""
    model = ContDiscreteNonlinearSSM(state_dim=3, emission_dim=2)
    params, props = model.initialize()

    # CD-NLSSM parameters
    # Imports
    from cd_dynamax.src.continuous_discrete_nonlinear_ssm.cdnlssm_utils import (
        ParamsCDNLSSM,
        ParamsCDNLSSMInitial,
        ParamsCDNLSSMDynamics,
        ParamsCDNLSSMEmissions,
    )

    # Checks
    assert isinstance(params, ParamsCDNLSSM)
    # Initial should be a ParamsCDNLSSMInitial
    assert hasattr(params, "initial")
    assert isinstance(params.initial, ParamsCDNLSSMInitial)
    # Dynamics should be a ParamsCDNLSSMDynamics
    assert hasattr(params, "dynamics")
    assert isinstance(params.dynamics, ParamsCDNLSSMDynamics)
    # Emissions should be a ParamsCDNLSSMEmissions
    assert hasattr(params, "emissions")
    assert isinstance(params.emissions, ParamsCDNLSSMEmissions)
    assert props is not None


def test_cdnlssm_sample_path():
    """Run a short forward sample to check integration with diffrax."""
    model = ContDiscreteNonlinearSSM(state_dim=3, emission_dim=2)
    params, _ = model.initialize()

    # Test model sampling
    check_cddynamax_model_sample_match(
        model=model, params=params, key=jr.PRNGKey(0), T=10
    )


def make_slds_params(num_states, state_dim, emission_dim):
    transition_matrix = jnp.ones((num_states, num_states)) / num_states
    return ParamsSLDS(
        discrete=DiscreteParamsSLDS(
            initial_distribution=jnp.ones(num_states) / num_states,
            transition_matrix=transition_matrix,
            proposal_transition_matrix=transition_matrix,
        ),
        linear_gaussian=LGParamsSLDS(
            initial_mean=jnp.zeros((num_states, state_dim)),
            initial_cov=jnp.tile(jnp.eye(state_dim)[None, :, :], (num_states, 1, 1)),
            dynamics_weights=jnp.tile(
                (0.9 * jnp.eye(state_dim))[None, :, :], (num_states, 1, 1)
            ),
            dynamics_cov=jnp.tile(
                (0.1 * jnp.eye(state_dim))[None, :, :], (num_states, 1, 1)
            ),
            dynamics_bias=jnp.zeros((num_states, state_dim)),
            dynamics_input_weights=jnp.zeros((num_states, state_dim, 1)),
            emission_weights=jnp.ones((num_states, emission_dim, state_dim)),
            emission_cov=jnp.tile(
                (0.1 * jnp.eye(emission_dim))[None, :, :], (num_states, 1, 1)
            ),
            emission_bias=jnp.zeros((num_states, emission_dim)),
            emission_input_weights=jnp.zeros((num_states, emission_dim, 1)),
            initialized=True,
        ),
    )


def test_slds_basic_init():
    """Test that an SLDS can be created and has correct dimensions."""
    model = SLDS(num_states=2, state_dim=3, emission_dim=2)
    assert model.num_states == 2
    assert model.state_dim == 3
    assert model.emission_dim == 2
    assert model.input_dim == 1
    assert model.inputs_shape == (1,)


def test_slds_sample():
    """Run a short SLDS forward sample and check output shapes."""
    model = SLDS(num_states=2, state_dim=3, emission_dim=2)
    params = make_slds_params(
        num_states=model.num_states,
        state_dim=model.state_dim,
        emission_dim=model.emission_dim,
    )
    dstates, cstates, emissions = model.sample(
        params, key=jr.PRNGKey(0), num_timesteps=10, inputs=jnp.zeros((10, 1))
    )

    assert dstates.shape == (10,)
    assert cstates.shape == (10, model.state_dim)
    assert emissions.shape == (10, model.emission_dim)
