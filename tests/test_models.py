import pytest
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm import models
import jax.random as jr

def test_cdlgssm_basic_init():
    """Test that a model can be created and has correct dimensions."""
    model = models.ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=1)
    assert model.state_dim == 2
    assert model.emission_dim == 1
    assert model.inputs_shape is None


def test_cdlgssm_initialize_defaults():
    """Check that default initialization returns params/props tuples."""
    model = models.ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=1)
    params, props = model.initialize()
    assert hasattr(params, "dynamics")
    assert hasattr(params, "emissions")
    assert props is not None


def test_cdlgssm_sample_path_runs():
    """Run a short forward sample to check integration with diffrax."""
    model = models.ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=1)
    params, _ = model.initialize()
    key = jr.PRNGKey(0)
    states, emissions = model.sample_path(params, key=key, num_timesteps=5)
    assert states.shape[0] == 5
    assert emissions.shape[0] == 5
