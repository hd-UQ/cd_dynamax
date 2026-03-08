import jax.random as jr

# CD-LGSSM basic tests
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.models import ContDiscreteLinearGaussianSSM
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
        ParamsCDLGSSMDynamics
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
    key = jr.PRNGKey(0)
    T=10
    states, emissions = model.sample_path(params, key=key, num_timesteps=T)
    assert states.shape == (T, model.state_dim)
    assert emissions.shape == (T, model.emission_dim)

# CD-NLGSSM basic tests
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import ContDiscreteNonlinearGaussianSSM
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
        ParamsCDNLGSSMEmissions
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
    key = jr.PRNGKey(0)
    T=10
    states, emissions = model.sample_path(params, key=key, num_timesteps=T)
    assert states.shape == (T, model.state_dim)
    assert emissions.shape == (T, model.emission_dim)

# CD-NLSSM basic tests
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import ContDiscreteNonlinearSSM
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
        ParamsCDNLSSMEmissions
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
    key = jr.PRNGKey(0)
    T=10
    states, emissions = model.sample_path(params, key=key, num_timesteps=T)
    assert states.shape == (T, model.state_dim)
    assert emissions.shape == (T, model.emission_dim)

