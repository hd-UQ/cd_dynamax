# tests/test_imports.py

def test_imports():
    """Basic smoke test to ensure core modules and deps are importable."""

    import cd_dynamax.dynamax as dynamax
    import cd_dynamax.src as src

    import jax
    import numpyro
    import optax
    import diffrax

    assert hasattr(dynamax, "__file__")
    assert hasattr(src, "__file__")

def test_create_model():
    from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm import models
    model = models.ContDiscreteLinearGaussianSSM(state_dim=2, emission_dim=1)
    assert model.state_dim == 2
    assert model.emission_dim == 1
