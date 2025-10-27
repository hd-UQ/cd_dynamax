# tests/test_imports.py

def test_imports():
    """Basic smoke test to ensure core modules and deps are importable."""

    import cd_dynamax.dynamax as dynamax
    import cd_dynamax.src as src

    import jax
    import optax
    import diffrax

    assert hasattr(dynamax, "__file__")
    assert hasattr(src, "__file__")

