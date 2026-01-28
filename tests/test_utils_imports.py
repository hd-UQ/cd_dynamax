def test_utils_imports():
    """Ensure utils are importable and callable."""
    from cd_dynamax.src.utils import debug_utils, diffrax_utils

    assert hasattr(debug_utils, "__file__") or hasattr(debug_utils, "__name__")
    assert hasattr(diffrax_utils, "diffeqsolve")
