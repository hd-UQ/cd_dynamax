# numpyro_extension/models/__init__.py
from .builders import build_params
from .builders_linear import build_params_linear

__all__ = [
    "build_params",
    "build_params_linear"
]
