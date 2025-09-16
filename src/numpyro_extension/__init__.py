# numpyro_extension/__init__.py

from .models import build_params, build_params_linear
from .plotting import triangle_plot

__all__ = [
    "build_params",
    "build_params_linear",
    "triangle_plot",
]
