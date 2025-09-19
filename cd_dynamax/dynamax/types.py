import jax
from typing import Optional, Union
from typing_extensions import Protocol
from jaxtyping import Array, Float
import jax._src.random as prng


PRNGKey = jax.Array   # simple alias; keys are just jax.Array now

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
