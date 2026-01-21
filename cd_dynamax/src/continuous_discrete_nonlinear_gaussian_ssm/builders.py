import equinox as eqx
import jax.numpy as jnp
import inspect


# -------------------------
# Callable + Constant wrappers
# -------------------------
class ConstantFunction(eqx.Module):
    constant: jnp.ndarray

    def f(self, x=None, u=None, t=None):
        return self.constant


class CallableFunction(eqx.Module):
    func: callable

    def f(self, x=None, u=None, t=None):
        sig = inspect.signature(self.func)
        kwargs = {}
        if "x" in sig.parameters:
            kwargs["x"] = x
        if "u" in sig.parameters:
            kwargs["u"] = u
        if "t" in sig.parameters:
            kwargs["t"] = t
        return self.func(**kwargs)


def wrap_function_or_constant(obj, expected_shape=None) -> eqx.Module:
    """
    Wraps a user-provided constant or callable into an eqx.Module with `.f(x, u, t)`.
    - If it's callable, only passes (x, u, t) args the function accepts
    - If it's a constant, returns a ConstantFunction that ignores inputs
    """
    if callable(obj):
        sig = inspect.signature(obj)
        extra = set(sig.parameters) - {"x", "u", "t"}
        if extra:
            raise ValueError(
                f"Callable has unexpected arguments: {extra}. Only (x, u, t) allowed."
            )
        return CallableFunction(func=obj)

    arr = jnp.asarray(obj)
    if expected_shape is not None:
        arr = jnp.broadcast_to(arr, expected_shape)
    return ConstantFunction(constant=arr)


# -------------------------
# Param containers
# -------------------------
class CDNLGSSMInitialParams(eqx.Module):
    mean: eqx.Module
    cov: eqx.Module


class CDNLGSSMDynamicsParams(eqx.Module):
    drift: eqx.Module
    diffusion_cov: eqx.Module
    diffusion_coefficient: eqx.Module
    approx_order: float = 1.0


class CDNLGSSMEmissionsParams(eqx.Module):
    emission_function: eqx.Module
    emission_cov: eqx.Module


class CDNLGSSMParams(eqx.Module):
    initial: CDNLGSSMInitialParams
    dynamics: CDNLGSSMDynamicsParams
    emissions: CDNLGSSMEmissionsParams


# -------------------------
# Param construction
# -------------------------


def _default(value, default):
    return value if value is not None else default


def build_params(
    state_dim,
    emission_dim,
    drift,
    initial_mean=None,
    initial_cov=None,
    emission_function=None,
    emission_cov=None,
    diffusion_coeff=None,
    diffusion_cov=None,
):
    """Build parameters for a CDNLGSSM model.
    Parameters
    ----------
    state_dim : int
        Dimension of the state space.
    emission_dim : int
        Dimension of the emission space.
    drift : array-like or callable
        Drift function for the state dynamics.
        Should only accept (x, u, t) or a subset of these (can also return a constant).
    initial_mean : array-like or callable, optional (default: zeros)
    initial_cov : array-like or callable, optional (default: identity matrix)
    emission_function : array-like or callable, optional (default: identity function)
        Function mapping state to emissions.
        Should only accept (x, u, t) or a subset of these.
    emission_cov : array-like or callable, optional (default: identity matrix)
        Covariance of the emissions.
        Should only accept (x, u, t) or a subset of these.
    diffusion_coeff : array-like or callable, optional (default: 1e-2 * ones(state_dim))
        Diffusion coefficient for the state dynamics.
        Should only accept (x, u, t) or a subset of these.
    diffusion_cov : array-like or callable, optional (default: identity matrix)
        Covariance of the diffusion process.
        Should only accept (x, u, t) or a subset of these.
    Returns
    -------
    CDNLGSSMParams
        An instance of CDNLGSSMParams containing the wrapped parameters.
    Raises
    ------
    ValueError
        If any of the provided callable parameters have unexpected arguments.
    Notes
    -----
    The callable parameters should only accept (x, u, t) or a subset of these.
    If they accept additional arguments, a ValueError will be raised.
    """

    return CDNLGSSMParams(
        initial=CDNLGSSMInitialParams(
            mean=wrap_function_or_constant(
                _default(initial_mean, jnp.zeros(state_dim))
            ),
            cov=wrap_function_or_constant(_default(initial_cov, jnp.eye(state_dim))),
        ),
        dynamics=CDNLGSSMDynamicsParams(
            drift=wrap_function_or_constant(drift),
            diffusion_cov=wrap_function_or_constant(
                _default(diffusion_cov, jnp.eye(state_dim))
            ),
            diffusion_coefficient=wrap_function_or_constant(
                _default(diffusion_coeff, 1e-2 * jnp.ones(state_dim))
            ),
        ),
        emissions=CDNLGSSMEmissionsParams(
            emission_function=wrap_function_or_constant(
                _default(emission_function, lambda x: x[:emission_dim])
            ),
            emission_cov=wrap_function_or_constant(
                _default(emission_cov, jnp.eye(emission_dim))
            ),
        ),
    )
