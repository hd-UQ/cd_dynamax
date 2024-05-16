# Diffrax for ODE solving with autodiff
import diffrax as dfx
import jax.numpy as jnp
import jax.debug as jdb
from jax import random as jr
from jax import vmap, lax
from pdb import set_trace as bp
import jax.debug as jdb

DEBUG = False

def reverse_rhs(rhs, t1, ref_var):
    if rhs is None:
        return None

    if isinstance(ref_var, tuple):
        def rev_rhs(s, y, args):
            foo = rhs(t1 - s, y, args)
            return tuple(-f for f in foo)
    else:
        def rev_rhs(s, y, args):
            return -rhs(t1 - s, y, args)

    return rev_rhs

def breakpoint_if_nan(x):
    is_nan = jnp.isnan(x).any()

    def true_fn(x):
        jdb.breakpoint()

    def false_fn(x):
        pass

    lax.cond(is_nan, true_fn, false_fn, x)

# Solve a differential equation
#   given a RHS. t0, t1, and initital conditions y0
def diffeqsolve(
    drift,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    reverse: bool = False,
    args = None,
    solver: dfx.AbstractSolver = None,
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    adjoint: dfx.AbstractAdjoint = dfx.DirectAdjoint(),
    dt0: float = 0.01,
    tol_vbt: float = 1e-1, # tolerance for virtual brownian tree
    diffusion = None,
    key = None,
    debug = DEBUG,
    **kwargs
) -> jnp.ndarray:

    if debug:
        # run hand-written Euler and/or Euler-Maruyama using a for loop with fixed step size dt0
        N = 6 # if this is too small, then the error will be too large and covariances can be very non-SPD.
        dt = (t1 - t0) / N
        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, N)

        for i in range(N):
            drift_i = drift(t0 + i * dt, y0, None)
            if isinstance(y0, tuple):
                # breakpoint_if_nan(y0[0])
                # breakpoint_if_nan(y0[1])
                # if jnp.any(jnp.isnan(y0[0])):
                #     jdb.print("NaN detected in y0[0] at step {}", i)
                # if jnp.any(jnp.isnan(y0[1])):
                #     jdb.print("NaN detected in y0[1] at step {}", i)
                # If y0 and drift_i are tuples, update each component
                y0 = tuple(y0_component + dt * drift_component for y0_component, drift_component in zip(y0, drift_i))
                if diffusion is not None:
                    diff = diffusion(t0 + i * dt, y0, None)
                    rnd = tuple(jr.normal(key=keys[i], shape=y0_component.shape) for y0_component in y0)
                    y0 = tuple(y0_component + jnp.sqrt(dt) * diff_component * rnd_component
                            for y0_component, diff_component, rnd_component in zip(y0, diff, rnd))
            else:
                # If y0 and drift_i are vectors, update directly
                y0 = y0 + dt * drift_i
                if diffusion is not None:
                    diff = diffusion(t0 + i * dt, y0, None)
                    rnd = jr.normal(key=keys[i], shape=y0.shape)
                    y0 = y0 + jnp.sqrt(dt) * diff * rnd

        # return the final state y0 with an additional first dimension
        if isinstance(y0, tuple):
            # Reshape to match the expected output of the solver
            return tuple(jnp.expand_dims(y0_component, axis=0) for y0_component in y0)
        else:
            return y0

    # set solver to default if not provided
    if solver is None:
        if diffusion is None:
            solver = dfx.Dopri5()
        else:
            solver = dfx.Heun()

    # allow for reverse-time integration
    # if t1 < t0, we assume that initial condition y0 is at t1
    if reverse:
        t0_new = 0
        t1_new = t1 - t0
        drift_new = reverse_rhs(drift, t1, y0)
        diffusion_new = reverse_rhs(diffusion, t1, y0)
    else:
        t0_new = t0
        t1_new = t1
        drift_new = drift
        diffusion_new = diffusion

    # set DE terms
    if diffusion_new is None:
        terms = dfx.ODETerm(drift_new)
    else:
        bm = dfx.VirtualBrownianTree(t0=t0_new, t1=t1_new, tol=tol_vbt, shape=y0.shape, key=key)
        terms = dfx.MultiTerm(dfx.ODETerm(drift_new), dfx.ControlTerm(diffusion_new, bm))

    # return a specific solver
    sol = dfx.diffeqsolve(
        terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=t0_new,
        t1=t1_new,
        y0=y0,
        args=args,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
        **kwargs
    ).ys

    return sol
