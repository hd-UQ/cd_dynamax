# Diffrax for ODE solving with autodiff
import diffrax as dfx
import jax.numpy as jnp
from jax import vmap

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
    diffusion = None,
    key = None,
    **kwargs
) -> jnp.ndarray:

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
        bm = dfx.UnsafeBrownianPath(shape=y0.shape, key=key)
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

