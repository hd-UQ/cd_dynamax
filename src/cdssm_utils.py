# Diffrax for ODE solving with autodiff
import diffrax as dfx
import jax.numpy as jnp

# Solve a differential equation
#   given a RHS. t0, t1, and initital conditions y0
def diffeqsolve(
    drift,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = None,
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    adjoint: dfx.AbstractAdjoint = dfx.DirectAdjoint(),
    dt0: float = 0.01,
    diffusion = None,
    key = None,
) -> jnp.ndarray:

    # set solver to default if not provided
    if solver is None:
        if diffusion is None:
            solver = dfx.Dopri5()
        else:
            solver = dfx.Heun()

    # set DE terms
    if diffusion is None:
        terms = dfx.ODETerm(drift)
    else:
        bm = dfx.UnsafeBrownianPath(shape=y0.shape, key=key)
        terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))

    # return a specific solver
    sol = dfx.diffeqsolve(
        terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
    ).ys

    return sol