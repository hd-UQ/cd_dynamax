# Diffrax for ODE solving with autodiff
import diffrax as dfx
import jax.numpy as jnp

# Solve a differential equation
#   given a RHS. t0, t1, and initital conditions y0
def diffeqsolve(
    rhs,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = dfx.Dopri5(),
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    dt0: float = 0.01,
) -> jnp.ndarray:
    return dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
    ).ys
