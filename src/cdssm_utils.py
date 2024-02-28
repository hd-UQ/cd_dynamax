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
    
# Aux functions for tests
def try_all_close(x, y, start_tol=-8, end_tol=-4):
    """Try all close with increasing tolerance"""
    # create list of tols 1e-8, 1e-7, 1e-6, ..., 1e1
    tol_list = jnp.array([10 ** i for i in range(start_tol, end_tol+1)])
    for tol in tol_list:
        if jnp.allclose(x, y, atol=tol):
            return True, tol
    return False, tol

def compare(x, x_ref, do_det=False, accept_failure=False):
    allclose, tol = try_all_close(x, x_ref)
    if allclose:
        print(f"\tAllclose passed with atol={tol}.")
    else:
        print(f"\tAllclose FAILED with atol={tol}.")

        # if x is 1d, add batch dim
        # TODO: use ensure_array_has_batch_dim instead
        if x.ndim == 1:
            x = x[:, None]
            x_ref = x_ref[:, None]

        # compute MSE of determinants over time
        if do_det:
            x = vmap(jnp.linalg.det)(x)
            x_ref = vmap(jnp.linalg.det)(x_ref)
            mse = (x - x_ref) ** 2
            rel_mse = mse / (x_ref**2)
        else:
            mse = jnp.mean((x - x_ref) ** 2, axis=1)
            rel_mse = mse / jnp.mean(x_ref ** 2, axis=1)

        print("\tInitial relative MSE: ", rel_mse[0])
        print("\tFinal relative MSE: ", rel_mse[-1])
        print("\tMax relative MSE: ", jnp.max(rel_mse))
        print("\tAverage relative MSE: ", jnp.mean(rel_mse))

        allclose, tol = try_all_close(rel_mse, 0, end_tol=-3)
        if not accept_failure:
            assert allclose, f"Relative MSE allclose FAILED with atol={tol}. UNACCEPTABLE!"
        else:
            print(f"Relative MSE allclose FAILED with atol={tol} but accepting this failure.")

    pass
