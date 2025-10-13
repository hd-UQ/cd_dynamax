### Code Setup
# Main imports
from itertools import count

# Import jax and utils
from jax import numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Float, Array

# Import jax tree utils
from jax.tree_util import tree_map

# Set the device to CPU
import jax
# To be able to debug
import os
os.environ["JAX_DISABLE_JIT"] = "1"
os.environ["EQX_ON_ERROR"] = "breakpoint"
os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = "100"

#jax.config.update("jax_platform_name", "cpu")

# JAX device check
print("************* Checking JAX device *************")
print('Running on jax device:{}'.format(
        jax.devices()
    )
)
print('Running on jax device platform:{}'.format(
        jax.devices()[0].platform
    )
)
print("***********************************************")

# Diffrax for ODE solving with autodiff
import diffrax as dfx

# Diffeqsolve wrapper
diffeqsolve_settings={}
# Solve a differential equation given a RHS. t0, t1, and initital conditions y0
def diffeqsolve(
    drift,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    reverse: bool = False,
    args = None,
    solver: dfx.AbstractSolver = None,
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
    dt0: float = 0.01,
    tol_vbt: float = 1e-1, # tolerance for virtual brownian tree
    max_steps: int = 1e5,
    diffusion = None,
    key = None,
    **kwargs
) -> jnp.ndarray:

    """
    Choosing solvers and adjoints based on diffrax website's recommendation for training neural ODEs.
        See: https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/
        See: https://docs.kidger.site/diffrax/api/adjoints/

        Note that choosing RecursionCheckpointAdjoint requires usage of reverse-mode auto-differentiation.
        Can use DirectAdjoint for flexible forward-mode + reverse-mode auto-differentiation.

        Defaults are chosen to be decent low-cost options for forward solves and backpropagated gradients.

        If you want high-fidelity solutions (and their gradients), it is recommended 
        - for ODEs: choose a higher-order solver (Tsit5) and an adaptive stepsize controller (PIDController).
        - for SDEs: follow diffrax website advice (definitely can choose dt0 very small with constant stepsize controller).

        Things to pay attention to (that we have incomplete understanding of):
        - checkpoints in RecursiveCheckpointAdjoint: this is used to save memory during backpropagation.
        - max_steps: reducing this can speed things up. But it is also used to set default number of checkpoints.
        ... unclear the optimal way to set these parameters.
    """

    max_steps = int(max_steps)

    # set solver to default if not provided
    if solver is None:
        if diffusion is None:
            solver = dfx.Dopri5()
            # Tsit5 may be another slightly better default method.
        else:
            solver = dfx.Heun()
            # sometimes called the improved Euler method

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
        max_steps=max_steps,
        **kwargs
    ).ys

    return sol

## True model set up
# We generate data from a Lorenz 63 system, from dynamics with the following stochastic differential equations:
#
# \begin{align*}
# \frac{d x}{d t} &= a(y-x) + \sigma w_x(t) \\
# \frac{d y}{d t} &= x(b-z) - y + \sigma w_y(t) \\
# \frac{d z}{d t} &= xy - cz + \sigma w_z(t),
# \end{align*}
#
# With parameters $a=10, b=28, c=8/3$, the system gives rise to chaotic behavior, and we choose $\sigma=1.0$ for diffusion.
#
# To generate data, we numerically approximate random path solutions to this SDE using Heun's method (i.e. improved Euler), as implemented in [Diffrax](https://docs.kidger.site/diffrax/api/solvers/sde_solvers/).
#

## Main settings
state_dim = 3
# Define a drift model
from typing import NamedTuple
class lorenz63_drift(NamedTuple):
    params: Float[Array, "state_dim"]

    def f(self, x, u=None, t=None):
        foo = jnp.array(
            [
                self.params[0] * (x[1] - x[0]),
                self.params[1] * x[0] - x[1] - x[0] * x[2],
                -self.params[2] * x[2] + x[0] * x[1],
            ]
        )
        return foo

# Define the true parameters of the drift function
true_l63_drift_params = jnp.array([10.0, 28.0, 8 / 3])
# And the corresponding Lorenz 63 system
true_drift = {
    "params": lorenz63_drift(
        params=true_l63_drift_params
    ),
}

### Simulate data
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
initial_state = jnp.ones(state_dim)

# Step function to iterate over time steps
def _step(prev_state, args):
    key, t0, t1 = args

    # SDE definition
    def drift(t, y, args):
        return true_drift['params'].f(y, None, t)

    def diffusion(t, y, args):
        Qc_t = jnp.eye(state_dim) 
        L_t = 1.0 * jnp.eye(state_dim)
        Q_sqrt = jnp.linalg.cholesky(Qc_t)
        combined_diffusion = L_t @ Q_sqrt
        return combined_diffusion

    # solve the SDE
    state = diffeqsolve(
        key=key,
        drift=drift,
        diffusion=diffusion,
        t0=t0,
        t1=t1,
        y0=prev_state,
        **diffeqsolve_settings
    )[0]
    
    return state, (state)

############## SUCCESSFUL case 
## Set up seed for simulation
initial_seed = 0
keys = map(jr.PRNGKey, count(start=initial_seed))

# Simulation parameters
T = 1
num_timesteps = 50

# make t_emissions a linspace of time points from [0, T_total]
t_emissions = jnp.linspace(0, T, num_timesteps)[:, None]

print('Simulating with seed={}, T={}, num_timesteps={}'.format(
    initial_seed, T, num_timesteps)
)

# Initial random state
key_init = next(keys)

# Sample the remaining emissions and states
next_keys = jr.split(next(keys), num_timesteps - 1)


# Sample the remaining states via scan
_, (next_states) = lax.scan(
        _step,
        initial_state,
        (next_keys, t_emissions[0:-1,0], t_emissions[1:,0])
)

print('Successful execution with seed={}, T={}, num_timesteps={}'.format(
    initial_seed, T, num_timesteps)
)

############## UNSUCCESSFUL case 
## Set up seed for simulation
initial_seed = 0
keys = map(jr.PRNGKey, count(start=initial_seed))

# Simulation parameters
T = 1
num_timesteps = 20

# make t_emissions a linspace of time points from [0, T_total]
t_emissions = jnp.linspace(0, T, num_timesteps)[:, None]

print('Simulating with seed={}, T={}, num_timesteps={}'.format(
    initial_seed, T, num_timesteps)
)

# Initial random state
key_init = next(keys)

# Sample the remaining emissions and states
next_keys = jr.split(next(keys), num_timesteps - 1)


# Sample the remaining states via scan
_, (next_states) = lax.scan(
        _step,
        initial_state,
        (next_keys, t_emissions[0:-1,0], t_emissions[1:,0])
)

print('Successful execution with seed={}, T={}, num_timesteps={}'.format(
    initial_seed, T, num_timesteps)
)