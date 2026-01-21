# JAX imports
import jax.numpy as jnp

# Typing imports
from typing import NamedTuple, Union
from jaxtyping import Array, Float, Integer

# Imports from dynamax
from cd_dynamax.dynamax.parameters import ParameterProperties

# Imports from the cd-dynamax codebase
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import (
    _get_params,
)
from cd_dynamax import adjust_rhs

### Learnable physics-based models: Lorenz63, Lorenz96, FitzHugh-Nagumo, Van der Pol, Rossler


# Learnable Lorenz63 model
class LearnableLorenz63_Drift(NamedTuple):
    r"""Learnable Lorenz63 dynamical system drift model.

    The Lorenz63 system is defined by the following ODEs:
        $dx/dt = sigma * (y - x)$
        $dy/dt = rho * x - y - x * z$
        $dz/dt = -beta * z + x * y$
    where sigma, rho, and beta are parameters of the system.
    """

    # Learnable parameters
    sigma: Union[Float[Array, "1"], ParameterProperties]
    rho: Union[Float[Array, "1"], ParameterProperties]
    beta: Union[Float[Array, "1"], ParameterProperties]

    # Function to compute the drift
    def f(self, x, u=None, t=None):
        foo = jnp.array(
            [
                self.sigma * (x[1] - x[0]),
                self.rho * x[0] - x[1] - x[0] * x[2],
                -self.beta * x[2] + x[0] * x[1],
            ]
        )

        # adjust the rhs so that
        # 1) the state stays within the bounds of -100 and 100
        # 2) the derivative stays within the bounds of -1000 and 1000
        rhs = adjust_rhs(
            x,
            foo,
            lower_bound=-100,
            upper_bound=100,
            lower_bound_derivative=-1000,
            upper_bound_derivative=1000,
        )

        return rhs

    def get_params(self):
        return _get_params(self)


# Learnable Lorenz96 model
class LearnableLorenz96_Drift(NamedTuple):
    r"""Learnable Lorenz96 dynamical system drift model.

    The Lorenz96 system is defined by the following ODEs:
        $dx_k/dt = (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F$
    where F is a parameter of the system.
    """

    # Learnable parameter
    F: Union[Float[Array, "1"], ParameterProperties]

    # Function to compute the drift
    def f(self, x, u=None, t=None):
        foo = (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + self.F

        # adjust the rhs so that
        # 1) the state stays within the bounds of -100 and 100
        # 2) the derivative stays within the bounds of -1000 and 1000
        rhs = adjust_rhs(
            x,
            foo,
            lower_bound=-100,
            upper_bound=100,
            lower_bound_derivative=-1000,
            upper_bound_derivative=1000,
        )

        return rhs

    def get_params(self):
        return _get_params(self)


# Learnable Lorenz96 multiscale model
class LearnableLorenz96MultiScale_Drift(NamedTuple):
    r"""Learnable Lorenz96 multiscale dynamical system drift model.
    The Lorenz96 multiscale system is defined by the following ODEs:
        $dX_k/dt = f_k(X) + h_x * Y_bar$
        $dY_{k,j}/dt = (1/eps) * r_j(X, Y)$
    where:
        $f_k(X) = -X_{k-1} * (X_{k-2} - X_{k+1}) - X_k + F$
        $r_j(X, Y) = -Y_{k,j+1} * (Y_{k,j+2} - Y_{k,j-1}) - Y_{k,j} + h_y * X_k$
    and Y_bar is the average of the fast variables Y over j.
    """

    # Learnable parameters
    F: Union[Float[Array, "1"], ParameterProperties]
    hx: Union[Float[Array, "1"], ParameterProperties]
    hy: Union[Float[Array, "1"], ParameterProperties]
    eps: Union[Float[Array, "1"], ParameterProperties]
    K: Union[Integer[Array, "1"], ParameterProperties]
    J: Union[Integer[Array, "1"], ParameterProperties]

    """    params are (F, hx, hy, eps, K, J)
    F: Forcing term for the slow system
    hx: Coupling parameter between slow and fast systems
    hy: Coupling parameter between fast variables and X
    eps: Scale-separation parameter
    K: Number of slow variables
    J: Number of fast variables per slow variable
    """

    # TODO: doublcheck the implementation of the multiscale system
    def f(self, x, u=None, t=None):
        # Reshape x to get X and Y
        X = x[: self.K]
        Y = x[self.K :].reshape(self.K, self.J)

        """
        Computes the right-hand side of the multiscale system.
        
        Args:
            X (jnp.array): Shape (K,), slow variables.
            Y (jnp.array): Shape (K, J), fast variables.
            eps (float): Scale-separation parameter.
            hx (float): Coupling parameter between slow and fast systems.
            hy (float): Coupling parameter between fast variables and X.
            F (float): Forcing term.
        
        """

        # Compute the averaged fast variable Y_bar
        Y_bar = jnp.mean(Y, axis=1)

        # Compute f_k(X)
        X_rolled_m1 = jnp.roll(X, -1)  # X_{k+1}
        X_rolled_m2 = jnp.roll(X, -2)  # X_{k+2}
        X_rolled_p1 = jnp.roll(X, 1)  # X_{k-1}

        fX = -X_rolled_p1 * (X_rolled_m2 - X_rolled_m1) - X + self.F
        dX = fX + self.hx * Y_bar

        # Compute r_j(X, Y)
        Y_rolled_m1 = jnp.roll(Y, -1, axis=1)  # Y_{k,j+1}
        Y_rolled_m2 = jnp.roll(Y, -2, axis=1)  # Y_{k,j+2}
        Y_rolled_p1 = jnp.roll(Y, 1, axis=1)  # Y_{k,j-1}

        rY = -Y_rolled_m1 * (Y_rolled_m2 - Y_rolled_p1) - Y + self.hy * X[:, None]
        dY = (1 / self.eps) * rY

        # Concatenate dX and dY
        rhs = jnp.concatenate((dX, dY.flatten()))

        # adjust the rhs so that
        # 1) the state stays within the bounds of -100 and 100
        # 2) the derivative stays within the bounds of -1e8 and 1e8
        rhs = adjust_rhs(
            x,
            rhs,
            lower_bound=-100,
            upper_bound=100,
            lower_bound_derivative=-1e8,
            upper_bound_derivative=1e8,
        )

        return rhs

    def get_params(self):
        return _get_params(self)


# Learnable FitzHugh-Nagumo model
class LearnableFitzHughNagumo(NamedTuple):
    r"""Learnable FitzHugh-Nagumo dynamical system drift model.

    The FitzHugh-Nagumo system is defined by the following ODEs:
        $dv/dt = v - v^3/3 - w + RI_{ext}$
        $dw/dt = (1/tau) * (v + a - b * w)$
    where a, b, tau, and RI_{ext} are parameters of the system.
    """

    # Learnable parameters
    a: Union[Float[Array, "1"], ParameterProperties]
    b: Union[Float[Array, "1"], ParameterProperties]
    tau: Union[Float[Array, "1"], ParameterProperties]
    RIext: Union[Float[Array, "1"], ParameterProperties]

    # Function to compute the drift
    def f(self, x, u=None, t=None):
        foo = jnp.array(
            [
                x[0] - x[0] ** 3 / 3 - x[1] + self.RIext,
                (1 / self.tau) * (x[0] + self.a - self.b * x[1]),
            ]
        )

        # adjust the rhs so that
        # 1) the state stays within the bounds of -10 and 10
        # 2) the derivative stays within the bounds of -500 and 500
        rhs = adjust_rhs(
            x,
            foo,
            lower_bound=-10,
            upper_bound=10,
            lower_bound_derivative=-500,
            upper_bound_derivative=500,
        )

        return rhs

    def get_params(self):
        return _get_params(self)


# Learnable Van der Pol model
class LearnableVanDerPol(NamedTuple):
    r"""Learnable Van der Pol dynamical system drift model.

    The Van der Pol system is defined by the following ODEs:
        $dx1/dt = x2$
        $dx2/dt = mu * (1 - x1^2) * x2 - x1$
    where mu is a parameter of the system.
    """

    # Learnable parameter
    mu: Union[Float[Array, "1"], ParameterProperties]

    # Function to compute the drift
    def f(self, x, u=None, t=None):
        foo = jnp.array(
            [
                x[1],
                self.mu * (1 - x[0] ** 2) * x[1] - x[0],
            ]
        )

        # adjust the rhs so that
        # 1) the state stays within the bounds of -10 and 10
        # 2) the derivative for x1 stays within the bounds of -10 and 10
        # 3) the derivative for x2 stays within the bounds of -mu*1000 and mu*1000
        rhs = adjust_rhs(
            x,
            foo,
            lower_bound=-10,
            upper_bound=10,
            lower_bound_derivative=-1000,
            upper_bound_derivative=1000,
        )

        return rhs

    def get_params(self):
        return _get_params(self)


# Learnable Rossler model
class LearnableRossler(NamedTuple):
    r"""Learnable Rossler dynamical system drift model.

    The Rossler system is defined by the following ODEs:
        $dx/dt = -y - z$
        $dy/dt = x + a * y$
        $dz/dt = b + z * (x - c)$
    where a, b, and c are parameters of the system.
    """

    # Learnable parameters
    a: Union[Float[Array, "1"], ParameterProperties]
    b: Union[Float[Array, "1"], ParameterProperties]
    c: Union[Float[Array, "1"], ParameterProperties]

    # Function to compute the drift
    def f(self, x, u=None, t=None):
        foo = jnp.array(
            [
                -x[1] - x[2],
                x[0] + self.a * x[1],
                self.b + x[2] * (x[0] - self.c),
            ]
        )

        # adjust the rhs so that
        # 1) the state stays within the bounds of -100 and 100
        # 2) the derivative stays within the bounds of -5000 and 5000
        rhs = adjust_rhs(
            x,
            foo,
            lower_bound=-100,
            upper_bound=100,
            lower_bound_derivative=-5000,
            upper_bound_derivative=5000,
        )

        return rhs

    def get_params(self):
        return _get_params(self)
