This tutorial performs filtering and parameter identification for data coming from the Lorenz 63 system.
In all cases, we assume:
- continuous underlying dynamics governed by an SDE
- noisy and irregulary-sampled observations of the system.

We generate data from a Lorenz 63 system, from dynamics with the following stochastic differential equations:

\begin{align*}
\frac{d x}{d t} &= a(y-x) + \sigma w_x(t) \\
\frac{d y}{d t} &= x(b-z) - y + \sigma w_y(t) \\
\frac{d z}{d t} &= xy - cz + \sigma w_z(t),
\end{align*}

With parameters $a=10, b=28, c=8/3$, the system gives rise to chaotic behavior, and we choose $\sigma=1.0$ for diffusion.

To generate data, we numerically approximate random path solutions to this SDE using Heun's method (i.e. improved Euler), as implemented in [Diffrax](https://docs.kidger.site/diffrax/api/solvers/sde_solvers/).


We assume the observation model is
\begin{align*}
y(t) &= H x(t) + r(t) \\
r(t) &\sim N(0,R),
\end{align*}
where we choose $R=I$. 

Namely, we impose partial observability with H=[1, 0, 0], with noisy observations, sampled at irregular time intervals.

This directory contains notebooks with the following content:

1. [Filtering demonstration](./cdnlgssm_filtering.ipynb)

2. Drift estimation
- In all cases, we assume that the drift is the only unknown object in the dynamics + observation model.
- We will use the following methods to estimate the drift:
    - Maximum likelihood estimation (via SGD)
    - Bayesian estimation of posterior distribution (via NUTS)
- We will study these problems under different observation scenarios:
    - Full observation of system coordinates ($x_1, x_2, x_3$)
    - Partial observation system coordinates (only $x_1$)
- We will also study different parametric assumptions on the drift:
    - Known parametric form with unknown parameters ($\sigma, \rho, \beta$)
    - Unknown form represented by a neural network (with unknown parameters)
    - A linear combination of hand-chosen dictionary functions (with unknown coefficients) + appropriate sparsifying prior over these coefficients (TODO)
    - A linear combination of basis functions chosen to be a truncated Karhunen-Loeve expansion of a Gaussian process (with unknown coefficients) + appropriate prior over these coefficients (TODO)
- Currently, there are 8 notebooks corresponding to: ((SGD, NUTS) x (full observation, partial observation) x (mechanistic parameters, neural network parameters)).
