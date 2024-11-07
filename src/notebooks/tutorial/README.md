This tutorial performs filtering and parameter identification for data coming from the Lorenz 63 system.

In all cases, we assume:

- continuous underlying dynamics governed by an SDE
- noisy and irregulary-sampled observations of the system.

## The model: Lorenz 63

We generate data from a Lorenz 63 system, from dynamics with the following stochastic differential equations:

$$
\frac{d x}{d t} &= a(y-x) + \sigma w_x(t) \\
\frac{d y}{d t} &= x(b-z) - y + \sigma w_y(t) \\
\frac{d z}{d t} &= xy - cz + \sigma w_z(t),
\end{align*}
$$

With parameters $a=10, b=28, c=8/3$, the system gives rise to chaotic behavior, and we choose $\sigma=1.0$ for diffusion.

To generate data, we numerically approximate random path solutions to this SDE using Heun's method (i.e. improved Euler), as implemented in [Diffrax](https://docs.kidger.site/diffrax/api/solvers/sde_solvers/).

We assume the observation model is
$$
y(t) &= H x(t) + r(t) \\
r(t) &\sim N(0,R),
$$
where we choose $R=I$. 

Namely, we impose partial observability with H=[1, 0, 0], with noisy observations, sampled at irregular time intervals.

## Tutorials

This directory contains notebooks with different cases of interest

### Filtering partially-observed, noisy and irregularly-sampled observations

In the [state estimation and forecasting notebook](./cdnlgssm_filtering.ipynb) we showcase how to learn partially observed dynamics, given a cd-nlgssm model.

- We show how to use cd-dynamax to estimate the latent state of a continuous-discrete (non-linear) Gaussian dynamical system, by running the following filtering alternatives:
    - The Extended Kalman Filter (EKF)
    - The Ensemble Kalman Filter (EnKF)
    - The Unscented Kalman filter (UKF)

### Drift estimation

In all the following tutorials, we assume that **the drift function** of a latent SDE is **the only unknown object** in the dynamics + observation model.

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
