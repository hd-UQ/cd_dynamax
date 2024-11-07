# Tutorials: cd-dynamax for filtering and drift estimation

This tutorial performs filtering and parameter identification for data drawn from a Lorenz 63 system.

In all cases, we assume:

- continuous underlying dynamics, governed by the Lorenz 63 SDE,
- with noisy and irregulary-sampled observations of the system.

When needed, we will use the following parameter estimation approaches:

- Maximum likelihood estimation:
    - we show how to use cd-dynamax and Stochastic Gradient Descent for computing the Maximum Likelihood Estimate for the parameters of a continuous-discrete (non-linear) dynamical system
    
- Bayesian estimation of posterior distribution:
    - we show how to use [blackjax](https://github.com/blackjax-devs/blackjax)'s MCMC implementations (HMC or NUTS) to sample from the parameter posterior $p(\theta|Y_K)$ of unknown parameters of a continuous-discrete (non-linear) dynamical system.

## The model: Lorenz 63

We generate data from a Lorenz 63 system, from dynamics with the following stochastic differential equations:

$$
\frac{d x}{d t} = a(y-x) + \sigma w_x(t)
$$
$$
\frac{d y}{d t} = x(b-z) - y + \sigma w_y(t)
$$
$$
\frac{d z}{d t} = xy - cz + \sigma w_z(t)
$$.

With parameters $a=10, b=28, c=8/3$, the system gives rise to chaotic behavior, and we choose $\sigma=1.0$ for diffusion.

- To generate data, we numerically approximate random path solutions to this SDE using Heun's method (i.e. improved Euler), as implemented in [Diffrax](https://docs.kidger.site/diffrax/api/solvers/sde_solvers/).

We assume the observation model is

$$y(t) = H x(t) + r(t)$$ with $$r(t) \sim N(0,R)$$,

where we choose $R=I$. 

- Note that we can impose partial observability with H=[1, 0, 0], with noisy observations, sampled at irregular time intervals.

## Tutorials

We describe the notebooks with different use cases of interest for continuous-discrete, non-linear Gaussian dynamical system models (cd-nlgssm).

### Filtering partially-observed, noisy and irregularly-sampled observations

In the [state estimation and forecasting notebook](./cdnlgssm_filtering.ipynb) we showcase how to learn partially observed dynamics, for an assumed cd-nlgssm model, given a set of observations $Y_K = [y(t_1),\\ \dots ,\\ y(t_K)]$.

We show how to use cd-dynamax to estimate and forecast the latent state of a cd-nlgssm, by running the following filtering alternatives:

- The Extended Kalman Filter (EKF)
- The Ensemble Kalman Filter (EnKF)
- The Unscented Kalman filter (UKF)

### Drift estimation

In all the following tutorials, we assume that **the drift function** of a latent SDE is **the only unknown object** in the dynamics and observation model.

- We study these problems under different observation scenarios:
    - Full observation of system coordinates ($x_1, x_2, x_3$)
    - Partial observation system coordinates (only $x_1$)

- We study different parametric assumptions on the drift:
    - Known parametric form with unknown parameters ($\sigma, \rho, \beta$)
    - Unknown form represented by a neural network (with unknown parameters)

- We use the following methods to estimate the drift:
    - Maximum likelihood estimation (via SGD)
    - Bayesian estimation of posterior distribution (via NUTS)
    
- Hence, there are 8 notebooks corresponding to: ((SGD, NUTS) x (full observation, partial observation) x (mechanistic parameters, neural network parameters)).    
    - [MLE (via SGD) of fully observed Lorenz 63 system parameters](./cdnlgssm_parameter_estimation_SGD.ipynb)
    - [MLE (via SGD) of a partially observed Lorenz 63 system parameters](./cdnlgssm_parameter_estimation_SGD_partialObs.ipynb)
    - [Bayesian estimation (via NUTS) of fully observed Lorenz 63 system parameters](./cdnlgssm_parameter_estimation_NUTS.ipynb)
    - [Bayesian estimation (via NUTS) of a partially observed Lorenz 63 system parameters](./cdnlgssm_parameter_estimation_NUTS_partial_initwithSGD.ipynb)
    - [MLE (via SGD) of a neural network drift function parameters for a fully observed Lorenz 63 system](./cdnlgssm_NeuralNetDrift_SGD.ipynb)
    - [MLE (via SGD) of a neural network drift function parameters for a partially observed Lorenz 63 system](./cdnlgssm_NeuralNetDrift_SGD_partialObs.ipynb)
    - [Bayesian estimation (via NUTS) of a neural network drift function parameters for a fully observed Lorenz 63 system](./cdnlgssm_NeuralNetDrift_NUTS_initwithSGD.ipynb)
    - [Bayesian estimation (via NUTS) of a neural network drift function parameters for a partially observed Lorenz 63 system](./cdnlgssm_NeuralNetDrift_NUTS_initwithSGD_partialObs.ipynb)

### Additional cd-dynamax' diffeqsolve solver tweaking

- We provide an [additional diffeqsolve settings analysis notebook](./diffeqsolve_settings_analysis.ipynb), in which we study the tolerance and other solver choices for SDEs 
