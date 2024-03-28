# cd-dynamax

Codebase to extend dynamax to deal with irregular sampling, via continuous-discrete dynamics modeling

## Continuous-discrete state-space models

In this repository, we aim to build an expanded toolkit for learning and predicting dynamical systems that underpin real-world messy time-series data.

In particular, we assume there exists a (possibly unknown) stochastic dynamical system of form
$$\dot{x} = f(x,t) + \Sigma(x,t) \dot{w}, \quad x(0)=x_0$$
where $x_0 \in \mathbb{R}^{d_x}$, $f$ a possibly time-dependent drift function, $\Sigma$ a possibly state and/or time-dependent diffusion coefficient, and $\dot{w}$ the derivative of a $d_x$ i.i.d. Brownian motions.

We further assume that data are available at arbitrary times $\{t_k\}_{k=1}^K$ and observed via a measurement process dictated by
$$y(t) = h(x(t), t) + \eta(t)$$
where $h: \mathbb{R}^{d_x} \mapsto \mathbb{R}^{d_y}$ creates an observation from the true state of the dynamical system $x(t)$ (i.e., a realization of the above SDE), and $\eta(t_k)$ applies additive (i.i.d. wrt $t$) Gaussian noise to the observation.

Thus, we assume we have access to data $Y = [y(t_1), \dots , y(t_K)]$ and wish to:
- Estimate $x(t_K) | Y$ (i.e., filter)
- Estimate $\{x(t)\}_t | Y$ (i.e., smooth)
- Estimate $x(t > t_K) | Y$ (i.e. predict)
- Estimate $f, \Sigma, h, \textrm{Law}(\eta) | Y$ (i.e. infer parameters)

All of these problems are deeply interconnected, with the parameter inference step importantly relying on marginalizing out unobserved states $\{x(t)\}_t$.
This marginalization can be performed (approximately, in cases of non-linear dynamics) via filtering/smoothing algorithms.
By implementing such filtering/smoothing algorithms in a fast, autodifferentiable framework, we enable usage of modern general-purpose tools for parameter inference (e.g., stochastic gradient descent, Hamiltonian Monte Carlo)

## Codebase status

- We have implemented [continuous-discrete linear and non-linear models](./src/README.md), along with filtering and smoothing algorithms.

- We provide notebooks for linear and nonlinear continuous-discrete filtering/smoothing under regular and irregular sampling
    - Linear dynamics:
        - [Tracking](./src/notebooks/linear/cdlgssm_tracking.ipynb)
        - [Parameter estimation that marginalizes out un-observed dynamics via auto-differentiable filtering (MLE via SGD; uncertainty quantification via HMC)](./src/notebooks/non_linear/cdnlgssm_hmc.ipynb)
    - Nonlinear dynamics:
        - Pendulum:
            - [Pendulum (mimicking original dynamax notebook)](./src/notebooks/non_linear/cd_ekf_ukf_pendulum.ipynb)
            - [Pendulum (demonstrating instability of the problem)](./src/notebooks/non_linear/cd_ekf_ukf_pendulum.ipynb)
        - Lorenz 63:
            - [regular sampling times](./src/notebooks/non_linear/cd_ekf_ukf_enkf_Lorenz63.ipynb),
            - [irregular sampling times](./src/notebooks/non_linear/cd_ekf_ukf_enkf_Lorenz63_irregular_times.ipynb)
            - PENDING: parameter estimation example

- We are leveraging [dynamax](https://github.com/probml/dynamax) code
    - Currently, based on a [dynamax pull at version '0.1.1+147.g3ad2ac5'](./dynamax)
        - Synching and updates to new dynamax version is PENDING

## Conda environment

- We provide a working conda environment
    - with dependencies installed using the pip-based requirements file

```bash
$ conda create --name hduq_nodynamax python=3.11.4
$ conda activate hduq_nodynamax
$ conda install pip
$ pip install -r hduq_pip_nodynamax_requirements.txt
```
