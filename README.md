# Overview of cd-dynamax

The primary goal of this codebase is to extend [dynamax](https://github.com/probml/dynamax) to a continuous-discrete (CD) state-space-modeling setting: that is, where underlying dynamics are continuous in time and measurements can arise at arbitrary (i.e., non-regular) discrete times.
To address these gaps, `cd-dynamax` modifies `dynamax` to accept irregularly sampled data and implements classical algorithms for continuous-discrete filtering and smoothing.

## Mathematical Framework: continuous-discrete state-space models

In this repository, we aim to build an expanded toolkit for learning and predicting dynamical systems that underpin real-world messy time-series data.
We move towards this goal by introducing the following flexible mathematical setting.

We assume there exists a (possibly unknown) stochastic dynamical system of form
$$\dot{x} = f(x,t) + L(x,t) \dot{w}, \quad x(0)=x_0$$
where $x \in \mathbb{R}^{d_x}$, $x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$, $f$ a possibly time-dependent drift function, $L$ a possibly state and/or time-dependent diffusion coefficient, and $\dot{w}$ the derivative of a $d_x$-dimensional Brownian motion with a covariance $Q$.

We further assume that data are available at arbitrary times $\\{t_k\\}_{k=1}^K$ and observed via a measurement process dictated by
$$y(t) = h\big(x(t)\big) + \eta(t)$$
where $h: \mathbb{R}^{d_x} \mapsto \mathbb{R}^{d_y}$ creates a $d_y$-dimensional observation from the $d_x$-dimensional state of the dynamical system $x(t)$ (a realization of the above SDE), and $\eta(t)$ applies additive Gaussian noise to the observation.

Note that we assume $\eta(t)$ i.i.d. w.r.t. $t$; this assumption places us in the "continuous (dynamics) - discrete (observation)" setting.
    - If $\eta(t)$ had temporal correlations, we would likely adopt a mathematical setting that defines the observation process continuously in time via its own SDE.
    - Other extensions of the above paradigm include categorical state-spaces and non-additive observation noise distributions; these can fit into our code framework (indeed, they are covered in `dynamax`), but have not been our focus; thus, we specify our mathematical setting to distinguish from these cases.

We denote the collection of all parameters as $\theta = \\{f,\\  L,\\  \mu_0,\\  \Sigma_0,\\  L,\\  Q,\\  h,\\  \textrm{Law}(\eta) \\}$.

Thus, we assume we have access to data $Y_K = [y(t_1),\\ \dots ,\\ y(t_K)]$ and wish to:
- Filter: estimate $x(t_K) \\ | \\ Y_K, \\ \theta$
- Smooth: estimate $\\{x(t)\\}_t \\ | \\ Y_K, \\ \theta$
- Predict: estimate $x(t > t_K)\\ |\\ Y_K, \\ \theta$
- Infer parameters: estimate $\theta \\ |\\ Y_K$

All of these problems are deeply interconnected, with the parameter inference step importantly relying on marginalizing out unobserved states $\\{x(t)\\}_t$.
    - This marginalization can be performed (approximately, in cases of non-linear dynamics) via filtering/smoothing algorithms.

By implementing such filtering/smoothing algorithms in a fast, autodifferentiable framework, we enable usage of modern general-purpose tools for parameter inference (e.g., stochastic gradient descent, Hamiltonian Monte Carlo).

In the codebase, we also allow for doing filtering, smoothing, and parameter inference for a single system under multiple trajectory observations ($[Y^{(1)}, \\ \dots \\, \\ Y^{(N)}]$.
    - In these cases, we assume that each trajectory represents an independent realization of the same dynamics-data model, which we may be interested in learning, filtering, smoothing, or predicting.
    - In the future, we would like to have options to perform hierarchical inference, where we assume that each trajectory came from a different, yet similar set of system-defining parameters $\theta^{(n)}$.

## Codebase status

- We are leveraging [dynamax](https://github.com/probml/dynamax) code
    - Currently, based on a local directory with [dynamax pull at version '0.1.1+147.g3ad2ac5'](./dynamax)
        - Synching and updates to new dynamax version is PENDING

- We have implemented [continuous-discrete linear and non-linear models](./src/README.md), along with filtering and smoothing algorithms.
    - If you are simulating data from a non-linear SDE, it is recommended to use [`model.sample(..., transition_type="path")`](./src/ssm_temissions.py#L208), which runs an SDE solver.
        - [Default behavior](./src/ssm_temissions.py#L204) is to perform Gaussian approximations to the SDE.

- We provide notebooks for linear and nonlinear continuous-discrete filtering/smoothing under regular and irregular sampling
    - Linear dynamics:
        - [Tracking](./src/notebooks/linear/cdlgssm_tracking.ipynb)
        - [Parameter estimation](./src/notebooks/non_linear/cdnlgssm_hmc.ipynb) that marginalizes out un-observed dynamics via auto-differentiable filtering (MLE via SGD; uncertainty quantification via HMC)
    - Nonlinear dynamics:
        - Pendulum:
            - [Pendulum (mimicking original dynamax notebook)](./src/notebooks/non_linear/cd_ekf_ukf_pendulum.ipynb)
            - [Pendulum (demonstrating instability of the problem)](./src/notebooks/non_linear/cd_ekf_ukf_pendulum.ipynb)
        - Lorenz 63:
            - [Tracking: regular sampling times](./src/notebooks/non_linear/cd_ekf_ukf_enkf_Lorenz63.ipynb),
            - [Tracking: irregular sampling times](./src/notebooks/non_linear/cd_ekf_ukf_enkf_Lorenz63_irregular_times.ipynb)
            - [Parameter estimation: mechanistic](./src/notebooks/non_linear/cdnlgssm_hmc_l63_partialObs_learnSigma.ipynb)
            - Pending: Parameter estimation for neural network

- Important pending features:
    - Support generic parameter estimation for problems with non-linear dynamics

## Conda environment

- We provide a working conda environment
    - with dependencies installed using the pip-based requirements file

```bash
$ conda create --name hduq_nodynamax python=3.11.4
$ conda activate hduq_nodynamax
$ conda install pip
$ pip install -r hduq_pip_nodynamax_requirements.txt
```
