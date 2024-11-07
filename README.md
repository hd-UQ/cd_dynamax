# Overview of cd-dynamax

The primary goal of this codebase is to extend [dynamax](https://github.com/probml/dynamax) to a continuous-discrete (CD) state-space-modeling setting:

- that is, to problems where the underlying dynamics are continuous in time and measurements can arise at arbitrary (i.e., non-regular) discrete times.

To address these gaps, `cd-dynamax` modifies `dynamax` to accept irregularly sampled data and implements classical algorithms for continuous-discrete filtering and smoothing.

## Mathematical Framework: continuous-discrete state-space models

In this repository, build an expanded toolkit for learning and predicting dynamical systems that underpin real-world messy time-series data.
We move towards this goal by introducing the following flexible mathematical setting.

We assume there exists a (possibly unknown) stochastic dynamical system of form

$$dx(t) = f(x(t),t) + L(x(t),t) dw(t)$$

where $x \in \mathbb{R}^{d_x}$, $x(0) \sim \mathcal{N}(\mu_0, \Sigma_0)$, $f$ a possibly time-dependent drift function, $L$ a possibly state and/or time-dependent diffusion coefficient, and $dw$ is the derivative of a $d_x$-dimensional Brownian motion with a covariance $Q$.

We further assume that data are available at arbitrary times $\\{t_k\\}_{k=1}^K$ and observed via a measurement process dictated by

$$y(t) = h(x(t)) + \eta(t)$$

where $h: \mathbb{R}^{d_x} \mapsto \mathbb{R}^{d_y}$ creates a $d_y$-dimensional observation from the $d_x$-dimensional state of the dynamical system $x(t)$ (a realization of the above SDE), and $\eta(t)$ applies additive Gaussian noise to the observation.

We denote the collection of all parameters as $\theta = \\{f,\\  L,\\  \mu_0,\\  \Sigma_0,\\  L,\\  Q,\\  h,\\  \textrm{Law}(\eta) \\}$.

Note:

- We assume $\eta(t)$ i.i.d. w.r.t. $t$:
    - This assumption places us in the *continuous (dynamics) - discrete (observation)* setting.
    - If $\eta(t)$ had temporal correlations, we would likely adopt a mathematical setting that defines the observation process continuously in time via its own SDE.

- Other extensions of the above paradigm include categorical state-spaces and non-additive observation noise distributions
    - These can fit into our code framework (indeed, some are covered in `dynamax`), but have not been our focus.

## cd-dynamax goals and approach

For a given set of observations $Y_K = [y(t_1),\\ \dots ,\\ y(t_K)]$, we wish to:
- Filter: estimate $x(t_K) \\ | \\ Y_K, \\ \theta$
- Smooth: estimate $\\{x(t)\\}_t \\ | \\ Y_K, \\ \theta$
- Predict: estimate $x(t > t_K)\\ |\\ Y_K, \\ \theta$
- Infer parameters: estimate $\theta \\ |\\ Y_K$

All of these problems are deeply interconnected.

- In cd-dynamax, we enable filtering, smoothing, and parameter inference for a single system under multiple trajectory observations ($[Y^{(1)}, \\ \dots \\, \\ Y^{(N)}]$.
   
    - In these cases, we assume that each trajectory represents an independent realization of the same dynamics-data model, which we may be interested in learning, filtering, smoothing, or predicting.
        - In the future, we would like to have options to perform hierarchical inference, where we assume that each trajectory came from a different, yet similar set of system-defining parameters $\theta^{(n)}$.

    - We implement such filtering/smoothing algorithms in a fast, autodifferentiable framework, we enable usage of modern general-purpose tools for parameter inference (e.g., stochastic gradient descent, Hamiltonian Monte Carlo).

- In cd-dynamax, we take onto the parameter inference case by relying on marginalizing out unobserved states $\\{x(t)\\}_t$
    
    - this is a design choice of ours, other alternatives are possible.
    - This marginalization is performed (approximately, in cases of non-linear dynamics) via filtering/smoothing algorithms.

## Codebase status

- We are leveraging [dynamax](https://github.com/probml/dynamax) code
    - Currently, based on a local directory with [dynamax pull at version '0.1.1+147.g3ad2ac5'](./dynamax)
        - Synching and updates to new dynamax versions is ONGOING, just making sure it all runs smoothly both in CPU and GPUs, stay tuned!

- We have implemented [continuous-discrete linear and non-linear models](./src/README.md), along with filtering and smoothing algorithms.
    - If you are simulating data from a non-linear SDE, it is recommended to use [`model.sample(..., transition_type="path")`](./src/ssm_temissions.py#L208), which runs an SDE solver.
        - [Default behavior](./src/ssm_temissions.py#L204) is to perform Gaussian approximations to the SDE.

- For comparison purposes, we provide example notebooks for linear continuous-discrete filtering/smoothing under regular and irregular sampling
    - [Tracking](./src/notebooks/linear/cdlgssm_tracking.ipynb)
    - [Parameter estimation](./src/notebooks/non_linear/cdnlgssm_hmc.ipynb) that marginalizes out un-observed dynamics via auto-differentiable filtering (MLE via SGD; uncertainty quantification via HMC)

- For more interesting continuous-discrete, nonlinear models, see our new [tutorials](./src/notebooks/tutorial) for examples of how to use the codebase.
    - We provide a [tutorial REAMDE](./src/notebooks/tutorial/README.md) describing each of the tutorials
    - Highlights include a [notebook for learning neural network based drift functions](./src/notebooks/tutorial/cdnlgssm_NeuralNetDrift_NUTS_initwithSGD_partialObs.ipynb) from partial, noisy, irregularly-spaced observations!

## Conda environment

- We provide a working conda environment
    - with dependencies installed using the pip-based requirements file

```bash
# For CPU
$ conda create --name hduq_nodynamax python=3.11.4
$ conda activate hduq_nodynamax
$ conda install pip
$ pip install -r hduq_pip_nodynamax_requirements.txt

# For GPU
$ conda create --name hduq_nodynamax_GPU python=3.11.4
$ conda activate hduq_nodynamax_GPU
$ conda install pip
$ pip install -r hduq_pip_nodynamax_requirements.txt
$ pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$ pip install jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

