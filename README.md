# Overview of cd-dynamax

The primary goal of this codebase is to extend [dynamax](https://github.com/probml/dynamax) to a continuous-discrete (CD) state-space-modeling setting:

- that is, to problems where the underlying dynamics are continuous in time and measurements can arise at arbitrary (i.e., non-regular) discrete times.

To address these gaps, `cd-dynamax` modifies `dynamax` to accept irregularly sampled data and implements classical algorithms for continuous-discrete filtering and smoothing.

## Mathematical Framework: continuous-discrete state-space models

In this repository, build an expanded toolkit for learning and predicting dynamical systems that underpin real-world messy time-series data.
We move towards this goal by introducing the following flexible mathematical setting.

We assume there exists a (possibly unknown) stochastic dynamical system of form

$$dx(t) = f(x(t),t)dt + L(x(t),t) dw(t)$$

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

<!-- ## Codebase status

- We are leveraging [dynamax](https://github.com/probml/dynamax) code
    - Currently, based on a local directory with [Dynamax release 0.1.5](https://github.com/probml/dynamax/releases/tag/0.1.5)

- We have implemented [continuous-discrete linear and non-linear models](./src/README.md), along with filtering and smoothing algorithms.
    - If you are simulating data from a non-linear SDE, it is recommended to use [`model.sample(..., transition_type="path")`](./src/ssm_temissions.py#L208), which runs an SDE solver.
        - [Default behavior](./src/ssm_temissions.py#L204) is to perform Gaussian approximations to the SDE. -->

<!-- - For comparison purposes, we provide example notebooks for linear continuous-discrete filtering/smoothing under regular and irregular sampling
    - [Tracking](./src/notebooks/linear/cdlgssm_tracking.ipynb)
    - [Parameter estimation](./src/notebooks/non_linear/cdnlgssm_hmc.ipynb) that marginalizes out un-observed dynamics via auto-differentiable filtering (MLE via SGD; uncertainty quantification via HMC) -->

<!-- - For more interesting continuous-discrete, nonlinear models, see our new [tutorials](./src/notebooks/tutorial) for examples of how to use the codebase.
    - We provide a [tutorial REAMDE](./src/notebooks/tutorial/README.md) describing each of the tutorials
    - Highlights include a [notebook for learning neural network based drift functions](./src/notebooks/tutorial/cdnlgssm_NeuralNetDrift_NUTS_initwithSGD_partialObs.ipynb) from partial, noisy, irregularly-spaced observations! -->

## Demos

### Numpyro-based API

We provide a set of [demos](./demos) that showcase key functionality of `cd-dynamax`, by interacting with the core-functionalities via `numpyro`.

These scripts illustrate how to learn components of continuous-discrete SDEs from data.

In particular, we provide easy demonstrations for using `numpyro` to define priors and perform parameter inference using HMC or SVI with likelihoods computed via continuous-discrete filtering in `cd_dynamax`. Demos include:

- Learning drift functions from noisy, irregularly-sampled data:
    - Lorenz 63 system:
        - Sparse dictionary learning (all 3 components observed): 
            ```bash
            python ./demos/numpyro/l63_LaplaceDict.py --emission_dim 3
            ```
        - Partial observations (only 1st component): 
            ```bash
            python ./demos/numpyro/l63_nn.py --emission_dim 1 --state_dim 3
            ```
    - Lorenz 96 system:
        - Sparse dictionary learning (5 components): 
        ```bash
        python ./demos/numpyro/l96_LaplaceDict.py --emission_dim 5 --state_dim 5
        ```
        - Sparse dictionary learning (10 components): 
        ```bash
        python ./demos/numpyro/l96_LaplaceDict.py --emission_dim 10 --state_dim 10 --N_particles 100
        ```

    - Linear SDE:
        - Learning a linear SDE from multiple i.i.d. noisy trajectories. In this setting, all trajectories are assumed to come from the *same* underlying linear system.  

        ```bash
        python ./demos/numpyro/LinearGaussian_MultiTraj.py --emission_dim 5 --state_dim 5 --N_trajectories 30
        ```
        - Hierarchical learning of linear SDEs from multiple noisy trajectories. Here, we assume each trajectory comes from a different, yet similar, linear SDE.  
        ```bash
        python ./demos/numpyro/LinearGaussian_MultiTraj_KF_Hierarchical.py --emission_dim 5 --state_dim 5 --N_trajectories 30
        ```
## Installation

We support installation via **Conda** (recommended) or via a standard Python virtual environment.

---

### Option 1: Conda (recommended)

```bash
# Create and activate a new environment with Python 3.11
conda create -n cd_dynamax python=3.11
conda activate cd_dynamax

# Install your package in editable mode (so local changes are picked up)
pip install -e .[dev]
```

This installs the core dependencies listed in `pyproject.toml`, along with optional developer tools (`pytest`, etc.) if you use `[dev]`.

#### GPU support
If you want GPU acceleration with JAX, you must install a CUDA-enabled `jaxlib` wheel.  
Check the [JAX installation docs](https://jax.readthedocs.io/en/latest/installation.html#installation) for the exact commands for your system.  
For example (CUDA 12):
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

### Option 2: Python venv + pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
.venv\Scripts\activate      # on Windows

# Upgrade pip
pip install --upgrade pip

# Install in editable mode
pip install -e .[dev]
```

---

### Notes

- `pip install -e .` puts the repo in *editable mode*, so changes to source code are immediately available without reinstalling.

- If you plan to use plotting features that rely on `graphviz`, make sure the system binary is installed:
  - **macOS:** `brew install graphviz`  
  - **Ubuntu/Debian:** `sudo apt install graphviz`  
  - **Windows (conda):** `conda install graphviz`
  
- The `[dev]` extra installs additional developer tools (like `pytest`).
    - Once your environment is installed, you can run automated tests:
    ```bash
    pytest
    ```
