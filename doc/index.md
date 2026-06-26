# Welcome to cd-dynamax!

The primary goal of this codebase is to extend [dynamax](https://github.com/probml/dynamax) to a continuous-discrete (CD) state-space-modeling setting, that is, to problems where 

- the underlying dynamics are continuous in time,
- and measurements can arise at arbitrary (i.e., non-regular) discrete times.

To address these gaps, `cd-dynamax` modifies `dynamax` to accept irregularly sampled data and implements classical algorithms for continuous-discrete filtering and smoothing.

## Mathematical Framework: continuous-discrete state-space models

In this repository, we build an expanded toolkit for filtering, forecasting and learning dynamical systems that underpin real-world messy time-series data.

We move towards this goal by working with the following flexible mathematical setting:

- We assume there exists a (possibly unknown) stochastic dynamical system of form

$$dx(t) = f(x(t), u(t), t)dt + L(x(t), u(t), t) dw(t)$$

where $x \in \mathbb{R}^{d_x}$, $x(0) \sim p(x_0; \varphi_{x_0})$, $f$ is a possibly time-dependent drift function, $L$ is a possibly state and/or time-dependent diffusion coefficient, $u(t)$ denotes optional input covariates, and $dw$ is the derivative of a $d_x$-dimensional Brownian motion with covariance $Q$.

- We assume data are available at arbitrary times $\\{t_k\\}_{k=1}^K$ and observed via a measurement process dictated by

$$p(y(t_k) \mid x(t_k), u(t_k), t_k; \varphi_y)$$

The library provides the Gaussian model classes `ContDiscreteLinearGaussianSSM` and `ContDiscreteNonlinearGaussianSSM`, together with a `ContDiscreteNonlinearSSM` class for generic initial conditions and observation distributions.

We denote the collection of all parameters as $\theta = \\{f,\\  L,\\  \varphi_{x_0},\\  Q,\\  \varphi_y \\}$.

Note:

- The Gaussian model classes `ContDiscreteLinearGaussianSSM` and `ContDiscreteNonlinearGaussianSSM` use Gaussian observation noise.
    - At a high level, $\varphi_y$ collects the corresponding emission mean and covariance parameters.
    - These models remain in the standard *continuous (dynamics) - discrete (observation)* setting with conditionally independent observation noise across observation times.

- `ContDiscreteNonlinearSSM` supports generic initial conditions and generic observation distributions for nonlinear CD-SSMs.
    - These observation distributions can depend on state, inputs, and time.
    - This includes non-Gaussian emissions such as Poisson observations.

- Other extensions of the overall framework include categorical state spaces and additional non-Gaussian observation models.
    - These can fit into our broader code framework, and some related cases are already covered in `dynamax`, but they have not been our main focus here.

### On the importance of continuous-time modeling

While continuous-time SSMs can be represented as discrete-time SSMs when sampling at fixed intervals, there remain fundamental differences between these two modeling paradigms: the former cannot be perfectly translated into the latter without loss of information or introduction of artifacts.

Succinctly put, the relationship between the discrete and continuous frameworks is one of approximation --- a mapping that may involve significant information loss: while it is possible to derive a discrete-time model from a continuous-time model through discretization, the reverse process of obtaining a continuous-time model from a discrete-time model is generally ill-posed and non-unique.

There are two fundamental issues introduced by discretization:

- **Information Loss**: Sampling inevitably obscures the system's true dynamics, distorting the signal in a process known as aliasing. Discretization results in the loss of inter-sample behavior, and hence, a system can appear stable at the sampling points while actually experiencing oscillations between them.

- **Artifact Creation**: The choice of a discrete-time representation of a model, along with the definition of its sampling interval, can create non-physical, artificial dynamics. Discretization choices can introduce entirely new behaviors not present in the original continuous-time system. For instance, naive sampling can induce the emergence (or destruction) of chaos in simple discrete maps (entirely absent, or assured, in their stable continuous-time counterparts) or instability of control-systems (where a stable continuous-time system can be rendered catastrophically unstable by choosing incorrect sampling intervals).

There are **significant benefits of a continuous-time treatment of dynamical systems**:

- *Data agnosticism*: continuous-time models are inherently suited to handle real-world, irregularly-spaced, and missing data: they model the underlying process, not the measurement grid. Thus, continuous-time models naturally generalize to arbitrary observation time grids without retraining or modification.

- *Discretize at the end, not at the beginning*: a continuous-time framing allows for discretization choices to be deferred until the final stages of analysis, enabling the use of adaptive solvers and multi-rate sampling strategies that can better capture the system's dynamics. A history of successes in numerical analysis has shown that delaying discretization until the final stages of computation often leads to more accurate and stable results.

- *Physical interpretability*: continuous-time model parameters represent fundamental, invariant physical properties of the system (e.g., reaction rates, physical constants, clearance rates), whereas discrete-time parameters are a conflation of physical properties and the choices of sampling intervals. In physics-aware modeling, prior knowledge is often most naturally expressed in a continuous-time formulation.

- *First-principles-based theory*: continuous-time models, expressed as differential equations, are the "first principles" foundation for many physical and life sciences. The discrete-time model is most accurately viewed as a subsequent numerical implementation or approximation of this theoretical truth.


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

- We implement such filtering/smoothing algorithms in an efficient, autodifferentiable framework.
    - We enable usage of modern general-purpose tools for parameter inference (e.g., stochastic gradient descent, Hamiltonian Monte Carlo).

- In cd-dynamax, we take onto the parameter inference case by relying on marginalizing out unobserved states $\\{x(t)\\}_t$
    - this is a design choice of ours, other alternatives are possible.
    - This marginalization is performed (approximately, in cases of non-linear dynamics) via filtering/smoothing algorithms.

## Codebase description and status

The `cd-dynamax` codebase extends the `dynamax` library to support continuous-discrete state space models, where observations are made at specified discrete times rather than at regular intervals.

- We leverage [dynamax](https://github.com/probml/dynamax) code
    - Currently, based on a local directory with [Dynamax release 0.1.5](https://github.com/probml/dynamax/releases/tag/0.1.5)

- We have implemented the [`cd-dynamax` codebase](https://github.com/hd-UQ/cd_dynamax) to deal with [continuous-discrete linear, nonlinear Gaussian, and generic nonlinear models](./api_reference/index.md), along with several filtering and smoothing algorithms.

- The codebase is organized into several key directories:
```
cd_dynamax/
├── src/                       # Source code for cd-dynamax library
│   ├── continuous_discrete_linear_gaussian_ssm/  # CD-LGSSM models and algorithms
│   ├── continuous_discrete_nonlinear_gaussian_ssm/ # CD-NLGSSM models and algorithms
│   ├── continuous_discrete_nonlinear_ssm/ # CD-NLSSM models with generic initial/emission distributions
│   ├── ssm_temissions.py      # Modified SSM class for discrete emissions
│   └── utils/               # Utility functions and example models
├── dynamax/                     # Original dynamax library (as a submodule)
demos/                       # Python demos showcasing cd-dynamax functionality
├── python/scripts/          # Python scripts for running demos
├── python/notebooks/        # Jupyter notebooks for interactive demos
├── python/configs/          # Configuration files for demos
tests/                       # Tests for cd-dynamax functionality
```

## [Examples](./examples/)

We provide a set of [examples](./examples/) that showcase key functionality of `cd-dynamax`.

These examples illustrate how to learn components of continuous-discrete SDEs from data.

For instance:

- [Filtering-based likelihood tutorial](./examples/notebooks/lorenz63_filter_based_likelihood_tutorial/) to filtering-based likelihood computation for continuous-discrete SDEs.

- [SGD-based model fitting tutorial](./examples/notebooks/lorenz63_sgd_fit_to_data_tutorial/) to SGD-based fitting of continuous-discrete SDE model to data.

- [MCMC-based model fitting tutorial](./examples/notebooks/lorenz63_mcmc_fit_to_data_tutorial/) to MCMC-based fitting of continuous-discrete SDE model to data.

## Tests

- Several [tests](https://github.com/hd-UQ/cd_dynamax/tree/public/tests) to establish cd-dynamax general functionality, as well as linear and non-linear filters/smoothers tests: e.g., checks that non-linear algorithms applied to linear problems return similar results as linear algorithms.

## [Makefile](./Makefile)

- We provide a [Makefile](./Makefile) to automate common tasks, such as running tests and demos.

- To run all tests, simply execute:
```bash
make test
```

- For linting, we use `ruff`:
```bash
make lint
```

- We can also format files using `ruff`:
```bash
make clean
```

- The docs can be built using `mkdocs` as:
```bash
make build_docs
```

# Installation

Install from **PyPI** (recommended), from source in editable mode, or with a Conda-managed environment.

---

### Option 1: Install from PyPI (recommended)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
.venv\Scripts\activate      # on Windows

# Upgrade pip
pip install --upgrade pip

# Install latest release from PyPI
pip install cd-dynamax
```

`cd-dynamax` is currently **not available on Conda Forge**.

---

### Option 2: Install from source (editable)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
.venv\Scripts\activate      # on Windows

# Upgrade pip
pip install --upgrade pip

# Install in editable mode for local development
pip install -e .[dev]
```

---

### Option 3: Conda environment + pip install

```bash
# Create and activate a Conda environment with Python 3.11
conda create -n cd_dynamax python=3.11
conda activate cd_dynamax

# Install latest release from PyPI
pip install cd-dynamax
```

#### GPU support
If you want GPU acceleration with JAX, you must install a CUDA-enabled `jaxlib` wheel.  

Check the [JAX installation docs](https://jax.readthedocs.io/en/latest/installation.html#installation) for the exact commands for your system.

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
