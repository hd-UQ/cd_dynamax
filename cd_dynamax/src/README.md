# cd-dynamax source code description

We provide the following modifications of the dynamax codebase, to accommodate continuous-discrete models, i.e., those where observations are not assumed to be regularly sampled.

## [Continuous-time State Space Models with Emissions at Specified Discrete Times](./ssm_temissions.py)

- A modified version of dynamax's ssm.py that incorporates non-regular emission time instants: i.e., the t_emissions array
    - `t_emissions` is an input argument
        - We use `t0` and `t1` refer to $t_k$ and $t_{k+1}$, not necessarily regularly sampled
    - `t_emissions` is a matrix of size $[\textrm{num observations} \times 1]$
        - it should facilitate batching
        - For `lax.scan()` operations, we recast them in vector shape (i.e., remove final dimension)
  
## [Continuous-Discrete Linear Gaussian State Space Models](./continuous_discrete_linear_gaussian_ssm)

- We define a [ContDiscreteLinearGaussianSSM model](./continuous_discrete_linear_gaussian_ssm/models.py#L39)
    - We do not currently provide a ContDiscreteLinearGaussianConjugateSSM model implementation, as CD parameter conjugate priors are non-trivial
    
    - The CD-LGSSM model is based on
        - A continuous-time [push-forward operation](./continuous_discrete_linear_gaussian_ssm/inference.py#L77) that [computes and returns matrices A and Q](./continuous_discrete_linear_gaussian_ssm/models.py#L213)
            - based on Equation (3.135) in [[1] Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
- [Continuous-Discrete Kalman filtering and smoothing algorithms are implemented](./continuous_discrete_linear_gaussian_ssm/README.md)

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE  
    - where the marginal log-likelihood is computed based on the CD-Kalman filter

## [Continuous-Discrete Nonlinear Gaussian State Space Models](./continuous_discrete_nonlinear_gaussian_ssm)

- We define a [ContDiscreteNonlinearGaussianSSM model](./continuous_discrete_nonlinear_gaussian_ssm/models.py#L112)
    
    - The CD-NLGSSM model is based on a continuous-time [push-forward operation](./continuous_discrete_nonlinear_gaussian_ssm/models.py#L50) that solves an SDE forward over the mean $x$ and covariance $P$ of the latent state
        - the parameters of the SDE function are provided in the [ParamsCDNLGSSM](./continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py#L161) object, which contains
            - The initial state's prior parameters in ParamsLGSSMInitial, as defined by dynamax
            - The dynamics function in [ParamsCDNLGSSMDynamics](./continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py#L58)
            - The emissions function in [ParamsCDNLGSSMEmissions](./continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py#L133)
                - These two latter are learnable functions            
    
- Different [filtering and smoothing algorithms are implemented](./continuous_discrete_nonlinear_gaussian_ssm/README.md)

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE
    - the marginal log-likelihood can be computed according to different implemented filtering methods (EKF, UKF, EnKF)

## [Continuous-Discrete Nonlinear State Space Models](./continuous_discrete_nonlinear_ssm)

- We define a [ContDiscreteNonlinearSSM model](./continuous_discrete_nonlinear_ssm/models.py#L102)
    - This interface supports generic initial conditions and generic observation distributions, not just Gaussian ones
    - Dynamics and emission laws can depend on state, optional inputs, and time
    - It is the nonlinear CD-SSM entry point to use when $p(x_0; \varphi_{x_0})$ or $p(y_{t_k} \mid x_{t_k}, u_{t_k}, t_k; \varphi_y)$ is non-Gaussian

- We implement [differentiable particle filtering](./continuous_discrete_nonlinear_ssm/inference_dpf.py) for this model family
    - This is the inference path currently used for nonlinear CD-SSMs with generic observation distributions

- The codebase includes utilities for non-Gaussian emissions such as [LearnablePoissonEmission](./continuous_discrete_nonlinear_ssm/cdnlssm_utils.py#L81)

## [utils](./utils)

- cd-dynamax example model defintions:
    - [data_driven_models.py](./utils/data_driven_models.py): example neural network, Gaussian Process, polynomial and dictionary-learning models
    - [physics_based_models.py](./utils/physics_based_models.py): example definition of mechanistic models

- [data_generator.py](./utils/data_generator.py)
    - functions to generate synthetic data from continuous-discrete state space (cd-dynamax) models, based on user-specified configuration

- [diffrax_utils.py](./utils/diffrax_utils.py)
    - implements a diffrax based, autodifferentiable ODEsolver

- [debug_utils.py](./utils/debug_utils.py)
    - Debugging in jax can be difficult---pre-compilation speedups cause typical usage of in-line python debuggers to fail. To make debugging easier, we implemented a wrapper for `lax.scan` which, with `debug=True`, runs a (slow, but in-line debuggable!) `for` loop instead of `lax.scan`.
    - To use this in a particular piece of code, simply add `from utils.debug_utils import lax_scan` and replace an existing `lax.scan` call you wish to debug with `lax_scan(..., debug=True)`.
    - This is an experimental feature, so please report any issues that arise from using this tool---we hope it helps ease the transition into using jax!

- [optimize_utils.py](./utils/optimize_utils.py): utility functions for optimization routines

- [test_utils.py](./utils/test_utils.py): utility functions for unit tests

- Several plotting utilities are implemented in:
    - [plotting_utils.py](./utils/plotting_utils.py)
    - [pliotting_chaos_utils.py](./utils/plotting_chaos_utils.py)

- Additional utility functions are implemented in:
    - [experiment_utils.py](./utils/experiment_utils.py)
    - [simulation_utils.py](./utils/simulation_utils.py)
    - [prior_utils.py](./utils/prior_utils.py)
    - [likelihood_eval_utils.py](./utils/likelihood_eval_utils.py)
