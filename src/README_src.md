# Notes on src modifications

## ssm_temissions.py

- A modified version of dynamax's ssm.py that incorporates non-discrete emission time instants: i.e., the t_emissions array
    - t_emissions is an input argument
    - t_emissions is a matrix of size $[num_observations \times 1]$
        - it should facilitate batching in inference
        - For lax.scan() operations, we recast them in vector shape (i.e., remove final dimension) 

- The codebase is based on algorithms as defined in
    - [Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
## continuous_discrete_linear_gaussian_ssm

- This codebase provides a continuous-time pushforward that returns matrices A and Q
    - These are computed based on Sarkka's thesis eq (3.135)
    - See compute_pushforward() in [./continuous_discrete_linear_gaussian_ssm/inference.py](./continuous_discrete_linear_gaussian_ssm/inference.py)
    
- Filtering and smoothing code is implemented

- We provide a parameter (point)-estimation via implementation of stochastic gradient descent based MLE
    - Based on fit_sgd() in [./ssm_temissions.py](./ssm_temissions.py)
    
    - Which leverages the log-marginal likelihood computed in closed form [here](https://github.com/iurteaga/hybrid_dynamics_uq/blob/main/src/continuous_discrete_linear_gaussian_ssm/inference.py#L507)
        - The analytical marginalized log-likelihood is possible given the Gaussian nature of the (recursive) transition and emission distributions
    
- We do not provide a parameter (point)-estimation via EM
    - The m-step requires MLE for continuous time parameters

## continuous_discrete_nonlinear_gaussian_ssm
