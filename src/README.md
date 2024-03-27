# Notes on cd-dynamax modifications

We provide the following modifications of the dynamax codebase, to accommodate continuous-discrete models, i.e., those where observations are not assumed to be regularly sampled.

## [ssm_temissions.py](./ssm_temissions.py)

- A modified version of dynamax's ssm.py that incorporates non-discrete emission time instants: i.e., the t_emissions array
    - t_emissions is an input argument
        - We use t0 and t1 refer to t_k and t_{k+1}, not necessarily regularly sampled
    - t_emissions is a matrix of size $[num_observations \times 1]$
        - it should facilitate batching in inference
        - For lax.scan() operations, we recast them in vector shape (i.e., remove final dimension) 
  
## [continuous_discrete_linear_gaussian_ssm](./continuous_discrete_linear_gaussian_ssm)

- Summarize and move details to directory README

- This codebase provides a continuous-time pushforward that returns matrices A and Q
    - These are computed based on Sarkka's thesis eq (3.135)
    - See compute_pushforward() in [./continuous_discrete_linear_gaussian_ssm/inference.py](./continuous_discrete_linear_gaussian_ssm/inference.py)
    
- Filtering and smoothing code is implemented

- We provide a parameter (point)-estimation via implementation of stochastic gradient descent based MLE
    - Based on fit_sgd() in [./ssm_temissions.py](./ssm_temissions.py)
    
    - Which leverages the log-marginal likelihood computed in closed form [here](https://github.com/iurteaga/hybrid_dynamics_uq/blob/main/src/continuous_discrete_linear_gaussian_ssm/inference.py#L507)
        - The analytical marginalized log-likelihood is possible given the Gaussian nature of the (recursive) transition and emission distributions

## [continuous_discrete_nonlinear_gaussian_ssm](./continuous_discrete_nonlinear_gaussian_ssm)

- Summarize and move details to directory README
