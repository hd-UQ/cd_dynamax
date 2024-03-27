# Continuous Discrete Linear Gaussian State Space Models

- Implementation of Continuous-discrete Linear Gaussian State Space Models
    - We provide [a model class definition](./models.py#L39)
        - NB: t0 and t1 refer to t_k and t_{k+1}, not necessarily regularly sampled

    - We provide a set of filtering and smoothing algorithms described below.
 
## Implemented algorithms

- The codebase is based on algorithms as defined in
    - [[1] Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
        
- The [Continuous Discrete Kalman Filter implementation](./inference.py#L378)
    - i.e., Algorithm 3.15 in [[1]](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)

- [Continuous Discrete Kalman Smoothers](./inference.py#L515):
    - The [Continuous Discrete Kalman Smoother type I implementation](./inference.py#L558)
        - i.e., Algorithm 3.17 in [[1]](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
      
    - The [Continuous Discrete Kalman Smoother type II implementation](./inference.py#L588)
        - i.e., Algorithm 3.18 in [[1]](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
        
## Parameter inference

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE
    - See fit_sgd() in [../ssm_temissions.py](../ssm_temissions.py#L443)
        - Which leverages the [marginal log-likelihood computed in closed form](./continuous_discrete_linear_gaussian_ssm/models.py#L233)
        - The analytical marginalized log-likelihood is possible given the Gaussian nature of the (recursive) transition and emission distributions

- We do not provide a parameter (point)-estimation via EM
    - The m-step requires MLE for continuous time parameters
    
## Pending

- Note that the codebase currently only supports inputs at measurement times, i.e., $u$ is observed at times $t_k$ as given in t_emissions.

- Note that even if the model definition seems to allow for time-varying emission weights, the implementation is not ready to do so (dynamax wasn't either)
