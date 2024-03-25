# Continuous Discrete Linear Gaussian State Space Models

- Implementation of the Continuous-discrete NonLinear Gaussian State Space Models
    - We provide a model class definition
    - NB: t0 and t1 refer to t_k and t_{k+1}, not necessarily regularly sampled

- Note that the codebase currently only supports inputs at measurement times, i.e., $u$ is observed at times $t_k$ as given in t_emissions.

- The codebase is based on algorithms as defined in
    - [Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
## Implemented algorithms
    
- The Continuous Discrete Kalman Filter I implementation
    - i.e., Algorithm 3.15 in (Sarkka's proper reference)

- The Continuous Discrete Kalman Smoother I implementation
    - i.e., Algorithm 3.17 in (Sarkka's proper reference)
    
- Parameter (point)-estimation via implementation of stochastic gradient descent based MLE
    - See fit_sgd() in [../ssm_temissions.py](../ssm_temissions.py)
    
- We do not provide a parameter (point)-estimation via EM
    - The m-step requires MLE for continuous time parameters

## TODO:

- Note that even if the model definition seems to allow for time-varying emission weights, the implementation is not ready to do so (dynamax wasn't either)
