# Continuous Discrete Linear Gaussian State Space Models

- Implementation of Continuous-discrete Linear Gaussian State Space Models
    - We provide [a model class definition](./models.py)
        - NB: t0 and t1 refer to t_k and t_{k+1}, not necessarily regularly sampled

    - We provide a set of filtering and smoothing algorithms described below.

 
## Implemented algorithms

## Implemented algorithms

- The codebase is based on algorithms as defined in
    - [Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
- The Continuous Discrete EKF implementation
    - i.e., Algorithm ?? in (Sarkka's proper reference)

- TODO: We provide the Continuous Discrete UKF implementation
    - i.e., Algorithm ?? in (Sarkka's proper reference)

## Parameter inference

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE
    - See fit_sgd() in [../ssm_temissions.py](../ssm_temissions.py)

- We do not provide a parameter (point)-estimation via EM
    - The m-step requires MLE for continuous time parameters
    
## Pending

- Note that the codebase currently only supports inputs at measurement times, i.e., $u$ is observed at times $t_k$ as given in t_emissions.
