# Continuous Discrete Linear Gaussian State Space Models

- Implementation of Continuous-discrete Non-Linear Gaussian State Space Models
    - We provide [a model class definition](./models.py#L112)
        - NB: t0 and t1 refer to t_k and t_{k+1}, not necessarily regularly sampled

    - We provide a set of filtering and smoothing algorithms described below.

## Implemented algorithms

- The codebase is based on algorithms as defined in
    - [[1] Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
- We implement [Extended-](./inference_ekf.py), [Unscented-](./inference_ukf.py) and [Ensemble-](./inference_enkf.py) Kalman filters and smoothers

### Extended Kalman Filter and Smoothers

- The [Continuous Discrete Extended Kalman filter](./inference_ekf.py#L162), with 

    - The [First-order Continuous Discrete Extended Kalman Filter implementation](./inference_ekf.py#L85)
        - i.e., Algorithm 3.21 in [[1]](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
    - The [Second-order Continuous Discrete Extended Kalman Filter implementation](./inference_ekf.py#L95)
        - i.e., Algorithm 3.22 in [[1]](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
     
- The [Continuous Discrete Extended Kalman smoother](./inference_ekf.py#L382), with 

    - The [First-order Continuous Discrete Extended Kalman Smoother implementation](./inference_ekf.py#L295)
        - i.e., Algorithm 3.23 in [[1]](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
### Unscented Kalman Filter

- The [Continuous Discrete Unscented Kalman filter](./inference_ukf.py#L191)

### Ensemble Kalman Filter

- The [Continuous Discrete Ensemble Kalman filter](./inference_enkf.py#L144)

## Parameter inference

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE
    - See fit_sgd() in [../ssm_temissions.py](../ssm_temissions.py#L443)
        - Which leverages the [marginal log-likelihood computed in closed form](./continuous_discrete_nonlinear_gaussian_ssm/models.py#L291)
        - The analytical marginalized log-likelihood is possible given the Gaussian nature of the (recursive) transition and emission distributions

- We do not provide a parameter (point)-estimation via EM
    - The m-step requires MLE for continuous time parameters
    
## Pending

- Note that the codebase currently only supports inputs at measurement times, i.e., $u$ is observed at times $t_k$ as given in t_emissions.

- UKS and EnKS implementations
