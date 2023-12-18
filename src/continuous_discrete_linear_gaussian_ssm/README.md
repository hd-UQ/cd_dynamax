# Continuous Discrete Nonlinear Gaussian State Space Models

- Implementation of the Continuous-discrete NonLinear Gaussian State Space Models
    - We provide a model class definition
    - NB: t0 and t1 refer to t_k and t_{k+1}

- Note that the codebase currently only supports inputs at measurement times, i.e., $u$ is observed at times $t_k$ as given in t_emissions.
    
- TODO: We provide the Continuous Discrete Kalman Filter I implementation
    - i.e., Algorithm 3.15 in (Sarkka's proper reference)

- TODO: We provide the Continuous Discrete Kalman Smoother I implementation
    - i.e., Algorithm 3.17 in (Sarkka's proper reference)
    
- TODO: We provide a parameter (point)-estimation via  implementation of stochastic gradient descent based MLE
    - See fit_sgd() in [../ssm_temissions.py](../ssm_temissions.py)
    
- TODO: We do not provide a parameter (point)-estimation via EM
    - The m-step requires MLE for continuous time parameters


