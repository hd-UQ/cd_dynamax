# Continuous Discrete Linear Gaussian State Space Models

- Implementation of the Continuous-discrete Linear Gaussian State Space Models
    - We provide a model class definition
    - NB: t0 and t1 refer to t_k and t_{k+1}

- Note that the codebase currently only supports inputs at measurement times, i.e., $u$ is observed at times $t_k$ as given in t_emissions.
    
- TODO: We provide the Continuous Discrete EKF implementation
    - i.e., Algorithm ?? in (Sarkka's proper reference)

- TODO: We provide the Continuous Discrete UKF implementation
    - i.e., Algorithm ?? in (Sarkka's proper reference)
    
- TODO: We provide a parameter (point)-estimation via  implementation of stochastic gradient descent based MLE
    - See fit_sgd() in [../ssm_temissions.py](../ssm_temissions.py)
    
- We do not provide a parameter (point)-estimation via EM


## TODO:

- Note that even if the model definition seems to allow for time-varying emission weights, the implementation is not ready to do so (dynamax wasn't either)
