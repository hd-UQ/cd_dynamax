# To dos

- Nonlinear transition function using diffrax (get this to work on nonlinear ssm ipynb notebooks) (DONE)
    - [ekf_ukf_pendulum_diffrax.ipynb](./notebooks/ekf_ukf_pendulum_diffrax.ipynb)
    - No parameter estimation yet!
    

## Extend dynamax to deal with irregular sampling
- add non-regular interval capability to dynamax codebase
- linear transition function using diffrax
- linear transition function using exact continuous solutions

## Parameter estimation
- Parameter estimation for the linear gaussian case
- parameter estimation for 1 non-linear pendulum
- hierarchical parameter estimation for multiple non-linear pendula

## New models

- lorenz
- hormone models
- glucose-insulin

## Big picture
- Learn a RHS with uncertainty (via GP or NN)...either purely data-driven and/or hybrid modeling.
