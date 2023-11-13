# To dos

- Understand what info version of code is for, and implement if needed

- Nonlinear transition function using diffrax (get this to work on nonlinear ssm ipynb notebooks) (DONE)
    - [ekf_ukf_pendulum_diffrax.ipynb](./notebooks/ekf_ukf_pendulum_diffrax.ipynb)
        - It takes nonlinear SSM pynotebook and uses difrax for nonlinear pushforward
        - We can take inspiration from these, to devise continuous-discrete nonlinear ssm code later 
    
    - No parameter estimation yet!
    
## Extend dynamax to deal with irregular sampling
- add non-regular interval capability to dynamax codebase
    - Done within continuous-discrete linear codebase (DONE)
        - [cdlgssm_tracking.ipynb](./src/cdlgssm_tracking.ipynb)
    - If it works, replicate for continuous-discrete nonlinear codebase
        - Implement continuous-discrete nonlinear codebase with non-regular intervals (t_emissions)
        - Implement continuous-discrete nonlinear pushforward in dyfrax
        - Implement continuous-discrete nonlinear filters
            - implement 3DVAR
            - implement Sarkka's solutions for UKF
            - implement Sarkka's solutions for EKF
    
- linear transition function using diffrax (DONE)
    - NOTE that linear transition function using exact continuous solutions is non-trivial (discarded)

## Extend our codebase to incorporate continuous-time inputs
- Currently, the codebase supports inputs only at measurement times
- Moreover, these inputs currently couple to the dynamics and measurements discretely (creating a discontinuity in the state and measurement trajectories):
    - The dynamics are pushed-forward between measurement timepoints $[t0,t1]$, then the state at time $t_1$ is updated (additively) by a linear function $B$ of input at time $t_0$, $Bu(t_0)$.
        - $ x(t_1) := Fx(t_0) + Bu(t_0) + noise_Q $
    - The emission at time $t_1$ is then updated (additively) by a linear function $D$ of input at time $t_0$, $Du(t_0)$.
        - $ y(t_1) := Hx(t_1) + Du(t_0) + noise_R $
- GOALS:
  - allow for input times to be different from measurement times
  - extend to continuous coupling of $u(t)$ to the state and measurement dynamics---how?
    - Do we interpolate $u(t)$ between input-measurement times?



## Parameter estimation
- Parameter estimation for the linear gaussian case
    - SGD
    - EM is not implemented
        - The m-step requires MLE for continuous-time linear parameters 
        - EM will not be trivial for nonlinear ssms
    - ContDiscreteLinearGaussianConjugateSSM:
        - Shall we keep this and modify it for continuous-time linear paraemeters

- hierarchical parameter estimation for the linear gaussian case
    - Incorporate priors over parameters
    - Define prior hyperparameters as new dynamax "parameters"
    - Use Monte Carlo to average over many realizations of parameters
    - Let SGD learn hyperparameters of prior via MC-based loss

- parameter estimation for 1 non-linear pendulum
- hierarchical parameter estimation for multiple non-linear pendula

- Check the conjugate version of the model class, and decide how to proceed

## New non-linear models
- lorenz
- glucose-insulin
- hormone models: Delayed ODEs!!

## Make all the above ready for hybrid learning

- Modify the pushforward to incorporate physics + NN
    - How to incorporate DL within Jax?
    

## Big picture
- Learn a RHS with uncertainty (via GP or NN)...either purely data-driven and/or hybrid modeling.
