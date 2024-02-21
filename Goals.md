# To dos

- Understand what info version of code is for, and implement if needed
- Can we get optax.Adam to ignore parameters that are not tensors? Somehow we need to be passing dynamics_function in params in non-linear setting.
  
## Extend dynamax to deal with irregular sampling
- Continuous-dscrete extension for linear gaussian systems is done. 
  - Test passes for regular sampling [cdlgssm_test_filter](./src/cdlgssm_test_filter.py).
  - Irregular sampling demo in [cdlgssm_tracking](./src/example_notebooks/cdlgssm_tracking.ipynb).
- Continuous-discrete filtering extension implemented for non-linear gaussian systems.
  - Implemented UKF, EKF, and EnKF.
  - Tests pass for linear case with regular sampling [cdnlgssm_test_filter_linear_TRegular](./src/cdnlgssm_test_filter_linear_TRegular.py).
  - Pending:
    - Test to show that {d-EKFs / d-UKF } == {cd-EKFs / cd-UKF} for linear system.
    - Improve EnKF:
      - try to get consistency on Linear Gaussian case.
      - can build jacobian-based observation H within EnKF (instead of particle approximations)
    - Notebook w/ pendulum at regular time intervals showing cd-UKF, cd-EKF, cd-EnKF vs d-UKF, d-EKF. 
      - Check why our simulation differs from original notebook?
    - Notebook w/ pendulum at irregular time intervals using cd-UKF, cd-EKF, cd-EnKF
- Continuous-discrete smoothing extension implemented for non-linear gaussian systems.
  - Implemented EKS.
  - Pending: 
    - UKS
    - EnKS
    - Test to show that discrete KS vs d-EKS vs d-UKS vs CD KS 1 vs CD KS 2 vs CD Extended KS 1 vs CD UKS in linear system case
    - Notebook w/ pendulum showing at regular time interval cd-UKS, cd-EKS vs d-UKS, d-EKS
    - Notebook w/ pendulum showing at irregular time intervals cd-UKS and cd-EKS

## Code optimization (All Pending)
- build a lax_scan_debug function that behaves like lax.scan but actually just implements a for loop for easy debugging
- Matt needs to un-install dynamax so that he can change dynamax code and have it work
- add predicted_means/covs to lgssm_filter (in dynamax code)
- Diffeqsolve
  — debug feature
  - pass *kwargs
- Build compare() for tests that just takes objects rather than arrays…can make tests even more succinct this way. e.g. compare(cd_ukf_object, d_ukf_object) will compare all identically-named attributes
- Use the improved compare() in all tests.


## Extend our codebase to incorporate continuous-time inputs
- Currently, the codebase supports inputs only at measurement times
- Moreover, these inputs currently couple to the dynamics and measurements discretely (creating a discontinuity in the state and measurement trajectories):
    - The dynamics are pushed-forward between measurement timepoints $[t_0,t_1]$, then the state at time $t_1$ is updated (additively) by a linear function $B$ of input at time $t_0$, $Bu(t_0)$.
        - $x(t_1) := Fx(t_0) + Bu(t_0) + noise_Q$
    - The emission at time $t_1$ is then updated (additively) by a linear function $D$ of input at time $t_0$, $Du(t_0)$.
        - $y(t_1) := Hx(t_1) + Du(t_0) + noise_R$
- GOALS:
  - allow for input times to be different from measurement times
  - extend to continuous coupling of $u(t)$ to the state and measurement dynamics---how?
    - Do we interpolate $u(t)$ between input-measurement times?

## Parameter estimation
- Parameter estimation for the linear gaussian case
    - SGD
        - TODO: Add notebook showcasing parameter estimation accuracy (port from add_validation branch)
    - EM is not implemented
        - The m-step requires MLE for continuous-time linear parameters 
        - EM will not be trivial for nonlinear ssms
    - ContDiscreteLinearGaussianConjugateSSM:
        - Shall we keep this and modify it for continuous-time linear paraemeters

- Uncertainty Quantification for the continuous-discrete linear gaussian case using Monte Carlo

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
