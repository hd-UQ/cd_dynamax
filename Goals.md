# CD-Dynamax 

Goal is to extend dynamax to deal with irregular sampling, via continuous-discrete dynamics modeling

## Codebase Progress and status

- Continuous-dscrete extension for linear gaussian systems is implemented. 
  - Test passes for regular sampling [cdlgssm_test_filter](./src/cdlgssm_test_filter.py).
  - Irregular sampling demo in [cdlgssm_tracking](./src/example_notebooks/cdlgssm_tracking.ipynb).
- Continuous-discrete filtering extension implemented for non-linear gaussian systems.
  - Implemented UKF, EKF, and EnKF.
  - Tests pass for linear case with regular sampling [cdnlgssm_test_filter_linear_TRegular](./src/cdnlgssm_test_filter_linear_TRegular.py).
  - Pending:
    - Fix the test to show that {d-EKFs / d-UKF } == {cd-EKFs / cd-UKF} for linear system.
        - After SGD learning (which is accurate), filtered means and covs are not accurate anymore
        
    - Improve EnKF:
      - try to get consistency on Linear Gaussian case.
      - can build jacobian-based observation H within EnKF (instead of particle approximations)
    
    - Notebook w/ pendulum at regular time intervals showing cd-UKF, cd-EKF, cd-EnKF vs d-UKF, d-EKF. 
      - Check why our simulation differs from original notebook?
    
    - Notebook w/ pendulum at irregular time intervals using cd-UKF, cd-EKF, cd-EnKF

- Continuous-discrete smoothing extension implemented for non-linear gaussian systems.
  - Implemented EKS
  - Pending: 
    - UKS
    - EnKS
    - Test to show that discrete KS vs d-EKS vs d-UKS vs CD KS 1 vs CD KS 2 vs CD Extended KS 1 vs CD UKS in linear system case
    - Notebook w/ pendulum showing at regular time interval cd-UKS, cd-EKS vs d-UKS, d-EKS
    - Notebook w/ pendulum showing at irregular time intervals cd-UKS and cd-EKS

## Parameter estimation codebase

- Fit SGD works for continuous-discrete linear and non-linear models
    - We can compute MLE model parameter estimates based on different filtering algorithms
    
- Pending: 
	- Check that parameter estimates are consistent between cd-l and cd-nl for linear models with no bias terms (=None).
	- Generalize learnable function params property to deal with multiple parameters (e.g., weights and biases).

- Parameter estimation for the linear gaussian case
    - SGD
        - TODO: Add notebook showcasing parameter estimation accuracy (port from add_validation branch)
    - EM is not implemented
        - The m-step requires MLE for continuous-time linear parameters 
        - EM will not be trivial for nonlinear ssms
    - ContDiscreteLinearGaussianConjugateSSM:
        - Shall we keep this and modify it for continuous-time linear paraemeters
        - Pending: Are there conjugate priors for the continuous-discrete linear case?

- Uncertainty Quantification for the continuous-discrete linear/non-linear gaussian case:
  - using Monte Carlo
  - using HMC
  - using other MCMC?

### Hierarchical parameter estimation

- hierarchical parameter estimation via Empirical Bayes
    - Incorporate priors over parameters
    - Define prior hyperparameters as new dynamax "parameters"
    - Use Monte Carlo to average over many realizations of parameters
    - Let SGD learn hyperparameters of prior via MC-based loss
    
# To dos

## Initial pending

- For SIAM UQ conference, show HMC working in a cd-linear setting assuming linear and non-linear models (i.e., show that EKF reproduces results of basic KF approach)
    - Plot true latent states Vs filtered-smoothed latents
    - Plot true emissions (no noise) Vs true observed data Vs estimated emissions (estimated mean + estimated covariance)
    
    - How hard it is to learn the initial distribution?
        - Help inference with fixed initial distribution, or at least not-learnable

### Code optimization (All Pending)

- Build compare() for tests that just takes objects rather than arrays…can make tests even more succinct this way. e.g. compare(cd_ukf_object, d_ukf_object) will compare all identically-named attributes

- Use the improved compare() in all tests.

- Pending:
    - Can CD-dynamax deal with noiseless state evolution?
        - i.e, ODE mode
        - i.e., What happens if Q=0
    - SGD fit with validation option given train-validation (in add_validation branch)   
    - Add new filtering functionalities to __init__
	- Be consistent when calling filters (always use `model.filter`, don't call inference files directly from a script)
	- build a lax_scan_debug function that behaves like lax.scan but actually just implements a for loop for easy debugging
	- Matt needs to un-install dynamax so that he can change dynamax code and have it work
	- add predicted_means/covs to lgssm_filter (in dynamax code)
	- use `output_fields` in filters to control granularity of returned posterior
	- Diffeqsolve
		- debug feature

## For v1 

- Tests for linear and nonlinear CD irregular sampling

- Notebooks for linear and nonlinear CD irregular sampling
    - Linear
    - Pendulum
    - Lorenz

- How to deal with MLE vs MAP
    - Simply editing log-priors? (Iñigo)
    - Editing fit_sgd with an argument?
    
### For v1.5

- Process inputs in latent dynamics

- Allow for input times to be different from measurement times?

## For v2

- How to modify learnable parameters, to have a parameter set
    - Build it for linear function with weights and biases

- Modify the pushforward to incorporate physics + NN
    - How to incorporate DL within Jax?
  
## New non-linear models

- lorenz

- glucose-insulin

- hormone models: Delayed ODEs!!

## Others

- Understand what info version of code is for, and implement if needed

- Can we have EM for CD?
    - linear case?
    - nonlinear case?
       
### Extend our codebase to incorporate continuous-time inputs

- Currently, the codebase supports inputs only at measurement times

- Moreover, these inputs currently couple to the dynamics and measurements discretely (creating a discontinuity in the state and measurement trajectories):
    - The dynamics are pushed-forward between measurement timepoints $[t_0,t_1]$, then the state at time $t_1$ is updated (additively) by a linear function $B$ of input at time $t_0$, $Bu(t_0)$.
        - $x(t_1) := Fx(t_0) + Bu(t_0) + noise_Q$
    - The emission at time $t_1$ is then updated (additively) by a linear function $D$ of input at time $t_0$, $Du(t_0)$.
        - $y(t_1) := Hx(t_1) + Du(t_0) + noise_R$

- GOALS:
  
  - to allow for input times to be different from measurement times
  
  - extend to continuous coupling of $u(t)$ to the state and measurement dynamics---how?
    - Do we interpolate $u(t)$ between input-measurement times?

# Publication plan

Hybrid modeling = dynamics defined by combination of mechastinic + ML functions 
 
## Methods contibution

- Hybrid modeling with Hierarchical UQ
    - Open Questions:
        - What is it out there? Relevant literature
        - How to deal-disentangle with uncertainty coming from ML Vs Mechanistic?
        - Is it worth-novel without Hierarchy?

## Application papers

1. UQ over CD mechanistic models
    - Hormones
        - Simulated example: parameter UQ under partial & noisy state observations
            - How to deal with Delayed ODE?
        - Contributions
            - first to consider CD observations for this model
            - UQ over models?
        
        - MLCH2024
        
    - Travis Gibson
        - Custom code exists, does he want to transfer to CD-dynamax?
        - Contribution:
            - CD more natural for this setting
    - Dave?
    - Melike?

2. Mechanistic + ML drift fitting to data (point estimate parameter learning)
    - Travis Gibson

3. JAMIA perspective on fast and flexible tools to do UQ over mechanistic models?
    - Due July 1st 2024
    
4. Hierarchical UQ over CD dynamics
    - Hormones
    - Travis?
    - Emily?
    
5. Hybrid modeling with (Hierarchical) UQ
    - Hormones
    - Travis 
    - Emily
  
? Continuous-Discrete time bandits
    - 
    
- CD-conjugate priors
    - Do they exist?
    - Can we derive them?


# Big picture

- CD-UQ
    - References out there
    - Solutions in the linear case? conjugate priors?
    
- Learn a RHS with uncertainty (via GP or NN)...either purely data-driven and/or hybrid modeling.

- What priors to use for hybrid UQ
    - Mechanistic case over parameters
        - validity of solution (protect from explosive cases)

    - Hybrid
        - Parameter Vs function uncertainty
        
    - Combined
        - Enforcing physical constraints
        
- Extend CD-Dynamax to
    - Non-Gaussian Emission distributions
    - Consider other state processes
        - e.g., gamma process for latent states
        
- Unidentifiability questions
    - Disentangling uncertainty
        - emissions Vs states
        - mechanistic Vs ML
    - Learning/identifying equiprovable regions of space
        - Learning low-d mappings for equiprovable regions of parameter space
