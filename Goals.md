# Release todos

- we keep dynamax
- we provide requirements.txt
- within our codebase
    - move cdssm_utils to utils dir
    - create a test_scripts dir, move tests there

- Revise our own README with
    - intro to problem
    - summary of what is implemented, by linking to READMEs within each folder
    - how to install
    - summary of notebooks
    - summary of tests

- Make sure the links work

- Merge all the above into main
    - double-check all is good
    - tag it with v0
- Make v0 public
    - via new brunch

# cd-dynamax 

Goal is to extend dynamax to deal with irregular sampling, via continuous-discrete dynamics modeling

## Codebase progress and status

- Continuous-discrete extension (filtering and smoothing) for linear gaussian systems is implemented.
    - Test passes for regular sampling-based [cdlgssm_test_filter_TRegular.py](./src/cdlgssm_test_filter_TRegular.py).
        - Note that after SGD learning, comparison between discrete and continuous-discrete models is not easy due to different parameterizations.
            - Although filtered means and covs are not exactly equal, plots showcase they are quite accurate in both models.

    - Test passes for regular sampling-based [cdlgssm_test_smoother_TRegular.py](./src/cdlgssm_test_smoother_TRegular.py)
        - CD smoother type 1, as in Sarkka's Algorithm 3.17 matches discrete-time solution
        - CD smoother type 2, as in Sarkka's Algorithm 3.18 does not match discrete-time solutions
            - Performance is close though: are these related to differential equation solver differences?

    - Notebooks        
        - Irregular sampling demo in [cdlgssm_tracking](./src/notebooks/linear/cdlgssm_tracking.ipynb).

- Continuous-discrete extension (filtering and smoothing) implemented for non-linear gaussian systems.
    - Implemented UKF, EKF, and EnKF.
    - Implemented Extended Kalman Smoother (EKS)
        - UKS and EnKS are PENDING
    
    - Test [cdnlgssm_test_filter_linear_TRegular](./src/cdnlgssm_test_filter_linear_TRegular.py) pass for linear model with regular sampling-based filters.
        - Namely, we compare that
            1. A CDNLGSSM model with linearity asusmptions provides same model as CDLGSSM model
                - Based on both first and second order approximations to SDE (equivalent for linear SDEs)
            2. A CDNLGSSM model with EKF filtering provides same results than a KF in CDLGSSM model
                - Based on first and second order EKF approximations (equivalent for linear SDEs)
                - Both for pre- and post-fit of parameters with SGD, using EKF for logmarginal computations
            
            3. A CDNLGSSM model with UKF filtering 
                - matches the CD-Kalman filtering performance
                
            4. A CDNLGSSM model with EnKF filtering 
                - provides a close-enough, but not exactly equal performance (even with increased number of particles)
                    - Pending improvements to EnKF:
                        - try to get consistency on Linear Gaussian case.
                        - can build jacobian-based observation H within EnKF (instead of particle approximations)

    - Test [cdnlgssm_test_smoother_linear_TRegular](./src/cdnlgssm_test_smoother_linear_TRegular.py) pass for linear model with regular sampling-based smoother.
        - Namely, we compare that
            1. A CDNLGSSM model with EKS smoothing (as in Sarkka's Algorithm 3.23)
            - CD-nonlinear-EKS (as in Sarkka's Algorithm 3.23) matches CD-linear-KS type 2 (as in Sarkka's Algorithm 3.18)
                - but does not match CD-linear-KS type 1 (as in Sarkka's Algorithm 3.17)
                    - Performance is close though: are these related to differential equation solver differences?

            2. A CDNLGSSM model with UKF smoothing PENDING
            
            3. A CDNLGSSM model with EnKF smoothing PENDING

    - Notebooks
        - Notebook w/ pendulum at regular time intervals showing cd-UKF, cd-EKF, cd-EnKF vs d-UKF, d-EKF. 

        - Notebook w/ pendulum at irregular time intervals using cd-UKF, cd-EKF, cd-EnKF

        - Notebooks w/ pendulum showing 
            - at regular time interval cd-UKS, cd-EKS vs d-UKS, d-EKS
            - at irregular time intervals cd-UKS and cd-EKS

### Parameter estimation

- Fit SGD works for continuous-discrete linear and non-linear models
    - We can compute MLE model parameter estimates based on different filtering algorithms
    - Does it make sense to use smoothing to compute logmarginal used by SGD?
    
- Pending: 
	- Generalize learnable function params property to deal with multiple parameters (e.g., weights and biases).
        - TODO: Add notebook showcasing parameter estimation accuracy (port from add_validation branch)
        
    - ContDiscreteLinearGaussianConjugateSSM:
        - Can we derive Conjugate priors for continuous-discrete linear dynamic paraemeters?

    - EM is not implemented

### Uncertainty Quantification

- Via Hamiltonian Monte Carlo (HMC)
    - [Example notebook](to be added, based on Initial pending below)

- Hierarchical uncertainty quantification
    - via Empirical Bayes
        - Incorporate priors over parameters
        - Define prior hyperparameters as new dynamax "parameters"
        - Use Monte Carlo to average over many realizations of parameters
        - Let SGD learn hyperparameters of prior via MC-based loss
    
# To dos

- Uncertainty quantification via HMC
    - How to deal with MLE vs MAP
        - Simply editing log-priors? (Iñigo)
        - Editing fit_sgd with an argument?
        
### Code optimization (All Pending)

- Pending:
    - Can CD-dynamax deal with noiseless state evolution?
        - i.e, ODE mode
        - i.e., What happens if Q=0
    - fit_SGD function with validation option, given train-validation data (preliminary existing in add_validation branch)   
	- build a lax_scan_debug function that behaves like lax.scan but actually just implements a for loop for easy debugging
	- Matt needs to un-install dynamax so that he can change dynamax code and have it work
	- add predicted_means/covs to lgssm_filter (in dynamax code)
	- use `output_fields` in filters to control granularity of returned posterior
	- Diffeqsolve
		- debug feature

## For v0 

- Tests for linear and nonlinear CD filtering and smoothing with regular and irregular sampling

- Notebooks for linear and nonlinear CD with regular and irregular sampling
    - Linear:
        - [Tracking](./src/notebooks/linear/cdlgssm_tracking.ipynb)
        - [Parameter learning (regular sampling times)](./src/notebooks/linear/cdlgssm_learnParams_oscillator_fixedSampleRate.ipynb)
        - [Parameter learning (irregular sampling times)](./src/notebooks/linear/cdlgssm_learnParams_oscillator_irregularSampleRate.ipynb)
    - [Pendulum](./src/notebooks/non_linear/cd_ekf_ukf_pendulum.ipynb)
    - Lorenz 63:
        - [regular sampling times](./src/notebooks/non_linear/cd_ekf_ukf_enkf_Lorenz63.ipynb),
        - [irregular sampling times](./src/notebooks/non_linear/cd_ekf_ukf_enkf_Lorenz63_irregular_times.ipynb)
    
### For v+

- Process inputs in dynamic functions

- Allow for input times to be different from measurement times?

- How to modify learnable parameters, to have a parameter set
    - Build it for linear function with weights and biases
- How to be able to initialize params across many NL functions
    - Maybe within function wrapper?

- Modify the pushforward to incorporate physics + NN
    - How to incorporate DL within Jax?

## Longer term ideas

## Uncertainty Quantification

- Via Monte Carlo approches
    - using plain Monte Carlo
    - using HMC
    - using other MCMC?

- Hierarchical uncertainty quantification
    - via Empirical Bayes
        - Incorporate priors over parameters
        - Define prior hyperparameters as new dynamax "parameters"
        - Use Monte Carlo to average over many realizations of parameters
        - Let SGD learn hyperparameters of prior via MC-based loss
        
## New non-linear models

- lorenz

- glucose-insulin

- hormone models
    - Clark model, is based on delayed ODEs
        - See original equation [in Appendix A here](https://arxiv.org/abs/1712.00117)
            - It seems that [we simulated with fixed delta (tau), by providing those delayed values via indexing](https://github.com/iurteaga/hmc/blob/master/src/clark_dde.m)
    - [Graham + Selgrade model](https://www.sciencedirect.com/science/article/abs/pii/S0022519317300073?via%3Dihub)
        - This is an ODE
    - [Reduced Graham, Elhadad + Albers model](https://www.sciencedirect.com/science/article/abs/pii/S0025556423000202?via%3Dihub)
        - A version available [in arxiv](https://arxiv.org/abs/2006.05034)
        - this is clearly an ODE!
        

## Others

- Understand what info version of code is for, and implement if needed

- Can we have EM for CD?
    - The m-step requires MLE for continuous-time linear parameters
    - EM will not be trivial for nonlinear ssms
       
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
