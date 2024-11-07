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

### For v0 

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

### For v0.1

- New [tutorials](./src/notebooks/tutorial) for examples of how to use the codebase.
    - We provide a [tutorial REAMDE](./src/notebooks/tutorial/README.md) describing each of the tutorials
    - We studied tolerance/solver choices for SDEs (Brownian Tree tolerance, etc.) in this [notebook](./src/notebooks/tutorial/diffeqsolve_settings_analysis.ipynb).

### For v0.2: Extend our codebase to incorporate continuous-time inputs

- Process inputs in dynamic functions

- Currently, the codebase supports inputs only at measurement times
    - Allow for input times to be different from measurement times? 
    
- Moreover, these inputs currently couple to the dynamics and measurements discretely (creating a discontinuity in the state and measurement trajectories):
    - The dynamics are pushed-forward between measurement timepoints $[t_0,t_1]$, then the state at time $t_1$ is updated (additively) by a linear function $B$ of input at time $t_0$, $Bu(t_0)$.
        - $x(t_1) := Fx(t_0) + Bu(t_0) + noise_Q$
    - The emission at time $t_1$ is then updated (additively) by a linear function $D$ of input at time $t_0$, $Du(t_0)$.
        - $y(t_1) := Hx(t_1) + Du(t_0) + noise_R$

- GOALS:
  
  - to allow for input times to be different from measurement times
  
  - extend to continuous coupling of $u(t)$ to the state and measurement dynamics---how?
    - Do we interpolate $u(t)$ between input-measurement times?

### Comments and others on codebase

- Understand what info version of code is for, and implement if needed

- ContDiscreteLinearGaussianConjugateSSM:
    - Can we derive Conjugate priors for continuous-discrete linear dynamic parameters?

- EM is not implemented
    - Can we have EM for CD?
        - The m-step requires MLE for continuous-time linear parameters
        - EM will not be trivial for nonlinear ssms

# To dos

- Clean private and push to public
    - describe tutorial notebooks
    - write the SIAM news
    - Ping Scott

- Can CD-dynamax deal with noiseless state evolution?
        - i.e, ODE mode
        - i.e., What happens if Q=0

- Understand likelihoods better
    -What does it mean to change the diffusion covariance from Bayesian perspective?
    -Are the likelihoods for the learned model very similar to the true model? Why / why not?

- How to be able to initialize params across many NL functions
    - Maybe within function wrapper?
    
## Code

- Implement new learnable models:
    - dictionary learning w/ learnable coefficients
    - KL expansion of GP w/ learnable coefficients

- UQ 
    - How to deal with MLE vs MAP
        - Definition of log-priors and how to use them
            - Pass a list of prior-dictionaries to initialize, where each prior-dictionary has keys "param_names", "param_prior", "sample2params", "params2sample".
            - Incorporate into existing log_prior function
            - Add a sample_prior function to sample from the prior

- Optimization related
    - Optax and Jaxopt

- For latest package branch and environment
    - Why is diffeqsolve giving errors?
    - Implement progress bars (e.g. for SGD) that are compatible with lax.scan    

- Robustness        
    - Check why L63 sample path can return NaNs in the long run

    - Ensure all COVs are PSD
        - We should think about how to implement all of our filters to behave better!
        - I believe many people have faced these issues, and have devised ways to deal with them.

- Flexibility
    - Implement a linear model using the nonlinear learnable function approach
        
    - Implement new non-linear models, as functions (by adding to [cdnlgssm_utils.py](./src/continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py))
        - glucose-insulin

        - [FitzHugh-Nagumo](https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model)

        - [Van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)

        - Other models studied in SINDy universe:
            - [PNAS paper](https://www.pnas.org/doi/10.1073/pnas.1517384113)
            - Partial observation papers:
                - [Nature Comm. Phys.](https://www.nature.com/articles/s42005-022-00987-z)
                - [Niall](https://arxiv.org/abs/2105.10068)
                - [Discrepancy models](https://arxiv.org/abs/1909.08574)

        - hormone models
            - Clark model, is based on delayed ODEs
                - See original equation [in Appendix A here](https://arxiv.org/abs/1712.00117)
                    - It seems that [we simulated with fixed delta (tau), by providing those delayed values via indexing](https://github.com/iurteaga/hmc/blob/master/src/clark_dde.m)
            - [Graham + Selgrade model](https://www.sciencedirect.com/science/article/abs/pii/S0022519317300073?via%3Dihub)
                - This is an ODE
            - [Reduced Graham, Elhadad + Albers model](https://www.sciencedirect.com/science/article/abs/pii/S0025556423000202?via%3Dihub)
                - A version available [in arxiv](https://arxiv.org/abs/2006.05034)
                - this is clearly an ODE!

    - Plan for data with multiple trajectories and train/validation splits.
        - Revise fit_SGD to have a validation option:
            - train and validation data (preliminary existing in add_validation branch)   
    
### Longer term ideas

- Implement autodiff SMC?

- Incoporate Optimal Transport ideas to filtering?

- Model error learning
    - NNs
        - not over the whole RHS
        - over specific variables, with specific variable dependencies
    - GPs
        - TODO
    
- Hierarchical uncertainty quantification
    - via Empirical Bayes
        - Incorporate priors over parameters
        - Define prior hyperparameters as new dynamax "parameters"
        - Use Monte Carlo to average over many realizations of parameters
        - Let SGD learn hyperparameters of prior via MC-based loss
    - working on eb_cddynamax branch
        - Implement more priors
            - e.g., from 2D simplex for linear models
        - Revise prior definition and handeling
            - now hard-coded, should be flexible

# Big picture

- CD-UQ
    - References out there
    - Solutions in the linear case? conjugate priors?
    
- Learn a RHS with uncertainty (via GP or NN)...either purely data-driven and/or hybrid modeling.
    - GP-based latent drift function lerning

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
    - Andrey work on canonical representations
    
    - Disentangling uncertainty
        - emissions Vs states
        - mechanistic Vs ML

    - Learning/identifying equiprovable regions of space
        - Learning low-d mappings for equiprovable regions of parameter space
        
## Publication plan

- Hybrid modeling = dynamics defined by combination of mechastinic + ML functions 
 
## Methods contibution

- Training with multi-scale/masking approach
    - Showcase improvements
    - Is this sufficient for conference paper?
    
- Training not on Y, but on sufficient statistics of Y: e.g., time-average statistics
    - How to do this
    - Is this sufficient for conference paper?
    
- Compare marginal loglikelihood-based approach to latent state estimation/matching approaches:
    - SINDY and follow-ups
    - Discuss and get input from Niall

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

3. Niall biological data
    - Uncertainty on parameters
    - ODE based model?
    
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
