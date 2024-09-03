# Discrete Vs Continuous-discrete

- Often performances are close, but not exactly equal:
    - These can be related to differential equation solver differences

    - After SGD learning, comparison between discrete and continuous-discrete models is not easy due to different model parameterizations.
        - Although filtered means and covs are not exactly equal, plots showcase they are quite accurate in both models.
        
    - When comparing cd-linear model with cd-nonlinear model (assumping linear model) and using EKF
        - Performance is equivalent when bias terms in linear models are not learned!
        - However, when bias terms are learned, then covariances are close, but not filtered and predicted means (hence, not marginal_logliks either):
            
            ```
            Fields that are close within tol=0.0001: ['predicted_covariances', 'filtered_covariances']            
            Fields that are different within tol=0.0001: ['predicted_means', 'marginal_loglik', 'filtered_means']
            ```
        - Why?
        
- dt_final
    - Due to how filtering algorithms are coded:
        - first we *condition_on* observation, then we run forward estimation (*predict*)
    - the continuous-discrete case requires a final extra (often ignored) prediction
        - this is determined by dt_final
            - with dt_final=1:
                - then discrete Vs continuous-discrete results match
            - with small dt_final 
                - we avoid numerical issues in this extra prediction

    - dt_final is incorporated as hyperparameter in filter_hyperparams (with default 1e-10)
        - user can edit, for instance, to pass discrete Vs continous-discrete tests
    
# Codebase maintenance

- As much as possible, we try to **not modify** original dynamax code
    - Instead, we write wrapper or utility functions if additional or complementary functionalities are needed.

