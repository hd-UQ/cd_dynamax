# cd-dynamax source code description

We provide the following modifications of the dynamax codebase, to accommodate continuous-discrete models, i.e., those where observations are not assumed to be regularly sampled.

## [Continuous-time State Space Models with Emissions at Specified Discrete Times](./ssm_temissions.py)

- A modified version of dynamax's ssm.py that incorporates non-regular emission time instants: i.e., the t_emissions array
    - `t_emissions` is an input argument
        - We use `t0` and `t1` refer to $t_k$ and $t_{k+1}$, not necessarily regularly sampled
    - `t_emissions` is a matrix of size $[\textrm{num observations} \times 1]$
        - it should facilitate batching
        - For `lax.scan()` operations, we recast them in vector shape (i.e., remove final dimension)
  
## [Continuous-Discrete Linear Gaussian State Space Models](./continuous_discrete_linear_gaussian_ssm)

- We define a [ContDiscreteLinearGaussianSSM model](./continuous_discrete_linear_gaussian_ssm/models.py#L39)
    - We do not currently provide a ContDiscreteLinearGaussianConjugateSSM model implementation, as CD parameter conjugate priors are non-trivial
    
    - The CD-LGSSM model is based on
        - A continuous-time [push-forward operation](./continuous_discrete_linear_gaussian_ssm/inference.py#L77) that [computes and returns matrices A and Q](./continuous_discrete_linear_gaussian_ssm/models.py#L213)
            - based on Equation (3.135) in [[1] Särkkä, Simo. Recursive Bayesian inference on stochastic differential equations. Helsinki University of Technology, 2006.](https://aaltodoc.aalto.fi/items/cc45c44e-ff66-4907-bfff-03293391fe1d)
    
- [Continuous-Discrete Kalman filtering and smoothing algorithms are implemented](./continuous_discrete_linear_gaussian_ssm/README.md)

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE  
    - where the marginal log-likelihood is computed based on the CD-Kalman filter

## [Continuous-Discrete Nonlinear Gaussian State Space Models](./continuous_discrete_nonlinear_gaussian_ssm)

- We define a [ContDiscreteNonlinearGaussianSSM model](./continuous_discrete_nonlinear_gaussian_ssm/models.py#L112)
    
    - The CD-NLGSSM model is based on a continuous-time [push-forward operation](./continuous_discrete_nonlinear_gaussian_ssm/models.py#L50) that solves an SDE forward over the mean $x$ and covariance $P$ of the latent state
        - the parameters of the SDE function are provided in the [ParamsCDNLGSSM](./continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py#L161) object, which contains
            - The initial state's prior parameters in ParamsLGSSMInitial, as defined by dynamax
            - The dynamics function in [ParamsCDNLGSSMDynamics](./continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py#L58)
            - The emissions function in [ParamsCDNLGSSMEmissions](./continuous_discrete_nonlinear_gaussian_ssm/cdnlgssm_utils.py#L133)
                - These two latter are learnable functions            
    
- Different [filtering and smoothing algorithms are implemented](./continuous_discrete_nonlinear_gaussian_ssm/README.md)

- Parameter (point)-estimation is possible via stochastic gradient descent based MLE
    - the marginal log-likelihood can be computed according to different implemented filtering methods (EKF, UKF, EnKF)

## [utils](./utils)

- [diffrax_utils.py](./utils/diffrax_utils.py)
    - implements a diffrax based, autodifferentiable ODEsolver
    
- [test_utils.py](./utils/test_utils.py)

- [plotting_utils.py](./utils/plotting_utils.py)

## [test_scripts](./test_scripts)

- [cdlgssm_test_filter_TRegular.py](./test_scripts/cdlgssm_test_filter_TRegular.py) checks discrete and continuous-discrete Linear filtering algorithms with regularly sampled observations
    - Note that after SGD learning, comparison between discrete and continuous-discrete models is not easy due to different parameterizations.
        - Although filtered means and covs are not exactly equal, plots showcase they are quite accurate in both models.

- [cdlgssm_test_smoother_TRegular.py](./test_scripts/cdlgssm_test_smoother_TRegular.py) checks discrete and continuous-discrete Linear smoothing algorithms with regularly sampled observations
    - CD smoother type 1, as in Sarkka's Algorithm 3.17 matches discrete-time solution
    - CD smoother type 2, as in Sarkka's Algorithm 3.18 does not match discrete-time solutions
        - Performance is close though: are these related to differential equation solver differences?

- [cdnlgssm_test_filter_linear_TRegular](./src/cdnlgssm_test_filter_linear_TRegular.py) checks continuous-discrete Linear and Non-Linear filtering algorithms with regularly sampled observations
    1. A CDNLGSSM model with linearity assumptions is equivalent to a CDLGSSM model
        - Which can be computed based on both first and second order approximations to SDE (equivalent to linear SDEs)

    2. A CDNLGSSM model with EKF filtering provides same results than a KF with a CDLGSSM model
        - Based on first and second order EKF approximations (equivalent for linear SDEs)
        - CD-EKF matches the CD-Kalman filtering performance
        - Both for pre- and post-fit of parameters with SGD, using EKF for logmarginal computations
    
    3. A CDNLGSSM model with UKF filtering
        - CD-UKF matches the CD-Kalman filtering performance
        
    4. A CDNLGSSM model with EnKF filtering 
        - CD-EnKF provides a close-enough, but not exactly equal performance (even with increased number of particles) to the CD-Kalman filter
            - Pending improvements to EnKF:
                - try to get consistency on Linear Gaussian case.
                - can build jacobian-based observation H within EnKF (instead of particle approximations)

- [cdnlgssm_test_smoother_linear_TRegular](./src/cdnlgssm_test_smoother_linear_TRegular.py) checks continuous-discrete Linear and Non-Linear smoothing algorithms with regularly sampled observations
    1. We compare that a CDNLGSSM model with EKS smoothing (as in Sarkka's Algorithm 3.23) matches CD-linear-KS type 2 (as in Sarkka's Algorithm 3.18)
        - We notice that EKS smoothing (as in Sarkka's Algorithm 3.23) does not match CD-linear-KS type 1 (as in Sarkka's Algorithm 3.17)
            - Performance is close though: are these related to differential equation solver differences?

## [notebooks](./notebooks)

- 
