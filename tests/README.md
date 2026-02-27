# Simple functionality tests

- [test_imports.py](./test_imports.py) checks that key packages can be imported without errors.

- [test_models.py](./test_models.py) checks that basic model creation and parameter setting works for cd-dynamax models.

- [test_utils_imports.py](./test_utils_imports.py) checks that key (debug and difrax) utility functions can be imported without errors.

# Tests for cd-dynamax performance

- [test_cdlgssm_test_filter_TRegular.py](./test_cdlgssm_filter_TRegular.py) checks discrete and continuous-discrete Linear filtering algorithms with regularly sampled observations
    - I.e., it is a comparison between dynamax implementations of Kalman filter and cd-dynamax's CD-Kalman filter (as in Sarkka's Algorithm 3.16)
    - CD-Kalman filter matches discrete-time Kalman filter results
    - Note that after SGD learning, comparison between discrete and continuous-discrete models is not easy due to different parameterizations.
        - Although filtered means and covs are not exactly equal, plots showcase they are quite accurate in both models.

- [test_cdlgssm_test_smoother_TRegular.py](./test_cdlgssm_smoother_TRegular.py) checks discrete and continuous-discrete Linear smoothing algorithms with regularly sampled observations
    - I.e., it is a comparison between dynamax implementations of Kalman smoother and cd-dynamax's CD-Kalman smoother (as in Sarkka's Algorithm 3.17 and 3.18)
    - CD smoother type 1, as in Sarkka's Algorithm 3.17 matches discrete-time solution
    - CD smoother type 2, as in Sarkka's Algorithm 3.18 does not match discrete-time solutions
        - Performance is close though: are these related to differential equation solver differences?

- [test_cdnlgssm_test_filter_linear_TRegular.py](./test_cdnlgssm_filter_linear_TRegular.py) checks continuous-discrete Linear and Non-Linear filtering algorithms with regularly sampled observations
    1. A CDNLGSSM model with linearity assumptions is equivalent to a CDLGSSM model
        - Which can be computed based on both first and second order approximations to SDE (equivalent to linear SDEs)

    2. A CDNLGSSM model with EKF filtering provides same results as a KF with a CDLGSSM model
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

- [test_cdnlgssm_test_smoother_linear_TRegular.py](./test_cdnlgssm_smoother_linear_TRegular.py) checks continuous-discrete Linear and Non-Linear smoothing algorithms with regularly sampled observations
    1. We compare that a CDNLGSSM model with EKS smoothing (as in Sarkka's Algorithm 3.23) matches CD-linear-KS type 2 (as in Sarkka's Algorithm 3.18)
        - We notice that EKS smoothing (as in Sarkka's Algorithm 3.23) does not match CD-linear-KS type 1 (as in Sarkka's Algorithm 3.17)
            - Performance is close though: are these related to differential equation solver differences?
