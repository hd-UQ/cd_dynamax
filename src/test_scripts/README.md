- [cdlgssm_test_filter_TRegular.py](./cdlgssm_test_filter_TRegular.py) checks discrete and continuous-discrete Linear filtering algorithms with regularly sampled observations
    - Note that after SGD learning, comparison between discrete and continuous-discrete models is not easy due to different parameterizations.
        - Although filtered means and covs are not exactly equal, plots showcase they are quite accurate in both models.

- [cdlgssm_test_smoother_TRegular.py](./cdlgssm_test_smoother_TRegular.py) checks discrete and continuous-discrete Linear smoothing algorithms with regularly sampled observations
    - CD smoother type 1, as in Sarkka's Algorithm 3.17 matches discrete-time solution
    - CD smoother type 2, as in Sarkka's Algorithm 3.18 does not match discrete-time solutions
        - Performance is close though: are these related to differential equation solver differences?

- [cdnlgssm_test_filter_linear_TRegular.py](./cdnlgssm_test_filter_linear_TRegular.py) checks continuous-discrete Linear and Non-Linear filtering algorithms with regularly sampled observations
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

- [cdnlgssm_test_smoother_linear_TRegular.py](./cdnlgssm_test_smoother_linear_TRegular.py) checks continuous-discrete Linear and Non-Linear smoothing algorithms with regularly sampled observations
    1. We compare that a CDNLGSSM model with EKS smoothing (as in Sarkka's Algorithm 3.23) matches CD-linear-KS type 2 (as in Sarkka's Algorithm 3.18)
        - We notice that EKS smoothing (as in Sarkka's Algorithm 3.23) does not match CD-linear-KS type 1 (as in Sarkka's Algorithm 3.17)
            - Performance is close though: are these related to differential equation solver differences?