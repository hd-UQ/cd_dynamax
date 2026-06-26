## Overview

The `cd_dynamax` API is organized into a few main components:

- A Base SSM Class
- Continuous-Discrete Linear Gaussian State-Space Models
    - Including inference via the Kalman filter
- Continuous-discrete nonlinear Gaussian state-space models
    - Including inference via the Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Ensemble Kalman Filter (EnKF).
- Continuous-discrete nonlinear state-space models with generic initial/emission distributions
    - Including inference via the differentiable particle filter (DPF).
- Utilities for simulation, optimization, evaluation, and plotting
