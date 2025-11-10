# cd-dynamax codebase

The `cd-dynamax` codebase extends the `dynamax` library to support continuous-discrete state space models, where observations are made at specified discrete times rather than at regular intervals.

The codebase is organized into several key directories:
```cd_dynamax/
├── src/                       # Source code for cd-dynamax library
│   ├── continuous_discrete_linear_gaussian_ssm/  # CD-LGSSM models and algorithms
│   ├── continuous_discrete_nonlinear_gaussian_ssm/ # CD-NLGSSM models and algorithms
│   ├── ssm_temissions.py      # Modified SSM class for discrete emissions
│   └── utils/               # Utility functions and example models
├── dynamax/                     # Original dynamax library (as a submodule)
```

The `src` directory contains the main implementation of continuous-discrete state space models, including both linear and nonlinear Gaussian models, along with their associated filtering and smoothing algorithms.

The `dynamax` directory contains the original `dynamax` library, which is used as a foundation for building the `cd-dynamax` extensions.
    - Precisely, we have included `dynamax` as in git submodule, to ensure compatibility and ease of updates.
    - We are currently using `dynamax` version `v0.1.5`.

## Key Imports from Dynamax

- We build upon the existing `dynamax` library, following the design choices made there.
    - Specifically, we adhere to the same coding style, type annotations, and functional programming paradigms (e.g., use of `lax.scan`, `jit`, etc.).

- To maintain modularity and avoid unnecessary dependencies, we import from `dynamax` in a targeted manner, importing only key components as needed.

- Specifically, we import the following from `dynamax`:
    - Types and parameters
        - from dynamax.types import PRNGKey, Scalar
        - from dynamax.parameters import ParameterSet, PropertySet

    - utils
        - from dynamax.parameters import to_unconstrained, from_unconstrained, log_det_jac_constrain
        - from dynamax.utils.utils import ensure_array_has_batch_dim, fallback_hessian
        - from dynamax.utils.utils import psd_solve
        - from dynamax.utils.bijectors import RealToPSDBijector
        - from dynamax.utils.utils import pytree_len

    - classes
        - from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions
        - from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

    - optimizations
        - from dynamax.utils.optimize import run_sgd as dynamax_run_sgd
        - from dynamax.utils.optimize import sample_minibatches