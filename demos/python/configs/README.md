# Configuration Files for cd-dynamax

This directory contains configuration files used for the cd-dynamax codebase in general, and for the demos in particular.

These configuration files define various parameters and settings for continuous-discrete state space models, filtering and smoothing algorithms, and other functionalities provided by the cd-dynamax library.

Specifically, there are configuration files provided according to the following structure:

```demos/python/configs/
├── README.md                  # This file
├── data/                    # Configuration files for data generation
├── model/                   # Configuration files for model definitions
├── solver/                  # Configuration files for SDE solver settings, as per difrax
├── filter/                  # Configuration files for filtering algorithms
├── fitting/                 # Configuration files for fitting (parameter learning) algorithms
├── prior/                   # Configuration files for prior settings   
```

Users can modify these configuration files to customize the behavior of the cd-dynamax library for their specific use cases.

When running demos or scripts, the appropriate configuration files can be loaded to set up the desired models and algorithms.

## Format

In general, config files are INI-style, parsed section-by-section, with values evaluated as Python expressions
    - e.g., `jnp.eye(state_dim)`, `LearnableVector(...)`, `ParameterProperties(trainable=True)`

Config files cross-reference each other by relative path, so a full experiment is assembled by picking one file per axis:
- a `data/` config points to the `model/` config used to generate "true" data (`true_model_config_file`)
- a `model/` config points to the `solver/` config used to integrate its dynamics (`solver_config_file`)
- a `filter/` config points to its own `solver/` config for the filter's internal integration (`diffeqsolve_settings_file`)
    - Note that this is not duplication of a `solver/` config: it is specifically for the filter's internal integration, which  can differ from the model's, e.g. to filter with a coarser/cheaper solver than the one used to generate data

## What each config file controls

We describe below, for each config type:
    - what it's for,
    - the section header(s) it must contain, and
    - the key attributes that live inside those sections.

### `data/`

- **Purpose**: how to generate (or, if desired, load) a trajectory to feed into filtering/fitting
    - the time grid to sample on and which model acts as the "true" generator.
- **Sections**: `[data_generation]`, `[data_saving]`.
- **Key attributes**:
    - `[data_generation]`: PRNG `key`, time range (`t0`, `t1`), `num_samples`, `irregular_samples` (regular vs. randomly-spaced emission times), `true_model_config_file` (the `model/` config to sample from).
    - `[data_saving]`: `data_save_file`, where the generated trajectory is pickled.

### `model/`

- **Purpose**: specification of the CD-SSM to filter/fit
    - its class, dimensions,  and every parameter of the initial state, dynamics, and emission  distributions, each tagged as fixed or trainable.
    
    - cd-dynamax model parameters are defined following dynamax convention:
        - i.e, each parameter contains a `{"params": ..., "props": ...}` pair:
            - `params` is the value or a learnable function,
            - `props` is `ParameterProperties` (or learnable-object of them) marking it `trainable` and giving a `constrainer` (e.g. `RealToPSDBijector()` for covariances) for unconstrained-space optimization.

- **Sections**: `[model]`, `[initial_values]`.
    - Note that `[initial_values]` refers to the initial values of all CD-SSM parameter

- **Key attributes**:
    - `[model]`: `class_name` (`CDLGSSM`, `CDNLGSSM`, or `CDNLSSM`), `state_dim`, `emission_dim`, `solver_config_file`.
    - `[initial_values]`: one entry per parameter
        - (`initial_mean`, `initial_cov`)
        - `dynamics_weights`/`dynamics_drift`, `dynamics_diffusion_coefficient`, `dynamics_diffusion_cov`,
        - `emission_weights`, `emission_cov`

### `solver/`

- **Purpose**: numerical settings for integrating the continuous-time dynamics
    - used both when generating "true" data and when a model/filter  pushes its state forward between observations.
- **Sections**: `[diffeqsolve_settings]`.
- **Key attributes**: passed straight to `diffrax.diffeqsolve`
    - `solver` (`None` defaults to Dopri5 for ODEs / Heun for SDEs),
    - `stepsize_controller`, `adjoint` (e.g. `RecursiveCheckpointAdjoint` vs. `DirectAdjoint`, trading off memory vs. speed for gradients through the solve),
    - `dt0`,
    - `max_steps`.

### `filter/`

- **Purpose**: which filtering/smoothing algorithm to run and its hyperparameters
    - plus display metadata for the plotting scripts.
- **Sections**: one named after the filter (`[KF]`, `[EKF]`, `[UKF]`, `[EnKF]`, `[DPF]`), plus `[filter_info]`.
- **Key attributes**:
    - filter section: algorithm-specific hyperparameters, e.g., 
        - EKF's `state_order`/`emission_order`;
        - DPF's `N_particles`/`proposal_method`/ `resample_method`),
        - plus `diffeqsolve_settings_file`.
    - `[filter_info]`:
        - display `name`/`desc`, and matplotlib style dicts (`filtered_style`, `forecasted_style`) the plotting scripts use to distinguish filters on shared plots.

### `fitting/`

- **Purpose**: which parameter-learning method to run and its hyperparameters
    - point estimation via SGD or scipy
    - Bayesian inference via MCMC.
- **Sections**: one named after the method
    - `[sgd]`, `[scipy]`, `[scipy_jaxopt]`,
    - or `[mcmc]` for every MCMC variant.
- **Key attributes**:
    - `[sgd]`: `optimizer`, `batch_size`, `num_epochs`.
    - `[scipy]` / `[scipy_jaxopt]`: `method`, `options`.
    - `[mcmc]`:
        - `type` selects the sampler (`nuts`, `rmh`, `additive_step_random_walk`, ...),
        - plus `n_samples`, `warmup_samples`, `key`, and
        - sampler-specific `parameters` (e.g. a `proposal` for random-walk variants).

### `prior/`

- **Purpose**: the prior over trainable parameters used during MCMC-based fitting.
    - This is the one exception to the INI-style configuration.
    - Each file is a plain `.py` module defining a class that subclasses `cd_dynamax`'s `Prior`.
- **Sections**: none
- **Key attributes**:
    - `sample(key, M)` and `log_prob(x)`, implemented directly in code rather than declared as static values, since priors need
  arbitrary sampling logic.
