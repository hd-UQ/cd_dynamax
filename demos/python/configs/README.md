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
- See the cd-dynamax config walkthrough in [../notebooks/cddynamax_experiment_config_tutorial.ipynb](../notebooks/cddynamax_experiment_config_tutorial.ipynb) 

## Format

In general, config files are INI-style, parsed section-by-section, with values evaluated as Python expressions
- e.g., `jnp.eye(state_dim)`, `LearnableVector(...)`, `ParameterProperties(trainable=True)`

**A config file is executable Python, not sandboxed data.** Only load configs from sources you trust.

Config files cross-reference each other by relative path, so a full experiment is assembled by picking one file per axis:
- a `data/` config points to the `model/` config used to generate "true" data (`true_model_config_file`)
- a `model/` config points to the `solver/` config used to integrate its dynamics (`solver_config_file`)
- a `filter/` config points to its own `solver/` config for the filter's internal integration (`diffeqsolve_settings_file`)
    - Note that this is not duplication of a `solver/` config: it is specifically for the filter's internal integration, which  can differ from the model's, e.g. to filter with a coarser/cheaper solver than the one used to generate data

## How configs are loaded and executed

Config files aren't parsed by a custom format engine. Each one is read with Python's `configparser.ConfigParser`, and then every value string is passed through `eval()`, evaluated in a namespace that already has cd-dynamax's models/utils, dynamax's bijectors, `diffrax as dfx`, and `optax` imported.
- That's the whole reason a config can write `jnp.eye(state_dim)`, `LearnableVector(...)`, `dfx.ConstantStepSize()`, or `optax.adam(1e-1)` directly, with no import statement of its own.

The functions doing this live in [`experiment_utils.py`](../../../cd_dynamax/src/utils/experiment_utils.py) (model/filter/solver configs) and [`data_generator.py`](../../../cd_dynamax/src/utils/data_generator.py) (data configs):

- **`create_cddynamax_model_from_config`**: reads a `model/` config, builds the model class, and `eval()`s every entry in `[initial_values]`.
    - If the config also has a `[prior]` section (optional, `prior_class_file` + `prior_init_key`), it dynamically imports the `.py` module named there (a `prior/` file) and instantiates it as the parameter prior used for MCMC-based fitting.
    
- **`create_cddynamax_filter_from_config`**: reads a `filter/` config.
    - The filter algorithm is inferred from whichever section appears **first** in the file (`KF`/`EKF`/`UKF`/`EnKF`/`DPF`), so that section must come before `[filter_info]`, not after.

- **`solver_settings_from_config`**: reads a `solver/` config into the `diffeqsolve_settings` dict passed to `diffrax.diffeqsolve`.

- **`mcmc_config_to_dict`**: reads the `[mcmc]` section of a `fitting/` config.

- **`generate_data_from_config`** (in `data_generator.py`): reads a `data/` config.
    - If its `data_save_file` already exists on disk, it loads that pickle instead of regenerating, so re-running a script is cheap unless you delete or rename the cached file.

- **`override_config`**: applies a `{"section.option": value}` dict on top of an already-parsed `ConfigParser`, before any `eval()`s happen.
    - This is what CLI flags use under the hood: e.g. `--enforce_twin_experiment` overrides `data_generation.true_model_config_file`, and `--data_key` overrides `data_generation.key`
    - This lets a script patch one field without editing the config file itself.

## What each config file controls

We describe below, for each config type:
- what it's for,
- the section header(s) it must contain, and
- the key attributes that live inside those sections.

### `data/`

- **Purpose**: how to generate (or, if desired, load) a trajectory to plot/analyze/filter. It needs
    - the time grid to sample on, and
    - which model acts as the "true" generator.
- **Sections**: `[data_generation]`, `[data_saving]`.
- **Key attributes**:
    - `[data_generation]`: PRNG `key`, time range (`t0`, `t1`), `num_samples`, `irregular_samples` (regular vs. randomly-spaced emission times), `true_model_config_file` (the `model/` config to sample from).
    - `[data_saving]`: `data_save_file`, where the generated trajectory is pickled.

### `model/`

- **Purpose**: specification of the CD-SSM to filter/fit
    - it defines the cd-dynamax class, state/space dimensions, and every parameter of the initial state, dynamics, and emission  distributions, each tagged as fixed or trainable.
    
    - cd-dynamax model parameters are defined following dynamax convention:
        - i.e, each parameter contains a `{"params": ..., "props": ...}` pair:
            - `params` is the value or a learnable function,
            - `props` is `ParameterProperties` (or learnable-object of them) marking it `trainable` and giving a `constrainer` (e.g. `RealToPSDBijector()` for covariances) for unconstrained-space optimization.

- **Sections**: `[model]`, `[initial_values]`.
    - Note that `[initial_values]` refers to the initial values of all CD-SSM parameters/props

- **Key attributes**:
    - `[model]`: `class_name` (`CDLGSSM`, `CDNLGSSM`, or `CDNLSSM`), `state_dim`, `emission_dim`, `solver_config_file`.
    - `[initial_values]`: one entry per parameter
        - `initial_mean`, `initial_cov`
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

## Annotated template examples

We provide here minimal, fully-commented examples for each config type.
- Use as starting point instead of reverse-engineering it from the working examples.

### `data/`

```ini
#### Data Config File
[data_generation]
key: 0                                          # PRNG key for reproducible sampling
t0: 0.0                                          # start time
t1: 10.0                                         # end time
num_samples: 1000                                # number of emission times to draw
irregular_samples: True                          # True: randomly-spaced emission times; False: a regular grid
true_model_config_file: model/true_l63_mech_x1   # model/ config used as the "true" generating process

[data_saving]
data_save_file: data/my_experiment_data.pkl      # where the generated trajectory is pickled -- also a cache: if this file already exists, it's loaded instead of regenerated
```

### `model/`

```ini
[model]
class_name: CDNLGSSM                             # CDLGSSM | CDNLGSSM | CDNLSSM
state_dim: 3
emission_dim: 1
solver_config_file: solver/dt1e-2_maxSteps1e5    # integrates the continuous-time dynamics

[initial_values]
# Every entry is a {"params": ..., "props": ...} pair:
#   params -> the value, or a learnable function
#   props  -> ParameterProperties (or a learnable-object of them), marking the entry
#             trainable and giving a constrainer for unconstrained-space optimization
initial_mean = {"params": jnp.zeros(state_dim),
    "props": ParameterProperties(trainable=False)}
initial_cov = {"params": 5.0 * jnp.eye(state_dim),
    "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())}

dynamics_drift = {"params": LearnableLorenz63_Drift(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
    "props": LearnableLorenz63_Drift(sigma=ParameterProperties(trainable=True),
                                      rho=ParameterProperties(trainable=True),
                                      beta=ParameterProperties(trainable=True))}
dynamics_diffusion_coefficient = {"params": jnp.eye(state_dim),
    "props": ParameterProperties(trainable=False)}

emission_cov = {"params": jnp.eye(emission_dim),
    "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())}

# Optional: a prior over trainable parameters, only needed for MCMC-based fitting
[prior]
prior_class_file: prior/l63_mech_drift_hi_info.py
prior_init_key: 0
```

### `solver/`

```ini
[diffeqsolve_settings]
solver: None                               # None defaults to Dopri5 (ODE) / Heun (SDE)
stepsize_controller: dfx.ConstantStepSize()
adjoint: dfx.RecursiveCheckpointAdjoint()   # vs. dfx.DirectAdjoint() -- a memory/speed trade-off for gradients through the solve
dt0: 0.01
max_steps: 1e5
tol_vbt: 5e-3
```

### `filter/`

```ini
[EKF]
dt_final: 1e-4
state_order: first                         # zeroth | first | second -- order of the Taylor approximation to the dynamics
emission_order: first
smooth_order: first
cov_rescaling: 1.0
diffeqsolve_settings_file: solver/dt1e-2_maxSteps1e5

[filter_info]
name=EKF (1st order)
desc=Extended Kalman Filter with first order approximation to state and emission functions
filtered_style = {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7}
forecasted_style = {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.7}
```
The filter-type section (here `[EKF]`) must be the *first* section in the file -- it's how the loader decides which filter class to build; `[filter_info]` must follow it.

### `fitting/`

```ini
[sgd]
optimizer: optax.adam(1e-1)
batch_size: 1
num_epochs: 1000
shuffle: False
return_param_history: True
key: 0
```
```ini
[mcmc]
type: nuts                                 # nuts | rmh | additive_step_random_walk
n_samples: 100
warmup_samples: 10
parameters: {}                             # sampler-specific, e.g. {'proposal': "blackjax.mcmc.random_walk.normal(0.1)"} for random-walk variants
verbose: True
key: 0
```

### `prior/`

```python
import jax.numpy as jnp
from cd_dynamax import Prior

class CDNLGSSM_Prior(Prior):
    def __init__(self, **kwargs):
        ...  # define one distribution per trainable parameter

    def sample(self, key, M):
        ...  # return M samples, in the same pytree shape as the trainable params

    def log_prob(self, x):
        ...  # return log p(x) for a pytree x shaped like the trainable params
```
