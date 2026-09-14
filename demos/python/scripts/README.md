# Overview: a config-driven experimentation harness

None of these scripts hardcode a model, filter, or optimizer.

Each one takes `--data_config_file`, `--model_config_file`, `--filter_config_file`, and (for fitting) `--fit_config_file` flags
    - These all point to files under [`../configs/`](../configs/README.md) (see there for what each config type contains)
    
The general flow of these scripts at runtime:

    1. generates or loads the data (`data/`),
    2. builds the cd-dynamax model object (`model/`)
        - which can be a     CD-LGSSM, CD-NLGSSM, or CD-NLSSM depending on the config's `class_name`,
    3. builds the filter (`filter/`)
        - and, for fitting scripts, the optimizer (`fitting/`),
    4. runs filtering/forecasting or parameter learning, and saves results.

## Design Motivation & Rationale

Swapping any one axis (e.g., a different filter, a stiffer solver, a different optimizer)
is a matter of **pointing to a different config file, not editing code.**

This is what makes it a *harness*:
    - the same driver script covers every model family and algorithm combination the config axes can express, so filter/model/solver/fitting comparisons are run identically and
are directly comparable.

## Mechanics worth knowing before the examples

A few mechanics worth knowing before diving into the examples below:

- **Results are keyed by the config combination.** `run_*` scripts write results under a directory built from the basenames of the config files  used
    - see `build_results_dir` in [`experiment_utils.py`](../../../cd_dynamax/src/utils/experiment_utils.py)),
    - This implies re-running the same combination lands in the same place, and different combinations naturally sit side by side for comparison.

- **The `plot_*` /  `compare_*` scripts don't recompute anything**
    - they take the *same* config flags as the `run_*` call and just locate and visualize its pickled output.

- **Seeds are decoupled.**
    - `--data_key` seeds synthetic data generation
    - `--ftf_key` (filter-then-forecast) or the fitting script's own key seeds filtering/forecasting or SGD minibatching, independently of the data draw
    
    - so you can check whether a result is robust to the random draw used at inference time without regenerating the underlying data.

- **Twin experiments** The `--enforce_twin_experiment` flag forces the data-generating model config to match the model config being fit/filtered 
    - Hence, sets up the the standard check that an inference procedure recovers known ground truth before trusting it on real data.
    
- **`--filter_spec` decides how the filter is chosen**:
    - `model` (default)  picks the filter implied by the model's class,
    - `filter` uses whatever the filter config says explicitly
        - this is relevant when testing a nonlinear filter (EKF/UKF/EnKF/DPF) against a model it wasn't specifically designed for.

- **Filters can be swept in one call**:
    - The `--filter_config_file all` flag expands  to a predefined list of CD-NLGSSM filters (EKF variants, EnKF, UKF), and `compare_filter_then_forecast.py` overlays all of their results on shared plots.

# Example python-scripts on how to use cd-dynamax for different goals

## Filter and forecasting

- These are example script-based runs to filter observed data using cd-dynamax, based on a given filter (as specified via --filter_config)

    - Kalman Filtering and forecasting with a continuous-discrete Gaussian State-space model data
    
    ```bash
    python3 run_filter_then_forecast.py --data_config_file data/true_cdlgssm_data_x1 --model_config_file model/cdlgssm_x1 --filter_config_file filter/kf
    ```

    - Filtering and forecasting a continuous-discrete Lorenz 63 model with EKF and EnKF filters
    
    ```bash
    python3 run_filter_then_forecast.py --filter_config_file filter/ekf_StateFirst_EmissionsFirst
    ```

    ```bash
    python3 run_filter_then_forecast.py --filter_config_file filter/ekf_StateFirst_EmissionsFirst --data_key 0 1 2
    ```

    ```bash
    python3 run_filter_then_forecast.py --filter_config_file filter/enkf_StateFirst --ftf_key 0 1 2
    ```

    - An example to run all (pre-defined) CD-NLGSSM filters in one simulation

    ```bash
    python3 run_filter_then_forecast.py --filter_config_file all --data_key 10 --ftf_key 10
    ```

- An example of a Differentiable Particle Filter for a CD-NLSSM model
```bash
python3 run_filter_then_forecast.py --data_config_file data/true_oudrift_poissondata --model_config_file model/cdnlssm_oudrift_poissondata --filter_config_file filter/dpf_bootstrap_stopg
```
    
### Plotting the filter and forecast results

- To plot results from the above scripts, run the `plot_filter_then_forecast.py` with the same config files as when using `run_filter_then_forecast.py script
    
    - For instance, to plot results for the KF run above    
    ```bash
    python3 plot_filter_then_forecast.py --data_config_file data/true_cdlgssm_data_x1 --model_config_file model/cdlgssm_x1 --filter_config_file filter/kf
    ```
    
    - For instance, to plot results for one of the EKFs above run
    
    ```bash
    python3 plot_filter_then_forecast.py --filter_config_file filter/ekf_StateFirst_EmissionsFirst
    ```

- To compare all executed CD-NLGSSM filters and plot their results, execute

```bash
python3 compare_filter_then_forecast.py --filter_config_file all --data_key 10 --ftf_key 10
```

- To plot the example of a Differentiable Particle Filter for a CD-NLSSM model
```bash
python3 plot_filter_then_forecast.py --data_config_file data/true_oudrift_poissondata --model_config_file model/cdnlssm_oudrift_poissondata --filter_config_file filter/dpf_bootstrap_stopg
```

# Parameter learning: i.e., fitting a cd-dynamax model to observed data

## Using SGD

- Run the parameter learning `run_fit_model_to_data.py` script, with corresponding fit_sgd config file and default filter

    - For a CD-LGSSM model
    
    ```bash
    python3 run_fit_model_to_data.py --data_config_file data/true_cdlgssm_data_x1 --model_config_file model/cdlgssm_x1_fit_to_data --filter_config_file filter/kf --fit_config_file fitting/fit_sgd_cdlgssm_x1 
    ```
    
    - For a Lorenz 63 CD-NLGSSM model with EKF (defaults in script) 
    
    ```bash
    python3 run_fit_model_to_data.py --fit_config_file fitting/fit_sgd
    ```
    
    - For a Lorenz 63 CD-NLGSSM model with selected EnKF filter

    ```bash
    python3 run_fit_model_to_data.py --fit_config_file fitting/fit_sgd --filter_config_file filter/enkf_StateFirst
    ```

- Parameter learning results can be plotted using the `plot_fitted_model.py` script, with same config files as above, for the SGD-based learning:
    
    - For the CD-LGSSM model
    
    ```bash
    python3 plot_fitted_model.py --data_config_file data/true_cdlgssm_data_x1 --model_config_file model/cdlgssm_x1_fit_to_data --filter_config_file filter/kf --fit_config_file fitting/fit_sgd_cdlgssm_x1 
    ```
    
    - For the Lorenz63 CD-NLGSSM model with EKF
    ```bash
    python3 plot_fitted_model.py --fit_config_file fitting/fit_sgd
    ```

## Using MCMC - Nuts

- Run the parameter learning `run_fit_model_to_data.py` script, with corresponding fit_nuts config file
```bash
python3 run_fit_model_to_data.py --fit_config_file fitting/fit_nuts
```

- Parameter learning results can be plotted using the `plot_fitted_model.py` script, with same config files as above, for the NUTS-based learning
```bash
python3 plot_fitted_model.py --fit_config_file fitting/fit_nuts
```

## Scipy

- Run the parameter learning `run_fit_model_to_data.py` script, with corresponding fit_scipy config file
```bash
python3 run_fit_model_to_data.py --fit_config_file fitting/fit_scipy
```

- Parameter learning results can be plotted using the `plot_fitted_model.py` script, with same config files as above
```bash
python3 plot_fitted_model.py --fit_config_file fitting/fit_scipy
```

