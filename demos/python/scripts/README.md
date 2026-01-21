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

    - An example to run all (pre-defined) filters in one simulation

    ```bash
    python3 run_filter_then_forecast.py --filter_config_file all --data_key 10 --ftf_key 10
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

- To compare all executed filters and plot their results, execute

```bash
python3 compare_filter_then_forecast.py --filter_config_file all --data_key 10 --ftf_key 10
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

