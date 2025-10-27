# Example python-scripts to use cd-dynamax 

## Filter and forecast examples

- These are examples to filter observed data from cd-dynamax, using a given filter (as specified via --filter_config)

```bash
python3 run_filter_then_forecast.py --filter_config_file configs/filter/ekf_StateFirst_EmissionsFirst --ftf_key 0 1 2
```

```bash
python3 run_filter_then_forecast.py --filter_config_file configs/filter/ekf_StateFirst_EmissionsFirst configs/filter/enkf_StateFirst --ftf_key 0 1 2
```

```bash
python3 run_filter_then_forecast.py --filter_config_file configs/filter/enkf_StateFirst
```

```bash
python3 run_filter_then_forecast.py --filter_config_file configs/filter/ekf_StateFirst_EmissionsFirst --data_key 0 1 2
```

```bash
python3 -m pdb run_filter_then_forecast.py --filter_config_file all --data_key 10 --ftf_key 10
```

### Plotting the filter and forecast results

- Simply run the `plot_filter_then_forecast.py` with same configs as when using `run_filter_then_forecast.py script

```bash
```

- Comparing all executed filters is as simple as 

```bash
python3 -m compare_filter_then_forecast.py --filter_config_file all --data_key 10 --ftf_key 10
```

# Parameter learning: i.e., fitting a cd-dynamax to observed data

## Using SGD

- Run the parameter learning `fit_model_to_data.py` script, with corresponding fit_sgd config file
```bash
python3 -m pdb fit_model_to_data.py --fit_config_file configs/fitting/fit_sgd
```

- Parameter learning results can be plotted using the `plot_fitted_model.py` script, with same config files as above
```bash
python3 -m pdb plot_fitted_model.py --fit_config_file configs/fitting/fit_sgd
```

## Using MCMC - Nuts

- Run the parameter learning `fit_model_to_data.py` script, with corresponding fit_nuts config file
```bash
python3 -m pdb fit_model_to_data.py --fit_config_file configs/fitting/fit_nuts
```

- Parameter learning results can be plotted using the `plot_fitted_model.py` script, with same config files as above
```bash
python3 -m pdb plot_fitted_model.py --fit_config_file configs/fitting/fit_nuts
```

## Scipy

- Run the parameter learning `fit_model_to_data.py` script, with corresponding fit_nuts config file
```bash
python3 -m pdb fit_model_to_data.py --fit_config_file configs/fitting/fit_scipy
```

- Parameter learning results can be plotted using the `plot_fitted_model.py` script, with same config files as above
```bash
python3 -m pdb plot_fitted_model.py --fit_config_file configs/fitting/fit_scipy
```

