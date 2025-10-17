import os
from configparser import ConfigParser
import pickle

# CD-NLGSSM imports
from data_generator import generate_data_from_config
from cd_dynamax.src.utils.experiment_utils import *
from cd_dynamax.src.utils.simulation_utils import filter_and_forecast
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import update_params

# Demo plotting import
from demo_plot_utils import (
    plot_filter_then_forecast_state_results,
    plot_filter_then_forecast_state_emission_results,
)

# Main function to plot filtering and forecasting experiment
def plot_filter_then_forecast(
    data_config_file,
    model_config_file,
    filter_config_file,
    output_dir,
    T_filter=0.8,
    enforce_twin_experiment=False,
    ftf_key=None,
    overrides={},
):
    '''Plot a filtering and forecasting experiment with specified configurations.

    Args:
        data_config_file (str): Path to the data configuration file.
        model_config_file (str): Path to the model configuration file.
        filter_config_file (str): Path to the filter configuration file.
        output_dir (str): Directory to save the results.
        T_filter (float): Fraction of the data to use for filtering
            remaining data will be used for forecasting.
            (default: 0.8)
        enforce_twin_experiment (bool): If True, will enforce the twin experiment setup.
            Done by resetting the true_model_config_file in the data_config_file = model_config_file.
        overrides (dict): Dictionary of configuration overrides.
    '''
    # Figure out the directory structure
    results_dir = build_results_dir(
        output_dir,
        data_config_file,
        model_config_file,
        filter_config_file,
        overrides=overrides,
    )

    # Check if the results directory exists, if not, print a warning
    if not os.path.exists(results_dir):
        print("Warning: Results directory does not exist at:", results_dir)
        print("Please run the filtering and forecasting experiment first to generate results.")
        return

    if enforce_twin_experiment:
        # Override the true_model_config_file in data_config_file with the model's config file
        overrides['data_generation.true_model_config_file'] = model_config_file
        print(f"Enforcing twin experiment by setting 'data_generation.true_model_config_file' to {model_config_file}.")
    
    # Load data: the same function used for generation is used, it should simply load existing data
    data, data_key = generate_data_from_config(
        data_config_file=data_config_file,
        data_save_file=None,
        overrides=overrides,
    )

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams, filter_info = create_cdnlgssm_filter_from_config(
        filter_config_file,
        overrides=overrides
    )

    # Assign forecasting and filtering time indices based on T_filter fraction.
    T0 = data['t_emissions'][0]
    T_forecast_end = data['t_emissions'][-1]
    T_filter_end = T0 + T_filter * (T_forecast_end - T0)

    # Set the key(s) for filtering and forecasting
    ftf_keys = [data_key + 10] if ftf_key is None else ftf_key  # i.e. 10 more than data_key or given ftf_key

    # Run filtering and forecasting, for each key in ftf_keys
    for ftf_key in ftf_keys if isinstance(ftf_keys, (list)) else [ftf_keys]:
        print(f"Loading {filter_info['name']} filtering from T={T0} up to T={T_filter_end} and forecasting up to T={T_forecast_end} (with ftf_key={ftf_key}) results...")

        # Load the results
        results_file = os.path.join(results_dir, 'results_ftfkey{}.pkl'.format(ftf_key))
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        # Print a message indicating the completion of the experiment
        print("Results loaded from:", results_file)

        # Plot the filtering and forecasting state results, along with their MSE
        plot_filter_then_forecast_state_results(
            data=data,
            results=results,
            results_file=results_file,
            filter_info=filter_info,
            plot_uncertainty=True,
            plot_observations=True,
            plot_mse=True,
        )

        # Plot the filtering and forecasting state and emission results
        plot_filter_then_forecast_state_emission_results(
            data=data,
            results=results,
            results_file=results_file,
            filter_info=filter_info,
            plot_uncertainty=True,
            plot_observations=True,
        )

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    
    # Create optional flags for comamand line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot the filter then forecast experiment with specified configurations.')
    parser.add_argument('--data_config_file', type=str, default='configs/data/true_l63_data_x1',
                        help='Data configuration file: (default: true_l63_data_x1)')
    parser.add_argument('--model_config_file', type=str, default='configs/model/true_l63_mech_x1',
                        help='Model configuration file: (default: true_l63_mech_x1)')
    parser.add_argument('--enforce_twin_experiment', type=bool, default=False,
                        help='If True, will enforce the twin experiment setup (default: False). Done by resetting the true_model_config_file in the data_config_file = model_config_file.')
    # Allow user to specify multiple filter config files or "all"
    parser.add_argument('--filter_config_file', nargs='+', default='configs/filter/ekf_StateFirst_EmissionsFirst', 
                        help='Filter configuration file: (default: ekf_StateFirst_EmissionsFirst), if all filters should be used, set to "all"'
                        )
    parser.add_argument('--output_dir', type=str, default='results/filter_then_forecast',
                        help='Output directory for results: (default: results/filter_then_forecast)')
    parser.add_argument('--T_filter', type=float, default=0.8,
                        help='Fraction of the data to use for filtering (default: 0.8)')   
    # Add optional data_key and ftf_key arguments, which can be a sequence of integers
    parser.add_argument('--data_key', type=int, nargs='+', default=None,
                        help='Optional key to use for data generation. If None, the key from the data config file will be used.')
    parser.add_argument('--ftf_key', type=int, nargs='+', default=None,
                        help='Key to use for filter-then-forecast. Default is None, in which case it will be set to data_key + 10.')
    args = parser.parse_args()

    # Process the filter_config_file argument to allow running multiple filter files
    if len(args.filter_config_file) == 1:
        filter_config_files = args.filter_config_file

        if filter_config_files[0].lower() == "all":
            filter_config_files = [
                "configs/filter/ekf_StateFirst_EmissionsFirst",
                "configs/filter/ekf_StateSecond_EmissionsFirst",
                "configs/filter/ekf_StateZeroth_EmissionsFirst",
                "configs/filter/enkf_StateFirst",
                "configs/filter/enkf_StateZero",
                "configs/filter/ukf_StateFirst",
                "configs/filter/ukf_StateZeroth",
            ]
    else:
        filter_config_files = args.filter_config_file

    # Iterate over filter config files
    print("Plotting filtering and forecasting experiment...")
    for filter_config_file in filter_config_files:
        # Prepare overrides dictionary for this run
        overrides = {}
        if args.data_key is not None:
            for data_key in args.data_key:
                overrides['data_generation.key'] = data_key

                print(f"\t with: {filter_config_file} and overrides: {overrides}")
                plot_filter_then_forecast(
                    data_config_file=args.data_config_file,
                    model_config_file=args.model_config_file,
                    filter_config_file=filter_config_file,
                    output_dir=args.output_dir,
                    T_filter=args.T_filter,
                    enforce_twin_experiment=args.enforce_twin_experiment,
                    ftf_key=args.ftf_key,
                    overrides=overrides
                )
        else:
            print(f"\t with: {filter_config_file} and no overrides")
            plot_filter_then_forecast(
                data_config_file=args.data_config_file,
                model_config_file=args.model_config_file,
                filter_config_file=filter_config_file,
                output_dir=args.output_dir,
                T_filter=args.T_filter,
                enforce_twin_experiment=args.enforce_twin_experiment,
                ftf_key=args.ftf_key,
            )