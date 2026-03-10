import os
import pickle

# cd-dynamax imports
from cd_dynamax.src.utils.data_generator import generate_data_from_config
from cd_dynamax.src.utils.experiment_utils import (
    build_results_dir,
    create_cddynamax_model_from_config,
    create_cddynamax_filter_from_config,
)
from cd_dynamax.src.utils.simulation_utils import (
    filter_and_forecast,
    cddynamax_emissions,
    tree_to_dict,
)


def run_filter_then_forecast(
    config_path,
    data_config_file,
    model_config_file,
    filter_config_file,
    output_dir,
    T_filter=0.8,
    enforce_twin_experiment=False,
    ftf_key=None,
    overrides={},
    filter_spec="model"
):
    """Run a filtering and forecasting experiment with specified configurations.

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
        filter_spec (str): Whether to decide the filtering type based on the model or the filter. Options are 'model' or 'filter' (default: 'model'). If 'model', will use the model type to decide what filter to use. If 'filter', will use the filter type to decide.
    """
    # Figure out and build the directory structure
    results_dir = build_results_dir(
        output_dir,
        data_config_file,
        model_config_file,
        filter_config_file,
        overrides=overrides,
    )

    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    print("Results directory created at:", results_dir)

    if enforce_twin_experiment:
        # Override the true_model_config_file in data_config_file with the model's config file
        overrides["data_generation.true_model_config_file"] = model_config_file
        print(
            f"Enforcing twin experiment by setting 'data_generation.true_model_config_file' to {model_config_file}."
        )

    # Generate or load data
    data, data_key = generate_data_from_config(
        config_path=config_path,
        data_config_file=data_config_file,
        data_save_file=None,
        overrides=overrides,
    )

    # Create and initialize the cd-dynamax model from the model config file
    model, params, props = create_cddynamax_model_from_config(
        config_path=config_path,
        true_model_config_file=model_config_file,
        overrides=overrides,
    )

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams, filter_info = create_cddynamax_filter_from_config(
        config_path=config_path,
        filter_config_file=filter_config_file,
        overrides=overrides,
    )

    # Assign forecasting and filtering time indices based on T_filter fraction.
    T0 = data["t_emissions"][0]
    T_forecast_end = data["t_emissions"][-1]
    T_filter_end = T0 + T_filter * (T_forecast_end - T0)

    # Set the key(s) for filtering and forecasting
    ftf_keys = (
        [data_key + 10] if ftf_key is None else ftf_key
    )  # i.e. 10 more than data_key or given ftf_key

    # Run filtering and forecasting, for each key in ftf_keys
    for ftf_key in ftf_keys if isinstance(ftf_keys, (list)) else [ftf_keys]:
        print(
            "Running filtering with {filter_name} from T={T0} up to T={T_filter_end} and forecasting up to T={T_forecast_end} (with ftf_key={ftf_key}).".format(
                filter_name=filter_info["name"]
                if filter_info is not None and "name" in filter_info
                else "the specified filter",
                T0=T0,
                T_filter_end=T_filter_end,
                T_forecast_end=T_forecast_end,
                ftf_key=ftf_key,
            )
        )
        (
            filtered,
            forecasted,
            start_idx_filter,
            stop_idx_filter,
            start_idx_forecast,
            stop_idx_forecast,
        ) = filter_and_forecast(
            model_params=params,
            filter_hyperparams=filter_hyperparams,
            t_emissions=data["t_emissions"],
            emissions=data["emissions"],
            T0=T0,
            T_filter_end=T_filter_end,
            T_forecast_end=T_forecast_end,
            key=ftf_key,
            filter_spec=filter_spec
        )

        # Compute emissions for filtered and forecasted states
        filtered_emissions, forecasted_emissions = cddynamax_emissions(
            model=model,
            model_params=params,
            t_emissions_filter=data["t_emissions"][:stop_idx_filter],
            filtered_state=filtered,
            t_emissions_forecast=data["t_emissions"][start_idx_forecast:stop_idx_forecast],
            forecasted_state=forecasted,
            inputs_state=None,
            inputs_forecast=None
        ):


        # Convert the filtered and forecasted results to dictionaries, and add additional fields.
        filtered_dict = tree_to_dict(filtered)
        filtered_dict["filtered_emissions_means"] = filtered_emissions.mean
        filtered_dict["filtered_emissions_covariances"] = filtered_emissions.cov
        forecasted_dict = tree_to_dict(forecasted)
        forecasted_dict["forecasted_emissions_means"] = forecasted_emissions.mean
        forecasted_dict["forecasted_emissions_covariances"] = forecasted_emissions.cov

        # Save the results as a dictionary
        results = {
            "filtered": filtered_dict,
            "forecasted": forecasted_dict,
            "start_idx_filter": start_idx_filter,
            "stop_idx_filter": stop_idx_filter,
            "start_idx_forecast": start_idx_forecast,
            "stop_idx_forecast": stop_idx_forecast,
        }

        # Save the results to a file
        with open(
            os.path.join(results_dir, "results_ftfkey{}.pkl".format(ftf_key)), "wb"
        ) as f:
            pickle.dump(results, f)

        # Print a message indicating the completion of the experiment
        print("Experiment completed. Results saved to:", results_dir)


# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    # Create optional flags for comamand line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a filter then forecast with specified configurations."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="../configs/",
        help="Path to configuration files: (default: ../configs/)",
    )
    parser.add_argument(
        "--data_config_file",
        type=str,
        default="data/true_l63_data_x1",
        help="Data configuration file: (default: true_l63_data_x1)",
    )
    parser.add_argument(
        "--model_config_file",
        type=str,
        default="model/true_l63_mech_x1",
        help="Model configuration file: (default: true_l63_mech_x1)",
    )
    parser.add_argument(
        "--enforce_twin_experiment",
        type=bool,
        default=False,
        help="If True, will enforce the twin experiment setup (default: False). Done by resetting the true_model_config_file in the data_config_file = model_config_file.",
    )
    # Allow user to specify multiple filter config files or "all"
    parser.add_argument(
        "--filter_config_file",
        nargs="+",
        default="filter/ekf_StateFirst_EmissionsFirst",
        help='Filter configuration file: (default: ekf_StateFirst_EmissionsFirst), if all filters should be used, set to "all"',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/filter_then_forecast",
        help="Output directory for results: (default: results/filter_then_forecast)",
    )
    parser.add_argument(
        "--T_filter",
        type=float,
        default=0.8,
        help="Fraction of the data to use for filtering (default: 0.8)",
    )
    # Add optional data_key and ftf_key arguments, which can be a sequence of integers
    parser.add_argument(
        "--data_key",
        type=int,
        nargs="+",
        default=None,
        help="Optional key to use for data generation. If None, the key from the data config file will be used.",
    )
    parser.add_argument(
        "--ftf_key",
        type=int,
        nargs="+",
        default=None,
        help="Key to use for filter-then-forecast. Default is None, in which case it will be set to data_key + 10.",
    )
    # Decide what defines filtering type: model or filter
    parser.add_argument(
        "--filter_spec",
        type=str,
        default="model",
        help="Whether to decide the filtering type based on the model or the filter. Options are 'model' or 'filter' (default: 'model'). If 'model', will use the model type to decide what filter to use. If 'filter', will use the filter type to decide.",
    )

    args = parser.parse_args()

    # Process the filter_config_file argument to allow running multiple filter files
    if len(args.filter_config_file) == 1:
        filter_config_files = args.filter_config_file

        if filter_config_files[0].lower() == "all":
            filter_config_files = [
                "filter/ekf_StateFirst_EmissionsFirst",
                "filter/ekf_StateSecond_EmissionsFirst",
                "filter/ekf_StateZeroth_EmissionsFirst",
                "filter/enkf_StateFirst",
                "filter/enkf_StateZero",
                "filter/ukf_StateFirst",
                "filter/ukf_StateZeroth",
            ]
    else:
        filter_config_files = args.filter_config_file

    # Iterate over filter config files
    print("Running filtering and forecasting experiment...")
    for filter_config_file in filter_config_files:
        # Prepare overrides dictionary for this run
        overrides = {}
        if args.data_key is not None:
            for data_key in args.data_key:
                overrides["data_generation.key"] = data_key

                print(f"\t with: {filter_config_file} and overrides: {overrides}")
                run_filter_then_forecast(
                    config_path=args.config_path,
                    data_config_file=args.data_config_file,
                    model_config_file=args.model_config_file,
                    filter_config_file=filter_config_file,
                    output_dir=args.output_dir,
                    T_filter=args.T_filter,
                    enforce_twin_experiment=args.enforce_twin_experiment,
                    ftf_key=args.ftf_key,
                    overrides=overrides,
                    filter_spec=args.filter_spec
                )
        else:
            print(f"\t with: {filter_config_file} and no overrides")
            run_filter_then_forecast(
                config_path=args.config_path,
                data_config_file=args.data_config_file,
                model_config_file=args.model_config_file,
                filter_config_file=filter_config_file,
                output_dir=args.output_dir,
                T_filter=args.T_filter,
                enforce_twin_experiment=args.enforce_twin_experiment,
                ftf_key=args.ftf_key,
                filter_spec=args.filter_spec
            )
