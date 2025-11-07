import os
import argparse
from configparser import ConfigParser
import pickle

# CD-NLGSSM imports
from cd_dynamax.src.utils.experiment_utils import *
from cd_dynamax.src.utils.data_generator import generate_data_from_config

# Demo plotting import
from parameter_learning_plotting_utils import *

# Main function to plot fitted model to data
def plot_fitted_model_to_data(
        config_path,
        data_config_file,
        model_config_file,
        filter_config_file,
        fit_config_file,
        output_dir,
        fit_key=None,
        overrides={},
    ):
    # Figure out and build the directory structure
    results_dir = build_results_dir(
        output_dir,
        data_config_file,
        model_config_file,
        filter_config_file,
        overrides=overrides,
    )

    # And add fit_config_file info
    results_dir = os.path.join(results_dir, os.path.splitext(os.path.basename(fit_config_file))[0])
    
    # Check if the results directory exists, if not, print a warning
    if not os.path.exists(results_dir):
        print("Warning: Results directory does not exist at:", results_dir)
        print("Please run the fit_model_to_data.py experiment to generate results.")
        return
    
    # Plotting dir will be inside results_dir
    plotting_dir = os.path.join(results_dir, "plots")
    os.makedirs(plotting_dir, exist_ok=True)
    
    # Generate or load data
    data, data_key = generate_data_from_config(
        config_path=config_path,
        data_config_file=data_config_file,
        data_save_file=None,
        overrides=overrides,
    )

    # Create and initialize the CD-NLGSSM model from the model config file
    learnable_model, learnable_params, learnable_props = create_cdnlgssm_model_from_config(
        config_path=config_path,
        true_model_config_file=model_config_file,
        overrides=overrides,
    )

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams, filter_info = create_cdnlgssm_filter_from_config(
        config_path=config_path,
        filter_config_file=filter_config_file,
        overrides=overrides
    )

    # Full path to the fit config file
    fit_config_filepath = os.path.join(config_path, fit_config_file)

    # Check if the config file exists
    if not os.path.exists(fit_config_filepath):
        raise FileNotFoundError(f"Configuration file '{fit_config_filepath}' not found.")

    ## Figure out series of inferences based on inference config file
    # Initialize fit config parser
    config = ConfigParser()
    config.read(fit_config_filepath)

    # For each optimization method specified in the config
    optim_configs = config.sections()

    # Run SGD for parameter estimation
    if 'sgd' in optim_configs:
        # SGD configuration
        sgd_config = config['sgd']

        # If fit_keys are provided
        fit_keys = fit_key if fit_key is not None else [sgd_config.getint('key', 0)]
        # Iterate over the fit_key list
        for idx, key in enumerate(fit_keys):
            print(f"Loading SGD MAP estimation results with fit_key: {key} (Index {idx + 1} of {len(fit_keys)})")

            # Load the results, for each fit_key separately
            with open(os.path.join(results_dir, f'sgd_model_fit_fitkey{key}.pkl'), 'rb') as f:
                sgd_results = pickle.load(f)

            print("SGD MAP estimation results loaded for fit_key:", key)
            # Results dictionary format
            '''
            sgd_results={
                'params_fitted': fit_results[0],
                'loss_history': fit_results[1],
                'params_history': None,
                'grad_history': None,
            }
            '''

            # Plot the marginal log likelihood learning curve
            plot_mll_learning_curve(
                true_model = data['dgmodel_info']['model'],
                true_params = data['dgmodel_info']['params'],
                true_emissions = data['emissions'],
                t_emissions = data['t_emissions'],
                marginal_lls = sgd_results['loss_history'],
                filter_hyperparams=filter_hyperparams,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'mll_learning_curve_fitkey{key}'
                )
            )

            # Extract interesting parameters in dataframe format for later plotting
            true_params_df = learnable_params_to_df(
                data['dgmodel_info']['params'], learnable_props
            )
            init_params_df = learnable_params_to_df(
                learnable_params, learnable_props
            )
            fitted_params_df = learnable_params_to_df(
                sgd_results['params_fitted'], learnable_props
            )
            fitted_params_history_df = learnable_params_to_df(
                sgd_results['params_history'], learnable_props
            )
            
            # Plot parameter trajectories over SGD iterations
            plot_param_sequences(
                param_history=fitted_params_history_df,
                true=true_params_df,
                init=init_params_df,
                pointwise_estimate=fitted_params_df,
                burn_in_frac=0.0,
                pairwise_plots=True,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'parameter_trajectories_fitkey{key}'
                )
            )

    # Run MCMC for parameter estimation
    if 'mcmc' in optim_configs:
        mcmc_config = config['mcmc']
        
        # If fit_keys are provided
        fit_keys = fit_key if fit_key is not None else [mcmc_config.getint('key', 0)]
        # Iterate over the fit_key list
        for idx, key in enumerate(fit_keys):
            print(f"Loading MCMC MAP estimation results with fit_key: {key} (Index {idx + 1} of {len(fit_keys)})")

            # Load the results, for each fit_key separately
            load_results_file = os.path.join(
                results_dir,
                mcmc_config.get('type', 'unknown') +
                f'_mcmc_model_fit_fitkey{key}.pkl'
            )
            with open(load_results_file, 'rb') as f:
                mcmc_results = pickle.load(f)

            print("MCMC MAP estimation results loaded for fit_key:", key)
            # Results dictionary format
            '''
            mcmc_results = {
                'warmup_param_samples': mcmc_result[0],
                'mcmc_param_samples': mcmc_result[1],
                'warmup_log_probs': mcmc_result[2],
                'mcmc_log_probs': mcmc_result[3],
            }
            '''

            # Plot the marginal log likelihood learning curve
            plot_mll_learning_curve(
                true_model = data['dgmodel_info']['model'],
                true_params = data['dgmodel_info']['params'],
                true_emissions = data['emissions'],
                t_emissions = data['t_emissions'],
                marginal_lls = mcmc_results['mcmc_log_probs'],
                filter_hyperparams=filter_hyperparams,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'mll_learning_curve_fitkey{key}'
                )
            )
   
            # Extract interesting parameters in dataframe format for later plotting
            true_params_df = learnable_params_to_df(
                data['dgmodel_info']['params'], learnable_props
            )
            init_params_df = learnable_params_to_df(
                learnable_params, learnable_props
            )
            mcmc_samples_df = learnable_params_to_df(
                mcmc_results['mcmc_param_samples'], learnable_props
            )
            
            # Plot parameter trajectories over MCMC iterations
            plot_param_sequences(
                param_history=mcmc_samples_df,
                true=true_params_df,
                init=init_params_df,
                pointwise_estimate=mcmc_samples_df.mean(axis=0).to_frame().T,
                burn_in_frac=0.0,
                pairwise_plots=True,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'parameter_trajectories_fitkey{key}'
                )
            )

            # Plot parameter distributions from MCMC samples
            plot_param_dist(
                samples=mcmc_samples_df,
                true=true_params_df,
                init=init_params_df,
                pointwise_estimate=mcmc_samples_df.mean(axis=0).to_frame().T,
                burn_in_frac=0.0,
                pairwise_plots=True,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'parameter_distributions_fitkey{key}'
                )
            )

    if 'scipy' in optim_configs or 'scipy_jaxopt' in optim_configs:
        config_key = 'scipy' if 'scipy' in optim_configs else 'scipy_jaxopt'
        scipy_config = config[config_key]

        # If fit_keys are provided
        fit_keys = fit_key if fit_key is not None else [scipy_config.getint('key', 0)]
        # Iterate over the fit_key list
        for idx, key in enumerate(fit_keys):
            print(f"Loading Scipy MAP estimation results with fit_key: {key} (Index {idx + 1} of {len(fit_keys)})")

            # Load the results, for each fit_key separately
            load_results_file = os.path.join(
                results_dir,
                f'_scipy_model_fit_fitkey{key}.pkl'
            )
            with open(load_results_file, 'rb') as f:
                scipy_results = pickle.load(f)

            print("Scipy MAP estimation results loaded for fit_key:", key)
            '''
            scipy_results = {
                'params_fitted': scipy_result.x,
                'fun_value': scipy_result.fun,
                'nfev': scipy_result.nfev,
                'success': scipy_result.success,
                'message': scipy_result.message,
                'params_history': Only for SciPy 
            }
            '''

            # Plot the marginal log likelihood learning curve
            plot_mll_learning_curve(
                true_model = data['dgmodel_info']['model'],
                true_params = data['dgmodel_info']['params'],
                true_emissions = data['emissions'],
                t_emissions = data['t_emissions'],
                marginal_lls = scipy_results['fun_value'],
                filter_hyperparams=filter_hyperparams,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'mll_learning_curve_fitkey{key}'
                )
            )

            # Extract interesting parameters in dataframe format for later plotting
            true_params_df = learnable_params_to_df(
                data['dgmodel_info']['params'], learnable_props
            )
            init_params_df = learnable_params_to_df(
                learnable_params, learnable_props
            )
            fitted_params_df = learnable_params_to_df(
                scipy_results['params_fitted'], learnable_props
            )
            fitted_params_history_df = learnable_params_to_df(
                scipy_results['params_history'], learnable_props
            )
            
            # Plot parameter trajectories over SGD iterations
            plot_param_sequences(
                param_history=fitted_params_history_df,
                true=true_params_df,
                init=init_params_df,
                pointwise_estimate=fitted_params_df,
                burn_in_frac=0.0,
                pairwise_plots=True,
                plot_save_path=os.path.join(
                    plotting_dir,
                    f'parameter_trajectories_fitkey{key}'
                )
            )

    # Print a message indicating the completion of the experiment
    print("Experiment completed. Results saved to:", results_dir)
    

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    # Create optional flags for comamand line arguments
    parser = argparse.ArgumentParser(description='Plot fitted model to data --- as run by fit_model_to_data.py according to specified configurations of data, model and filtering.')
    parser.add_argument('--config_path', type=str, default='../configs/',
                        help='Path to configuration files: (default: ../configs/)')
    parser.add_argument('--data_config_file', type=str, default='data/true_l63_data',
                        help='Data configuration file: (default: true_l63_data)')
    parser.add_argument('--model_config_file', type=str, default='model/l63_mech',
                        help='Model configuration file: (default: l63_mech)')
    parser.add_argument('--filter_config_file', type=str, default='filter/ekf_StateFirst_EmissionsFirst', 
                        help='Filter configuration file: (default: ekf_StateFirst_EmissionsFirst)')
    parser.add_argument('--fit_config_file', type=str, default='fitting/fit_sgd',
                        help='Fitting algorithm configuration file: (default: fit_sgd)')
    # Add optional data_key and ftf_key arguments, which can be a sequence of integers
    parser.add_argument('--fit_key', type=int, nargs='+', default=None,
                        help='Optional key to use for fitting. If None, the key from the fit config file will be used.')
    parser.add_argument('--output_dir', type=str, default='results/fit_model_to_data',
                        help='Output directory for results: (default: results/fit_model_to_data)')

    args = parser.parse_args()

    # Revise this if overrides are needed
    overrides = {}
    if args.fit_key is not None:
        pass

    # Run the plot_fitted_model_to_data function with the provided arguments
    plot_fitted_model_to_data(**args.__dict__)
