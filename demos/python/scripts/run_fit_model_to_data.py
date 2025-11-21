import argparse
import os
from configparser import ConfigParser
import jax.random as jr
import pickle

# CD-NLGSSM imports
from cd_dynamax.src.utils.experiment_utils import *
from cd_dynamax.src.utils.data_generator import generate_data_from_config

def fit_model_to_data(
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
    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    print("Results directory created at:", results_dir)

    # Generate or load data
    data, data_key = generate_data_from_config(
        config_path=config_path,
        data_config_file=data_config_file,
        data_save_file=None,
        overrides=overrides,
    )

    # Create and initialize the CD-NLGSSM model from the model config file
    model, params, props = create_cddynamax_model_from_config(
        config_path=config_path,
        true_model_config_file=model_config_file,
        overrides=overrides,
    )

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams, filter_info = create_cddynamax_filter_from_config(
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
        import optax  # Local import to avoid unnecessary dependency if not using SGD
        # SGD configuration
        sgd_config = config['sgd']

        # If fit_keys are provided
        fit_keys = fit_key if fit_key is not None else [sgd_config.getint('key', 0)]
        # Iterate over the fit_key list
        for idx, key in enumerate(fit_keys):
            print(f"Running SGD MAP estimation with fit_key: {key} (Index {idx + 1} of {len(fit_keys)})")
            
            # SGD MAP estimation via cd-dynamax model
            return_param_history = sgd_config.getboolean('return_param_history', False)
            return_grad_history = sgd_config.getboolean('return_grad_history', False)
            fit_results = model.fit_sgd(
                initial_params = params,
                props=props,
                emissions=data['emissions'],
                t_emissions=data['t_emissions'],
                filter_hyperparams=filter_hyperparams,
                inputs=None,
                optimizer=eval(
                    sgd_config.get('optimizer', 'optax.adam(1e-3)')
                ),
                batch_size = sgd_config.getint('batch_size', 1),
                num_epochs = sgd_config.getint('num_epochs', 50),
                shuffle = sgd_config.getboolean('shuffle', False),
                return_param_history = return_param_history,
                return_grad_history = return_grad_history,
                key = jr.PRNGKey(key)
            )
            print("SGD MAP estimation complete for fit_key:", key)
            
            # Reformat the results into a dictionary
            sgd_results={
                'params_fitted': fit_results[0],
                'loss_history': fit_results[1],
                'params_history': None,
                'grad_history': None,
            }
            if return_param_history and return_grad_history:
                sgd_results['params_history'] = fit_results[2]
                sgd_results['grad_history'] = fit_results[3]
            elif return_param_history:
                sgd_results['params_history'] = fit_results[2]
            elif return_grad_history:
                sgd_results['grad_history'] = fit_results[2]

            # Save the results, for each fit_key separately
            with open(os.path.join(results_dir, f'sgd_model_fit_fitkey{key}.pkl'), 'wb') as f:
                pickle.dump(sgd_results, f)

    # Run MCMC for parameter estimation
    if 'mcmc' in optim_configs:
        mcmc_config = config['mcmc']
        
        # If fit_keys are provided
        fit_keys = fit_key if fit_key is not None else [mcmc_config.getint('key', 0)]
        # Iterate over the fit_key list
        for idx, key in enumerate(fit_keys):
            # MCMC MAP estimation via cd-dynamax model
            print("Using fit results from SGD for MCMC initialization." if 'sgd' in optim_configs else "No previous SGD fit; using initial parameters for MCMC initialization.")
            print(f"Running MCMC MAP estimation with fit_key: {key} (Index {idx + 1} of {len(fit_keys)})")
            mcmc_result = model.fit_mcmc(
                initial_params=sgd_results['params_fitted'] if 'sgd' in optim_configs else params,
                props=props,
                emissions=data['emissions'],
                t_emissions=data['t_emissions'],
                filter_hyperparams=filter_hyperparams,
                inputs=None,
                mcmc_algorithm=mcmc_config_to_dict(mcmc_config),
                verbose=mcmc_config.getboolean('verbose', True),
                key=jr.PRNGKey(key)
            )
            print("MCMC MAP estimation complete for fit_key:", key)

            # Reformat the results into a dictionary
            mcmc_results = {
                'warmup_param_samples': mcmc_result[0],
                'mcmc_param_samples': mcmc_result[1],
                'warmup_log_probs': mcmc_result[2],
                'mcmc_log_probs': mcmc_result[3],
            }

            # Save the results
            save_results_file = os.path.join(
                results_dir,
                mcmc_config.get('type', 'unknown') +
                f'_mcmc_model_fit_fitkey{key}.pkl'
            )
            with open(save_results_file, 'wb') as f:
                pickle.dump(mcmc_results, f)

    if 'scipy' in optim_configs or 'scipy_jaxopt' in optim_configs:
        config_key = 'scipy' if 'scipy' in optim_configs else 'scipy_jaxopt'
        scipy_config = config[config_key]

        # If fit_keys are provided
        fit_keys = fit_key if fit_key is not None else [scipy_config.getint('key', 0)]
        # Iterate over the fit_key list
        for idx, key in enumerate(fit_keys):
            print(f"Running {config_key} optimization MAP estimation with fit_key: {key} (Index {idx + 1} of {len(fit_keys)})")

            return_param_history = scipy_config.getboolean('return_param_history', False)
            # SciPy optimization MAP estimation via cd-dynamax model
            scipy_result = model.fit_scipy(
                initial_params=sgd_results['params_fitted'] if 'sgd' in optim_configs else params,
                props=props,
                emissions=data['emissions'],
                t_emissions=data['t_emissions'],
                filter_hyperparams=filter_hyperparams,
                inputs=None,
                method=scipy_config.get('method', 'l-bfgs-b'),
                options={
                    'maxiter': scipy_config.getint('maxiter', 100),
                    'disp': scipy_config.getboolean('disp', True),
                },
                return_param_history=return_param_history,
            )
            print(f"{config_key} optimization MAP estimation complete for fit_key: {key}")

            # Reformat the results into a dictionary
            scipy_results = {
                'params_fitted': scipy_result[0],
                'fun_value': scipy_result[-1].fun,
                'nfev': scipy_result[-1].nfev,
                'success': scipy_result[-1].success,
                'message': scipy_result[-1].message,
                'loss_history': scipy_result[1],
            }

            if return_param_history:
                scipy_results['params_history'] = scipy_result[2]

            # Save the results
            with open(os.path.join(results_dir, f'scipy_model_fit_fitkey{key}.pkl'), 'wb') as f:
                pickle.dump(scipy_results, f)

    # Print a message indicating the completion of the experiment
    print("Experiment completed. Results saved to:", results_dir)
    

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    # Create optional flags for comamand line arguments
    parser = argparse.ArgumentParser(description='Fit model to data, according to specified configurations of data, model and filtering.')
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
    
    # Run the fit_model_to_data function with the provided arguments
    fit_model_to_data(**args.__dict__)
