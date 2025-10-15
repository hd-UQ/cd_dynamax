import os
from configparser import ConfigParser
import jax.numpy as jnp
import jax.random as jr
import optax
import pickle

# CD-NLGSSM imports
from cd_dynamax.src.utils.experiment_utils import *
from data_generator import generate_data_from_config

def run_exp(
    data_config_file,
    model_config_file,
    filter_config_file,
    param_est_config_file,
    output_dir,
):
    # Figure out the directory structure
    # Join the four names of the file,
    # omitting their directory and extensions
    # into a saving directory path
    results_dir = os.path.join(
        output_dir,
        os.path.basename(data_config_file).split('.')[0],
        os.path.basename(model_config_file).split('.')[0],
        os.path.basename(filter_config_file).split('.')[0],
        os.path.basename(param_est_config_file,).split('.')[0]
    )
    os.makedirs(results_dir, exist_ok=True)

    ## Load data from file (or generate it if needed)--- [DATA CONFIG FILE]
    data, data_key = generate_data_from_config(data_config_file)

    # Create and initialize the CD-NLGSSM model from the model config file
    model, params, props = create_cdnlgssm_model_from_config(model_config_file)

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams = create_cdnlgssm_filter_from_config(filter_config_file)

    ## Run series of inferences based on inference config --- [INFERENCE CONFIG FILE]
    config = ConfigParser()
    config.read(param_est_config_file,)

    # For each optimization method specified in the config
    optim_configs = config.sections()

    # Run SGD for parameter estimation
    if 'sgd' in optim_configs:
        sgd_config = config['sgd']
        
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
            key = jr.PRNGKey(sgd_config.getint('key', 0))
        )
        print("SGD MAP estimation complete.")

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

        # Save the results
        with open(os.path.join(results_dir, 'sgd_results.pkl'), 'wb') as f:
            pickle.dump(sgd_results, f)

    # Run MCMC for parameter estimation
    if 'mcmc' in optim_configs:
        mcmc_config = config['mcmc']
        
        # MCMC MAP estimation via cd-dynamax model
        mcmc_result = model.fit_mcmc(
            initial_params=sgd_results['params_fitted'] if 'sgd' in optim_configs else params,
            props=props,
            emissions=data['emissions'],
            t_emissions=data['t_emissions'],
            filter_hyperparams=filter_hyperparams,
            inputs=None,
            mcmc_algorithm=mcmc_config_to_dict(mcmc_config),
            verbose=mcmc_config.getboolean('verbose', True),
            key=jr.PRNGKey(mcmc_config.getint('key', 0))
        )
        print("MCMC MAP estimation complete")

        # Reformat the results into a dictionary
        mcmc_results = {
            'warmup_param_samples': mcmc_result[0],
            'mcmc_param_samples': mcmc_result[1],
            'warmup_log_probs': mcmc_result[2],
            'mcmc_log_probs': mcmc_result[3],
        }

        # Save the results
        with open(os.path.join(results_dir, 'mcmc_results.pkl'), 'wb') as f:
            pickle.dump(mcmc_results, f)

    # Print a message indicating the completion of the experiment
    print("Experiment completed. Results saved to:", results_dir)
    

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    # Create optional flags for comamand line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run experiment with specified configurations.')
    parser.add_argument('--data_config_file', type=str, default='configs/data/true_l63_data',
                        help='Data configuration file: (default: true_l63_data)')
    parser.add_argument('--model_config_file', type=str, default='configs/model/true_l63_mech',
                        help='Model configuration file: (default: true_l63_mech)')
    parser.add_argument('--filter_config_file', type=str, default='configs/filter/ekf_StateFirst_EmissionsFirst', 
                        help='Filter configuration file: (default: ekf_StateFirst_EmissionsFirst), if all filters should be used, set to "all"')
    parser.add_argument('--param_est_config_file', type=str, default='configs/param_estimation/all',
                        help='Parameter estimation configuration file: (default: all)')
    parser.add_argument('--output_dir', type=str, default='results/exp_runner',
                        help='Output directory for results: (default: results/exp_runner)')
    
    args = parser.parse_args()
    run_exp(**args.__dict__)
