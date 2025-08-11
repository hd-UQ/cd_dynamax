# Imports
from configparser import ConfigParser
import pickle

from continuous_discrete_nonlinear_gaussian_ssm.models import ContDiscreteNonlinearGaussianSSM
from utils.experiment_utils import *

# TODO: make sure this loads if file exists, otherwise generate data
def generate_data_from_config(config_file=None, config=None):
    """
    Load data from a configuration file.
    
    Args:
        config_file (str): Path to the configuration file.
        config (ConfigParser): Configuration parser object. If None, will create a new one from the config_file.
        
    Returns:
        dict: Configuration parameters.
    """
    if config is None:
        config = ConfigParser()
        config.read(config_file)


    if 'data_generation' in config:
        data_gen_config = config['data_generation']
        
        # Model and solver configuration files
        true_model_config_file = data_gen_config.get('true_model_config_file', None)
        
        # Create and initialize the CD-NLGSSM model
        true_model, true_params, true_props = create_cdnlgssm_model_from_config(true_model_config_file)
       
        # Sample from the model
        key = int(data_gen_config.get('key', 0))
        # Generate more keys, copy from experiment_utils
        keys=make_key_sequence(key)
        
        # Generate the time vector
        t0 = float(data_gen_config.get('t0', 0.0))
        t1 = float(data_gen_config.get('t1', 10.0))
        num_samples = int(data_gen_config.get('num_samples', 1000))
        irregular_samples = data_gen_config.getboolean('irregular_samples', True)
        num_timesteps, t_emissions = generate_t_emissions(
            t0,
            t1,
            num_samples,
            irregular_samples=irregular_samples,
            key=next(keys)
        )

        # Transition type for data generation
        transition_type = data_gen_config.get('transition_type', 'path')  # Default to 'path'
        
        # Actually sample from the model
        states, emissions = true_model.sample(
            true_params,
            next(keys),
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=None, # TODO: add inputs if needed
            transition_type=transition_type
        )

        # Print message data generation is complete
        print("Data generation complete. States and emissions sampled from the true model.")

        # Save the generated data
        generated_data = {
            'states': states,
            'emissions': emissions,
            't_emissions': t_emissions,
        } 
        
        return generated_data
    else:
        raise ValueError("Configuration file must contain 'data_generation' section.")

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python data_generator.py <config_file> <data_save_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = ConfigParser()
    config.read('config_file')

    # Generate data from the configuration file
    generated_data = generate_data_from_config(config_file)

    # Check if the data save file is provided in the config file
    if 'data_saving' in config:
        data_save_file = config['data_saving'].get('data_save_file', None)

        # Warning message that data_save_file will be rewritten
        print(f"Warning: 'data_save_file' is set to '{data_save_file}'. This will be overwritten by the data generation process.")
    else:
        data_save_file = sys.argv[2]
    
    # Save the generated data to the specified file, using 
    # Check if file exists and warn the user
    import os
    if os.path.exists(data_save_file):
        print(f"Warning: The file '{data_save_file}' already exists. It will be overwritten.")   

    with open(data_save_file, 'wb') as f:
        pickle.dump(generated_data, f)

    print(f"Generated data saved to {data_save_file}.")
