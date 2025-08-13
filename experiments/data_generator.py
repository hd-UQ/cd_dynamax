# Imports
import os
from configparser import ConfigParser
import pickle

# CD-NLGSSM imports
import sys
sys.path.append('..')
sys.path.append('../src')
from continuous_discrete_nonlinear_gaussian_ssm.models import ContDiscreteNonlinearGaussianSSM
from utils.experiment_utils import *
from utils.simulation_utils import *
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import update_params

def generate_data_from_config(
        data_config_file=None,
        data_config=None,
        data_save_file=None,
        model_config=None,
        param_reset_dict={},
        ):
    """
    Load data from a configuration file.
    
    Args:
        config_file (str): Path to the configuration file.
        config (ConfigParser): Configuration parser object. If None, will create a new one from the config_file.
        
    Returns:
        dict: Configuration parameters.
    """

    # Check if the config file exists
    if data_config is None:
        data_config = ConfigParser()
        data_config.read(data_config_file)

    # Check if a data_save_file is provided
    if data_save_file is None:
        # Check if the data save file is provided in the config file
        if 'data_saving' in data_config:
            data_save_file = data_config['data_saving'].get('data_save_file', None)

    else:
        # Check if the data save file is provided in the config file
        if 'data_saving' in data_config:
            # Warning message that data_save_file will be rewritten
            print(f"DATA Warning: 'data_save_file' in config is set to={data_config['data_saving'].get('data_save_file')} \n we are using the provided data_save_file={data_save_file} instead.")

    # Check if the data_save_file exists
    if os.path.exists(data_save_file):
        print(f"NO DATA GENERATED: The file '{data_save_file}' already exists, data is loaded from the file.")
        with open(data_save_file, 'rb') as f:
            generated_data = pickle.load(f)
    else:
        if 'data_generation' in data_config:
            data_gen_config = data_config['data_generation']
            
            # Model and solver configuration files
            true_model_config_file = data_gen_config.get('true_model_config_file', None)
            
            # Create and initialize the CD-NLGSSM model
            if model_config is None:
                true_model, true_params, true_props = create_cdnlgssm_model_from_config(true_model_config_file)
            else:
                true_model, true_params, true_props = create_cdnlgssm_model_from_config(config=model_config)

            # Reset the parameters according to the provided reset dictionary
            # (if empty, it will not change the parameters)
            true_params = update_params(true_params, param_reset_dict)
        
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

            # Save the generated data
            generated_data = {
                'states': states,
                'emissions': emissions,
                't_emissions': t_emissions,
            } 

            # Print message data generation is complete
            print("DATA GENERATED: States and emissions sampled from the true model.")

        else:
            raise ValueError("Configuration file must contain 'data_generation' section.")

        # Save the generated data to the specified file, using 
        os.makedirs(os.path.dirname(data_save_file), exist_ok=True)
        # Check if file exists and warn the user
        if os.path.exists(data_save_file):
            print(f"Warning: The file '{data_save_file}' already exists. It will be overwritten.")   

        with open(data_save_file, 'wb') as f:
            pickle.dump(generated_data, f)

        print(f"Data saved to {data_save_file}.")
    return generated_data

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python data_generator.py <config_file> <data_save_file>")
        sys.exit(1)   
    
    # Generate data from the configuration file
    generated_data = generate_data_from_config(
        data_config_file=sys.argv[1],
        data_save_file=sys.argv[2]
    )
    
    
