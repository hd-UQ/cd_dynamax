# Imports
import os
from configparser import ConfigParser
import pickle

from cd_dynamax.src.utils.experiment_utils import *
from cd_dynamax.src.utils.simulation_utils import *

def generate_data_from_config(
        config_path: str = '../configs/',
        data_config_file=None,
        data_save_file=None,
        overrides=None,
        ):
    """
    Load data from a configuration file.
    
    Args:
        data_config_file (str): Path to the configuration file.
        data_save_file (str): Path to save or load the generated data.
        overrides (dict): Dictionary of configuration overrides.
        
    Returns:
        dict: Configuration parameters.
    """

    # Full path to the config file
    data_config_filepath = os.path.join(config_path, data_config_file)

    # Check if the config file exists
    if not os.path.exists(data_config_filepath):
        raise FileNotFoundError(f"Configuration file '{data_config_filepath}' not found.")
    
    # Initialize data config parser
    data_config = ConfigParser()
    data_config.read(data_config_filepath)
    
    # Apply overrides if provided
    if overrides is not None:
        data_config = override_config(
            cfg=data_config,
            overrides=overrides
        )

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

    # If data_generation.key was a key on overrides dictionary,
    # append its value to data_save_file name
    if overrides is not None and 'data_generation.key' in overrides:
        if data_save_file is not None:
            base, ext = os.path.splitext(data_save_file)
            data_save_file = f"{base}_datakey{overrides['data_generation.key']}{ext}"

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
            true_model, true_params, true_props = create_cdnlgssm_model_from_config(
                config_path=config_path,
                true_model_config_file=true_model_config_file,
                overrides=overrides
            )
        
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

            # Save the true data generating model
            true_model_info = {
                'model': true_model,
                'params': true_params,
                # No need to save props for now, they are not mandatory for data generation
            }

            # Save the generated data
            generated_data = {
                'dgmodel_info': true_model_info,
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

        # Save the generated data
        with open(data_save_file, 'wb') as f:
            pickle.dump(generated_data, f)

        print(f"Data saved to {data_save_file}.")

    # Return the generated data and the data key
    return generated_data, data_config['data_generation'].getint('key', 0)

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python data_generator.py <config_file> <data_save_file>")
        sys.exit(1)   
    
    # Generate data from the configuration file
    generated_data, data_key = generate_data_from_config(
        data_config_file=sys.argv[1],
        data_save_file=sys.argv[2]
    )
    
    
