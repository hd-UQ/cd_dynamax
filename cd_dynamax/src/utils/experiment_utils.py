# System imports
import os
from configparser import ConfigParser
# Jax imports
import jax.numpy as jnp
import jax.random as jr
# Import floating types, to avoid float64' is not defined error
from numpy import float64, float32, array
from typing import Tuple, NamedTuple, Union

# CD-Nonlinear Gaussian models
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import *
from cd_dynamax.dynamax.parameters import ParameterProperties

# Useful utility functions
from .simulation_utils import *
from .physics_based_models import *
from .data_driven_models import *


def generate_t_emissions(
        t0: float,
        t1: float,
        num_samples: int,
        irregular_samples: bool = True,
        key=jr.PRNGKey(0),
    ) -> Tuple[int, Array]:
    """    Generate time points for emissions, either uniformly or irregularly sampled.
    Args:
        t0 (float): Start time.
        t1 (float): End time.
        num_samples (int): Number of samples to generate.
        irregular_samples (bool): Whether to generate irregular samples.
        key (jr.PRNGKey): JAX random key.
    Returns:
        Tuple[int, Array]: Number of time steps and the time emissions array.
    """
    if irregular_samples:
        # Generate irregular time points
        u = jr.uniform(key, (num_samples,), minval=0, maxval=1)
        s = jnp.cumsum(u)  # Convert them into sorted cumulative sum
        # Normalize to [t0, t1] interval
        t_emissions = t0 + (s / s[-1]) * (t1 - t0)

        # throw error if duplicate time points
        if jnp.any(jnp.diff(t_emissions) < 1e-12):
            raise ValueError("Generated time points contain duplicates (within 1e-12).")
    else:
        # Generate uniform time points
        t_emissions = jnp.linspace(t0, t1, num_samples)

    # Ensure t_emissions is a column vector
    t_emissions = t_emissions[:, None]

    return len(t_emissions), t_emissions


### Python script utilities
# Useful auxiliary functions for the python scripts
# Build results directory based on config file names
def build_results_dir(
        output_dir,
        data_config_file,
        model_config_file,
        filter_config_file,
        overrides=None
    ):
    '''
        Builds the results directory path based on configuration file names.
        
        Joins the names of the config files, omitting their directory and extensions.
    '''
    os.makedirs(output_dir, exist_ok=True)

    # Default subdirectory names
    data_subdir = os.path.basename(data_config_file).split('.')[0]
    model_subdir = os.path.basename(model_config_file).split('.')[0]
    if filter_config_file is not None:
        filter_subdir = os.path.basename(filter_config_file).split('.')[0]
    else:
        filter_subdir = ""
    # Preallocate list for data overrides
    data_override_parts = []
    # Process overrides to create a unique subdirectory if needed
    if overrides is not None and len(overrides) > 0:
        # data_config_file overrides
        for k, v in overrides.items():
            if 'data_generation.' in k:
                # Remove data_generation. prefix and append the value
                data_override_parts.append(f"{k.replace('data_generation.', 'data')}_{str(v).replace('/', '_')}")

    # Join all parts to form the results directory path
    results_dir = os.path.join(
        output_dir,
        data_subdir + ("_" + "_".join(data_override_parts) if data_override_parts else ""),
        model_subdir,
        filter_subdir
    )
    return results_dir

# Override ConfigParser with a dictionary of overrides
def override_config(
        cfg: ConfigParser,
        overrides: dict | None
    ) -> ConfigParser:
    if overrides is not None and isinstance(overrides, dict):    
        for k, v in overrides.items():
            if "." in k:
                sect, opt = k.split(".", 1)
            else:
                raise ValueError(f"Override key {k} must be of the form 'section.option'")
            # Set the option in the configuration
            if sect not in cfg:
                cfg.add_section(sect)
            cfg[sect][opt] = str(v)
    
    return cfg

# Convert MCMC configuration from config file to dictionary
def mcmc_config_to_dict(mcmc_config):
    """
    Convert MCMC configuration from a config file to a dictionary.

    Args:
        mcmc_config (ConfigParser): MCMC configuration section.

    Returns:
        dict: MCMC configuration parameters.
    """
    import blackjax # In case the MCMC parameter evaluation contains blackjax functions
    
    mcmc_config_dict = {
        'type': mcmc_config.get('type', 'nuts'),
        'n_samples': mcmc_config.getint('n_samples', 100),
        'warmup_samples': mcmc_config.getint('warmup_samples', 10),
        'parameters': eval(mcmc_config.get('parameters', '{}')),
    }

    return mcmc_config_dict

# Create CD-NLGSSM model from configuration files
def create_cdnlgssm_model_from_config(
        true_model_config_file: str = None,
        overrides = None,
    ) -> Tuple[ContDiscreteNonlinearGaussianSSM, ParamsCDNLGSSM, ParameterProperties]:
    r"""Create CD-NLGSSM model from configuration files

    Args:
        :param true_model_config_file: path to the model configuration file
        :param overrides: dictionary of overrides to apply to the configuration file
    Returns:
        :return: Tuple of CD-NLGSSM model, parameters and properties
    """
    # Load the model configuration file
    config = ConfigParser()
    config.read(true_model_config_file)

    # The model section contains
    '''
    [model]
    class_name: "CDNLGSSM"
    state_dim: 3
    emission_dim: 3
    solver_config_file: "lorenz63_solver.py"
    '''
    if config.get('model', 'class_name') != "CDNLGSSM":
        raise ValueError(f"Unknown model class name: {config.get('model', 'class_name')}")
    
    # Apply overrides if provided
    if overrides is not None:
        config = override_config(
            cfg=config,
            overrides=overrides
        )
    
    # Create the model
    state_dim = config.getint('model', 'state_dim')
    emission_dim = config.getint('model', 'emission_dim')
    true_model = ContDiscreteNonlinearGaussianSSM(
        state_dim=state_dim,
        emission_dim=emission_dim,
        diffeqsolve_settings=solver_settings_from_config(
            config.get('model', 'solver_config_file', fallback=None)
        )
    )

    # Load the initial values from the configuration file,
    # by iterating over rows in section [initial_values]
    initial_config_values = config['initial_values']

    # Typing aliases: for eval to work properly
    Array = array
    
    # Iterate over the rows in the initial values section (we don't know the order or which parameters are present)
    initial_params = {}
    for key, val in initial_config_values.items():
        initial_params[key] = eval(val)

    # Load the prior from the configuration file, if present
    prior_config_file = config.get('prior', 'prior_class_file', fallback=None)
    if prior_config_file is not None:
        # convert / to . in prior_config_file
        prior_config_import_nm = prior_config_file.replace('/', '.').rstrip('.py')
        ## WARNING: if your file has a "dash" in the name, it will not work as a module name!
        from importlib import import_module
        module = import_module(prior_config_import_nm)
        prior = module.CDNLGSSM_Prior()
        prior_init_key = jr.PRNGKey(config.getint('prior', 'prior_init_key', fallback=0))
    else:
        prior = None
        prior_init_key = jr.PRNGKey(0)
    
    # Initialize the model with the provided parameters
    true_model_params, true_model_props = init_cdnlgssm_params(
        default_params=true_model._default_cdnlgssm_params(),
        init_params = initial_params,
        init_prior = prior,
        key = prior_init_key,
    )

    return true_model, true_model_params, true_model_props

def solver_settings_from_config(
        config_file: str,
        overrides = None,
    ) -> dict:
    r"""Load solver settings from a configuration file

    Args:
        :param config_file: path to the configuration file
        :param overrides: dictionary of overrides to apply to the configuration file

    Returns:
        :return: dictionary of solver settings
    """
    
    import diffrax as dfx
    
    # Load the solver configuration file
    solver_config = ConfigParser()
    solver_config.read(config_file)
    
    # Apply overrides if provided
    if overrides is not None:
        solver_config = override_config(
            cfg=solver_config,
            overrides=overrides
        )

    # Extract the solver settings into a dictionary
    diffeqsolve_settings = {}
    diffeqsolve_settings['solver'] = eval(
        solver_config.get('diffeqsolve_settings', 'solver', fallback='None')
    )
    diffeqsolve_settings['stepsize_controller'] = eval(
        solver_config.get('diffeqsolve_settings', 'stepsize_controller', fallback='dfx.ConstantStepSize()')
    )
    diffeqsolve_settings['adjoint'] = eval(
        solver_config.get('diffeqsolve_settings', 'adjoint', fallback='dfx.RecursiveCheckpointAdjoint()')
    )
    diffeqsolve_settings['dt0'] = solver_config.getfloat('diffeqsolve_settings', 'dt0', fallback=0.01)
    diffeqsolve_settings['tol_vbt'] = solver_config.getfloat('diffeqsolve_settings', 'tol_vbt', fallback=1e-1)
    diffeqsolve_settings['max_steps'] = solver_config.getfloat('diffeqsolve_settings', 'max_steps', fallback=1e2)

    return diffeqsolve_settings

# Create CD-NLGSSM filter from configuration file
def create_cdnlgssm_filter_from_config(
        filter_config_file: str,
        overrides = None,
    ) -> NamedTuple:
    r"""Create CD-NLGSSM filter from configuration file
    Args:
        :param filter_config_file: path to the filter configuration file
        :param overrides: dictionary of overrides to apply to the configuration file
    Returns:
        :return: NamedTuple of filter settings
    """

    # Load the filter configuration file
    filter_config = ConfigParser()
    filter_config.optionxform = str  # Preserve case sensitivity (needed for "N_particles")
    filter_config.read(filter_config_file)

    # Apply overrides if provided
    if overrides is not None:
        filter_config = override_config(
            cfg=filter_config,
            overrides=overrides
        )

    # The filter type is specified as a section within the configuration file
    # The section name is the filter type, e.g. EKF, UKF, EnKF
    section = filter_config.sections()[0] if filter_config.sections() else None
    
    # Get the section name and check if it is a valid filter type
    if section not in ['EKF', 'UKF', 'EnKF']:
        raise ValueError(f"Unknown filter type: {section}")

    filter_class_str='{}HyperParams'.format(section)
    hyperparam_dict = dict(filter_config.items(section))
    hyperparam_dict['dt_final'] = float(hyperparam_dict.get('dt_final', 1e-4))
    hyperparam_dict['cov_rescaling'] = float(hyperparam_dict.get('cov_rescaling', 1.0))
    hyperparam_dict['diffeqsolve_settings'] = solver_settings_from_config(hyperparam_dict['diffeqsolve_settings_file'])
    # drop the diffeqsolve_settings_file key
    hyperparam_dict.pop('diffeqsolve_settings_file', None)
    if section == 'EKF':
        pass
    elif section == 'UKF':
        hyperparam_dict['alpha'] = float(eval(hyperparam_dict.get('alpha', 'jnp.sqrt(3.0)')))
        hyperparam_dict['beta'] = float(eval(hyperparam_dict.get('beta', '2')))
        hyperparam_dict['kappa'] = float(eval(hyperparam_dict.get('kappa', '0')))
    elif section == 'EnKF':
        hyperparam_dict['N_particles'] = int(hyperparam_dict.get('N_particles', 30))
        hyperparam_dict['perturb_measurements'] = eval(hyperparam_dict.get('perturb_measurements', 'True'))

    filter_hyperparams = eval(filter_class_str)(**hyperparam_dict)

    # Also process filter_info section if present
    filter_info = None
    if filter_config.has_section('filter_info'):
        filter_info = {}
        for k, v in filter_config.items('filter_info'):
            filter_info[k] = str(v)
    
    # return both hyperparams and filter_info
    return filter_hyperparams, filter_info

