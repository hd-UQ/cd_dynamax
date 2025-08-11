import os
from configparser import ConfigParser
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import pickle
import equinox as eqx
import sys
sys.path.append('..')
sys.path.append('../..')

from utils.experiment_utils import *
from data_generator import generate_data_from_config
from utils.simulation_utils import filter_and_forecast

def build_results_dir(data_config_file, model_config_file, filter_config_file, sweep_config_file, output_dir):
    '''Builds the results directory path based on configuration file names.'''
    return os.path.join(
        output_dir,
        os.path.basename(data_config_file).split('.')[0],
        os.path.basename(model_config_file).split('.')[0],
        os.path.basename(filter_config_file).split('.')[0],
        os.path.basename(sweep_config_file).split('.')[0]
    )

def run_likelihood_sweep(
    data_config_file,
    model_config_file,
    filter_config_file,
    sweep_config_file,
    output_dir,
):
    '''Run a filtering and forecasting experiment with specified configurations.

    Args:
        data_config_file (str): Path to the data configuration file.
        model_config_file (str): Path to the model configuration file.
        filter_config_file (str): Path to the filter configuration file.
        sweep_config_file (str): Path to the sweep configuration file.
        output_dir (str): Directory to save the results.
        T_filter (float): Fraction of the data to use for filtering
            remaining data will be used for forecasting.
            (default: 0.8)
    '''
    # Figure out the directory structure
    # Join the four names of the file,
    # omitting their directory and extensions
    # into a saving directory path
    results_dir = build_results_dir(
        data_config_file,
        model_config_file,
        filter_config_file,
        sweep_config_file,
        output_dir
    )
    os.makedirs(results_dir, exist_ok=True)

    ## Load data from file (or generate it if needed)--- [DATA CONFIG FILE]
    data = generate_data_from_config(data_config_file)


    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams = create_cdnlgssm_filter_from_config(filter_config_file)

    # Create and initialize the CD-NLGSSM model from the model config file
    model, params, props = create_cdnlgssm_model_from_config(model_config_file)

    # --- Read sweep config ---
    sweep_config = ConfigParser()
    sweep_config.read(sweep_config_file)

    # Extract sweep ranges into a dict
    param_sweeps = {}
    for section in sweep_config.sections():
        low = float(sweep_config[section]['low'])
        high = float(sweep_config[section]['high'])
        num = int(sweep_config[section]['num_samples'])
        param_sweeps[section] = jnp.linspace(low, high, num)

    # Create meshgrid and flatten
    mesh = jnp.meshgrid(*param_sweeps.values(), indexing='ij')
    grid_points = jnp.stack([m.flatten() for m in mesh], axis=-1)  # shape: (num_combos, num_params)

    param_names = list(param_sweeps.keys())

    # Function to replace params at given point
    def get_log_likelihood(point):
        new_params = params
        for name, value in zip(param_names, point):
            # This assumes params.<path> corresponds exactly to the section name
            # Adapt this mapping if section names != attribute names
            new_params = eqx.tree_at(
                lambda p, n=name: getattr(p.dynamics.drift, n), 
                new_params,
                value
            )

        filtered = cdnlgssm_filter(
            params=new_params,
            emissions=data['emissions'],
            t_emissions=data['t_emissions'],
            filter_hyperparams=filter_hyperparams,
        )
        return filtered.marginal_loglik

    # vmap over grid points
    log_likelihoods = vmap(get_log_likelihood)(grid_points)

    # Save results
    results = {
        'true_params': params,
        'param_names': param_names,
        'grid_points': grid_points,  # all tested combinations
        'log_likelihoods': log_likelihoods
    }

    # Save the results to a file
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Print a message indicating the completion of the experiment
    print("Experiment completed. Results saved to:", results_dir)

def eval_sweep_experiment(
    list_of_result_dirs,
    eval_output_dir,
):
    ''' Evaluate the results of a likelihood sweep experiment.
    Args:
        list_of_result_dirs (list): List of directories containing the results of the sweep experiment.
        eval_output_dir (str): Directory to save the evaluation results and plots.
    Raises:
        FileNotFoundError: If the results file is not found in any of the specified directories
        ValueError: If the results file does not contain the expected keys or shapes
    Description:
        This function loads the results from the specified directories, checks for the expected keys and shapes,
        and then generates a plot of the likelihood sweep experiment.
        It expects each result directory to contain a 'results.pkl' file with the following structure:
        - 'param_names': list[str] of length 1 (the parameter being swept)
        - 'grid_points': (N, 1) array of grid points for the parameter
        - 'log_likelihoods': (N,) array of log likelihoods for the parameter
        - 'true_params': optional; if present, tries to access .dynamics.drift.<param_name>
        The function will raise an error if any of the expected keys are missing or if the shapes do not match the expected format.
        The function will generate a plot of the likelihood sweep experiment and save it to the specified evaluation output directory.
        
        Can only plot 1D sweeps, i.e., each method must sweep a single parameter.
    '''

    # Make sure eval_output_dir exists
    os.makedirs(eval_output_dir, exist_ok=True)

    # Loop through each result directory and load the results
    results_dict = {}
    for results_dir in list_of_result_dirs:
        # Get the name of the filter from the directory as one directory back from base
        filter_name = os.path.basename(os.path.dirname(results_dir))

        # Load the results from the file
        if not os.path.exists(os.path.join(results_dir, 'results.pkl')):
            raise FileNotFoundError(f"Results file not found for {filter_name} in {results_dir}.")  

        # Load the results from the file
        with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
        results_dict[filter_name] = results
        print(f"Loaded results for filter: {filter_name}")

    # Now make a figure
    plot_sweep_experiment(
        results_dict=results_dict,
        eval_output_dir=eval_output_dir,
        save_name="likelihood_sweep.png",
    )

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------
# Pretty names and styling
# -------------------------------

PRETTY_NAME_MAP = {
    "ekf_StateFirst_EmissionsFirst": "EKF (1st order, Emiss-1st)",
    "ekf_StateSecond_EmissionsFirst": "EKF (2nd order, Emiss-1st)",
    "ekf_StateZeroth_EmissionsFirst": "EKF (0th order, Emiss-1st)",
    "enkf_StateFirst": "EnKF (1st order)",
    "enkf_StateZero": "EnKF (0th order)",
    "ukf_StateFirst": "UKF (1st order)",
    "ukf_StateZeroth": "UKF (0th order)",
}

def pretty_name(method_key: str) -> str:
    return PRETTY_NAME_MAP.get(method_key, method_key)

def infer_order(method_key: str) -> str:
    # Returns 'Zeroth' | 'Zero' | 'First' | 'Second' | 'Unknown'
    if re.search(r'zeroth|zero', method_key, re.IGNORECASE):
        return 'Zeroth'
    if re.search(r'first', method_key, re.IGNORECASE):
        return 'First'
    if re.search(r'second', method_key, re.IGNORECASE):
        return 'Second'
    return 'Unknown'

def infer_family(method_key: str) -> str:
    # Returns 'ekf' | 'enkf' | 'ukf' | 'other'
    if method_key.lower().startswith('ekf'):  return 'ekf'
    if method_key.lower().startswith('enkf'): return 'enkf'
    if method_key.lower().startswith('ukf'):  return 'ukf'
    return 'other'

def build_style_registry(method_keys):
    """
    Map each method to a style dict. Now both family and order determine the color.
    """
    # Base palette for families
    family_palette = {
        'ekf':  '#1f77b4',  # blue
        'enkf': '#d62728',  # red
        'ukf':  '#2ca02c',  # green
        'other': '#7f7f7f', # gray
    }

    # Lightness variation (or shade tweak) by order
    order_shade_factor = {
        'Zeroth': 0.6,   # darker
        'Zero':   0.6,
        'First':  1.0,   # base
        'Second': 1.3,   # lighter
        'Unknown': 1.0,
    }

    # Helper: adjust color brightness
    import matplotlib.colors as mcolors
    def adjust_lightness(color, factor):
        c = mcolors.to_rgb(color)
        return tuple(min(1, max(0, ch * factor)) for ch in c)

    # Linestyle/marker now free to be something else (e.g., solid for all, or per-order)
    # I’ll keep order on linestyle here, but you can swap to something else if needed.
    order_to_ls = {
        'Zeroth':  '--',
        'Zero':    '--',
        'First':   '-',
        'Second':  '-.',
        'Unknown': ':',
    }
    family_to_marker = {
        'ekf':  'o',
        'enkf': 's',
        'ukf':  '^',
        'other':'x',
    }

    styles = {}
    for key in method_keys:
        order = infer_order(key)
        fam = infer_family(key)

        base_color = family_palette.get(fam, family_palette['other'])
        color = adjust_lightness(base_color, order_shade_factor.get(order, 1.0))

        styles[key] = {
            'label': pretty_name(key),
            'color': color,
            'linestyle': order_to_ls.get(order, order_to_ls['Unknown']),
            'marker': family_to_marker.get(fam, family_to_marker['other']),
            'linewidth': 1.8,
            'markersize': 3.0,
            'zorder': 2,
        }
    return styles

TRUTH_STYLE = {
    'label': 'Truth',
    'color': 'black',
    'linestyle': '-',
    'linewidth': 2.4,
    'zorder': 3,
}

# -------------------------------
# Main plotting function
# -------------------------------
def plot_sweep_experiment(
    results_dict: dict,
    eval_output_dir: str,
    save_name: str | None = None,
    dpi: int = 300,
):
    """
    1D eval plotter for the NEW sweep results format.

    Expects each method entry to contain:
      - 'param_names': list[str] of length 1
      - 'grid_points': (N, 1)
      - 'log_likelihoods': (N,)
      - 'true_params': optional; if present, tries .dynamics.drift.<param_name>

    Plots all methods with styles from `build_style_registry`.
    Errors if any entry sweeps more than one parameter or names differ.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if not results_dict:
        raise ValueError("results_dict is empty; nothing to plot.")

    method_keys = list(results_dict.keys())
    styles = build_style_registry(method_keys)

    # Validate 1D sweeps and collect arrays
    X, Y, param_names = {}, {}, []
    for key in method_keys:
        res = results_dict[key]
        for req in ("param_names", "grid_points", "log_likelihoods"):
            if req not in res:
                raise ValueError(f"[{key}] Missing required key: '{req}'.")

        names = res["param_names"]
        if not isinstance(names, (list, tuple)) or len(names) != 1:
            print("Only 1D sweep plots are supported currently.")
            raise ValueError(f"[{key}] 'param_names' must be a list of length 1; got {names}")

        gp = np.asarray(res["grid_points"])
        if gp.ndim != 2 or gp.shape[1] != 1:
            raise ValueError(f"[{key}] grid_points must be shape (N, 1); got {gp.shape}")

        ll = np.asarray(res["log_likelihoods"]).reshape(-1)
        if gp.shape[0] != ll.shape[0]:
            raise ValueError(f"[{key}] N mismatch: grid_points={gp.shape[0]} vs log_likelihoods={ll.shape[0]}")

        param_names.append(names[0])
        X[key] = gp[:, 0].astype(float)
        Y[key] = (-ll.astype(float))  # plot negative log-likelihood

    # Ensure all methods sweep the same single parameter
    unique_names = sorted(set(param_names))
    if len(unique_names) != 1:
        raise ValueError(f"All methods must sweep the same single parameter. Found: {unique_names}")
    pname = unique_names[0]

    # Try to get true value
    true_val = None
    first_key = method_keys[0]
    if "true_params" in results_dict[first_key]:
        try:
            tp = results_dict[first_key]["true_params"]
            true_attr = getattr(getattr(getattr(tp, "dynamics"), "drift"), pname)
            true_val = float(np.squeeze(np.asarray(true_attr)))
        except Exception:
            true_val = None

    # Nicify axis label for common greek names
    greek_map = {"sigma": r"\sigma", "rho": r"\rho", "beta": r"\beta"}
    xlabel = rf"${greek_map.get(pname, pname)}$"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)

    for key in method_keys:
        style = styles[key]
        x, y = X[key], Y[key]
        order = np.argsort(x)
        ax.plot(
            x[order], y[order],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            zorder=style["zorder"],
        )

    if true_val is not None:
        ax.axvline(
            true_val,
            color=TRUTH_STYLE.get("color", "black"),
            linestyle=TRUTH_STYLE.get("linestyle", "-"),
            linewidth=TRUTH_STYLE.get("linewidth", 2.4),
            zorder=TRUTH_STYLE.get("zorder", 3),
            label=TRUTH_STYLE.get("label", "Truth"),
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"$-\log p(y\,|\,\theta)$", fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.6)

    # De-dupe legend
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            H.append(h); L.append(l); seen.add(l)
    ax.legend(H, L, frameon=False, fontsize=9, ncol=2, loc="best")

    ax.set_title(f"Likelihood Sweep over {xlabel}", fontsize=12)

    if save_name is None:
        save_name = f"likelihood_sweep_{pname}.png"
    outpath = os.path.join(eval_output_dir, save_name)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {outpath}")


# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    # Create optional flags for comamand line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run experiment with specified configurations.')
    parser.add_argument('--data_config_file', type=str, default='configs/data/l63_data')
    parser.add_argument('--model_config_file', type=str, default='configs/model/l63_mech')
    parser.add_argument('--filter_config_file', type=str, default='configs/filter/ekf_StateFirst_EmissionsFirst')
    parser.add_argument('--sweep_config_file', type=str, default='configs/likelihood_sweep/sigma')
    parser.add_argument('--output_dir', type=str, default='results/likelihood_sweep')

    args = parser.parse_args()
    run_likelihood_sweep(**args.__dict__)

    # Full experiment
    
    # Run the filtering and forecasting for each filter config
    filter_names = [
        "ekf_StateFirst_EmissionsFirst",
        "ekf_StateSecond_EmissionsFirst",
        "ekf_StateZeroth_EmissionsFirst",
        "enkf_StateFirst",
        "enkf_StateZero",
        "ukf_StateFirst",
        "ukf_StateZeroth",
    ]

    for name in filter_names:
        filter_config_file = f"configs/filter/{name}"
        run_likelihood_sweep(
            data_config_file=args.data_config_file,
            model_config_file=args.model_config_file,
            filter_config_file=filter_config_file,
            sweep_config_file=args.sweep_config_file,
            output_dir=args.output_dir,
        )
    
    # Evaluate the experiment
    result_dirs = [
        build_results_dir(args.data_config_file, args.model_config_file, f"configs/filter/{name}", args.sweep_config_file, args.output_dir)
        for name in filter_names
    ]

    eval_output_dir = os.path.join(args.output_dir, 'eval', os.path.basename(args.sweep_config_file))
    os.makedirs(eval_output_dir, exist_ok=True)

    eval_sweep_experiment(
        list_of_result_dirs=result_dirs,
        eval_output_dir=eval_output_dir,
    )
