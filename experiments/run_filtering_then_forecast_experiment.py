import os
from configparser import ConfigParser
import pickle

# CD-NLGSSM imports
import sys
sys.path.append('..')
sys.path.append('../src')
from data_generator import generate_data_from_config
from utils.experiment_utils import *
from utils.simulation_utils import filter_and_forecast

def build_results_dir(data_config_file, model_config_file, filter_config_file, output_dir):
    '''Builds the results directory path based on configuration file names.'''
    return os.path.join(
        output_dir,
        os.path.basename(data_config_file).split('.')[0],
        os.path.basename(model_config_file).split('.')[0],
        os.path.basename(filter_config_file).split('.')[0]
    )

def _make_data(
    data_config_file,
    model_config_file,
    enforce_twin_experiment=False,
    data_key=None,
):
    ## Load data from file (or generate it if needed)--- [DATA CONFIG FILE]
    data_config = ConfigParser()
    data_config.read(data_config_file)
    if enforce_twin_experiment:
        # If enforcing twin experiment, set the true model config file to be the same as the model config file
        data_config['data_generation']['true_model_config_file'] = model_config_file
    
    if data_key is None:
        # If no key is provided, use the key from the data config file
        data_key = data_config['data_generation']['key']
        # This is just extracted to create the data save file name below
    else:
        # If a key is provided, set it in the data config
        data_config['data_generation']['key'] = str(data_key)

    # Generate data
    # By calling the generate_data_from_config function with the modified config
    # And indicating the new data_save_file
    data = generate_data_from_config(
        data_config=data_config,
        data_save_file=os.path.join(
            'data',
            '{}_{}'.format(os.path.basename(model_config_file), f'data_{data_key}.pkl')
        )
    )
    return data, int(data_key)


def run_filter_then_forecast(
    data_config_file,
    model_config_file,
    filter_config_file,
    output_dir,
    T_filter=0.8,
    enforce_twin_experiment=False,
    data_key=None,
    ftf_key=None,
):
    '''Run a filtering and forecasting experiment with specified configurations.

    Args:
        data_config_file (str): Path to the data configuration file.
        model_config_file (str): Path to the model configuration file.
        filter_config_file (str): Path to the filter configuration file.
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
        output_dir
    )
    os.makedirs(results_dir, exist_ok=True)

    data, data_key = _make_data(
        data_config_file=data_config_file,
        model_config_file=model_config_file,
        enforce_twin_experiment=enforce_twin_experiment,
        data_key=data_key,
    )


    # Create and initialize the CD-NLGSSM model from the model config file
    model, params, props = create_cdnlgssm_model_from_config(model_config_file)

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams = create_cdnlgssm_filter_from_config(filter_config_file)

    # Assign forecasting and filtering time indices based on T_filter fraction.
    T0 = data['t_emissions'][0]
    T_forecast_end = data['t_emissions'][-1]
    T_filter_end = T0 + T_filter * (T_forecast_end - T0)

    # Set ftf key
    if ftf_key is None:
        ftf_key = data_key + 10

    # Run filtering and forecasting
    filtered, forecasted, start_idx_filter, stop_idx_filter, start_idx_forecast, stop_idx_forecast = filter_and_forecast(
        model_params=params,
        filter_hyperparams=filter_hyperparams,
        t_emissions=data['t_emissions'],
        emissions=data['emissions'],
        T0=T0,
        T_filter_end=T_filter_end,
        T_forecast_end=T_forecast_end,
        key=ftf_key
    )

    # convert the filtered and forecasted results to dictionaries for easily adding fields.
    filtered_dict = tree_to_dict(filtered)
    forecasted_dict = tree_to_dict(forecasted)

    # Generate emissions means/covs, then update the output dictionaries
    emissions_mean, emissions_cov = cdnlgssm_emissions(
        params=params,
        t_states=data['t_emissions'][:stop_idx_filter],
        state_means=filtered.filtered_means,
        state_covs=filtered.filtered_covariances,
        inputs=None,
    )
    filtered_dict['filtered_means_emissions'] = emissions_mean
    filtered_dict['filtered_covariances_emissions'] = emissions_cov

    # Generate emissions means/covs, then update the output dictionaries
    emissions_mean, emissions_cov = cdnlgssm_emissions(
        params=params,
        t_states=data['t_emissions'][:stop_idx_filter],
        state_means=filtered.predicted_means,
        state_covs=filtered.predicted_covariances,
        inputs=None,
    )
    filtered_dict['predicted_means_emissions'] = emissions_mean
    filtered_dict['predicted_covariances_emissions'] = emissions_cov

    # Generate emissions means/covs, then update the output dictionaries
    emissions_mean, emissions_cov = cdnlgssm_emissions(
        params=params,
        t_states=data['t_emissions'][start_idx_forecast:stop_idx_forecast],
        state_means=forecasted.forecasted_state_means,
        state_covs=forecasted.forecasted_state_covariances,
        inputs=None,
    )
    forecasted_dict['forecasted_emission_means'] = emissions_mean
    forecasted_dict['forecasted_emission_covariances'] = emissions_cov

    # Save the results
    results = {
        'filtered': filtered_dict,
        'forecasted': forecasted_dict,
        'start_idx_filter': start_idx_filter,
        'stop_idx_filter': stop_idx_filter,
        'start_idx_forecast': start_idx_forecast,
        'stop_idx_forecast': stop_idx_forecast,
    }

    # Save the results to a file
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Print a message indicating the completion of the experiment
    print("Experiment completed. Results saved to:", results_dir)

def eval_filter_then_forecast_experiment(
    data_config_file,
    list_of_result_dirs,
    enforce_twin_experiment=False,
    model_config_file=None,
    eval_output_dir='eval',
    data_key=None,
    ftf_key=None,
):
    '''Evaluate the filtering and forecasting experiment with specified configurations.

    Args:
        data_config_file (str): Path to the data configuration file.
        list_of_result_dirs (list): List of directories containing the results of the filtering and forecasting.
        enforce_twin_experiment (bool): If True, will enforce the twin experiment setup.
            Done by resetting the true_model_config_file in the data_config_file = model_config_file.
        model_config_file (str): Path to the model configuration file.
        eval_output_dir (str): Directory to save the evaluation results.
        data_key (int): Key for the data generation. If None, will use the key from the data config file.
        ftf_key (int): Key for the filtering and forecasting. If None, will use data_key + 10.
    '''

    # Make sure eval_output_dir exists
    os.makedirs(eval_output_dir, exist_ok=True)

    data, data_key = _make_data(
        data_config_file=data_config_file,
        model_config_file=model_config_file,
        enforce_twin_experiment=enforce_twin_experiment,
        data_key=data_key,
    )

    # Loop through each result directory and load the results
    results_dict = {}
    for results_dir in list_of_result_dirs:
        # Get the name of the filter from the directory
        filter_name = os.path.basename(results_dir)

        # Load the results from the file
        if not os.path.exists(os.path.join(results_dir, 'results.pkl')):
            raise FileNotFoundError(f"Results file not found for {filter_name} in {results_dir}.")  

        # Load the results from the file
        with open(os.path.join(results_dir, 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
        results_dict[filter_name] = results
        print(f"Loaded results for filter: {filter_name}")

    # Now make a figure
    plot_filter_then_forecast_states(
        results_dict=results_dict,
        data=data,
        eval_output_dir=eval_output_dir,
        save_name="filter_then_forecast_states.png",
        plot_uncertainty=False,  # flip to True if your covariances are present & you want 95% bands
    )

import re
import numpy as np

# -------------------------------
# Pretty names and styling
# -------------------------------

PRETTY_NAME_MAP = {
    "ekf_StateFirst_EmissionsFirst": "EKF (1st order)",
    "ekf_StateSecond_EmissionsFirst": "EKF (2nd order)",
    "ekf_StateZeroth_EmissionsFirst": "EKF (0th order)",
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
    Map each method to a style dict. Use distinct hues by order within family
    (not just lightness), and vary markers by order for grayscale robustness.
    """
    import matplotlib.colors as mcolors

    def pick_color(fam, order):
        # Order-specific palettes per family (high contrast)
        order_palette_by_family = {
            'ekf': {   # blue family split into blue/cyan
                'Zeroth': '#0b3c6f',  # dark navy
                'First':  '#1f77b4',  # blue
                'Second': '#17becf',  # cyan
                'Unknown':'#1f77b4',
            },
            'enkf': {  # red family split into red/orange
                'Zeroth': '#7f1d1d',  # dark red
                'First':  '#d62728',  # red
                'Second': '#ff7f0e',  # orange
                'Unknown':'#d62728',
            },
            'ukf': {   # green family split into green/teal
                'Zeroth': '#145a14',  # dark green
                'First':  '#2ca02c',  # green
                'Second': '#1abc9c',  # teal
                'Unknown':'#2ca02c',
            },
            'other': {
                'Zeroth': '#4d4d4d',
                'First':  '#7f7f7f',
                'Second': '#b3b3b3',
                'Unknown':'#7f7f7f',
            }
        }
        fam = fam if fam in order_palette_by_family else 'other'
        pal = order_palette_by_family[fam]
        return pal.get(order, pal['Unknown'])

    order_to_ls = {
        'Zeroth':  '--',
        'Zero':    '--',
        'First':   '-',
        'Second':  '-.',
        'Unknown': ':',
    }
    # Change marker by order (family still tweaks shape in your old code; this is clearer)
    order_to_marker = {
        # 'Zeroth':  'v',
        # 'Zero':    'v',
        # 'First':   'o',
        # 'Second':  'D',
        # 'Unknown': 'x',
    }

    def infer_family(method_key: str) -> str:
        mk = method_key.lower()
        if mk.startswith('ekf'):  return 'ekf'
        if mk.startswith('enkf'): return 'enkf'
        if mk.startswith('ukf'):  return 'ukf'
        return 'other'

    styles = {}
    for key in method_keys:
        order = infer_order(key)
        fam = infer_family(key)
        color = pick_color(fam, order)
        styles[key] = {
            'label': pretty_name(key),
            'color': color,
            'linestyle': order_to_ls.get(order, order_to_ls['Unknown']),
            'marker': order_to_marker.get(order, None),
            'linewidth': 1.0,
            'markersize': 1.0,
            'zorder': 2,
        }
    return styles

TRUTH_STYLE = {
    'label': 'Truth',
    'color': 'black',
    'linestyle': '--',
    'linewidth': 2.0,
    'zorder': 0,
}

# -------------------------------
# Utility
# -------------------------------

def _window_mean_sq_error(truth, est, i0, i1):
    """Mean squared error over [i0, i1) across states."""
    err = est[i0:i1] - truth[i0:i1]           # (T, D)
    return float(np.mean(err**2))

def _per_time_mse(truth, est):
    """Per-time MSE across states, shape (T,)."""
    return np.mean((est - truth)**2, axis=-1)

# -------------------------------
# Main plotting function
# -------------------------------
def plot_filter_then_forecast_states(
    results_dict: dict,
    data: dict,
    eval_output_dir: str,
    save_name: str = "filter_then_forecast_states.png",
    plot_uncertainty: bool = False,
    uncertainty_alpha: float = 0.15,
    dpi: int = 300,
    show_filtered_mse: bool = True,
    show_cumavg_filtered_mse: bool = True,
    # NEW: observed data overlays
    plot_observations: bool = True,
    obs_marker: str = "x",
    obs_markersize: float = 3.0,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    from matplotlib.ticker import MaxNLocator

    method_keys = list(results_dict.keys())
    styles = build_style_registry(method_keys)

    t = np.asarray(data['t_emissions']).reshape(-1)
    Xtrue = np.asarray(data['states'])          # (T_total, D)
    T_total, D = Xtrue.shape

    # Optional emissions
    Y = None
    E = 0
    if plot_observations:
        if 'emissions' not in data:
            warnings.warn("plot_observations=True but data['emissions'] is missing; skipping observation overlay.")
            plot_observations = False
        else:
            Y = np.asarray(data['emissions'])   # (T_total, E) expected
            if Y.ndim != 2 or Y.shape[0] != T_total:
                warnings.warn("plot_observations=True but emissions shape is unexpected; skipping overlay.")
                plot_observations = False
            else:
                E = Y.shape[1]
                n_overlay = min(D, E)
                if n_overlay < 1:
                    plot_observations = False
                else:
                    warnings.warn(
                        f"Overlaying observed emissions on state plots: using first {n_overlay} emission dims over first {n_overlay} state dims."
                    )

    first_key = method_keys[0]
    r0 = results_dict[first_key]
    i0_filt = int(r0['start_idx_filter'])
    i1_filt = int(r0['stop_idx_filter'])
    i0_fcst = int(r0['start_idx_forecast'])
    i1_fcst = int(r0['stop_idx_forecast'])
    t_split = t[i1_filt-1] if i1_filt > 0 else t[i0_fcst]

    # Layout
    bottom_rows = int(show_filtered_mse) + int(show_cumavg_filtered_mse)
    nrows = D + bottom_rows

    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(7.0, 1.8 * nrows),
        sharex=True, constrained_layout=False
    )
    if nrows == 1:
        axes = [axes]
    fig.subplots_adjust(right=0.76, hspace=0.15)

    method_handles, method_labels = [], []
    obs_handle = None  # for legend

    # === State plots ===
    for d in range(D):
        ax = axes[d]
        ax.plot(t, Xtrue[:, d], **TRUTH_STYLE)

        # Optional obs overlay (match times t)
        if plot_observations and d < min(D, E):
            # Small red x's, no line
            (obs_handle_line,) = ax.plot(
                t, Y[:, d], linestyle="None", marker=obs_marker,
                markersize=obs_markersize, color="red", alpha=0.4, zorder=50,
                label="observations"
            )
            if obs_handle is None:
                obs_handle = obs_handle_line  # keep one for legend

        for key in method_keys:
            r = results_dict[key]
            f_means = np.asarray(r['filtered']['filtered_means'])

            ln, = ax.plot(
                t[i0_filt:i1_filt], f_means[i0_filt:i1_filt, d],
                color=styles[key]['color'],
                linestyle=styles[key]['linestyle'],
                linewidth=styles[key]['linewidth'],
                marker=styles[key]['marker'],
                markersize=styles[key]['markersize'],
                zorder=styles[key]['zorder']
            )
            if d == 0:
                method_handles.append(ln)
                method_labels.append(styles[key]['label'])

            if plot_uncertainty and 'filtered_covariances' in r['filtered']:
                covs = np.asarray(r['filtered']['filtered_covariances'])
                std = np.sqrt(np.clip(covs[i0_filt:i1_filt, d, d], 0, np.inf))
                mean_seg = f_means[i0_filt:i1_filt, d]
                ax.fill_between(
                    t[i0_filt:i1_filt], mean_seg - 2*std, mean_seg + 2*std,
                    color=styles[key]['color'], alpha=uncertainty_alpha, linewidth=0
                )

            fc_means = np.asarray(r['forecasted']['forecasted_state_means'])
            t_fc = t[i0_fcst:i1_fcst]
            ax.plot(
                t_fc, fc_means[:, d],
                color=styles[key]['color'],
                linestyle=styles[key]['linestyle'],
                linewidth=styles[key]['linewidth'],
                marker=styles[key]['marker'],
                markersize=styles[key]['markersize'],
                zorder=styles[key]['zorder']
            )

            if plot_uncertainty and 'forecasted_state_covariances' in r['forecasted']:
                covs_f = np.asarray(r['forecasted']['forecasted_state_covariances'])
                std_f = np.sqrt(np.clip(covs_f[:, d, d], 0, np.inf))
                ax.fill_between(
                    t_fc, fc_means[:, d] - 2*std_f, fc_means[:, d] + 2*std_f,
                    color=styles[key]['color'], alpha=uncertainty_alpha, linewidth=0
                )

        ax.axvline(t_split, color='k', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_ylabel(f"$x_{d+1}$", fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    # === Helpers ===
    def compute_est_full(r):
        est_full = np.zeros_like(Xtrue)
        f_means = np.asarray(r['filtered']['filtered_means'])
        est_full[i0_filt:i1_filt, :] = f_means[i0_filt:i1_filt, :]
        fc_means = np.asarray(r['forecasted']['forecasted_state_means'])
        est_full[i0_fcst:i1_fcst, :] = fc_means
        return est_full

    mse_handles, mse_labels = [], []
    cum_handles, cum_labels = [], []

    # === Bottom: per-time MSE ===
    bottom_ax_idx = D
    if show_filtered_mse:
        axm = axes[bottom_ax_idx]
        bottom_ax_idx += 1
        axm.set_ylabel("MSE(t)\n(filtered)", fontsize=10)
        eps = 1e-16

        for key in method_keys:
            est_full = compute_est_full(results_dict[key])
            mse_t = _per_time_mse(Xtrue, est_full)
            mse_filt = np.clip(mse_t[i0_filt:i1_filt], eps, None)

            h, = axm.plot(
                t[i0_filt:i1_filt], mse_filt,
                color=styles[key]['color'],
                marker=styles[key]['marker'],
                markersize=styles[key]['markersize'],
                linewidth=styles[key]['linewidth'],
                zorder=styles[key]['zorder']
            )

            mse_filt_mean = _window_mean_sq_error(Xtrue, est_full, i0_filt, i1_filt)
            mse_fcst_mean = _window_mean_sq_error(Xtrue, est_full, i0_fcst, i1_fcst)
            label = f"{styles[key]['label']} | filt={mse_filt_mean:.3g}, fcst={mse_fcst_mean:.3g}"

            mse_handles.append(h)
            mse_labels.append(label)

        axm.axvline(t_split, color='k', linestyle='--', linewidth=1.2, alpha=0.8)
        axm.grid(True, alpha=0.25, linewidth=0.6)
        axm.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        axm.set_yscale('log')

    # === Bottom: cumulative MSE ===
    if show_cumavg_filtered_mse:
        axc = axes[bottom_ax_idx]
        axc.set_ylabel("Cum. MSE\n(filtered)", fontsize=10)
        eps = 1e-16

        for key in method_keys:
            est_full = compute_est_full(results_dict[key])
            mse_t = _per_time_mse(Xtrue, est_full)
            mse_filt = mse_t[i0_filt:i1_filt]
            cumavg = np.cumsum(mse_filt) / np.arange(1, len(mse_filt) + 1)

            h, = axc.plot(
                t[i0_filt:i1_filt], np.clip(cumavg, eps, None),
                color=styles[key]['color'],
                marker=styles[key]['marker'],
                markersize=styles[key]['markersize'],
                linewidth=styles[key]['linewidth'],
                zorder=styles[key]['zorder']
            )

            final_val = cumavg[-1]
            cum_labels.append(f"{styles[key]['label']} | final={final_val:.3g}")
            cum_handles.append(h)

        axc.axvline(t_split, color='k', linestyle='--', linewidth=1.2, alpha=0.8)
        axc.grid(True, alpha=0.25, linewidth=0.6)
        axc.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        axc.set_yscale('log')
        axc.set_xlabel("Time", fontsize=11)

    # === Legends ===
    dedup = {}
    for h, l in zip(method_handles, method_labels):
        if l not in dedup:
            dedup[l] = h
    method_labels = list(dedup.keys())
    method_handles = [dedup[l] for l in method_labels]

    # Add observations to the state legend (first, so it appears on top)
    if obs_handle is not None:
        method_handles = [obs_handle] + method_handles
        method_labels = ["observations"] + method_labels

    fig.legend(
        handles=method_handles, labels=method_labels,
        loc='center left', bbox_to_anchor=(0.78, 0.80),
        frameon=False, title="Methods (states)", fontsize=9, title_fontsize=10
    )
    if mse_handles:
        fig.legend(
            handles=mse_handles, labels=mse_labels,
            loc='center left', bbox_to_anchor=(0.78, 0.45),
            frameon=False, title="Filtered MSE (window means)", fontsize=9, title_fontsize=10
        )
    if cum_handles:
        fig.legend(
            handles=cum_handles, labels=cum_labels,
            loc='center left', bbox_to_anchor=(0.78, 0.25),
            frameon=False, title="Final CumMSE", fontsize=9, title_fontsize=10
        )

    fig.suptitle(
        "Filter-then-Forecast on States (vertical line = forecast start)",
        fontsize=12, y=0.995
    )
    outpath = os.path.join(eval_output_dir, save_name)
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {outpath}")

# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    # Create optional flags for comamand line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run experiment with specified configurations.')
    parser.add_argument('--data_config_file', type=str, default='configs/data/l63_data_x1')
    parser.add_argument('--model_config_file', type=str, default='configs/model/true_l63_mech_x1')
    # parser.add_argument('--filter_config_file', type=str, default='configs/filter/ekf_StateFirst_EmissionsFirst')
    parser.add_argument('--output_dir', type=str, default='results/filter_then_forecast')
    parser.add_argument('--T_filter', type=float, default=0.8,
                        help='Fraction of the data to use for filtering (default: 0.8)')
    parser.add_argument('--do_run', type=int, default=1)
    parser.add_argument('--do_eval', type=int, default=1)
    parser.add_argument('--enforce_twin_experiment', type=int, default=1,
                        help='If True, will enforce the twin experiment setup (default: False). Done by resetting the true_model_config_file in the data_config_file = model_config_file.')
    parser.add_argument('--data_key', type=int, default=None,
                        help='Optional key to use for data generation. If None, the key from the data config file will be used.')
    parser.add_argument('--ftf_key', type=int, default=None,
                        help='Key to use for filter-then-forecast. Default is None, in which case it will be set to data_key + 10.')
    args = parser.parse_args()
    # run_filter_then_forecast(**args.__dict__)

    # Full experiment looping over many filter configurations
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
    if args.do_run:
        print("Running filtering and forecasting for each filter configuration...")
        for name in filter_names:
            filter_config_file = f"configs/filter/{name}"
            run_filter_then_forecast(
                data_config_file=args.data_config_file,
                model_config_file=args.model_config_file,
                filter_config_file=filter_config_file,
                output_dir=args.output_dir,
                T_filter=args.T_filter,
                enforce_twin_experiment=args.enforce_twin_experiment,
                data_key=args.data_key,
                ftf_key=args.ftf_key
            )
    
    if args.do_eval:
        print("Evaluating the filtering and forecasting experiment...")
        # Evaluate the experiment
        eval_output_dir = os.path.join(args.output_dir,
                                    os.path.basename(args.data_config_file),
                                    os.path.basename(args.model_config_file),
                                    'eval')

        os.makedirs(eval_output_dir, exist_ok=True)
        result_dirs = [
            build_results_dir(args.data_config_file, args.model_config_file, f"configs/filter/{name}", args.output_dir)
            for name in filter_names
        ]
        eval_filter_then_forecast_experiment(
            data_config_file=args.data_config_file,
            list_of_result_dirs=result_dirs,
            enforce_twin_experiment=args.enforce_twin_experiment,
            model_config_file=args.model_config_file,
            eval_output_dir=eval_output_dir,
            data_key=args.data_key,
            ftf_key=args.ftf_key
        )
