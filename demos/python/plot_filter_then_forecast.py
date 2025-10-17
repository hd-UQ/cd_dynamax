import os
from configparser import ConfigParser
import pickle

# CD-NLGSSM imports
from data_generator import generate_data_from_config
from cd_dynamax.src.utils.experiment_utils import *
from cd_dynamax.src.utils.simulation_utils import filter_and_forecast
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import update_params

def plot_filter_then_forecast(
    data_config_file,
    model_config_file,
    filter_config_file,
    output_dir,
    T_filter=0.8,
    enforce_twin_experiment=False,
    ftf_key=None,
    overrides={},
):
    '''Plot a filtering and forecasting experiment with specified configurations.

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
    '''
    # Figure out the directory structure
    results_dir = build_results_dir(
        output_dir,
        data_config_file,
        model_config_file,
        filter_config_file,
        overrides=overrides,
    )

    # Check if the results directory exists, if not, print a warning
    if not os.path.exists(results_dir):
        print("Warning: Results directory does not exist at:", results_dir)
        print("Please run the filtering and forecasting experiment first to generate results.")
        return

    if enforce_twin_experiment:
        # Override the true_model_config_file in data_config_file with the model's config file
        overrides['data_generation.true_model_config_file'] = model_config_file
        print(f"Enforcing twin experiment by setting 'data_generation.true_model_config_file' to {model_config_file}.")
    
    # Load data: the same function used for generation is used, it should simply load existing data
    data, data_key = generate_data_from_config(
        data_config_file=data_config_file,
        data_save_file=None,
        overrides=overrides,
    )

    # Create and initialize the CD-NLGSSM model from the model config file
    model, params, props = create_cdnlgssm_model_from_config(
        model_config_file,
        overrides=overrides,
    )

    # Figure-out the filtering/smoothing settings from config
    filter_hyperparams = create_cdnlgssm_filter_from_config(
        filter_config_file,
        overrides=overrides
    )

    # Assign forecasting and filtering time indices based on T_filter fraction.
    T0 = data['t_emissions'][0]
    T_forecast_end = data['t_emissions'][-1]
    T_filter_end = T0 + T_filter * (T_forecast_end - T0)

    # Set the key(s) for filtering and forecasting
    ftf_keys = [data_key + 10] if ftf_key is None else ftf_key  # i.e. 10 more than data_key or given ftf_key

    # Run filtering and forecasting, for each key in ftf_keys
    for ftf_key in ftf_keys if isinstance(ftf_keys, (list)) else [ftf_keys]:
        print(f"Loading filtering from T={T0} up to T={T_filter_end} and forecasting up to T={T_forecast_end} (with ftf_key={ftf_key}) results...")

        # Load the results
        results_file = os.path.join(results_dir, 'results_ftfkey{}.pkl'.format(ftf_key))
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        # Print a message indicating the completion of the experiment
        print("Results loaded from:", results_file)

        # Plot the filtering and forecasting state results, along with their MSE
        plot_filter_then_forecast_state_results(
            data=data,
            results=results,
            results_file=results_file,
            plot_uncertainty = True,
            plot_observations = True,
            plot_mse = True,
        )

        # Plot the filtering and forecasting state and emission results
        plot_filter_then_forecast_state_emission_results(
            data=data,
            results=results,
            results_file=results_file,
            plot_uncertainty=True,
            plot_observations=True,
        )


# Plotting functions
import matplotlib.pyplot as plt
import numpy as np

# Figure out the data dimensions
def figure_out_data_dimensions(data):
    '''Figure out the data dimensions from the data dictionary.
    Args:
        data (dict): Dictionary containing the data.
    Returns:
        d_states (int): Dimension of the states.
        d_emissions (int): Dimension of the emissions.
        t_emissions (np.ndarray): Time vector for the emissions.
    '''
    
    t_emissions = data['t_emissions'][:,0]
    assert data['states'].shape[0] == t_emissions.shape[0], "States and time vector length mismatch."
    assert data['emissions'].shape[0] == t_emissions.shape[0], "Emissions and time vector length mismatch."
    d_states = data['states'].shape[1]
    d_emissions = data['emissions'].shape[1]

    return d_states, d_emissions, t_emissions

# Plotting helpers
def data_plotter(
    ax,
    t_idx,
    states,
    state_label,
    state_style,
    emissions,
    emission_label,
    emissions_style = {'color': 'gray', 'linestyle': 'None', 'marker': 'x', 'markersize': 3, 'alpha': 0.5},
    plot_observations: bool = True,
):
    '''Plot data on the given axes.
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        t_idx (np.ndarray): Time indices.
        data (np.ndarray): Data values.
        label (str): Label for the plot.
        style (dict): Style for the plot.
    '''
    if states is not None:
        ax.plot(
            t_idx,
            states,
            label=state_label,
            **state_style
        )
    
    # Emissions
    if plot_observations:
        ax.plot(
            t_idx,
            emissions,
            label=emission_label,
            **emissions_style
        )

def ts_plotter(
    ax,
    t_idx,
    f_mean,
    f_std,
    label,
    line_style,
    fill_style,
    plot_uncertainty: bool = True,
):
    '''Plot a time series with optional uncertainty intervals.
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        t_idx (np.ndarray): Time indices.
        f_mean (np.ndarray): Mean values.
        f_std (np.ndarray): Standard deviation values.
        label (str): Label for the plot.
        line_style (dict): Line style for the mean plot.
        fill_style (dict): Fill style for the uncertainty intervals.
        plot_uncertainty (bool): Whether to plot uncertainty intervals. (default: True)
    '''
    ax.plot(
        t_idx,
        f_mean,
        label=label,
        **line_style
    )
    if plot_uncertainty:
        ax.fill_between(
            t_idx,
            f_mean - 2*f_std,
            f_mean + 2*f_std,
            **fill_style
        )

# Plot vertical line at the split time
def plot_vertical_split_line(
        ax,
        t_split,
        vertical_line_style = {'color': 'k', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.8},
    ):
    '''Plot a vertical line at the split time.
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        t_split (float): Time to plot the vertical line at.
        vertical_line_style (dict): Style for the vertical line.
    '''
    ax.axvline(
        t_split,
        **vertical_line_style
    )

# Function to plot the filtering and forecasting results for the states
def plot_filter_then_forecast_state_results(
        data,
        results,
        results_file,
        plot_uncertainty: bool = True,
        plot_observations: bool = True,
        plot_mse: bool = False,
        true_state_style = {'color': 'black', 'linestyle': '-', 'linewidth': 1.5},
        true_emission_style = {'color': 'gray', 'linestyle': 'None', 'marker': 'x', 'markersize': 3, 'alpha': 0.5},
        filtered_style = {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7},
        forecasted_style = {'color': 'green', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7},
        plot_dpi: int = 300,
        ):
    
    '''Plot the filtering and forecasting results.
    Args:
        data (dict): Dictionary containing the data.
        results (dict): Dictionary containing the filtering and forecasting results.
        plot_uncertainty (bool): Whether to plot uncertainty intervals. (default: True)
        plot_observations (bool): Whether to plot observations. (default: True)
    '''

    # Figure out data dimensions
    d_states, d_emissions, t_emissions = figure_out_data_dimensions(data)

    # Figure out filtering and forecasting indices
    start_idx_filter=results['start_idx_filter']
    stop_idx_filter=results['stop_idx_filter']
    start_idx_forecast=results['start_idx_forecast']
    stop_idx_forecast=results['stop_idx_forecast']

    # Last time index used for filtering
    t_split = t_emissions[stop_idx_filter - 1]  if stop_idx_filter > 0 else t_emissions[start_idx_forecast]
    
    # Plot Layout
    plot_rows = d_states
    plot_cols = 1

    # Plot MSE if required, in second column
    if plot_mse:
        plot_cols += 1

    fig, axes = plt.subplots(
        nrows=plot_rows,
        ncols=plot_cols,
        sharex=True,
        constrained_layout=True
    )
    if plot_rows == 1:
        axes = [axes]

    # === State plots ===
    for d in range(d_states):
        ### True data
        data_plotter(
            ax=axes[d] if plot_mse == False else axes[d][0],
            t_idx=t_emissions,
            states=data['states'][:, d],
            state_label='True State' if d == 0 else "", # Add label only for the first plot
            state_style=true_state_style,
            emissions=data['emissions'][:, d],
            emission_label='True Observation' if d == 0 else "", # Add label only for the first plot
            emissions_style=true_emission_style,
            plot_observations=plot_observations,
        )

        ### Filtered time-series
        ts_plotter(
            ax=axes[d] if plot_mse == False else axes[d][0],
            t_idx=t_emissions[start_idx_filter:stop_idx_filter],
            f_mean=results['filtered']['filtered_means'][:, d],
            f_std=np.sqrt(
                np.clip(
                    np.asarray(results['filtered']['filtered_covariances'])[:, d, d],
                    0, np.inf
                )
            ),
            label='Filtered State' if d == 0 else "", # Add label only for the first plot
            line_style=filtered_style,
            fill_style={
                'color': filtered_style['color'],
                'alpha': filtered_style['alpha'],
                'linewidth': 0
            },
            plot_uncertainty=plot_uncertainty
        )

        ### Filtered MSE
        if plot_mse:
            # Compute MSE for filtered states
            filtered_mse = (data['states'][start_idx_filter:stop_idx_filter, d] - results['filtered']['filtered_means'][:, d])**2

            # In new column, plot the MSE
            axes[d][1].plot(
                t_emissions[start_idx_filter:stop_idx_filter],
                filtered_mse,
                label='Filtered MSE' if d == 0 else "",
                color='red',
                linestyle='--',
                linewidth=1.0,
                alpha=0.7
            )

        ### Forecasted time-series
        ts_plotter(
            ax=axes[d] if plot_mse == False else axes[d][0],
            t_idx=t_emissions[start_idx_forecast:stop_idx_forecast],
            f_mean=results['forecasted']['forecasted_state_means'][:, d],
            f_std=np.sqrt(
                np.clip(
                    np.asarray(results['forecasted']['forecasted_state_covariances'])[:, d, d],
                    0, np.inf
                )
            ),
            label='Forecasted State' if d == 0 else "", # Add label only for the first plot
            line_style=forecasted_style,
            fill_style={
                'color': forecasted_style['color'],
                'alpha': forecasted_style['alpha'],
                'linewidth': 0
            },
            plot_uncertainty=plot_uncertainty
        )

        ### Forecasted MSE
        if plot_mse:
            # Compute MSE for forecasted states
            forecasted_mse = (data['states'][start_idx_forecast:stop_idx_forecast, d] - results['forecasted']['forecasted_state_means'][:, d])**2

            # In new column, plot the MSE
            axes[d][1].plot(
                t_emissions[start_idx_forecast:stop_idx_forecast],
                forecasted_mse,
                label='Forecasted MSE' if d == 0 else "",
                color='orange',
                linestyle='--',
                linewidth=1.0,
                alpha=0.7
            )
        
        # Plot vertical line at the split time
        plot_vertical_split_line(
            axes[d] if plot_mse == False else axes[d][0],
            t_split,
            vertical_line_style={'color': 'k', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.8}
        )

        # Set titles and labels
        (axes[d] if plot_mse == False else axes[d][0]).set_title(f"State $x_{d+1}$ over time", fontsize=8)
        (axes[d] if plot_mse == False else axes[d][0]).set_ylabel(f"$x_{d+1}$", fontsize=8)
        (axes[d] if plot_mse == False else axes[d][0]).grid(True)
        if plot_mse:
            axes[d][1].set_title(f"State $x_{d+1}$ MSE over time", fontsize=8)
            axes[d][1].set_ylabel(f"MSE of $x_{d+1}$", fontsize=8)
            axes[d][1].grid(True)

    # time xlabel for the last state plot
    (axes[d] if plot_mse == False else axes[d][0]).set_xlabel("Time", fontsize=8)
    if plot_mse:
        axes[d][1].set_xlabel("Time", fontsize=8)

    # Overall figure legend
    fig.legend(
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8
    )

    fig.suptitle(
        "Filter-then-Forecast on States (vertical line = forecast start)",
        fontsize=10
    )

    # Save the figure, within a figures directory at results directory
    plot_results_dir = os.path.join(os.path.dirname(results_file), 'figures')
    os.makedirs(plot_results_dir, exist_ok=True)
    plot_file = os.path.join(
        plot_results_dir,
        'filter_then_forecast_states_ftfkey{}.png'.format(
            results_file.split('ftfkey')[-1].split('.pkl')[0]
        )
    )
    fig.savefig(
        plot_file,
        dpi=plot_dpi,
        bbox_inches='tight' # Use bbox_inches='tight' as a safeguard
    )
    print("Filter-then-Forecast states plot saved to:", plot_file)
    plt.close(fig)

# Function to plot the filtering and forecasting results for the states and emissions
def plot_filter_then_forecast_state_emission_results(
        data,
        results,
        results_file,
        plot_uncertainty: bool = True,
        plot_observations: bool = True,
        true_state_style = {'color': 'black', 'linestyle': '-', 'linewidth': 1.5},
        true_emission_style = {'color': 'gray', 'linestyle': 'None', 'marker': 'x', 'markersize': 3, 'alpha': 0.5},
        filtered_style = {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7},
        forecasted_style = {'color': 'green', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7},
        plot_dpi: int = 300,
        ):
    '''Plot the filtering and forecasting results for states and emissions.
    Args:
        data (dict): Dictionary containing the data.
        results (dict): Dictionary containing the filtering and forecasting results.
        plot_uncertainty (bool): Whether to plot uncertainty intervals. (default: True)
        plot_observations (bool): Whether to plot observations. (default: True)
        plot_dpi (int): Dots per inch for the plot. (default: 300)
    '''

    # Figure out data dimensions
    d_states, d_emissions, t_emissions = figure_out_data_dimensions(data)

    # Figure out filtering and forecasting indices
    start_idx_filter=results['start_idx_filter']
    stop_idx_filter=results['stop_idx_filter']
    start_idx_forecast=results['start_idx_forecast']
    stop_idx_forecast=results['stop_idx_forecast']
    # Last time index used for filtering
    t_split = t_emissions[stop_idx_filter - 1]  if stop_idx_filter > 0 else t_emissions[start_idx_forecast]
    
    # Plot Layout: two columns, one for states, one for emissions
    plot_rows = d_states
    plot_cols = 2  # States and Emissions

    fig, axes = plt.subplots(
        nrows=plot_rows,
        ncols=plot_cols,
        sharex=True,
        constrained_layout=True
    )

    if plot_rows == 1:
        axes = [axes]

    # === State plots ===
    for d in range(d_states):

        # Left column: States
        ### True data
        data_plotter(
            ax=axes[d][0],
            t_idx=t_emissions,
            states=data['states'][:, d],
            state_label='True State' if d == 0 else "", # Add label only for the first plot
            state_style=true_state_style,
            emissions=None,
            emission_label=None,
            emissions_style=None,
            plot_observations=False,
        )
        ### Filtered
        ts_plotter(
            ax=axes[d][0],
            t_idx=t_emissions[start_idx_filter:stop_idx_filter],
            f_mean=results['filtered']['filtered_means'][:, d],
            f_std=np.sqrt(
                np.clip(
                    np.asarray(results['filtered']['filtered_covariances'])[:, d, d],
                    0, np.inf
                )
            ),
            label='Filtered State' if d == 0 else "", # Add label only for the first plot
            line_style=filtered_style,
            fill_style={
                'color': filtered_style['color'],
                'alpha': filtered_style['alpha'],
                'linewidth': 0
            },
            plot_uncertainty=plot_uncertainty
        )
        ### Forecasted
        ts_plotter(
            ax=axes[d][0],
            t_idx=t_emissions[start_idx_forecast:stop_idx_forecast],
            f_mean=results['forecasted']['forecasted_state_means'][:, d],
            f_std=np.sqrt(
                np.clip(
                    np.asarray(results['forecasted']['forecasted_state_covariances'])[:, d, d],
                    0, np.inf
                )
            ),
            label='Forecasted State' if d == 0 else "", # Add label only for the first plot
            line_style=forecasted_style,
            fill_style={
                'color': forecasted_style['color'],
                'alpha': forecasted_style['alpha'],
                'linewidth': 0
            },
            plot_uncertainty=plot_uncertainty
        )

        # Plot vertical line at the split time
        plot_vertical_split_line(
            axes[d][0],
            t_split,
            vertical_line_style={'color': 'k', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.8}
        )

        # Set titles and labels
        axes[d][0].set_title(f"State $x_{d+1}$ over time", fontsize=8)
        axes[d][0].set_ylabel(f"$x_{d+1}$", fontsize=8)
        axes[d][0].grid(True)

    # === Emission plots ===
    for d in range(d_emissions):
        # Right column: Emissions
        ### True data
        data_plotter(
            ax=axes[d][1],
            t_idx=t_emissions,
            states=None,  # No states to plot here
            state_label="",
            state_style={},
            emissions=data['emissions'][:, d],
            emission_label='True Observation' if d == 0 else "", # Add label only for the first plot
            emissions_style=true_emission_style,
            plot_observations=plot_observations,
        )
        ### Filtered emissions
        ts_plotter(
            ax=axes[d][1],
            t_idx=t_emissions[start_idx_filter:stop_idx_filter],
            f_mean=results['filtered']['filtered_emissions_means'][:,d],
            f_std=np.sqrt(
                np.clip(
                    np.asarray(results['filtered']['filtered_emissions_covariances'])[:, d, d],
                    0, np.inf
                )
            ),
            label='Filtered Emission' if d == 0 else "", # Add label only for the first plot
            line_style=filtered_style,
            fill_style={
                'color': filtered_style['color'],
                'alpha': filtered_style['alpha'],
                'linewidth': 0
            },
            plot_uncertainty=plot_uncertainty
        )
        ### Forecasted emissions
        ts_plotter(
            ax=axes[d][1],
            t_idx=t_emissions[start_idx_forecast:stop_idx_forecast],
            f_mean=results['forecasted']['forecasted_emissions_means'][:,d],
            f_std=np.sqrt(
                np.clip(
                    np.asarray(results['forecasted']['forecasted_emissions_covariances'])[:,d, d],
                    0, np.inf
                )
            ),
            label='Forecasted Emission' if d == 0 else "", # Add label only for the first plot
            line_style=forecasted_style,
            fill_style={
                'color': forecasted_style['color'],
                'alpha': forecasted_style['alpha'],
                'linewidth': 0
            },
            plot_uncertainty=plot_uncertainty
        )

        # Plot vertical line at the split time
        plot_vertical_split_line(
            axes[d][1],
            t_split,
            vertical_line_style={'color': 'k', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.8}
        )

        # Set titles and labels
        axes[d][1].set_title(f"Emission $y_{d+1}$ over time", fontsize=8)
        axes[d][1].set_ylabel(f"$y_{d+1}$", fontsize=8)
        axes[d][1].grid(True)

    # time xlabel for the last state plot
    axes[d][0].set_xlabel("Time", fontsize=8)
    axes[d][1].set_xlabel("Time", fontsize=8)

    # Overall figure legend
    fig.legend(
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8
    )

    fig.suptitle(
        "Filter-then-Forecast on States and Emissions (vertical line = forecast start time)",
        fontsize=8
    )

    # Save the figure, within a figures directory at results directory
    plot_results_dir = os.path.join(os.path.dirname(results_file), 'figures')
    os.makedirs(plot_results_dir, exist_ok=True)
    plot_file = os.path.join(
        plot_results_dir,
        'filter_then_forecast_states_emissions_ftfkey{}.png'.format(
            results_file.split('ftfkey')[-1].split('.pkl')[0]
        )
    )
    fig.savefig(
        plot_file,
        bbox_inches='tight',
        dpi=plot_dpi
    )
    print("Filter-then-Forecast states and emissions plot saved to:", plot_file)
    plt.close(fig)



# Main script gets two arguments: config file and data save file
if __name__ == "__main__":
    
    # Create optional flags for comamand line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot the filter then forecast experiment with specified configurations.')
    parser.add_argument('--data_config_file', type=str, default='configs/data/true_l63_data_x1',
                        help='Data configuration file: (default: true_l63_data_x1)')
    parser.add_argument('--model_config_file', type=str, default='configs/model/true_l63_mech_x1',
                        help='Model configuration file: (default: true_l63_mech_x1)')
    parser.add_argument('--enforce_twin_experiment', type=bool, default=False,
                        help='If True, will enforce the twin experiment setup (default: False). Done by resetting the true_model_config_file in the data_config_file = model_config_file.')
    # Allow user to specify multiple filter config files or "all"
    parser.add_argument('--filter_config_file', nargs='+', default='configs/filter/ekf_StateFirst_EmissionsFirst', 
                        help='Filter configuration file: (default: ekf_StateFirst_EmissionsFirst), if all filters should be used, set to "all"'
                        )
    parser.add_argument('--output_dir', type=str, default='results/filter_then_forecast',
                        help='Output directory for results: (default: results/filter_then_forecast)')
    parser.add_argument('--T_filter', type=float, default=0.8,
                        help='Fraction of the data to use for filtering (default: 0.8)')   
    # Add optional data_key and ftf_key arguments, which can be a sequence of integers
    parser.add_argument('--data_key', type=int, nargs='+', default=None,
                        help='Optional key to use for data generation. If None, the key from the data config file will be used.')
    parser.add_argument('--ftf_key', type=int, nargs='+', default=None,
                        help='Key to use for filter-then-forecast. Default is None, in which case it will be set to data_key + 10.')
    args = parser.parse_args()

    # Process the filter_config_file argument to allow running multiple filter files
    if len(args.filter_config_file) == 1:
        filter_config_files = args.filter_config_file

        if filter_config_files[0].lower() == "all":
            filter_config_files = [
                "configs/filter/ekf_StateFirst_EmissionsFirst",
                "configs/filter/ekf_StateSecond_EmissionsFirst",
                "configs/filter/ekf_StateZeroth_EmissionsFirst",
                "configs/filter/enkf_StateFirst",
                "configs/filter/enkf_StateZero",
                "configs/filter/ukf_StateFirst",
                "configs/filter/ukf_StateZeroth",
            ]
    else:
        filter_config_files = args.filter_config_file

    # Iterate over filter config files
    print("Plotting filtering and forecasting experiment...")
    for filter_config_file in filter_config_files:
        # Prepare overrides dictionary for this run
        overrides = {}
        if args.data_key is not None:
            for data_key in args.data_key:
                overrides['data_generation.key'] = data_key

                print(f"\t with: {filter_config_file} and overrides: {overrides}")
                plot_filter_then_forecast(
                    data_config_file=args.data_config_file,
                    model_config_file=args.model_config_file,
                    filter_config_file=filter_config_file,
                    output_dir=args.output_dir,
                    T_filter=args.T_filter,
                    enforce_twin_experiment=args.enforce_twin_experiment,
                    ftf_key=args.ftf_key,
                    overrides=overrides
                )
        else:
            print(f"\t with: {filter_config_file} and no overrides")
            plot_filter_then_forecast(
                data_config_file=args.data_config_file,
                model_config_file=args.model_config_file,
                filter_config_file=filter_config_file,
                output_dir=args.output_dir,
                T_filter=args.T_filter,
                enforce_twin_experiment=args.enforce_twin_experiment,
                ftf_key=args.ftf_key,
            )