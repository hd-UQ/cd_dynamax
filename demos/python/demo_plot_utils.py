import os
from configparser import ConfigParser
import pickle

# Plotting
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
        filter_info,
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
        "{} Filter-then-Forecast on States (vertical line = forecast start)".format(
            filter_info['name'] if filter_info is not None and 'name' in filter_info else ""
        ),
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
        filter_info,
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
        "{} Filter-then-Forecast on States (vertical line = forecast start)".format(
            filter_info['name'] if filter_info is not None and 'name' in filter_info else ""
        ),
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

# Function to compare the filtering and forecasting results for the states
def compare_filter_then_forecast_state_results(
        data,
        all_filter_results,
        all_filter_info,
        plot_filename,
        plot_uncertainty: bool = True,
        plot_observations: bool = True,
        plot_mse: bool = False,
        true_state_style = {'color': 'black', 'linestyle': '-', 'linewidth': 1.5},
        true_emission_style = {'color': 'gray', 'linestyle': 'None', 'marker': 'x', 'markersize': 3, 'alpha': 0.5},
        default_filtered_style = {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7},
        default_forecasted_style = {'color': 'green', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7},
        plot_dpi: int = 300,
        ):
    '''Compare the filtering and forecasting results for the states.
    Args:
        data (dict): Dictionary containing the data.
        all_filter_results (list): List of dictionaries containing the filtering and forecasting results for each filter.
        all_filter_info (list): List of dictionaries containing the filter information for each filter.
        results_files (list): List of result file paths for each filter.
        plot_uncertainty (bool): Whether to plot uncertainty intervals. (default: True)
        plot_observations (bool): Whether to plot observations. (default: True)
    '''

    # Figure out data dimensions
    d_states, d_emissions, t_emissions = figure_out_data_dimensions(data)

    # Figure out filtering and forecasting indices
    # Any index should work since they are the same for all filters
    results = all_filter_results[0]
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
        # Loop over each filter result to plot
        for filter_idx, results in enumerate(all_filter_results):
            ## Filter name and style
            this_filter_name = all_filter_info[filter_idx]['name'] if 'name' in all_filter_info[filter_idx] else "Filter {}".format(filter_idx + 1)
            this_filter_style = eval(all_filter_info[filter_idx]['filtered_style']) if 'filtered_style' in all_filter_info[filter_idx] else default_filtered_style

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
                label='Filtered State - {}'.format(
                    this_filter_name
                ) if d == 0 else "", # Add label only for the first plot
                line_style=this_filter_style,
                fill_style={
                    'color': this_filter_style['color'],
                    'alpha': this_filter_style['alpha'],
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
                    label='Filtered MSE - {}'.format(
                        this_filter_name
                    ) if d == 0 else "",
                    color=this_filter_style['color'],
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.7
                )
            
            ### Forecast name and style
            this_forecast_style = eval(all_filter_info[filter_idx]['forecasted_style']) if 'forecasted_style' in all_filter_info[filter_idx] else default_forecasted_style

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
                label="", #'Forecasted State' if d == 0 else "", # Add label only for the first plot
                line_style=this_forecast_style,
                fill_style={
                    'color': this_forecast_style['color'],
                    'alpha': this_forecast_style['alpha'],
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
                    label="", #'Forecasted MSE' if d == 0 else "",
                    color=this_forecast_style['color'],
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
        "Comparison of Filter-then-Forecast on States (vertical line = forecast start)",
        fontsize=10
    )

    # Save the figure, within a figures directory at results directory
    
    fig.savefig(
        plot_filename,
        dpi=plot_dpi,
        bbox_inches='tight' # Use bbox_inches='tight' as a safeguard
    )
    print("Compare Filter-then-Forecast states plot saved to:", plot_filename)
    plt.close(fig)