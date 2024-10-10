# Imports
import numpy as np
from matplotlib import pyplot as plt
import jax


def plot_simple(
    time_grid=None,
    true_states=None,
    true_emissions_noisy=None,
    filtered_states=None,
    t_start=None,
    t_end=None,
):
    """
    Wrapper function to plot a simple time series.

    Parameters:
    - time_grid (array-like, optional): The time points corresponding to the true states and emissions.
    - true_states (ndarray, optional): The ground truth states, shape (T, n_states).
    - true_emissions_noisy (ndarray, optional): Noisy emissions from the true model, shape (T, n_emissions).
    - filtered_states (ndarray, optional): Filtered states using the learned model, shape (T, n_states).
    - t_start (float, optional): The start time for plotting.
    - t_end (float, optional): The end time for plotting.

    """

    plot_advanced(
        time_grid_all=time_grid,
        true_states=true_states,
        true_emissions_noisy=true_emissions_noisy,
        model_filtered_states=filtered_states,
        t_start=t_start,
        t_end=t_end,
    )


def plot_advanced(
    time_grid_all=None,
    time_grid_filter=None,
    time_grid_forecast=None,
    true_states=None,
    true_filtered_states=None,
    model_filtered_states=None,
    true_filtered_covariances=None,
    model_filtered_covariances=None,
    model_forecast_covariances=None,
    true_forecast_states=None,
    model_forecast_states=None,
    true_emissions_noisy=None,
    emission_function=None,
    t_start=None,
    t_end=None,
):
    """
    Plot the true states and emissions, as well as filtered and forecasted states and emissions from true and learned models.

    Parameters:
    - time_grid_all (array-like, optional): The time points corresponding to the true states and emissions.
    - time_grid_filter (array-like, optional): The time points corresponding to the filtered states and emissions.
    - time_grid_forecast (array-like, optional): The time points corresponding to the forecasted states and emissions.
    - true_states (ndarray, optional): The ground truth states, shape (T, n_states).
    - true_filtered_states (ndarray, optional): Filtered states using the true model, shape (T, n_states).
    - model_filtered_states (ndarray, optional): Filtered states using the learned model, shape (T, n_states).
    - true_filtered_covariances (ndarray, optional): Covariance matrices for filtered states using the true model, shape (T, n_states, n_states).
    - model_filtered_covariances (ndarray, optional): Covariance matrices for filtered states using the learned model, shape (T, n_states, n_states).
    - model_forecast_covariances (ndarray, optional): Covariance matrices for forecasted states using the learned model, shape (T, n_states, n_states).
    - true_forecast_states (ndarray, optional): Forecasted states using the true model, shape (T, n_states).
    - model_forecast_states (ndarray, optional): Forecasted states using the learned model, shape (T, n_states).
    - true_emissions_noisy (ndarray, optional): Noisy emissions from the true model, shape (T, n_emissions).
    - emission_function (LearnableLinear, optional): A function that computes emissions from states.
    - t_start (float, optional): The start time for plotting.
    - t_end (float, optional): The end time for plotting.
    """
    # Set default time grid if not specified
    plot_divider = True
    if time_grid_forecast is None:
        time_grid_forecast = time_grid_all
        plot_divider = False
    if time_grid_filter is None:
        time_grid_filter = time_grid_all
        plot_divider = False
    if time_grid_all is None:
        raise ValueError("time_grid_all must be specified if time_grid_filter or time_grid_forecast are not provided.")

    # Ensure at least one time grid is specified
    if time_grid_all is None and time_grid_filter is None and time_grid_forecast is None:
        raise ValueError("One of time_grid_all, time_grid_filter, or time_grid_forecast must be specified.")

    # squeeze time grids to ensure they are 1D arrays
    time_grid_all = np.squeeze(time_grid_all)
    time_grid_filter = np.squeeze(time_grid_filter)
    time_grid_forecast = np.squeeze(time_grid_forecast)

    # Determine indices based on t_start and t_end for all time grids
    if t_start is not None:
        start_idx_all = np.searchsorted(time_grid_all, t_start)
        start_idx_filter = np.searchsorted(time_grid_filter, t_start)
        start_idx_forecast = np.searchsorted(time_grid_forecast, t_start)
    else:
        start_idx_all = start_idx_filter = start_idx_forecast = 0

    if t_end is not None:
        end_idx_all = np.searchsorted(time_grid_all, t_end, side="right")
        end_idx_filter = np.searchsorted(time_grid_filter, t_end, side="right")
        end_idx_forecast = np.searchsorted(time_grid_forecast, t_end, side="right")
    else:
        end_idx_all = len(time_grid_all)
        end_idx_filter = len(time_grid_filter)
        end_idx_forecast = len(time_grid_forecast)

    # Subset the time grids based on start and end indices
    time_grid_all = time_grid_all[start_idx_all:end_idx_all]
    time_grid_filter = time_grid_filter[start_idx_filter:end_idx_filter]
    time_grid_forecast = time_grid_forecast[start_idx_forecast:end_idx_forecast]

    # Subset the time and state arrays based on start and end indices for each grid
    if true_states is not None:
        true_states = true_states[start_idx_all:end_idx_all, :]
    if true_filtered_states is not None:
        true_filtered_states = true_filtered_states[start_idx_filter:end_idx_filter, :]
    if model_filtered_states is not None:
        model_filtered_states = model_filtered_states[start_idx_filter:end_idx_filter, :]
    if true_filtered_covariances is not None:
        true_filtered_covariances = true_filtered_covariances[start_idx_filter:end_idx_filter, :, :]
    if model_filtered_covariances is not None:
        model_filtered_covariances = model_filtered_covariances[start_idx_filter:end_idx_filter, :, :]
    if true_forecast_states is not None:
        true_forecast_states = true_forecast_states[start_idx_forecast:end_idx_forecast, :]
    if model_forecast_states is not None:
        model_forecast_states = model_forecast_states[start_idx_forecast:end_idx_forecast, :]
    if model_forecast_covariances is not None:
        model_forecast_covariances = model_forecast_covariances[start_idx_forecast:end_idx_forecast, :, :]
    if true_emissions_noisy is not None:
        true_emissions_noisy = true_emissions_noisy[start_idx_all:end_idx_all, :]

    # Determine the number of rows for subplots
    n_states = true_states.shape[1] if true_states is not None else 0
    n_emissions = (
        true_emissions_noisy.shape[1]
        if true_emissions_noisy is not None
        else (
            emission_function.f(true_states[0]).shape[0]
            if true_states is not None and emission_function is not None
            else 0
        )
    )
    n_rows = max(n_states, n_emissions)

    # Create a canvas with subplots for states and emissions
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 2 * n_rows), sharex=True)

    # Use vmap to apply emission_function across the batch of states
    if emission_function is not None:
        vmap_emission = jax.vmap(emission_function.f, in_axes=(0, None, None))

    # Define color and linestyle profiles for different data types
    true_color = "k"
    true_filtered_color = "gray"
    model_filtered_color = "C0"
    linestyle_true = "-"
    linestyle_filtered = "--"
    linestyle_forecast = ":"
    marker_noisy_emission = "x"

    # Plot the states in the first column
    if true_states is not None:
        for i in range(n_states):
            # Plot true state values
            axes[i, 0].plot(
                time_grid_all, true_states[:, i], linestyle_true, color=true_color, alpha=0.5, linewidth=2, label=f"True State {i}"
            )
            # Plot filtered state values from true model
            if true_filtered_states is not None:
                axes[i, 0].plot(
                    time_grid_filter,
                    true_filtered_states[:, i],
                    color=true_filtered_color,
                    linestyle=linestyle_filtered,
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Filtered State {i}",
                )
                # Plot uncertainty bounds if available
                if true_filtered_covariances is not None and true_filtered_covariances.shape[0] == time_grid_filter.shape[0]:
                    axes[i, 0].fill_between(
                        time_grid_filter,
                        true_filtered_states[:, i] - np.sqrt(true_filtered_covariances[:, i, i]),
                        true_filtered_states[:, i] + np.sqrt(true_filtered_covariances[:, i, i]),
                        color=true_filtered_color,
                        alpha=0.3,
                    )
            # Plot filtered state values from learned model
            if model_filtered_states is not None:
                axes[i, 0].plot(
                    time_grid_filter,
                    model_filtered_states[:, i],
                    color=model_filtered_color,
                    linestyle=linestyle_filtered,
                    alpha=0.7,
                    linewidth=2,
                    label=f"Model Filtered State {i}",
                )
                # Plot uncertainty bounds if available
                if model_filtered_covariances is not None and model_filtered_covariances.shape[0] == time_grid_filter.shape[0]:
                    axes[i, 0].fill_between(
                        time_grid_filter,
                        model_filtered_states[:, i] - np.sqrt(model_filtered_covariances[:, i, i]),
                        model_filtered_states[:, i] + np.sqrt(model_filtered_covariances[:, i, i]),
                        color=model_filtered_color,
                        alpha=0.3,
                    )
            # Plot forecast state values from true model
            if true_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid_forecast,
                    true_forecast_states[:, i],
                    color=true_color,
                    linestyle=linestyle_forecast,
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Forecast State {i}",
                )
            # Plot forecast state values from learned model
            if model_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid_forecast,
                    model_forecast_states[:, i],
                    color=model_filtered_color,
                    linestyle=linestyle_forecast,
                    alpha=0.7,
                    linewidth=2,
                    label=f"Model Forecast State {i}",
                )
                # Plot uncertainty bounds if available
                if model_forecast_covariances is not None and model_forecast_covariances.shape[0] == time_grid_forecast.shape[0]:
                    axes[i, 0].fill_between(
                        time_grid_forecast,
                        model_forecast_states[:, i] - np.sqrt(model_forecast_covariances[:, i, i]),
                        model_forecast_states[:, i] + np.sqrt(model_forecast_covariances[:, i, i]),
                        color=model_filtered_color,
                        alpha=0.3,
                    )

            # Set y-axis label and add legend
            axes[i, 0].set_ylabel(f"State {i}")
            axes[i, 0].legend(loc="lower left")

    # Plot the emissions in the second column
    for i in range(n_emissions):
        # Plot noisy true emissions if available
        if true_emissions_noisy is not None:
            axes[i, 1].plot(
                time_grid_all,
                true_emissions_noisy[:, i],
                marker_noisy_emission,
                color=true_color,
                alpha=0.5,
                ms=3,
                label=f"Noisy True Emission {i}",
            )

        # Plot filtered emission values from true model
        if emission_function is not None and true_filtered_states is not None:
            true_filtered_emissions = vmap_emission(true_filtered_states, None, None)
            axes[i, 1].plot(
                time_grid_filter,
                true_filtered_emissions[:, i],
                color=true_filtered_color,
                linestyle=linestyle_filtered,
                alpha=0.7,
                linewidth=2,
                label=f"True Filtered Emission {i}"
            )
        # Plot filtered emission values from learned model
        if emission_function is not None and model_filtered_states is not None:
            model_filtered_emissions = vmap_emission(model_filtered_states, None, None)
            axes[i, 1].plot(
                time_grid_filter,
                model_filtered_emissions[:, i],
                color=model_filtered_color,
                linestyle=linestyle_filtered,
                alpha=0.7,
                linewidth=2,
                label=f"Model Filtered Emission {i}",
            )
        # Plot forecast emission values from true model
        if emission_function is not None and true_forecast_states is not None:
            true_forecast_emissions = vmap_emission(true_forecast_states, None, None)
            axes[i, 1].plot(
                time_grid_forecast,
                true_forecast_emissions[:, i],
                color=true_color,
                linestyle=linestyle_forecast,
                alpha=0.7,
                linewidth=2,
                label=f"True Forecast Emission {i}"
            )
        # Plot forecast emission values from learned model
        if emission_function is not None and model_forecast_states is not None:
            model_forecast_emissions = vmap_emission(model_forecast_states, None, None)
            axes[i, 1].plot(
                time_grid_forecast,
                model_forecast_emissions[:, i],
                color=model_filtered_color,
                linestyle=linestyle_forecast,
                alpha=0.7,
                linewidth=2,
                label=f"Model Forecast Emission {i}"
            )

        # Set y-axis label and add legend
        axes[i, 1].set_ylabel(f"Emission {i}")
        axes[i, 1].legend(loc="lower left")

    # Plot a vertical line to indicate the switch between filtered and forecasted states if both are provided
    if plot_divider:
        switch_time = time_grid_filter[-1]
        for i in range(n_rows):
            axes[i, 0].axvline(x=switch_time, color="k", linestyle="--", linewidth=1, label="Filter/Forecast Boundary")
            axes[i, 1].axvline(x=switch_time, color="k", linestyle="--", linewidth=1)

    # Set x-axis label only on the bottom subplots
    if n_rows > 0:
        for ax in axes[-1, :]:
            ax.set_xlabel("Time $t$")

    # Set a super title for the entire figure
    plt.suptitle("True vs Filtered vs Forecast States and Emissions: True and Learned Models")

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the title
    plt.show()
