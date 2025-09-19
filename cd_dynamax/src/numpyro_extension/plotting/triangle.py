import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def triangle_plot(true_values, posterior_samples, burn_in_frac=0.0, variables=None):
    """
    Make a triangle plot of posterior samples, comparing with true values.
    Handles scalar, vector, and matrix parameters by flattening into scalars.

    Parameters
    ----------
    true_values : dict
        Dictionary of true parameter values, with keys matching posterior_samples.
        Scalars, vectors, and matrices supported.
    posterior_samples : dict
        Dictionary of posterior samples; each entry shape (N,), (N, D), or (N, D1, D2).
    burn_in_frac : float, optional
        Fraction of initial samples to discard as burn-in.
    variables : list, optional
        List of variable names to include (default: all keys in posterior_samples).
    """
    # -------------------------
    # Flatten samples with unique separator "__"
    # -------------------------
    samples = {}
    if variables is None:
        variables = list(posterior_samples.keys())

    for var in variables:
        arr = np.asarray(posterior_samples[var])
        if arr.ndim == 1:  # scalar
            samples[var] = arr
        elif arr.ndim == 2:  # vector
            for i in range(arr.shape[1]):
                samples[f"{var}__{i}"] = arr[:, i]
        elif arr.ndim == 3:  # matrix
            for i in range(arr.shape[1]):
                for j in range(arr.shape[2]):
                    samples[f"{var}__{i}_{j}"] = arr[:, i, j]
        else:
            raise ValueError(f"Unsupported shape {arr.shape} for variable '{var}'")

    # -------------------------
    # Burn-in removal
    # -------------------------
    burn_idx = int(len(next(iter(samples.values()))) * burn_in_frac)
    for k in samples:
        samples[k] = samples[k][burn_idx:]

    # -------------------------
    # Helper to extract true value given flattened name
    # -------------------------
    def get_true_value(name):
        if "__" not in name:  # scalar
            return true_values[name]
        base, idx_str = name.split("__", 1)
        if "_" not in idx_str:  # vector index
            return true_values[base][int(idx_str)]
        else:  # matrix index "i_j"
            i, j = map(int, idx_str.split("_"))
            return true_values[base][i, j]

    # -------------------------
    # Prepare plot
    # -------------------------
    param_names = list(samples.keys())
    ndim = len(param_names)
    fig, axes = plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if i == j:
                sns.kdeplot(samples[param_names[i]], fill=True, ax=ax, linewidth=1.2)
                ax.axvline(get_true_value(param_names[i]), color="red", linestyle="--")
            elif i > j:
                sns.kdeplot(
                    x=samples[param_names[j]],
                    y=samples[param_names[i]],
                    fill=True, ax=ax, cmap="Blues", thresh=0.05
                )
                ax.plot(get_true_value(param_names[j]),
                        get_true_value(param_names[i]),
                        "r*", markersize=10)
            else:
                ax.axis("off")
            if j == 0:
                ax.set_ylabel(param_names[i])
            if i == ndim - 1:
                ax.set_xlabel(param_names[j])

    plt.tight_layout()
    return fig
