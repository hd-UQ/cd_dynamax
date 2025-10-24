import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import gaussian_kde
from typing import Sequence

# -----------------------------------------------------------------------------
# Helper: Plot Time Series
# -----------------------------------------------------------------------------
def _plot_time_series(t, states_true, states_learned, state_labels):
    fig, axs = plt.subplots(states_true.shape[1], 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(state_labels):
        axs[i].plot(t, states_true[:, i], label=f"True {lbl}", color='C0')
        axs[i].plot(t, states_learned[:, i], label=f"Learned {lbl}", color='C1', alpha=0.7)
        axs[i].set_ylabel(lbl)
        axs[i].legend()
    axs[-1].set_xlabel("Time")
    fig.suptitle("Lorenz-63 Time Series")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Helper: Phase Portraits
# -----------------------------------------------------------------------------
def _plot_phase_portraits(states_true, states_learned):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [(0,1), (1,2), (0,2)]
    for ax, (i,j) in zip(axs, pairs):
        ax.plot(states_true[:, i], states_true[:, j], label="True", color="C0", alpha=0.8)
        ax.plot(states_learned[:, i], states_learned[:, j], label="Learned", color="C1", alpha=0.6)
        ax.set_xlabel(f"x[{i}]"); ax.set_ylabel(f"x[{j}]")
        ax.legend()
    fig.suptitle("Phase Portraits (2D projections)")
    plt.tight_layout()
    plt.show()

    # Optional 3D plot
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states_true[:,0], states_true[:,1], states_true[:,2], color='C0', label='True', alpha=0.8)
    ax.plot(states_learned[:,0], states_learned[:,1], states_learned[:,2], color='C1', label='Learned', alpha=0.6)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Helper: KDE of Trajectories
# -----------------------------------------------------------------------------
def _plot_kde_comparison_2d(data_true, data_learned, title="KDE Comparison 2D"):
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    pairs = [(0,1), (1,2), (0,2)]
    for ax, (i,j) in zip(axs, pairs):
        kde_true = gaussian_kde(np.vstack([data_true[:,i], data_true[:,j]]))
        kde_learned = gaussian_kde(np.vstack([data_learned[:,i], data_learned[:,j]]))
        # grid
        x_min, x_max = np.min(data_true[:,i]), np.max(data_true[:,i])
        y_min, y_max = np.min(data_true[:,j]), np.max(data_true[:,j])
        X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z_true = np.reshape(kde_true(positions).T, X.shape)
        Z_learn = np.reshape(kde_learned(positions).T, X.shape)

        ax.contour(X, Y, Z_true, levels=6, cmap="Blues")
        ax.contour(X, Y, Z_learn, levels=6, cmap="Reds")
        ax.set_xlabel(f"x[{i}]"); ax.set_ylabel(f"x[{j}]")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def _plot_kde_comparison(data_true, data_learned, title="KDE Comparison"):
    fig, axs = plt.subplots(data_true.shape[1], 1, figsize=(10, 8))
    for i in range(data_true.shape[1]):
        kde_true = gaussian_kde(data_true[:,i])
        kde_learned = gaussian_kde(data_learned[:,i])
        x_min = min(np.min(data_true[:,i]), np.min(data_learned[:,i]))
        x_max = max(np.max(data_true[:,i]), np.max(data_learned[:,i]))
        x_grid = np.linspace(x_min, x_max, 200)
        axs[i].plot(x_grid, kde_true(x_grid), label="True", color='C0')
        axs[i].plot(x_grid, kde_learned(x_grid), label="Learned", color='C1')
        axs[i].set_ylabel(f"KDE x[{i}]")
        axs[i].legend()
    axs[-1].set_xlabel("State Value")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Helper: Autocorrelation
# -----------------------------------------------------------------------------
def _plot_autocorr(states_true, states_learned, max_lag=500):
    def autocorr(x):
        x = x - jnp.mean(x)
        result = jnp.correlate(x, x, mode='full')
        result = result[result.size // 2:]
        return result / result[0]

    fig, axs = plt.subplots(states_true.shape[1], 1, figsize=(10,6), sharex=True)
    lags = jnp.arange(max_lag)
    for i in range(states_true.shape[1]):
        axs[i].plot(lags, autocorr(states_true[:max_lag,i]), label="True", color='C0')
        axs[i].plot(lags, autocorr(states_learned[:max_lag,i]), label="Learned", color='C1')
        axs[i].set_ylabel(f"ACF x[{i}]")
        axs[i].legend()
    axs[-1].set_xlabel("Lag")
    fig.suptitle("Autocorrelation Functions")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Helper: Power Spectral Density
# -----------------------------------------------------------------------------
def _plot_psd(states_true, states_learned, fs):
    fig, axs = plt.subplots(states_true.shape[1], 1, figsize=(10,8), sharex=True)
    for i in range(states_true.shape[1]):
        f_true, Pxx_true = welch(states_true[:,i], fs=fs, nperseg=2048)
        f_learn, Pxx_learn = welch(states_learned[:,i], fs=fs, nperseg=2048)
        axs[i].semilogy(f_true, Pxx_true, label="True", color='C0')
        axs[i].semilogy(f_learn, Pxx_learn, label="Learned", color='C1')
        axs[i].set_ylabel(f"PSD x[{i}]")
        axs[i].legend()
    axs[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Power Spectral Density")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Helper: Divergence (Volume Contraction)
# -----------------------------------------------------------------------------
def _estimate_divergence(states, drift_fn, eps=1e-5):
    """Estimate divergence via finite difference Jacobian trace."""
    n = states.shape[1]
    divs = []
    for x in states:
        J = []
        f0 = drift_fn(x)
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            f1 = drift_fn(x + dx)
            J.append((f1 - f0)/eps)
        divs.append(np.trace(np.stack(J, axis=1)))
    return np.array(divs)

def _plot_divergence(states_true, states_learned, drift_true, drift_learn):
    div_true = _estimate_divergence(states_true[::100], drift_true)
    div_learn = _estimate_divergence(states_learned[::100], drift_learn)
    plt.figure(figsize=(8,4))
    plt.hist(div_true, bins=40, alpha=0.6, label="True")
    plt.hist(div_learn, bins=40, alpha=0.6, label="Learned")
    plt.axvline(np.mean(div_true), color='C0', linestyle='--')
    plt.axvline(np.mean(div_learn), color='C1', linestyle='--')
    plt.xlabel("Divergence (Tr(J))")
    plt.ylabel("Count")
    plt.title("Volume Contraction Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Helper: Lyapunov Exponent (largest)
# -----------------------------------------------------------------------------
def _estimate_lyapunov(states, drift_fn, dt=0.01, eps=1e-6, T_max=10000):
    x = np.array(states[0])
    v = np.random.randn(x.size)
    v /= np.linalg.norm(v)
    log_sum = 0.0
    count = 0
    for t in range(min(len(states)-1, T_max)):
        f0 = drift_fn(x)
        x_next = x + dt * f0
        # propagate perturbation
        J = []
        for i in range(x.size):
            dx = np.zeros_like(x); dx[i] = eps
            J.append((drift_fn(x + dx) - f0)/eps)
        J = np.stack(J, axis=1)
        v = v + dt * J @ v
        norm_v = np.linalg.norm(v)
        log_sum += np.log(norm_v)
        v /= norm_v
        x = x_next
        count += 1
    return log_sum / (count * dt)

# -----------------------------------------------------------------------------
# Helper: Correlation Dimension Estimation
# -----------------------------------------------------------------------------
def plot_correlation_dimension_comparison(
    states_true, states_learned, n_eps=30, sample_size=5000, label_true="True", label_learn="Learned"
):
    def _estimate(states):
        X = np.asarray(states)
        N = X.shape[0]
        if N > sample_size:
            idx = np.random.choice(N, sample_size, replace=False)
            X = X[idx]
            N = sample_size

        # Pairwise distances
        dists = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(-1))
        tri = np.triu_indices(N, k=1)
        d = dists[tri]

        eps_min, eps_max = np.percentile(d, 5), np.percentile(d, 95)
        epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
        C = np.array([np.mean(d < eps) for eps in epsilons])

        log_eps = np.log(epsilons)
        log_C = np.log(C)
        slope = np.gradient(log_C, log_eps)

        start = n_eps // 4
        end = 3 * n_eps // 4
        D2 = np.mean(slope[start:end])
        return log_eps, log_C, D2, start, end

    # Estimate for both
    log_eps_t, log_C_t, D2_true, s_t, e_t = _estimate(states_true)
    log_eps_l, log_C_l, D2_learn, s_l, e_l = _estimate(states_learned)

    # Plot comparison
    plt.figure(figsize=(7,5))
    plt.plot(log_eps_t, log_C_t, 'o-', color='C0', label=f"{label_true} (D₂ ≈ {D2_true:.2f})")
    plt.plot(log_eps_t[s_t:e_t], log_C_t[s_t:e_t], 'C0', lw=3)

    plt.plot(log_eps_l, log_C_l, 'o-', color='C1', label=f"{label_learn} (D₂ ≈ {D2_learn:.2f})")
    plt.plot(log_eps_l[s_l:e_l], log_C_l[s_l:e_l], 'C1', lw=3)

    plt.xlabel("log ε")
    plt.ylabel("log C(ε)")
    plt.title("Correlation Dimension (Grassberger–Procaccia)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Estimated Correlation Dimension D2:")
    print(f"  {label_true:<8}: {D2_true:.3f}")
    print(f"  {label_learn:<8}: {D2_learn:.3f}")

    return D2_true, D2_learn

# -----------------------------------------------------------------------------
# MAIN WRAPPER FUNCTION
# -----------------------------------------------------------------------------
def analyze_chaotic_dynamics(
    states_true_long,
    states_long_learned,
    t,
    burnin_frac=0.5,
    drift_true=None,
    drift_learn=None,
    state_labels: Sequence[str] = ("x", "y", "z"),
    fs: float = 100.0,
):

    # Apply burn-in
    burnin_idx = int(burnin_frac * t.shape[0])
    t = t[burnin_idx:]
    states_true_long = states_true_long[burnin_idx:]
    states_long_learned = states_long_learned[burnin_idx:]
    
    # 1. Time series comparison
    _plot_time_series(t, states_true_long, states_long_learned, state_labels)

    # 2. Phase portraits
    _plot_phase_portraits(states_true_long, states_long_learned)

    # 3. KDE comparison
    _plot_kde_comparison(states_true_long, states_long_learned, "KDE of states: True vs Learned")
    _plot_kde_comparison_2d(states_true_long, states_long_learned, "KDE of states: True vs Learned")

    # 4. Autocorrelation
    _plot_autocorr(states_true_long, states_long_learned)

    # 5. PSD
    _plot_psd(states_true_long, states_long_learned, fs)

    # 6. Divergence (optional)
    if drift_true is not None and drift_learn is not None:
        # _plot_divergence(states_true_long, states_long_learned, drift_true, drift_learn)

        # 7. Lyapunov exponent
        le_true = _estimate_lyapunov(states_true_long, drift_true, dt=t[1]-t[0])
        le_learn = _estimate_lyapunov(states_long_learned, drift_learn, dt=t[1]-t[0])
        print(f"Estimated Largest Lyapunov Exponent:")
        print(f"  True   : {le_true:.3f}")
        print(f"  Learned: {le_learn:.3f}")
    
    # 8. Correlation dimension
    D2_true, D2_learn = plot_correlation_dimension_comparison(
        states_true_long, states_long_learned, label_true="True", label_learn="Learned"
    )
    print(f"Estimated Correlation Dimensions: True D2={D2_true:.3f}, Learned D2={D2_learn:.3f}")
