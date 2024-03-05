""" Generate illustrations using matplotlib.
    Mostly implemented by ChatGPT 4
"""
import os
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tools.common import sliding_average
import tools
plt.style.use('seaborn-v0_8-whitegrid')


def generate_illustrations(
    data: np.ndarray,
    results_dirname: str = None,
    random: np.random.Generator = None
):
    """ generate illustrations of "Distance-Based Metrics of tvFC study"
    """
    if results_dirname is not None:
        illustrations_dir = os.path.join(results_dirname, "illustrations")
        os.mkdir(illustrations_dir)
    # Sample x-axis data
    x = np.linspace(0, 2*np.pi, 500)
    # Two signals for demonstration
    y1 = np.sin(x)
    y2 = np.sin(x) + np.cos(x)
    plt.figure()

    # Plot zoomed-out view with equal aspect ratio and a rectangular window
    ax = plt.subplot(aspect='equal')
    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.fill_between(
        x, y1, y2, facecolor='none', hatch='|', edgecolor='lightcoral', interpolate=True)

    # Determine the zoomed-in window
    intersections = np.where(np.isclose(y1, y2, atol=0.01))[0]
    if len(intersections) > 0:
        center_index = intersections[0]
        start_index = max(center_index - 10, 0)
        end_index = min(center_index + 10, len(x) - 1)

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "sine-signals.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    plt.figure()
    # Plot zoomed-in view around the first intersection with equal aspect ratio
    ax = plt.subplot(aspect='equal')
    zoom_range = slice(start_index, end_index)
    ax.plot(x[zoom_range], y1[zoom_range])
    ax.plot(x[zoom_range], y2[zoom_range])
    ax.fill_between(
        x[zoom_range],
        y1[zoom_range],
        y2[zoom_range],
        facecolor='none', hatch='|', edgecolor='lightcoral', interpolate=True)

    plt.tight_layout()
    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "sine-signals-zoom.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    dy1 = np.gradient(y1, x)
    dy2 = np.gradient(y2, x)

    # Plotting the signals in 3D space
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plotting the first signal in 3D
    ax.plot(x, y1, dy1, label='Signal 1: Sine Wave')
    # Plotting the second signal in 3D
    ax.plot(x, y2, dy2, label='Signal 2: Sine + Cosine Wave')

    line_interval = 10
    # Add vertical lines between the two signals at intervals
    for i in range(0, len(x), line_interval):
        xi, y1i, y2i, dy1i, dy2i = x[i], y1[i], y2[i], dy1[i], dy2[i]
        ax.plot([xi, xi], [y1i, y2i], [dy1i, dy2i], color='gray', linewidth=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_zlabel('Derivative')
    ax.view_init(elev=15, azim=-20)

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "sine-signals-3d.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    # # Calculate absolute distances and derivatives
    abs_distances_amplitude = np.abs(y1 - y2)
    abs_distances_derivative = np.abs(dy1 - dy2)

    # Calculate average distances
    # expanded_distances = (abs_distances_amplitude + abs_distances_derivative) / 2
    expanded_distances = np.sqrt(abs_distances_amplitude ** 2 + abs_distances_derivative ** 2)

    # Distance-based estimates
    ax = plt.subplot(aspect='equal')
    ax.plot(x, abs_distances_amplitude, label='Amplitude Distance', color='blue')
    ax.plot(x, abs_distances_derivative, label='Derivative Distance', color='green')
    ax.plot(x, expanded_distances, label='Expanded Distance', color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance')
    ax.set_ylim([0, 2.5])
    ax.legend()

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "distances.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    # plot selected empirical signals
    plt.figure()
    # Assuming x is common for all signals in `data`
    x = np.linspace(0, 2*np.pi, 100)  # Assuming data.shape[1] matches the length of the x-axis data
    # Plotting selected signals
    nodes = random.choice(data.shape[0], size=10, replace=False)
    for node in nodes:  # Plot up to 10 signals
        plt.plot(x, data[node, 500:600])

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "empirical-signals.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    # create tvFC edges plot for all edges
    window_size = 29
    edge_time_series = tools.swd(data, window_size=window_size)
    edge_significance = tools.significant_estimates(edge_time_series)
    # Determine global color scale across multiple datasets
    sns.heatmap(edge_significance, cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Edge')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "edge-timeseries.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()

    # create edges plot for significant time-averaged FC
    sc_data = tools.sc(data, random)
    timeavg_estimates_empirical = tools.swd(data, data.shape[-1])
    timeavg_estimates_sc = tools.swd(sc_data, sc_data.shape[-1])
    h1_edges_of_interest = tools.get_edges_of_interest(
        timeavg_estimates_empirical,
        timeavg_estimates_sc,
        alpha=0
    )
    h1_edges = [
        i for i, is_edge_significant in enumerate(h1_edges_of_interest)
        if is_edge_significant]
    sns.heatmap(edge_significance[h1_edges], cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Edge')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "edge-timeseries-h1.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()


    # create edges plot for significant edge variance
    scc_data = tools.laumann(data, random)
    scc_edge_time_series = tools.swd(scc_data, window_size=window_size)
    h2_edges_of_interest = tools.get_edges_of_interest(
        np.var(edge_time_series, axis=-1),
        np.var(scc_edge_time_series, axis=-1),
        one_side=True,
        alpha=0.05
    )
    h2_edges = [
        i for i, is_edge_significant in enumerate(h2_edges_of_interest)
        if is_edge_significant]
    sns.heatmap(edge_significance[h2_edges], cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Edge')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "edge-timeseries-h2.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()


    # create edges plot for null edges
    h1h2_edges_of_interest = h1_edges_of_interest + h2_edges_of_interest
    h1h2_edges_of_interest[h1h2_edges_of_interest != 0] = 1
    null_edges = [
        i for i, is_edge_significant in enumerate(h1h2_edges_of_interest)
        if not is_edge_significant]
    sns.heatmap(edge_significance[null_edges], cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Edge')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "edge-timeseries-null.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()

    h1h2_edges = [
        i for i, is_edge_significant in enumerate(h1h2_edges_of_interest)
        if is_edge_significant]
    sns.heatmap(edge_significance[h1h2_edges], cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Edge')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "edge-timeseries-h1h2.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()

    edge_significance = tools.significant_estimates(edge_time_series, null=edge_time_series[null_edges])
    # Determine global color scale across multiple datasets
    sns.heatmap(edge_significance, cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Edge')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "edge-timeseries-filtered.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()

    # nodewise power spectra
    empirical_data = data[:, 0:500]
    empirical_fft = np.fft.rfft(empirical_data, axis=-1)
    empirical_fft_amplitude = np.abs(empirical_fft)
    empirical_freqs = np.fft.rfftfreq(empirical_data.shape[-1], 0.72)

    # Filter frequencies and corresponding amplitudes to include only 0 to 0.3 Hz
    mask = (empirical_freqs >= 0) & (empirical_freqs <= 0.3)
    filtered_freqs = empirical_freqs[mask]
    filtered_amplitudes = empirical_fft_amplitude[:, mask]

    sns.heatmap(filtered_amplitudes, cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    # Adjusting x-ticks to show frequency values
    nticks = 6
    tick_positions = np.linspace(0, len(filtered_freqs) - 1, nticks, dtype=int)
    tick_labels = [f"{filtered_freqs[pos]:.2f}" for pos in tick_positions]
    plt.xticks(tick_positions + 0.5, tick_labels, fontsize=10, rotation=45)

    plt.xlabel('Frequency')
    plt.ylabel('Node')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "node-power-spectra.png")
        plt.savefig(figpath, transparent=True, dpi=600)
        plt.close()

    # covariance plots
    plt.figure()
    covariance = np.cov(data)
    sns.heatmap(covariance, cmap="seismic", cbar=False, xticklabels=False, yticklabels=False, square=True)
    plt.xlabel('Node')
    plt.ylabel('Node')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "covariance.png")
        plt.savefig(figpath, transparent=True, dpi=600)
        plt.close()

    # create node time series plot
    sns.heatmap(data, cmap="seismic", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Node')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "node-timeseries.png")
        plt.savefig(figpath, transparent=True, dpi=1200)
        plt.close()


    swd_data = {
        'SWD': [],
        'Edges': [],
        'Window Size': []
    }
    window_sizes = [9, 19, 29, 39, 99]

    sc_data = tools.sc(data, random)
    timeavg_estimates_empirical = tools.swd(data, data.shape[-1])
    timeavg_estimates_sc = tools.swd(sc_data, sc_data.shape[-1])
    h1_edges_of_interest = tools.get_edges_of_interest(
        timeavg_estimates_empirical,
        timeavg_estimates_sc,
        alpha=0
    )

    scc_data = tools.laumann(data, random)
    timepoint_samples = random.choice(data.shape[-1] - window_sizes[-1], 400, replace=False)
    for ws in window_sizes:
        empirical_swd = tools.swd(data, ws)
        scc_edge_time_series = tools.swd(scc_data, window_size=ws)
        h2_edges_of_interest = tools.get_edges_of_interest(
            np.var(empirical_swd, axis=-1),
            np.var(scc_edge_time_series, axis=-1),
            one_side=True,
            alpha=0.05
        )
        h1h2_edges_of_interest = h1_edges_of_interest + h2_edges_of_interest
        h1h2_edges_of_interest[h1h2_edges_of_interest != 0] = 1
        null_edges = [
            i for i, is_edge_significant in enumerate(h1h2_edges_of_interest)
            if not is_edge_significant]

        null_swd = empirical_swd[null_edges][:, timepoint_samples].flatten()
        empirical_swd = empirical_swd[:, timepoint_samples].flatten()
        swd_data['SWD'].extend(empirical_swd.tolist() + null_swd.tolist())
        swd_data['Edges'].extend(['All'] * len(empirical_swd) + ['Null'] * len(null_swd))
        swd_data['Window Size'].extend([ws] * (len(empirical_swd) + len(null_swd)))

    df = pd.DataFrame.from_dict(swd_data)

    sns.violinplot(data=df, x='Window Size', y='SWD', hue='Edges', split=True, gap=.1, inner='quart', palette={'All': 'skyblue', 'Null': 'lightcoral'})

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "empirical-null-distributions.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    swd_data = {
        'SWD': [],
        'Window Size': []
    }
    window_sizes = [9, 19, 29, 39, 99]

    timepoint_samples = random.choice(data.shape[-1] - window_sizes[-1], 400, replace=False)
    for ws in window_sizes:
        empirical_swd = tools.swd(data, ws)
        empirical_swd = empirical_swd[:, timepoint_samples].flatten()
        swd_data['SWD'].extend(empirical_swd.tolist())
        swd_data['Window Size'].extend([ws] * len(empirical_swd))

    df = pd.DataFrame.from_dict(swd_data)

    sns.violinplot(data=df, x='Window Size', y='SWD', inner='quart', color='skyblue')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "swd-distributions.png")
        plt.savefig(figpath, transparent=True)
        plt.close()
