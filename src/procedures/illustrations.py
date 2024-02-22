""" Generate illustrations using matplotlib.
    Mostly implemented by ChatGPT 4
"""
import os
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


    plt.figure()
    # Assuming x is common for all signals in `data`
    x = np.linspace(0, 2*np.pi, 100)  # Assuming data.shape[1] matches the length of the x-axis data
    # Plotting selected signals
    nodes = random.choice(data.shape[0], size=10, replace=False)
    for node in nodes:  # Plot up to 10 signals
        plt.plot(x, data[node, 2000:2100])

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "empirical-signals.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    # generate violin plots of SWD distributions
    fig, ax = plt.subplots()
    window_sizes = [9, 19, 29, 39, 49, 99]
    timepoint_samples = random.choice(data.shape[-1] - window_sizes[-1], 200, replace=False)
    swd_per_window_size = [tools.swd(data[:,timepoint_samples], window_size).flatten() for window_size in window_sizes]
    window_size_labels = [str(window_size) for window_size in window_sizes]
    quantiles = [[0, 0.25, 0.5, 0.75, 1] for _ in window_sizes]
    ax.violinplot(swd_per_window_size, quantiles=quantiles, showextrema=False)
    ax.set_xticks(np.arange(1, len(window_size_labels) + 1))
    ax.set_xticklabels(window_size_labels)
    ax.set_xlabel('Window Size')
    ax.set_ylabel('SWD')

    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "empirical-swd.png")
        plt.savefig(figpath, transparent=True)
        plt.close()
