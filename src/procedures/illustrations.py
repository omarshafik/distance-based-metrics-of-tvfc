""" Generate illustrations using matplotlib.
    Mostly implemented by ChatGPT 4
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tools.common import sliding_average
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
        # zoom_range_x = [x[start_index], x[end_index]]
        # zoom_range_y = [
        #     min(y1[start_index:end_index].min(), y2[start_index:end_index].min()),
        #     max(y1[start_index:end_index].max(), y2[start_index:end_index].max())]

        # # Draw a rectangle around the zoomed-in area on the zoomed-out plot
        # rect = plt.Rectangle(
        #     (zoom_range_x[0], zoom_range_y[0]),
        #     zoom_range_x[1]-zoom_range_x[0],
        #     zoom_range_y[1]-zoom_range_y[0],
        #     linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

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
    average_distances = (abs_distances_amplitude + abs_distances_derivative) / 2

    ax = plt.subplot(aspect='equal')
    ax.plot(x, abs_distances_amplitude, label='Amplitude Distance', color='blue')
    ax.plot(x, abs_distances_derivative, label='Derivative Distance', color='green')
    ax.plot(x, average_distances, label='Average Distance', color='red')
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

    plt.subplot(aspect='equal')
    plt.xlim(0, 4)
    plt.ylim(-2, 2)
    # Invert the log-transformed values by multiplying by -1
    y_min, y_max = -2, 2
    x_min_log = np.exp(-1 * y_min)  # Convert y_min back to x range for logarithmic function
    x_max_log = np.exp(-1 * y_max)  # Convert y_max back to x range for logarithmic function

    log_transformed_values_inverted = -1 * np.log(average_distances)
    log_x_range_adjusted = np.linspace(x_min_log, x_max_log, 500)  # Adjusted x range for logarithmic function
    log_y_range_adjusted = np.log(log_x_range_adjusted)  # Calculating log values for the adjusted x range
    plt.plot(log_x_range_adjusted, -1 * log_y_range_adjusted, color='blue')
    plt.scatter(average_distances, log_transformed_values_inverted, color='red', s=10)
    plt.xlabel('Mean Distance')
    plt.ylabel('Inverted Log (Mean Distance)')

    plt.tight_layout()
    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "log-distances.png")
        plt.savefig(figpath, transparent=True)
        plt.close()

    # Function for applying sliding window average
    sampling_rate = len(x) / (2 * np.pi)
    # def sliding_window_average(signal, window_size_seconds):
    #     window_size_samples = int(window_size_seconds * sampling_rate)
    #     window = np.ones(window_size_samples) / window_size_samples
    #     return np.convolve(signal, window, mode='valid')

    # Sliding window sizes in seconds
    window_sizes = [1 / sampling_rate, 1, 3]

    # New figure for sliding window averages
    plt.figure()
    ax = plt.subplot(aspect='equal')
    for i, window_size in enumerate(window_sizes):
        window_size_samples = int(window_size * sampling_rate)
        smoothed_distances = sliding_average(log_transformed_values_inverted, window_size_samples, kaiser_beta=5)
        if i == 0:
            label = "point-wise"
        else:
            label = f'{window_size} seconds window'
        ax.plot(x[:len(smoothed_distances)], smoothed_distances, label=label)

    ax.set_ylim([0, 2.5])
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance')
    ax.legend()

    # Save or display the figure
    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "sliding-window-distances.png")
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
