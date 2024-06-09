""" Generate illustrations using matplotlib.
    Mostly implemented by ChatGPT 4
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({'font.size': 12})

TRANSPARENT = True
DPI = 600

def generate_illustrations(
    data: any,
    sinusoid_simulation_stats_filepath: str,
    wihtin_subject_stats_filepath: dict,
    surrogate_stats_filepath: str,
    between_subjects_stats_filepath: str,
    results_dirname: str = None,
    lpf_window_size = 10,
    random: np.random.Generator = None
):
    """ generate illustrations of "Distance-Based Metrics of tvFC study"
    """
    if random is None:
        random = np.random
    if isinstance(data, str):
        data = tools.prep_emp_data(np.loadtxt(data).T, smooth=lpf_window_size)

    if results_dirname is not None:
        illustrations_dir = os.path.join(results_dirname, "illustrations")
        os.mkdir(illustrations_dir)

    ##########################################################################################
    # 1. sinusoids plot ######################################################################
        
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
        plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
        plt.close()

    ##########################################################################################
    # 2. zoomed-in sinuoid plot ##############################################################

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
        plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
        plt.close()

    dy1 = np.gradient(y1, x)
    dy2 = np.gradient(y2, x)

    ##########################################################################################
    # 3. 3D sinusoid plot ####################################################################

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
        plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
        plt.close()


    ##########################################################################################
    # 4. distance measures plot ##############################################################

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
        plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
        plt.close()

    ##########################################################################################
    # 5. sinusoid simulation results plot ####################################################

    # Load the CSV data
    csv_data = pd.read_csv(sinusoid_simulation_stats_filepath)

    metrics = csv_data['metric'].unique()

    # Create plots for each surrogate method and LPF window size
    subset = csv_data

    plt.figure()
    for metric in metrics:
        metric_data = subset[csv_data['metric'] == metric]
        plt.plot(metric_data['window'], metric_data['uncertainty'], label=metric.upper())
    
    plt.xlabel('Window Size')
    plt.ylabel('Uncertainty')
    plt.legend()
    plt.grid(True)
    
    if results_dirname is None:
        plt.show()
    else:
        figpath = os.path.join(illustrations_dir, "sinuoid-simulation.png")
        plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
        plt.close()

    ##########################################################################################
    # 6. Edge dynamics significance plot #####################################################

    # Load the CSV data
    csv_data = pd.read_csv(surrogate_stats_filepath)
    
    # Aggregate the data
    aggregated_data = csv_data.groupby(
        ['window_size', 'metric', 'surrogate_method', 'lpf_window_size']
    )['edge_variance_significance'].mean().reset_index()

    surrogate_methods = aggregated_data['surrogate_method'].unique()
    metrics = aggregated_data['metric'].unique()
    lpf_window_sizes = aggregated_data['lpf_window_size'].unique()

    # Create plots for each surrogate method and LPF window size
    for method in surrogate_methods:
        for lpf_size in lpf_window_sizes:
            subset = aggregated_data[
                (aggregated_data['surrogate_method'] == method) & (aggregated_data['lpf_window_size'] == lpf_size)]

            plt.figure()
            for metric in metrics:
                metric_data = subset[subset['metric'] == metric]
                plt.plot(metric_data['window_size'], metric_data['edge_variance_significance'] * 100, label=metric.upper())
            
            # percetage always spans from 0 to 60%
            plt.ylim(0, 60)

            # Add a horizontal line at 5% significance level
            plt.axhline(y=5, color='r', linestyle='--', linewidth=1, label='5% Band')
    
            plt.xlabel('Window Size')
            plt.ylabel('Significance Percentage')
            plt.legend(loc='upper right')
            plt.grid(True)
            
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(illustrations_dir, f"edge-variance-significance-{method}-{lpf_size}.png")
                plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
                plt.close()

    ##########################################################################################
    # 7. between-sessions ANOVA plots ##########################################################

    # Load the CSV data
    csv_data = pd.read_csv(wihtin_subject_stats_filepath['anova'])

    metrics = csv_data['metric'].unique()
    lpf_window_sizes = csv_data['lpf_window_size'].unique()

    # ANOVA for mean
    for lpf_window_size in lpf_window_sizes:
        plt.figure()
        lpf_data = csv_data[csv_data['lpf_window_size'] == lpf_window_size]
        for metric in metrics:
            metric_data = lpf_data[lpf_data['metric'] == metric]
            plt.plot(metric_data['window_size'], metric_data['mean_anova_pvalue'], label=metric.upper())
        
        plt.ylim(-0.1, 1.1)
        plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='0.05 Band')
        plt.xlabel('Window Size')
        plt.ylabel('p-value')
        plt.legend()
        plt.grid(True)
        
        if results_dirname is None:
            plt.show()
        else:
            figpath = os.path.join(illustrations_dir, f"between-sessions-ANOVA-mean-lpf-{lpf_window_size}.png")
            plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
            plt.close()

    # ANOVA for variance
    for lpf_window_size in lpf_window_sizes:
        plt.figure()
        lpf_data = csv_data[csv_data['lpf_window_size'] == lpf_window_size]
        for metric in metrics:
            metric_data = lpf_data[lpf_data['metric'] == metric]
            plt.plot(metric_data['window_size'], metric_data['variance_anova_pvalue'], label=metric.upper())
        
        plt.ylim(-0.1, 1.1)
        plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='0.05 Band')
        plt.xlabel('Window Size')
        plt.ylabel('p-value')
        plt.legend()
        plt.grid(True)
        
        if results_dirname is None:
            plt.show()
        else:
            figpath = os.path.join(illustrations_dir, f"between-sessions-ANOVA-variance-lpf-{lpf_window_size}.png")
            plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
            plt.close()


    ##########################################################################################
    # 8. wihtin-session Stationarity plots ######################################################


    ##########################################################################################
    # 9. between-subjects ANOVA plots ########################################################

    # Load the CSV data
    csv_data = pd.read_csv(between_subjects_stats_filepath)

    metrics = csv_data['metric'].unique()
    lpf_window_sizes = csv_data['lpf_window_size'].unique()

    # ANOVA for mean
    for lpf_window_size in lpf_window_sizes:
        plt.figure()
        lpf_data = csv_data[csv_data['lpf_window_size'] == lpf_window_size]
        plt.ylim(-0.1, 1.1)
        for metric in metrics:
            metric_data = lpf_data[lpf_data['metric'] == metric]
            plt.plot(metric_data['window_size'], metric_data['mean_anova_pvalue'], label=metric.upper())
        
        plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='0.05 Band')
        plt.xlabel('Window Size')
        plt.ylabel('p-value')
        plt.legend()
        plt.grid(True)
        
        if results_dirname is None:
            plt.show()
        else:
            figpath = os.path.join(illustrations_dir, f"between-subjects-ANOVA-mean-lpf-{lpf_window_size}.png")
            plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
            plt.close()

    # ANOVA for variance
    for lpf_window_size in lpf_window_sizes:
        plt.figure()
        lpf_data = csv_data[csv_data['lpf_window_size'] == lpf_window_size]
        for metric in metrics:
            metric_data = lpf_data[lpf_data['metric'] == metric]
            plt.plot(metric_data['window_size'], metric_data['variance_anova_pvalue'], label=metric.upper())
        
        plt.ylim(-0.1, 1.1)
        plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='0.05 Band')
        plt.xlabel('Window Size')
        plt.ylabel('p-value')
        plt.legend()
        plt.grid(True)
        
        if results_dirname is None:
            plt.show()
        else:
            figpath = os.path.join(illustrations_dir, f"between-subjects-ANOVA-variance-lpf-{lpf_window_size}.png")
            plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
            plt.close()

    ##########################################################################################
    # 10. time-averaged and time-resolved statistics plots ###################################
    
    # 10.a time-averaged FC hypothesis ########################################################
    # Load the CSV data
    csv_data = pd.read_csv(surrogate_stats_filepath)
    
    # Aggregate the data
    aggregated_data = csv_data.groupby(
        ['window_size', 'metric', 'surrogate_method', 'lpf_window_size']
    )['divergence_h1'].mean().reset_index()

    surrogate_methods = aggregated_data['surrogate_method'].unique()
    metrics = aggregated_data['metric'].unique()
    lpf_window_sizes = aggregated_data['lpf_window_size'].unique()

    # Create plots for each surrogate method and LPF window size
    for method in surrogate_methods:
        for lpf_size in lpf_window_sizes:
            subset = aggregated_data[
                (aggregated_data['surrogate_method'] == method) & (aggregated_data['lpf_window_size'] == lpf_size)]

            plt.figure()
            for metric in metrics:
                metric_data = subset[subset['metric'] == metric]
                plt.plot(metric_data['window_size'], metric_data['divergence_h1'], label=metric.upper())
    
            plt.xlabel('Window Size')
            plt.ylabel('Divergence')
            plt.legend()
            plt.grid(True)
            
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(illustrations_dir, f"h1-divergence-fixed-null-{method}-{lpf_size}.png")
                plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
                plt.close()

    # 10.a edge variance hypothesis ###########################################################
    # Load the CSV data
    csv_data = pd.read_csv(surrogate_stats_filepath)
    
    # Aggregate the data
    aggregated_data = csv_data.groupby(
        ['window_size', 'metric', 'surrogate_method', 'lpf_window_size']
    )['divergence_h2'].mean().reset_index()

    surrogate_methods = aggregated_data['surrogate_method'].unique()
    metrics = aggregated_data['metric'].unique()
    lpf_window_sizes = aggregated_data['lpf_window_size'].unique()

    # Create plots for each surrogate method and LPF window size
    for method in surrogate_methods:
        for lpf_size in lpf_window_sizes:
            subset = aggregated_data[
                (aggregated_data['surrogate_method'] == method) & (aggregated_data['lpf_window_size'] == lpf_size)]

            plt.figure()
            for metric in metrics:
                metric_data = subset[subset['metric'] == metric]
                plt.plot(metric_data['window_size'], metric_data['divergence_h2'], label=metric.upper())
    
            plt.xlabel('Window Size')
            plt.ylabel('Divergence')
            plt.legend()
            plt.grid(True)
            
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(illustrations_dir, f"h2-divergence-fixed-null-{method}-{lpf_size}.png")
                plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
                plt.close()

    # 10.c time-averaged FC & edge variance hypothesis ########################################
    # Load the CSV data
    csv_data = pd.read_csv(surrogate_stats_filepath)
    
    # Aggregate the data
    aggregated_data = csv_data.groupby(
        ['window_size', 'metric', 'surrogate_method', 'lpf_window_size']
    )['divergence_h1h2_updated'].mean().reset_index()

    surrogate_methods = aggregated_data['surrogate_method'].unique()
    metrics = aggregated_data['metric'].unique()
    lpf_window_sizes = aggregated_data['lpf_window_size'].unique()

    # Create plots for each surrogate method and LPF window size
    for method in surrogate_methods:
        for lpf_size in lpf_window_sizes:
            subset = aggregated_data[
                (aggregated_data['surrogate_method'] == method) & (aggregated_data['lpf_window_size'] == lpf_size)]

            plt.figure()
            for metric in metrics:
                metric_data = subset[subset['metric'] == metric]
                plt.plot(metric_data['window_size'], metric_data['divergence_h1h2_updated'], label=metric.upper())
    
            plt.xlabel('Window Size')
            plt.ylabel('Divergence')
            plt.legend()
            plt.grid(True)
            
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(illustrations_dir, f"h1h2-divergence-empirical-null-{method}-{lpf_size}.png")
                plt.savefig(figpath, transparent=TRANSPARENT, dpi=DPI)
                plt.close()
