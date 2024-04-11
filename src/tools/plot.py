"""
visualization helper functions
"""
import warnings
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from tools.common import find_segments

PLOT = True
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore", "is_categorical_dtype")

def plot_timeseries_and_estimates(
    timseries_array: np.ndarray,
    estimates_array: np.ndarray,
    timeseries_labels: list,
    estimates_labels: list,
    significant_timepoints: np.ndarray = None,
    out: str = None,
    yscale=None):
    """ plot given pair of fMRI timeseries, \
        their corresponding TVC estimates, and time points belonging to significant samples

    Args:
        timseries_array (np.ndarray): array of timeseries pair
        estimates_array (np.ndarray): array of estimates to plot
        timeseries_labels (list): a label for each timeseries
        estimates_labels (list): a label for each estimate
        significant_timepoints (np.ndarray, optional): \
            one-dimensional array of points belonging to (in)significant samples. Defaults to None.
        out (str, optional): output file path to save the plot to. Defaults to None.
        yscale (_type_, optional): y-axis scale to apply for the stimates plot. Defaults to None.
    """
    if not PLOT:
        return
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    for ts_idx, timeseries in enumerate(timseries_array):
        plt.plot(timeseries, label=timeseries_labels[ts_idx])

    # Plot background color for significant timepoints
    if significant_timepoints is not None:
        positive_sig_tps = np.where(significant_timepoints > 0, significant_timepoints, 0)
        negative_sig_tps = np.where(significant_timepoints < 0, significant_timepoints, 0)
        for start, end in find_segments(positive_sig_tps):
            plt.axvspan(start, end, color='red', alpha=0.2)
        for start, end in find_segments(negative_sig_tps):
            plt.axvspan(start, end, color='blue', alpha=0.2)

    plt.xlabel('Time (TR)')
    plt.ylabel('Amplitude')
    plt.title('Parcel Time Series')
    plt.legend()

    plt.subplot(2, 1, 2)
    for sim_idx, estimates in enumerate(estimates_array):
        plt.plot(estimates, label=estimates_labels[sim_idx])

    plt.xlabel('Time (TR)')
    plt.ylabel('Estimate')
    plt.title('Edge Time Series')
    plt.legend()
    if yscale is not None:
        plt.yscale(yscale)

    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_timeseries(
    timseries_array: np.ndarray,
    timeseries_labels: list,
    out: str = None):
    """ plot given pair of fMRI timeseries

    Args:
        timseries_array (np.ndarray): array of timeseries pair
        timeseries_labels (list): a label for each timeseries
        out (str, optional): output file path to save the plot to. Defaults to None.
    """
    if not PLOT:
        return
    plt.figure(figsize=(12, 6))

    plt.subplot()
    for ts_idx, timeseries in enumerate(timseries_array):
        plt.plot(timeseries, label=timeseries_labels[ts_idx])

    plt.xlabel('Time (TR)')
    plt.ylabel('Amplitude')
    plt.title('Parcel Time Series')
    plt.legend()

    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_distribution(
    timeseries: np.ndarray,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    out: str = None,
    fit: bool = False,
    density=True,
    bins='auto'):
    """ plot distribution of given timeseries data (TVC estimates)

    Args:
        timeseries (np.ndarray): timeseries array
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".
        title (str, optional): Defaults to "".
        out (str, optional): output file path to save the plot to. Defaults to None.
        bins (_type_, optional): bins to use for np.histogram. Defaults to None.
    """
    if not PLOT:
        return
    _, ax = plt.subplots()
    ax.hist(timeseries.flatten(), bins=bins, density=density)

    if fit:
        # Fit a normal distribution to the data
        mu, std = stats.norm.fit(timeseries.flatten())
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, color='blue', linewidth=1, label='Normal Distribution Fit')
        plt.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    # Save the plot to an image file
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_overlapping_distributions(
    timeseries_list: list,
    labels: list,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    density=True,
    out: str = None):
    """ plot distribution of given timeseries data (TVC estimates)

    Args:
        timeseries_list (list(np.ndarray)): list of timeseries. \
            Each list element consists of multivariate timeseries
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".
        title (str, optional): Defaults to "".
        out (str, optional): output file path to save the plot to. Defaults to None.
        bins (_type_, optional): bins to use for np.histogram. Defaults to None.
    """
    if not PLOT:
        return
    _, ax = plt.subplots(sharex=True, sharey=True)
    # for elementidx, ts_element in enumerate(timeseries_list):
    ax.hist(
        [timeseries.flatten() for timeseries in timeseries_list],
        bins='auto',
        density=density,
        alpha=0.5,
        histtype="stepfilled",
        label=labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Save the plot to an image file
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_global_timeseries(
    timeseries: np.ndarray,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    out: str = None):
    """ plot global timeseries of given array of timeseries

    Args:
        timeseries (np.ndarray): timeseries array
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".
        title (str, optional): Defaults to "".
        out (str, optional): output file path to save the plot to. Defaults to None.
    """
    if not PLOT:
        return
    global_connectivity_ts = np.mean(timeseries, axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(global_connectivity_ts)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    # Save the plot to an image file
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_grid(
    x_array: np.ndarray,
    y_array: np.ndarray,
    out: str = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = ""):
    """ generic plot function

    Args:
        x_array (np.ndarray): Data array for x-axis (array is flattened)
        y_array (np.ndarray): Data array for y-axis (array is flattened)
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".
        title (str, optional): Defaults to "".
    """
    if not PLOT:
        return
    plt.plot(x_array.flatten(), y_array.flatten())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Save the plot to an image file
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_autocorrelation(
    time_series: np.ndarray,
    max_lag: int = 40,
    out: str = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = ""):
    """ plot autocorrelation as a function of time lag

    Args:
        time_series (np.ndarray): timeseries array
        max_lag (int, optional): maximum lag value to plot. Defaults to 40.
        out (str, optional): output file path to save the plot to. Defaults to None.
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".
        title (str, optional): Defaults to "".
    """
    if not PLOT:
        return
    # Plot autocovariance
    acf_values = np.zeros((time_series.shape[0], max_lag))
    for i in range(time_series.shape[0]):
        acf_values[i, :] = acf(time_series[i,], nlags=max_lag - 1)

    # Calculate the average autocorrelation values for each lag across all elements
    average_acf = np.mean(acf_values, axis=0)

    # Plot the average autocorrelation function (ACF)
    plt.plot(np.arange(max_lag), average_acf)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Save the plot to an image file
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_qq(timeseries: np.ndarray, title: str = "", out: str = None):
    """ plot quantile-quantile graph of given timeseries array

    Args:
        time_series (np.ndarray): timeseries array
        title (str, optional): Defaults to ""
        out (str, optional): output file path to save the plot to. Defaults to None.
    """
    if not PLOT:
        return
    plt.figure(figsize=(6, 6))
    stats.probplot(timeseries.flatten(), plot=plt)
    plt.title(title)
    if out is None:
        plt.show()
    else:
        plt.savefig(out)
        plt.close()

def plot_correlation_matrices(
    timeseries_list: list,
    labels: list,
    out: str = None):
    """ plot correlation matrices of given timeseries data (TVC estimates)

    Args:
        timeseries_list (list(np.ndarray)): list of timeseries. \
            Each list element consists of multivariate timeseries
        title (str, optional): Defaults to "".
        out (str, optional): output file path to save the plot to. Defaults to None.
    """
    if not PLOT:
        return
    ntimeseries = len(timeseries_list)
    # Define figure and axes with increased figure size and appropriate spacing
    fig, ax_list = plt.subplots(
        1, ntimeseries, figsize=(5 * ntimeseries, 5), gridspec_kw={'wspace': 0.4})

    if ntimeseries == 1:
        ax_list = [ax_list]

    for i, axi in enumerate(ax_list):
        # Calculate the correlation matrix
        cov_matr = np.corrcoef(timeseries_list[i])
        # Display the correlation matrix with a more scientific colormap (diverging color map)
        caxi = axi.matshow(cov_matr, cmap='seismic', vmin=-1, vmax=1)
        # Add colorbar with a bit of padding for better aesthetics
        fig.colorbar(caxi, ax=axi, fraction=0.046, pad=0.04)
        # Set title with proper padding and font size for clarity
        axi.set_title(labels[i], pad=20, fontsize=12)
        # Improve tick visibility and aesthetics
        axi.tick_params(axis='both', which='both', length=0)

    # Save the plot to an image file or display it
    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')
        plt.close()

def scatter_fc_edges(avg_fc,
    edge_variance: np.ndarray,
    edge_sr: np.ndarray,
    out: str = None
):
    """ plot a scatter diagram of average FC (x-axis), edge variance (y-axis), \
        and edge significance rate (hue)
    """
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=avg_fc, y=edge_variance, hue=edge_sr, palette='viridis')
    scatter.set(xlabel='Average FC', ylabel='Edge Variance', title='Scatter plot of FC, Edge Variance, and Significance Rate')

    # Save the plot to an image file or display it
    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')
        plt.close()
