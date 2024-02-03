""" Procedures of within-subject ensemble parameters analysis
"""
import os
from itertools import combinations
import numpy as np
import statsmodels.api as sm
import tools
from tools import print_info

def analyze_within_subject_ensemble_statistics(
    filename: str,
    results_dirname: str,
    metric_name: str = "swd",
    metric: callable = None,
    window_sizes: np.ndarray = None,
    random: np.random.Generator = None):
    """ run procedures for analyzing within-subject ensemble statistics of SWD,
    and compare with SWC

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if random is None:
        random = np.random
    if metric is None:
        if metric_name == "swc":
            metric = tools.swc
        elif metric_name == "mtd":
            metric = tools.mtd
        else:
            metric = tools.swd
    print_info("##########################################################################", results_dirname)
    print_info(f"INFO: analyzing within-subject {metric_name} ensemble statistics", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)

    emp_dir = os.path.join(results_dirname, "within-subject-empirical-ensemble-statistics")
    os.mkdir(emp_dir)

    if window_sizes is None:
        window_sizes = [emp_data.shape[-1]]
    for window_size in window_sizes:
        print_info(f"# window size = {window_size} ###############################################", results_dirname)
        # Compute and plot SWD
        estimates = metric(emp_data, window_size=window_size)
        print_info(f"INFO: empirical {metric_name.upper()}-based estimates mean, variance: " +
            f"{np.mean(estimates), np.var(estimates)}", results_dirname)
        tools.plot_distribution(
            estimates,
            xlabel="Estimate",
            ylabel="Density",
            title=f"{metric_name.upper()} (w = {window_size})",
            out=os.path.join(
                emp_dir,
                f"{metric_name}-sampling-distribution-{window_size}.png"
            ))

        # Compute and plot SWD (no log transform)
        estimates_no_log = metric(emp_data, window_size=window_size, transform=False)
        print_info(f"INFO: empirical {metric_name.upper()}-based (without transformation) estimates mean, variance: " +
            f"{np.mean(estimates_no_log), np.var(estimates_no_log)}", results_dirname)
        tools.plot_distribution(
            estimates_no_log,
            xlabel="Estimate",
            ylabel="Density",
            title=f"SWD without Transformation (w = {window_size})",
            out=os.path.join(
                emp_dir,
                f"swd-no-log-sampling-distribution-{window_size}.png"
            ))

        if window_size == emp_data.shape[-1]:
            continue
        # Plot edge-averaged SWD-based tvFC estimates
        tools.plot_global_timeseries(
            estimates,
            xlabel="Time (TR)",
            ylabel="Estimate",
            title=f"w = {window_size}",
            out=os.path.join(
                emp_dir,
                f"global-swd-timeseries-{window_size}.png"))
        # compute and test the stationarity of SWD-based estimates' ensemble parameters
        mean_stationary_pval = sm.tsa.adfuller(np.mean(estimates, axis=0))[1]
        var_stationary_pval = sm.tsa.adfuller(np.var(estimates, axis=0))[1]
        print_info("Stationarity (mean, variance): " + \
            f"({mean_stationary_pval}, {var_stationary_pval})", results_dirname)
        print_info("Distribution stats (skewness, kurtosis): " + \
            f"{tools.test_distribution(estimates)}", results_dirname)
        # End of window size loop

def analyze_within_subject_swd_swc_correlation(
    filename: str,
    results_dirname: str):
    """ run procedures for analyzing within-subject SWD and SWC correlation

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    print_info("##########################################################################", results_dirname)
    print_info("INFO: analyzing within-subject SWD-SWC correlation", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

    emp_dir = os.path.join(results_dirname, "within-subject-empirical-ensemble-statistics")
    if not os.path.exists(emp_dir):
        os.mkdir(emp_dir)
    window_sizes = [1, 9, 19, 29, 39, 49, 59, 69, 99, emp_data.shape[-1]]

    swd_swc_correlations = []
    window_sizes = np.linspace(5, max(emp_data.shape[-1], 2400), 100)
    window_sizes = [int(window_size) for window_size in window_sizes]
    for window_size in window_sizes:
        swd_estimates = tools.swd(emp_data, window_size, pairs=pairs)
        swc_estimates = tools.swc(emp_data, window_size, pairs=pairs)
        swd_swc_correlations.append(
            np.corrcoef(swd_estimates.flatten(), swc_estimates.flatten())[0, 1])

    tools.plot_grid(
        np.array(window_sizes),
        np.array(swd_swc_correlations),
        xlabel="Window Size (TR)",
        ylabel="Correlation",
        out=os.path.join(
            emp_dir,
            "average-swd-swc-correlations.png"
        ))
