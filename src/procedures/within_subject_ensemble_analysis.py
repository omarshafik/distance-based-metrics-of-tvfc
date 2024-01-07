""" Procedures of within-subject ensemble parameters analysis
"""
import os
import numpy as np
import statsmodels.api as sm
import tools

def analyze_within_subject_ensemble_statistics(
    filename: str,
    results_dirname: str):
    """ run procedures for analyzing within-subject ensemble statistics of SWD,
    and compare with SWC

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    emp_data = emp_data[:, 0:(emp_data.shape[-1] // 2)]

    emp_dir = os.path.join(results_dirname, "within-subject-empirical-ensemble-statistics")
    os.mkdir(emp_dir)
    window_sizes = [1, 9, 19, 29, 39, 49, 59, 69, 99, 299, 499, emp_data.shape[-1]]

    for window_size in window_sizes:
        print(f"# window size = {window_size} ###############################################")
        # Compute and plot SWD
        swd_estimates = tools.swd(
            emp_data, window_size=window_size)
        print(f"INFO: empirical SWD-based estimates mean, variance: " +
            f"{np.mean(swd_estimates), np.var(swd_estimates)}")
        tools.plot_distribution(
            swd_estimates,
            xlabel="Estimate",
            ylabel="Density",
            title=f"Sampling Distribution of SWD-based Estimates (window size = {window_size})",
            out=os.path.join(
                emp_dir,
                f"empirical-swd-sampling-distribution-{window_size}.png"
            ))

        if window_size == 1:
            continue

        # Compute and plot SWC
        swc_estimates = tools.swc(emp_data, window_size=window_size)
        tools.plot_distribution(
            swc_estimates,
            xlabel="Estimate",
            ylabel="Density",
            title=f"Sampling Distribution of SWC-based Estimates (window size = {window_size})",
            out=os.path.join(
                emp_dir,
                f"empirical-swc-sampling-distribution-{window_size}.png"))

        if window_size == emp_data.shape[-1]:
            continue

        # Plot edge-averaged SWD-based tvFC estimates
        tools.plot_global_timeseries(
            swd_estimates,
            xlabel="Time (TR)",
            ylabel="Estimate",
            title=f"Edge-averaged SWD-based tvFC estimates  (window size = {window_size})",
            out=os.path.join(
                emp_dir,
                f"global-swd-timeseries-{window_size}.png"))
        # compute and test the stationarity of SWD-based estimates' ensemble parameters
        mean_stationary_pval = sm.tsa.adfuller(np.mean(swd_estimates, axis=0))[1]
        var_stationary_pval = sm.tsa.adfuller(np.var(swd_estimates, axis=0))[1]
        print(f"Stationarity (mean, variance): " + \
            f"({mean_stationary_pval}, {var_stationary_pval})")
        # End of window size loop

    swd_swc_correlations = []
    window_sizes = np.linspace(5, emp_data.shape[-1], 100)
    window_sizes = [int(window_size) for window_size in window_sizes]
    for window_size in window_sizes:
        swd_estimates = tools.swd(emp_data, window_size)
        swc_estimates = tools.swc(emp_data, window_size)
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
