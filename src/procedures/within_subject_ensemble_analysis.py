""" Procedures of within-subject ensemble parameters analysis
"""
import os
from itertools import combinations
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import tools
from tools import print_info

METRICS = {
    "mtd": tools.mtd,
    "swc": tools.swc,
    "swd": tools.swd
}

def analyze_within_subject_ensemble_statistics(
    filename: str,
    results_dirname: str,
    metrics: dict = None,
    window_sizes: np.ndarray = None,
    lpf_window_sizes: list = None,
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
    if metrics is None:
        metrics = METRICS
    if lpf_window_sizes is None:
        lpf_window_sizes = [0, 10]
    if window_sizes is None:
        window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]

    emp_dir = os.path.join(results_dirname, "within-subject-ensemble-statistics")
    os.mkdir(emp_dir)

    print_info(
        "##########################################################################", results_dirname)
    print_info(
        f"INFO: Caryying out within-subject ensemble statistics analysis for {os.path.basename(filename)}",
        results_dirname)

    anova_results = {
        'filename': [],
        'lpf_window_size': [],
        'metric': [],
        'window_size': [],
        'mean_anova_statistic': [],
        'mean_anova_pvalue': [],
        'variance_anova_statistic': [],
        'variance_anova_pvalue': []
    }

    adfuller_results = {
        'filename': [],
        'lpf_window_size': [],
        'metric': [],
        'window_size': [],
        'session': [],
        'mean_adfuller_statistic': [],
        'mean_adfuller_pvalue': [],
        'variance_adfuller_statistic': [],
        'variance_adfuller_pvalue': []
    }

    for lpf_window_size in lpf_window_sizes:
        emp_data = tools.prep_emp_data(np.loadtxt(filename).T, smooth=lpf_window_size)
        for metric_name, metric in metrics.items():
            print_info("##########################################################################", emp_dir)
            print_info(f"INFO: analyzing within-subject {metric_name.upper()} ensemble statistics", emp_dir)
            for window_size in window_sizes:
                print_info(f"# window size = {window_size} ###############################################", emp_dir)

                estimates = metric(emp_data, window_size=window_size)
                print_info(f"INFO: empirical {metric_name.upper()}-based estimates mean, variance: " +
                    f"{np.mean(estimates), np.var(estimates)}", emp_dir)

                session_length = estimates.shape[-1] // 4
                edgeavg_means_per_session = []
                edgeavg_variances_per_session = []
                subsession_length = 200
                for session in range(4):
                    start = session * session_length
                    end = start + session_length
                    subsession = estimates[:, start:end]
                    edgeavg_means = np.mean(subsession, axis=0)
                    edgeavg_variances = np.var(subsession, axis=0)

                    mean_stationary = sm.tsa.adfuller(edgeavg_means)
                    var_stationary = sm.tsa.adfuller(edgeavg_variances)
                    print_info(f"Session {session} Stationarity of mean: {mean_stationary[0:2]}", emp_dir)
                    print_info(f"Session {session} Stationarity of variance: {var_stationary[0:2]}", emp_dir)

                    subsession_means = tools.sliding_average(edgeavg_means, subsession_length)[::subsession_length]
                    edgeavg_means_per_session.append(subsession_means)
                    subsession_variances = tools.sliding_average(edgeavg_variances, subsession_length)[::subsession_length]
                    edgeavg_variances_per_session.append(subsession_variances)
                    
                    adfuller_results['filename'].append(os.path.basename(filename))
                    adfuller_results['session'].append(session)
                    adfuller_results['lpf_window_size'].append(lpf_window_size)
                    adfuller_results['metric'].append(metric_name)
                    adfuller_results['window_size'].append(window_size)
                    adfuller_results['mean_adfuller_statistic'].append(mean_stationary[0])
                    adfuller_results['mean_adfuller_pvalue'].append(mean_stationary[1])
                    adfuller_results['variance_adfuller_statistic'].append(var_stationary[0])
                    adfuller_results['variance_adfuller_pvalue'].append(var_stationary[1])

                means_anova = stats.f_oneway(*edgeavg_means_per_session)
                variances_anova = stats.f_oneway(*edgeavg_variances_per_session)
                print_info(F"INFO: Between Sessions ANOVA's statistics for mean parameter: {means_anova}", emp_dir)
                print_info(F"INFO: Between Sessions ANOVA's statistics for variance parameter: {variances_anova}", emp_dir)

                anova_results['filename'].append(os.path.basename(filename))
                anova_results['lpf_window_size'].append(lpf_window_size)
                anova_results['metric'].append(metric_name)
                anova_results['window_size'].append(window_size)
                anova_results['mean_anova_statistic'].append(means_anova[0])
                anova_results['mean_anova_pvalue'].append(means_anova[1])
                anova_results['variance_anova_statistic'].append(variances_anova[0])
                anova_results['variance_anova_pvalue'].append(variances_anova[1])

                print_info("Distribution stats (skewness, kurtosis): " + \
                    f"{tools.test_distribution(estimates)}", emp_dir)
    
    anova_results_filepath = os.path.join(emp_dir, "anova-stats.csv")
    pd.DataFrame(anova_results).to_csv(
        anova_results_filepath,
        index=False
    )
    adfuller_results_filepath = os.path.join(emp_dir, "adfuller-stats.csv")
    pd.DataFrame(adfuller_results).to_csv(
        adfuller_results_filepath,
        index=False
    )

    return anova_results_filepath, adfuller_results_filepath

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
