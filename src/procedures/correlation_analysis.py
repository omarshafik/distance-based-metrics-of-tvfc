""" Procedures of within-subject ensemble parameters analysis
"""
import os
from itertools import combinations
import numpy as np
from scipy import stats
import tools
from tools import print_info

def analyze_metrics_correlation(
    input_filenames: list,
    results_dirname: str,
    metrics: dict = None,
    random: np.random.Generator = None):
    """ run procedures for analyzing the correlations of different metrics \
        in their time-averaged estimates

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if random is None:
        random = np.random
    if metrics is None:
        metrics = {
            "mtd": tools.mtd,
            "swc": tools.swc,
            "swd": tools.swd
        }
    print_info("##########################################################################", results_dirname)
    print_info("analyzing metrics time-averaged estimates correlation", results_dirname)

    number_of_subjects = 30
    random_file_indices = random.choice(len(input_filenames), number_of_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"Selected files {', '.join(selected_subject_nums)}", results_dirname)


    # correlate tvFC significance
    filename = input_filenames[0]
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T, smooth=10)
    emp_data_no_filter = tools.prep_emp_data(np.loadtxt(filename).T, smooth=0)
    session_length = emp_data.shape[-1] // 4
    emp_data = emp_data[:, 0:session_length]
    emp_data_no_filter = emp_data_no_filter[:, 0:session_length]
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))
    window_sizes = [5, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 299, 599]
    for window_size in window_sizes:
        print_info(f"# window size = {window_size} ##########################################", results_dirname)
        significance = {}
        variance = {}
        for metric_name, metric in metrics.items():
            window_metric_estimates = metric(emp_data, window_size, pairs=pairs)
            window_metric_estimates_no_filter = metric(emp_data_no_filter, window_size, pairs=pairs)
            significance[metric_name] = tools.significant_estimates(window_metric_estimates, 0.08).flatten()
            window_metric_significance_no_filter = tools.significant_estimates(window_metric_estimates_no_filter, 0.08).flatten()
            variance[metric_name] = np.var(window_metric_estimates, axis=-1).flatten()
            window_metric_variance_no_filter = np.var(window_metric_estimates_no_filter, axis=-1).flatten()
            significance_self_correlation = stats.spearmanr(
                np.array([significance[metric_name], window_metric_significance_no_filter]),
                axis=1)
            variance_self_correlation = stats.spearmanr(
                np.array([variance[metric_name], window_metric_variance_no_filter]),
                axis=1)
            print_info(f"{metric_name.upper()} filter-nofilter Significance correlation: " + \
                        f"{significance_self_correlation}", results_dirname)
            print_info(f"{metric_name.upper()} filter-nofilter Variance correlation: " + \
                        f"{variance_self_correlation}", results_dirname)


        metrics_significance = np.array(list(significance.values()))
        metrics_variance = np.array(list(variance.values()))
        significance_correlations = stats.spearmanr(metrics_significance, axis=1)
        print_info("MTD-SWC Significance correlation: " + \
                    f"{significance_correlations[0][0, 1]}", results_dirname)
        print_info("MTD-SWD Significance correlation: " + \
                    f"{significance_correlations[0][0, 2]}", results_dirname)
        print_info("SWC-SWD Significance correlation: " + \
                    f"{significance_correlations[0][1, 2]}", results_dirname)
        variance_correlations = stats.spearmanr(metrics_variance, axis=1)
        print_info("MTD-SWC Variance correlation: " + \
                    f"{variance_correlations[0][0, 1]}", results_dirname)
        print_info("MTD-SWD Variance correlation: " + \
                    f"{variance_correlations[0][0, 2]}", results_dirname)
        print_info("SWC-SWD Variance correlation: " + \
                    f"{variance_correlations[0][1, 2]}", results_dirname)

    print_info("# session-length window ##########################################", results_dirname)
    estimates = {
        metric_name: np.array([])
        for metric_name in metrics.keys()
    }
    for fileidx in random_file_indices:
        filename = input_filenames[fileidx]
        emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
        pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))
        for metric_name, metric in metrics.items():
            subject_metric_estimates = metric(emp_data, emp_data.shape[-1], pairs=pairs)
            estimates[metric_name] = np.append(estimates[metric_name], subject_metric_estimates)

    metrics_correlations = np.corrcoef(list(estimates.values()))
    print_info("MTD-SWC correlation: " + \
                f"{metrics_correlations[0, 1]}", results_dirname)
    print_info("MTD-SWD correlation: " + \
                f"{metrics_correlations[0, 2]}", results_dirname)
    print_info("SWC-SWD correlation: " + \
                f"{metrics_correlations[1, 2]}", results_dirname)

    metrics_correlations = stats.spearmanr(list(estimates.values()), axis=1)
    print_info("MTD-SWC Spearman correlation: " + \
                f"{metrics_correlations[0][0, 1]}", results_dirname)
    print_info("MTD-SWD Spearman correlation: " + \
                f"{metrics_correlations[0][0, 2]}", results_dirname)
    print_info("SWC-SWD Spearman correlation: " + \
                f"{metrics_correlations[0][1, 2]}", results_dirname)
