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
    print_info("INFO: analyzing metrics time-averaged estimates correlation", results_dirname)

    number_of_subjects = 30
    random_file_indices = random.choice(len(input_filenames), number_of_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"INFO: Selected files {', '.join(selected_subject_nums)}", results_dirname)


    # correlate tvFC significance
    filename = input_filenames[0]
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    session_length = emp_data.shape[-1] // 4
    emp_data = emp_data[:, 0:session_length]
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))
    window_sizes = [5, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 299, 599, emp_data.shape[-1]]
    for window_size in window_sizes:
        print_info(f"# window size = {window_size} ##########################################", results_dirname)
        significance = {}
        variance = {}
        for metric_name, metric in metrics.items():
            window_metric_estimates = metric(emp_data, window_size, pairs=pairs)
            significance[metric_name] = tools.significant_estimates(window_metric_estimates, 0.1).flatten()
            variance[metric_name] = np.var(window_metric_estimates, axis=-1).flatten()
        metrics_significance = np.array(list(significance.values()))
        metrics_variance = np.array(list(variance.values()))
        significance_correlations = stats.spearmanr(metrics_significance, axis=1)
        print_info("INFO: Significance MTD-SWC correlation: " + \
                    f"{significance_correlations[0][0, 1]}")
        print_info("INFO: Significance MTD-SWD correlation: " + \
                    f"{significance_correlations[0][0, 2]}")
        print_info("INFO: Significance SWC-SWD correlation: " + \
                    f"{significance_correlations[0][1, 2]}")
        variance_correlations = np.corrcoef(metrics_variance)
        print_info("INFO: Variance MTD-SWC correlation: " + \
                    f"{variance_correlations[0, 1]}")
        print_info("INFO: Variance MTD-SWD correlation: " + \
                    f"{variance_correlations[0, 2]}")
        print_info("INFO: Variance SWC-SWD correlation: " + \
                    f"{variance_correlations[1, 2]}")

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
    print_info("INFO: MTD-SWC correlation: " + \
                f"{metrics_correlations[0, 1]}")
    print_info("INFO: MTD-SWD correlation: " + \
                f"{metrics_correlations[0, 2]}")
    print_info("INFO: SWC-SWD correlation: " + \
                f"{metrics_correlations[1, 2]}")

    metrics_correlations = stats.spearmanr(list(estimates.values()), axis=1)
    print_info("INFO: MTD-SWC Spearman correlation: " + \
                f"{metrics_correlations[0][0, 1]}")
    print_info("INFO: MTD-SWD Spearman correlation: " + \
                f"{metrics_correlations[0][0, 2]}")
    print_info("INFO: SWC-SWD Spearman correlation: " + \
                f"{metrics_correlations[0][1, 2]}")
