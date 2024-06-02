""" Procedures of between-subjects ensemble parameters analysis
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import tools
from tools import print_info

METRICS = {
    "mtd": tools.mtd,
    "swc": tools.swc,
    "swd": tools.swd
}

def analyze_between_subjects_ensemble_statistics(
    input_filenames: list,
    results_dirname: str,
    n_subjects: int = 30,
    window_sizes: np.ndarray = None,
    lpf_window_size: int = 10,
    metrics: dict = None,
    random: np.random.Generator = None):
    """ run procedures for analyzing and visualizing between-subjects ensemble statistics
    Args:
        input_dirname (str): parent directory of all parcel time series files
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if random is None:
        random = np.random
    if metrics is None:
        metrics = METRICS
    if window_sizes is None:
        window_sizes = [19, 49, 79, 119]
    print_info("##########################################################################", results_dirname)
    print_info("analyzing between-subject ensemble statistics", results_dirname)
    between_subject_dir = os.path.join(results_dirname, "between-subjects-ensemble-statistics")
    os.mkdir(between_subject_dir)

    random_file_indices = random.choice(len(input_filenames), n_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"Selected files {', '.join(selected_subject_nums)}", results_dirname)
    subsession_length = 200

    results = {
        'lpf_window_size': [],
        'metric': [],
        'window_size': [],
        'mean_anova_statistic': [],
        'mean_anova_pvalue': [],
        'mean_kruksal_statistic': [],
        'mean_kruksal_pvalue': [],
        'variance_anova_statistic': [],
        'variance_anova_pvalue': [],
        'variance_kruksal_statistic': [],
        'variance_kruksal_pvalue': []
    }

    print_info(f"Using low-pass filter window size = {lpf_window_size}", between_subject_dir)
    for metric_name, metric in metrics.items():
        print_info(f"Analyzing {metric_name.upper()} metric", between_subject_dir)
        for window_size in window_sizes:
            print_info(f"Using window size = {window_size}", between_subject_dir)
            print_info("#####################################", between_subject_dir)
            edgeavg_means_per_subject = []
            edgeavg_variances_per_subject = []
            for fileidx in random_file_indices:
                emp_data = tools.prep_emp_data(np.loadtxt(input_filenames[fileidx]).T, smooth=lpf_window_size)
                session_length = emp_data.shape[-1] // 4
                emp_data = emp_data[:, 0:session_length]
                estimates_empirical = metric(emp_data, window_size=window_size)
                edgeavg_means = tools.sliding_average(
                    np.mean(estimates_empirical, axis=0),
                    subsession_length)[::subsession_length]
                edgeavg_means_per_subject.append(edgeavg_means)
                edgeavg_variances = tools.sliding_average(
                    np.var(estimates_empirical, axis=0),
                    subsession_length)[::subsession_length]
                edgeavg_variances_per_subject.append(edgeavg_variances)

            means_kruksal = stats.kruskal(*edgeavg_means_per_subject)
            variances_kruksal = stats.kruskal(*edgeavg_variances_per_subject)
            print_info(F"Kruksal's statistics for mean parameter: {means_kruksal}", between_subject_dir)
            print_info(F"Kruksal's statistics for variance parameter: {variances_kruksal}", between_subject_dir)

            means_anova = stats.f_oneway(*edgeavg_means_per_subject)
            variances_anova = stats.f_oneway(*edgeavg_variances_per_subject)
            print_info(F"ANOVA's statistics for mean parameter: {means_anova}", between_subject_dir)
            print_info(F"ANOVA's statistics for variance parameter: {variances_anova}", between_subject_dir)

            results['lpf_window_size'].append(lpf_window_size)
            results['metric'].append(metric_name)
            results['window_size'].append(window_size)
            results['mean_anova_statistic'].append(means_anova[0])
            results['mean_anova_pvalue'].append(means_anova[1])
            results['mean_kruksal_statistic'].append(means_kruksal[0])
            results['mean_kruksal_pvalue'].append(means_kruksal[1])
            results['variance_anova_statistic'].append(variances_anova[0])
            results['variance_anova_pvalue'].append(variances_anova[1])
            results['variance_kruksal_statistic'].append(variances_kruksal[0])
            results['variance_kruksal_pvalue'].append(variances_kruksal[1])
            print_info("", between_subject_dir)
            print_info("#################################################", between_subject_dir)

    results_filepath = os.path.join(between_subject_dir, "between-subjects-stats.csv")
    pd.DataFrame(results).to_csv(
        results_filepath,
        index=False
    )
    return results_filepath
