""" Procedures of between-subjects ensemble parameters analysis
"""
import os
import numpy as np
import scipy.stats as stats
import tools
from tools import print_info

def analyze_between_subjects_ensemble_statistics(
    input_filenames: list,
    results_dirname: str,
    random: np.random.Generator = None,
    metric: callable = tools.swd):
    """ run procedures for analyzing and visualizing between-subjects ensemble statistics
    Args:
        input_dirname (str): parent directory of all parcel time series files
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if random is None:
        random = np.random
    print_info("##########################################################################", results_dirname)
    print_info("INFO: analyzing between-subject ensemble statistics", results_dirname)
    between_subject_dir = os.path.join(results_dirname, "between-subjects-ensemble-statistics")
    os.mkdir(between_subject_dir)

    window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    number_of_subjects = 30
    random_file_indices = random.choice(len(input_filenames), number_of_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"INFO: Selected files {', '.join(selected_subject_nums)}", results_dirname)
    subsession_length = 200
    for window_size in window_sizes:
        print_info(f"# window size = {window_size} ###############################################", results_dirname)
        edgeavg_means_per_subject = []
        edgeavg_variances_per_subject = []
        for fileidx in random_file_indices:
            emp_data = tools.prep_emp_data(np.loadtxt(input_filenames[fileidx]).T)
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

        tools.plot_distribution(
            np.array([edgeavg_means_per_subject]),
            xlabel="Mean",
            ylabel="Count",
            title=f"n = {number_of_subjects}, w = {window_size}",
            density=False,
            out=os.path.join(between_subject_dir,
            f"means-distribution-window-{window_size}.png"))
        tools.plot_distribution(
            np.array([edgeavg_variances_per_subject]),
            xlabel="Variance",
            ylabel="Count",
            title=f"n = {number_of_subjects}, w = {window_size}",
            density=False,
            out=os.path.join(between_subject_dir,
            f"variances-distribution-window-{window_size}.png"))

        means_kruksal = stats.kruskal(*edgeavg_means_per_subject)
        variances_kruksal = stats.kruskal(*edgeavg_variances_per_subject)
        print_info(F"INFO: Kruksal's statistics for mean parameter: {means_kruksal}", results_dirname)
        print_info(F"INFO: Kruksal's statistics for variance parameter: {variances_kruksal}", results_dirname)

        means_anova = stats.f_oneway(*edgeavg_means_per_subject)
        variances_anova = stats.f_oneway(*edgeavg_variances_per_subject)
        print_info(F"INFO: ANOVA's statistics for mean parameter: {means_anova}", results_dirname)
        print_info(F"INFO: ANOVA's statistics for variance parameter: {variances_anova}", results_dirname)
