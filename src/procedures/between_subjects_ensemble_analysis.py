""" Procedures of between-subjects ensemble parameters analysis
"""
import os
import numpy as np
import scipy.stats as stats
import tools
from tools import print_info

def analyze_between_subjects_ensemble_statistics(
    input_filenames: list,
    results_dirname: str):
    """ run procedures for analyzing and visualizing between-subjects ensemble statistics
    Args:
        input_dirname (str): parent directory of all parcel time series files
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    print_info("##########################################################################", results_dirname)
    print_info("INFO: analyzing between-subject ensemble statistics", results_dirname)
    between_subject_dir = os.path.join(results_dirname, "between-subjects-ensemble-statistics")
    os.mkdir(between_subject_dir)

    window_sizes = [9, 19, 29, 39, 49, 59, 69]
    number_of_subjects = 30
    random_file_indices = np.random.choice(len(input_filenames), number_of_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"INFO: Selected files {', '.join(selected_subject_nums)}", results_dirname)
    for window_size in window_sizes:
        print_info(f"# window size = {window_size} ###############################################", results_dirname)
        means = []
        variances = []
        samples = []
        for fileidx in random_file_indices:
            emp_data = tools.prep_emp_data(np.loadtxt(input_filenames[fileidx]).T)
            emp_data = emp_data[:, 2000:2500]
            estimates_empirical = tools.swd(emp_data, window_size=window_size)
            means.append(np.mean(estimates_empirical))
            variances.append(np.var(estimates_empirical))
            estimates_empirical_flat = estimates_empirical.flatten()
            samples.append(
                estimates_empirical_flat[
                    np.random.choice(estimates_empirical_flat.shape[0], 1000, replace=False)])

        tools.plot_distribution(
            np.array([means]),
            xlabel="Mean",
            ylabel="Count",
            title=f"n = {number_of_subjects}, w = {window_size}",
            density=False,
            out=os.path.join(between_subject_dir,
            f"means-distribution-window-{window_size}.png"))
        tools.plot_distribution(
            np.array([variances]),
            xlabel="Variance",
            ylabel="Count",
            title=f"n = {number_of_subjects}, w = {window_size}",
            density=False,
            out=os.path.join(between_subject_dir,
            f"variances-distribution-window-{window_size}.png"))

        between_variation = np.sum([(mean - np.mean(means)) ** 2 for mean in means])
        within_variation = np.sum(variances)
        f1_score = between_variation / within_variation
        print_info(F"INFO: Levene's statistics: {stats.levene(*samples)}", results_dirname)
        print_info(F"INFO: Flinger's statistics: {stats.fligner(*samples)}", results_dirname)
        print_info(F"INFO: Bartlett's statistics: {stats.bartlett(*samples)}", results_dirname)
        print_info(F"INFO: F1 score: {f1_score}", results_dirname)
