""" Procedures of between-subjects ensemble parameters analysis
"""
import os
import numpy as np
import tools

def analyze_between_subjects_ensemble_statistics(
    input_filenames: list,
    results_dirname: str):
    """ run procedures for analyzing and visualizing between-subjects ensemble statistics
    Args:
        input_dirname (str): parent directory of all node time series files
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    between_subject_dir = os.path.join(results_dirname, "between-subjects-ensemble-statistics")
    os.mkdir(between_subject_dir)

    window_sizes = [9, 19, 29, 39, 49, 59, 69]
    number_of_subjects = 100
    random_file_indices = np.random.choice(len(input_filenames), number_of_subjects, replace=False)
    for window_size in window_sizes:
        print(f"# window size = {window_size} ###############################################")
        means = []
        variances = []
        for fileidx in random_file_indices:
            emp_data = tools.prep_emp_data(np.loadtxt(input_filenames[fileidx]).T)
            emp_data = emp_data[:, 2000:2500]
            estimates_empirical = tools.swd(
                emp_data, window_size=window_size)
            means.append(np.mean(estimates_empirical))
            variances.append(np.var(estimates_empirical))

        tools.plot_distribution(
            np.array([means]),
            xlabel="Mean",
            ylabel="Count",
            title="Ensemble Means Distribution for a Sample of Subjects " + \
                f"(n = {number_of_subjects})",
            density=False,
            out=os.path.join(between_subject_dir,
            f"means-distribution-window-{window_size}.png"))
        tools.plot_distribution(
            np.array([variances]),
            xlabel="Variance",
            ylabel="Count",
            title="Ensemble Means Distribution for a Sample of Subjects " + \
                f"(n = {number_of_subjects})",
            density=False,
            out=os.path.join(between_subject_dir,
            f"variances-distribution-window-{window_size}.png"))
