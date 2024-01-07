""" Procedures of within-subject sample statistics analysis
"""
import os
from itertools import combinations
import numpy as np
import tools

def analyze_sample_statistics(
    filename: str,
    results_dirname: str):
    """ run procedures for analyzing and visualizing subject-level sample statistics

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))
    presentation_edges = np.random.choice(len(pairs), size=30, replace=False)
    start = 2000
    end = 3000

    # generate Surrogate data with the same frequency spectrum,
    # and autocorrelation as empirical
    white_noise = np.random.randn(*emp_data.shape)
    power_spectrum = np.mean(np.abs(np.fft.fft(emp_data, axis=-1)), axis=0)
    simulated_spectrum = power_spectrum \
        * np.exp(1j * np.angle(np.fft.fft(white_noise, axis=-1)))
    sc_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    emp_cov = np.cov(emp_data)
    chol_decomposition = np.linalg.cholesky(emp_cov)
    scc_data = np.dot(chol_decomposition, sc_data)
    sc_data = tools.normalized(sc_data, axis=-1)
    scc_data = tools.normalized(scc_data, axis=-1)

    sample_stats_dir = os.path.join(results_dirname, "within-subject-sample-statistics-analysis")
    os.mkdir(sample_stats_dir)

    window_sizes = [9, 19, 29, 39, 49, 59, 69]
    for window_size in window_sizes:
        sample_stats_window_dir = os.path.join(sample_stats_dir, f"window-{window_size}")
        os.mkdir(sample_stats_window_dir)

        print(f"# window size = {window_size} ####################################################")
        estimates_empirical = tools.swd(emp_data, window_size=window_size)

        estimates_significance = tools.significant_estimates(estimates_empirical)
        # only process selected presentation edges
        estimates_significance = estimates_significance[presentation_edges]
        edges_of_interest = tools.get_edges_of_interest(
            emp_data,
            sc_data,
            pairs,
            window_size=window_size,
            h1=True
        )
        edges_of_interest += tools.get_edges_of_interest(
            emp_data,
            scc_data,
            pairs,
            window_size=window_size,
            h2=True
        )
        edges_of_interest[edges_of_interest != 0] = 1
        filtered_significance = (estimates_significance.T * edges_of_interest[presentation_edges]).T
        significant_timepoints = tools.significant_time_points(filtered_significance, window_size)

        # pad estimates before plotting to match empirical data time dimension
        estimates_empirical = tools.pad_timeseries(estimates_empirical, window_size)

        loop_idx = 0
        for pair_idx in presentation_edges:
            (idx1, idx2) = pairs[pair_idx]
            tools.plot_timeseries_and_estimates(
                [emp_data[idx1, start:end], emp_data[idx2, start:end]],
                [estimates_empirical[pair_idx, start:end]],
                timeseries_labels=[f"Region {idx1}", f"Region {idx2}"],
                estimates_labels=["SWD"],
                significant_timepoints=significant_timepoints[loop_idx, start:end],
                out=os.path.join(
                    sample_stats_window_dir, f"pairs-{idx1}-{idx2}-and-tvfc-estimates.png")
            )
            loop_idx += 1
