""" Procedures of within-subject sample statistics analysis
"""
import os
from itertools import combinations
import numpy as np
import tools
from tools import print_info

def analyze_sample_statistics(
    filename: str,
    results_dirname: str,
    random: np.random.Generator = None,
    metric: callable = tools.swd):
    """ run procedures for analyzing and visualizing subject-level sample statistics

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if random is None:
        random = np.random
    print_info("##########################################################################", results_dirname)
    print_info("analyzing within-subject sample statistics", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))
    presentation_edges = random.choice(len(pairs), size=30, replace=False)
    start = 2000
    end = 3000

    # generate Surrogate data with the same frequency spectrum,
    # and autocorrelation as empirical
    sc_data = tools.sc(emp_data, random)
    scc_data = tools.laumann(emp_data, random)

    timeavg_estimates_empirical = metric(emp_data, emp_data.shape[-1], pairs=pairs)
    timeavg_estimates_sc = metric(sc_data, sc_data.shape[-1], pairs=pairs)

    sample_stats_dir = os.path.join(results_dirname, "within-subject-sample-statistics-analysis")
    os.mkdir(sample_stats_dir)

    window_sizes = [9, 19, 29, 39, 49, 59, 69]
    for window_size in window_sizes:
        sample_stats_window_dir = os.path.join(sample_stats_dir, f"window-{window_size}")
        os.mkdir(sample_stats_window_dir)

        print_info(f"# window size = {window_size} ####################################################", results_dirname)
        estimates_empirical = metric(emp_data, window_size=window_size)
        estimates_scc = metric(scc_data, window_size=window_size)

        edges_of_interest = tools.get_edges_of_interest(
            timeavg_estimates_empirical,
            timeavg_estimates_sc,
            alpha=0
        )
        edges_of_interest += tools.get_edges_of_interest(
            np.var(estimates_empirical, axis=-1),
            np.var(estimates_scc, axis=-1),
            one_side=True,
            alpha=0.05
        )
        edges_of_interest[edges_of_interest != 0] = 1
        insig_edge_indices = [
            i for i, is_edge_significant in enumerate(edges_of_interest)
            if not is_edge_significant]
        estimates_significance = tools.significant_estimates(
            estimates_empirical,
            null=estimates_empirical[insig_edge_indices])
        # only process selected presentation edges
        estimates_significance = estimates_significance[presentation_edges]
        significant_timepoints = tools.significant_time_points(estimates_significance, window_size)

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
