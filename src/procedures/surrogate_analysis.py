""" Procedures of surrogate data analysis
"""
import os
from itertools import combinations
import numpy as np
import pandas as pd
import tools
from tools import print_info


SURROGATE_METHODS = {
    "PR": tools.pr,
    "Laumann": tools.laumann
}

METRICS = {
    "mtd": tools.mtd,
    "swc": tools.swc,
    "swd": tools.swd
}

RESULTS = {
        'filename': [],
        'surrogate_method': [],
        'lpf_window_size': [],
        'metric': [],
        'window_size': [],
        'edge_variance_significance': [],
        'likelihood_h1': [],
        'likelihood_h2': [],
        'likelihood_h1h2': [],
        'likelihood_h1h2_updated': [],
        'posterior_h1': [],
        'posterior_h2': [],
        'posterior_h1h2': [],
        'posterior_h1h2_updated': [],
        'divergence_h1': [],
        'divergence_h2': [],
        'divergence_h1h2': [],
        'divergence_h1h2_updated': [],
    }


def surrogate_analysis(
        data: any,
        window_size: int,
        results_dirname: str,
        metric_name: str = "swd",
        metric: callable = None,
        lpf_window_size: int = 0,
        sc_data: np.ndarray = None,
        surrogate_method: str = "PR",
        scc_data: np.ndarray = None,
        pairs: np.ndarray = None,
        info: bool = True,
        results: dict = None,
        random: np.random.Generator = None):
    """ run procedures for analyzing empirical against surrogate statistics of given metric
    """
    if random is None:
        random = np.random
    tools.common.PRINT = info
    if metric is None:
        metric = METRICS[metric_name]
    if results is None:
        results = RESULTS

    if isinstance(data, str):
        emp_data = tools.prep_emp_data(np.loadtxt(data).T, smooth=lpf_window_size)
    else:
        emp_data = data

    if pairs is None:
        pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

    # generate Surrogate data with the same frequency spectrum,
    # and autocorrelation as empirical
    if sc_data is None:
        sc_data = tools.sc(emp_data, random=random)

    if scc_data is None:
        scc_data = SURROGATE_METHODS[surrogate_method]

    timeavg_estimates_empirical = metric(emp_data, emp_data.shape[-1], pairs=pairs)
    timeavg_estimates_sc = metric(sc_data, sc_data.shape[-1], pairs=pairs)

    # Compute tvFC estimates
    estimates_empirical = metric(emp_data, window_size=window_size, pairs=pairs)
    estimates_scc = metric(scc_data, window_size=window_size, pairs=pairs)

    edge_variance_empirical = np.var(estimates_empirical, axis=-1)
    edge_variance_scc = np.var(estimates_scc, axis=-1)

    estimates_significance = np.abs(
        tools.significant_estimates(estimates_empirical, null=estimates_empirical))
    total_significance_count_nofilter = np.sum(estimates_significance)
    significance_rate_nofilter = total_significance_count_nofilter / \
        np.size(estimates_significance)
    print_info("total number of significant tvFC estimates (before filtering): " +
        f"{total_significance_count_nofilter}, {significance_rate_nofilter}",
        results_dirname)

    # Test time-averaged estimates null hypothesis (H1)
    interest_edges_h1 = tools.get_edges_of_interest(
        timeavg_estimates_empirical,
        timeavg_estimates_sc,
        alpha=0
    )
    estimates_significance_h1 = (
        estimates_significance.T * interest_edges_h1).T
    significance_count_h1 = np.sum(estimates_significance_h1)
    significance_rate_h1 = significance_count_h1 / \
        np.size(estimates_significance)
    h1_likelihood = tools.likelihood(estimates_significance, interest_edges_h1)
    h1_posterior = tools.posterior(estimates_significance, interest_edges_h1)
    h1_divergence = tools.kl_divergence(estimates_significance, interest_edges_h1)
    print_info(
        f"significant edge count of H1: {np.sum(interest_edges_h1)}", results_dirname)
    print_info("significant tvFC estimates count of H1 " +
                "(time-averaged estimates' null): " +
                f"{significance_count_h1}, {significance_rate_h1}", results_dirname)
    print_info(
        f"H1 Likelihood (w={window_size}): {np.mean(h1_likelihood)}", results_dirname)
    print_info(
        f"H1 posterior significance rate (w={window_size}): {np.mean(h1_posterior)}",
        results_dirname)
    print_info(
        f"H1 Divergence from null (chance): {h1_divergence}", results_dirname)
    results["likelihood_h1"].append(np.mean(h1_likelihood))
    results["posterior_h1"].append(np.mean(h1_posterior))
    results["divergence_h1"].append(h1_divergence)

    # Test edge variance null hypothesis (H2)
    interest_edges_h2 = tools.get_edges_of_interest(
        edge_variance_empirical,
        edge_variance_scc,
        one_side=True,
        alpha=0.05
    )
    estimates_significance_h2 = (
        estimates_significance.T * interest_edges_h2).T
    significance_count_h2 = np.sum(estimates_significance_h2)
    significance_rate_h2 = significance_count_h2 / \
        np.size(estimates_significance)
    h2_likelihood = tools.likelihood(estimates_significance, interest_edges_h2)
    h2_posterior = tools.posterior(estimates_significance, interest_edges_h2)
    h2_divergence = tools.kl_divergence(estimates_significance, interest_edges_h2)
    print_info(f"significant edge count of H2 (w={window_size}): " +
                f"{np.sum(interest_edges_h2)}", results_dirname)
    print_info("significant tvFC estimates count of H2 (edge variance null): " +
                f"{significance_count_h2}, {significance_rate_h2}", results_dirname)
    print_info(
        f"H2 Likelihood (w={window_size}): {np.mean(h2_likelihood)}",
        results_dirname)
    print_info(
        f"H2 posterior significance rate (w={window_size}): {np.mean(h2_posterior)}",
        results_dirname)
    print_info(
        f"H2 Divergence from null (chance): {h2_divergence}",
        results_dirname)
    results["likelihood_h2"].append(np.mean(h2_likelihood))
    results["posterior_h2"].append(np.mean(h2_posterior))
    results["divergence_h2"].append(h2_divergence)
    results['edge_variance_significance'].append(np.sum(interest_edges_h2) / np.size(interest_edges_h2))

    # find significantly variant/different tvFC estimates that belong to
    #   edges with significant statistics (time-averaged estimate or edge variance)
    interest_edges_h1h2 = interest_edges_h1 + interest_edges_h2
    interest_edges_h1h2[interest_edges_h1h2 != 0] = 1
    estimates_significance_h1h2 = (
        estimates_significance.T * interest_edges_h1h2).T
    significance_count_h1h2 = np.sum(estimates_significance_h1h2)
    significance_rate_h1h2 = significance_count_h1h2 / \
        np.size(estimates_significance)
    h1h2_likelihood = tools.likelihood(estimates_significance, interest_edges_h1h2)
    h1h2_posterior = tools.posterior(estimates_significance, interest_edges_h1h2)
    h1h2_divergence = tools.kl_divergence(estimates_significance, interest_edges_h1h2)
    print_info(f"significant edge count of H1 & H2 (w={window_size}):" +
                f" {np.sum(interest_edges_h1h2)}", results_dirname)
    print_info("significant tvFC estimates count of H1 & H2: " +
                f"{significance_count_h1h2}, {significance_rate_h1h2}", results_dirname)
    print_info(
        f"H1 & H2 Likelihood (w={window_size}): {np.mean(h1h2_likelihood)}",
        results_dirname)
    print_info(
        f"H1 & H2 posterior significance rate (w={window_size}): {np.mean(h1h2_posterior)}",
        results_dirname)
    print_info(
        f"H1 & H2 Divergence from null (chance): {h1h2_divergence}",
        results_dirname)
    results["likelihood_h1h2"].append(np.mean(h1h2_likelihood))
    results["posterior_h1h2"].append(np.mean(h1h2_posterior))
    results["divergence_h1h2"].append(h1h2_divergence)

    insig_edge_indices = [
        i for i, is_edge_significant in enumerate(interest_edges_h1h2)
        if not is_edge_significant]

    estimates_significance = np.abs(tools.significant_estimates(
        estimates_empirical,
        null=estimates_empirical[insig_edge_indices]))
    total_significance_count_filter = np.sum(estimates_significance)
    significance_rate_filter = total_significance_count_filter / \
        np.size(estimates_significance)
    print_info("total number of significant tvFC estimates (after filtering): " +
                f"{total_significance_count_filter}, {significance_rate_filter}", results_dirname)
    change_in_significance_rate = (
        significance_rate_filter - significance_rate_nofilter
    ) / significance_rate_nofilter
    h1h2_likelihood = tools.likelihood(estimates_significance, interest_edges_h1h2)
    h1h2_posterior = tools.posterior(estimates_significance, interest_edges_h1h2)
    h1h2_divergence = tools.kl_divergence(estimates_significance, interest_edges_h1h2)
    print_info(
        f"H1 & H2 Likelihood (w={window_size}): {np.mean(h1h2_likelihood)}", results_dirname)
    print_info(
        f"H1 & H2 posterior significance rate (w={window_size}): {np.mean(h1h2_posterior)}",
        results_dirname)
    print_info(
        f"Total Divergence from null with new confidence levels: {h1h2_divergence}", results_dirname)
    print_info(
        f"Change in significance rate: {change_in_significance_rate}", results_dirname)
    results["likelihood_h1h2_updated"].append(np.mean(h1h2_likelihood))
    results["posterior_h1h2_updated"].append(np.mean(h1h2_posterior))
    results["divergence_h1h2_updated"].append(h1h2_divergence)

    return results


def metrics_surrogates_evaluation(
    input_filenames: str,
    results_dirname: str,
    metrics: dict = None,
    n_subjects: int = 30,
    surrogate_methods: dict = None,
    lpf_window_sizes: list = None,
    window_sizes: list = None,
    random: np.random.Generator = None):
    """ runs surrogate statistical analyses on given metrics, using given data files, \
        surrogate methods, low-pass filter window sizes, and window sizes
    """
    if random is None:
        random = np.random
    if metrics is None:
        metrics = METRICS
    if n_subjects is None:
        n_subjects = 30
    if surrogate_methods is None:
        surrogate_methods = SURROGATE_METHODS
    if lpf_window_sizes is None:
        lpf_window_sizes = [0, 10]
    if window_sizes is None:
        window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]

    results = RESULTS

    evaluation_dir = os.path.join(results_dirname, "metrics-evaluation-analysis")
    os.mkdir(evaluation_dir)

    print_info(
        "##########################################################################", results_dirname)
    print_info(f"Caryying out tvFC metrics evaluation analysis for {n_subjects} subjects", results_dirname)

    random_file_indices = random.choice(len(input_filenames), n_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"Selected files {', '.join(selected_subject_nums)}", results_dirname)

    for fileidx in random_file_indices:
        filename = input_filenames[fileidx]
        print_info(f"Analyzing {os.path.basename(filename)}...", evaluation_dir)
        for lpf_winsize in lpf_window_sizes:
            print_info(f"Using low-pass filter window size = {lpf_winsize}", evaluation_dir)
            emp_data = tools.prep_emp_data(np.loadtxt(filename).T, smooth=lpf_winsize)
            pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

            sc_data = tools.sc(emp_data, random=random)
            surrogates = {
                method_name: method(emp_data, random=random)
                for method_name, method in surrogate_methods.items()
            }
            for surrogate_method, surrogate_data in surrogates.items():
                print_info(f"Using {surrogate_method} surrogates", evaluation_dir)
                for metric_name, metric in metrics.items():
                    print_info(f"Analyzing {metric_name.upper()} metric", evaluation_dir)
                    for winsize in window_sizes:
                        print_info(f"Using window size = {winsize}", evaluation_dir)
                        print_info("#####################################", evaluation_dir)
                        results['window_size'].append(winsize)
                        results['metric'].append(metric_name)
                        results['filename'].append(os.path.basename(filename))
                        results['surrogate_method'].append(surrogate_method)
                        results['lpf_window_size'].append(lpf_winsize)
                        surrogate_analysis(
                            emp_data,
                            window_size=winsize,
                            results_dirname=evaluation_dir,
                            lpf_window_size=lpf_winsize,
                            metric_name=metric_name,
                            metric=metric,
                            sc_data=sc_data,
                            surrogate_method=surrogate_method,
                            scc_data=surrogate_data,
                            pairs=pairs,
                            info=True,
                            results=results
                        )
                        print_info("", evaluation_dir)
                    print_info("#########################################", evaluation_dir)
                print_info("#############################################", evaluation_dir)
            print_info("#################################################", evaluation_dir)

    results_filepath = os.path.join(evaluation_dir, "surrogate-stats.csv")
    pd.DataFrame(results).to_csv(
        results_filepath,
        index=False
    )
    return results_filepath
