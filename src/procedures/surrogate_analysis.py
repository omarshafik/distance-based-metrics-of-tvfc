""" Procedures of surrogate data analysis
"""
import os
from itertools import combinations
import numpy as np
import pandas as pd
import tools
from tools import print_info


def analyze_surrogate_statistics(
        data: any,
        results_dirname: str,
        metric_name: str = "swd",
        metric: callable = None,
        filename: str = None,
        sc_data: np.ndarray = None,
        scc_data: np.ndarray = None,
        pairs: np.ndarray = None,
        window_sizes: np.ndarray = None,
        plot: bool = True,
        info: bool = True,
        results: dict = None,
        random: np.random.Generator = None):
    """ run procedures for analyzing empirical against surrogate statistics of given metric

    Args:
        data (str | np.ndarray): filename to use or the data itself
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if random is None:
        random = np.random
    tools.plot.PLOT = plot
    tools.common.PRINT = info
    if results is None:
        results = {
            'filename': [],
            'metric': [],
            'window_size': [],
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
    if metric is None:
        if metric_name == "swc":
            metric = tools.swc
        elif metric_name == "mtd":
            metric = tools.mtd
        else:
            metric = tools.swd

    print_info("##########################################################################", results_dirname)
    print_info(f"INFO: analyzing {metric_name.upper()} surrogate statistics", results_dirname)
    if isinstance(data, str):
        emp_data = tools.prep_emp_data(np.loadtxt(data).T)
    else:
        emp_data = data

    if pairs is None:
        pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

    # generate Surrogate data with the same frequency spectrum,
    # and autocorrelation as empirical
    if sc_data is None:
        sc_data = tools.sc(emp_data, random=random)

    if scc_data is None:
        scc_data = tools.pr(emp_data, random=random)

    avg_sc_data = tools.sc(emp_data, average_spectrum=True, random=random)

    surrogate_dir = os.path.join(results_dirname, f"{metric_name}-surrogate-analysis")
    if plot:
        os.mkdir(surrogate_dir)

    # plot static Correlation matrices
    tools.plot_correlation_matrices(
        [emp_data],
        ["Empirical Correlation"],
        out=os.path.join(surrogate_dir, "correlation-plots-empirical.png"))
    tools.plot_correlation_matrices(
        [sc_data],
        ["SC Surrogate Correlation"],
        out=os.path.join(surrogate_dir, "correlation-plots-SC.png"))
    tools.plot_correlation_matrices(
        [scc_data],
        ["SCC Surrogate Correlation"],
        out=os.path.join(surrogate_dir, "correlation-plots-SCC.png"))

    tools.plot_autocorrelation(
        emp_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="Empirical",
        out=os.path.join(surrogate_dir, "autcorrelation-empirical.png"))
    tools.plot_autocorrelation(
        sc_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="SC",
        out=os.path.join(surrogate_dir, "autcorrelation-SC-surrogate.png"))
    tools.plot_autocorrelation(
        scc_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="SCC",
        out=os.path.join(surrogate_dir, "autcorrelation-SCC-surrogate.png"))

    timeavg_estimates_empirical = metric(emp_data, emp_data.shape[-1], pairs=pairs)
    timeavg_estimates_sc = metric(sc_data, sc_data.shape[-1], pairs=pairs)
    tools.plot_overlapping_distributions(
        [timeavg_estimates_empirical, timeavg_estimates_sc],
        ["Empirical", "SC"],
        xlabel="Estimate",
        ylabel="Density",
        title=f"w = {emp_data.shape[-1]}",
        out=os.path.join(
            surrogate_dir,
            f"tvFC-estimates-SC-overlapping-distribution-{emp_data.shape[-1]}.png"
        ))


    if window_sizes is None:
        if metric_name == "swd":
            window_sizes = [1, 5, 9, 19, 29, 39, 49, 59, 69,
                            79, 89, 99, emp_data.shape[-1]]
            # window_sizes = [29, 69, emp_data.shape[-1]]
        else:
            window_sizes = [5, 9, 19, 29, 39, 49, 59, 69,
                            79, 89, 99, emp_data.shape[-1]]
            # window_sizes = [29, 69, emp_data.shape[-1]]
    for window_size in window_sizes:
        if window_size == emp_data.shape[-1]:
            continue
        print_info(
            f"# window size = {window_size} ####################################################", results_dirname)
        # Compute and plot tvFC estimates
        estimates_empirical = metric(emp_data, window_size=window_size, pairs=pairs)
        estimates_sc = metric(sc_data, window_size=window_size, pairs=pairs)
        estimates_scc = metric(scc_data, window_size=window_size, pairs=pairs)

        tools.plot_overlapping_distributions(
            [estimates_empirical, estimates_sc],
            ["Empirical", "SC"],
            xlabel="Estimate",
            ylabel="Density",
            title=f"w = {window_size}",
            out=os.path.join(
                surrogate_dir,
                f"tvFC-estimates-SC-overlapping-distribution-{window_size}.png"
            ))
        tools.plot_overlapping_distributions(
            [estimates_empirical, estimates_scc],
            ["Empirical", "SCC"],
            xlabel="Estimate",
            ylabel="Density",
            title=f"w = {window_size}",
            out=os.path.join(
                surrogate_dir,
                f"tvFC-estimates-SCC-overlapping-distribution-{window_size}.png"
            ))

        if window_size > 1 and window_size < 100:
            results['window_size'].append(window_size)
            results['metric'].append(metric_name)
            results['filename'].append(filename)
            edge_variance_empirical = np.var(estimates_empirical, axis=-1)
            edge_variance_scc = np.var(estimates_scc, axis=-1)
            tools.plot_overlapping_distributions(
                [
                    edge_variance_empirical,
                    edge_variance_scc,
                ],
                ["Empirical", "SCC"],
                xlabel="Edge Variance",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"edge-variance-distribution-{window_size}.png"
                ))

            estimates_significance = np.abs(
                tools.significant_estimates(estimates_empirical, null=estimates_empirical))
            total_significance_count_nofilter = np.sum(estimates_significance)
            significance_rate_nofilter = total_significance_count_nofilter / \
                np.size(estimates_significance)
            print_info("INFO: total number of significant tvFC estimates (before filtering): " +
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
            h1_divergence = tools.kl_divergence(h1_posterior, significance_rate_nofilter)
            print_info(
                f"INFO: significant edge count of H1: {np.sum(interest_edges_h1)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H1 " +
                       "(time-averaged estimates' null): " +
                       f"{significance_count_h1}, {significance_rate_h1}", results_dirname)
            print_info(
                f"INFO: H1 Likelihood (w={window_size}): {np.mean(h1_likelihood)}", results_dirname)
            print_info(
                f"INFO: H1 posterior significance rate (w={window_size}): {np.mean(h1_posterior)}",
                results_dirname)
            print_info(
                f"INFO: H1 Divergence from null (chance): {h1_divergence}", results_dirname)
            results["likelihood_h1"].append(np.mean(h1_likelihood))
            results["posterior_h1"].append(np.mean(h1_posterior))
            results["divergence_h1"].append(h1_divergence)

            insig_edge_indices = [
                i for i, is_edge_significant in enumerate(interest_edges_h1)
                if not is_edge_significant]
            tools.plot_overlapping_distributions(
                [
                    estimates_empirical,
                    estimates_empirical[insig_edge_indices],
                ],
                [
                    "All",
                    "H1 filtered",
                ],
                xlabel="Estimate",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"filtered-h1-overlapping-distributions-{window_size}.png"
                ))

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
            h2_divergence = tools.kl_divergence(h2_posterior, significance_rate_nofilter)
            print_info(f"INFO: significant edge count of H2 (w={window_size}): " +
                       f"{np.sum(interest_edges_h2)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H2 (edge variance null): " +
                       f"{significance_count_h2}, {significance_rate_h2}", results_dirname)
            print_info(
                f"INFO: H2 Likelihood (w={window_size}): {np.mean(h2_likelihood)}",
                results_dirname)
            print_info(
                f"INFO: H2 posterior significance rate (w={window_size}): {np.mean(h2_posterior)}",
                results_dirname)
            print_info(
                f"INFO: H2 Divergence from null (chance): {h2_divergence}",
                results_dirname)
            results["likelihood_h2"].append(np.mean(h2_likelihood))
            results["posterior_h2"].append(np.mean(h2_posterior))
            results["divergence_h2"].append(h2_divergence)

            insig_edge_indices = [
                i for i, is_edge_significant in enumerate(interest_edges_h2)
                if not is_edge_significant]
            tools.plot_overlapping_distributions(
                [
                    estimates_empirical,
                    estimates_empirical[insig_edge_indices],
                ],
                [
                    "All",
                    "H2 filtered",
                ],
                xlabel="Estimate",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"filtered-h2-overlapping-distributions-{window_size}.png"
                ))

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
            h1h2_divergence = tools.kl_divergence(h1h2_posterior, significance_rate_nofilter)
            print_info(f"INFO: significant edge count of H1 & H2 (w={window_size}):" +
                       f" {np.sum(interest_edges_h1h2)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H1 & H2: " +
                       f"{significance_count_h1h2}, {significance_rate_h1h2}", results_dirname)
            print_info(
                f"INFO: H1 & H2 Likelihood (w={window_size}): {np.mean(h1h2_likelihood)}",
                results_dirname)
            print_info(
                f"INFO: H1 & H2 posterior significance rate (w={window_size}): {np.mean(h1h2_posterior)}",
                results_dirname)
            print_info(
                f"INFO: H1 & H2 Divergence from null (chance): {h1h2_divergence}",
                results_dirname)
            results["likelihood_h1h2"].append(np.mean(h1h2_likelihood))
            results["posterior_h1h2"].append(np.mean(h1h2_posterior))
            results["divergence_h1h2"].append(h1h2_divergence)

            insig_edge_indices = [
                i for i, is_edge_significant in enumerate(interest_edges_h1h2)
                if not is_edge_significant]
            tools.plot_overlapping_distributions(
                [
                    estimates_empirical,
                    estimates_empirical[insig_edge_indices],
                ],
                [
                    "All",
                    "H1 & H2 filtered",
                ],
                xlabel="Estimate",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"filtered-h1h2-overlapping-distributions-{window_size}.png"
                ))

            estimates_significance = np.abs(tools.significant_estimates(
                estimates_empirical,
                null=estimates_empirical[insig_edge_indices]))
            total_significance_count_filter = np.sum(estimates_significance)
            significance_rate_filter = total_significance_count_filter / \
                np.size(estimates_significance)
            print_info("INFO: total number of significant tvFC estimates (after filtering): " +
                       f"{total_significance_count_filter}, {significance_rate_filter}", results_dirname)
            change_in_significance_rate = (
                significance_rate_filter - significance_rate_nofilter
            ) / significance_rate_nofilter
            h1h2_likelihood = tools.likelihood(estimates_significance, interest_edges_h1h2)
            h1h2_posterior = tools.posterior(estimates_significance, interest_edges_h1h2)
            h1h2_divergence = tools.kl_divergence(h1h2_posterior, significance_rate_filter)
            print_info(
                f"INFO: H1 & H2 Likelihood (w={window_size}): {np.mean(h1h2_likelihood)}", results_dirname)
            print_info(
                f"INFO: H1 & H2 posterior significance rate (w={window_size}): {np.mean(h1h2_posterior)}",
                results_dirname)
            print_info(
                f"INFO: Total Divergence from null with new confidence levels: {h1h2_divergence}", results_dirname)
            print_info(
                f"INFO: Change in significance rate: {change_in_significance_rate}", results_dirname)
            results["likelihood_h1h2_updated"].append(np.mean(h1h2_likelihood))
            results["posterior_h1h2_updated"].append(np.mean(h1h2_posterior))
            results["divergence_h1h2_updated"].append(h1h2_divergence)

            empirical_significance_rate = tools.scaled_significance_rate(
                estimates_significance
            )
            estimates_avg_sc = metric(avg_sc_data, window_size, pairs=pairs)
            null_significance = np.abs(tools.significant_estimates(
                estimates_avg_sc,
                null=estimates_avg_sc))
            null_significance_rate = tools.scaled_significance_rate(
                null_significance
            )
            tools.plot_overlapping_distributions(
                [empirical_significance_rate, null_significance_rate],
                ["Empirical", "SC"],
                xlabel="Significance Rate per Edge",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"discriminability-distributions-{window_size}.png"))

            # plot scatter diagram of avgFC, edge variance, and edge significance rate
            tools.scatter_fc_edges(timeavg_estimates_empirical,
                edge_variance_empirical,
                empirical_significance_rate,
                out=os.path.join(
                    surrogate_dir,
                    f"avgfc-variance-sr-scatter-{window_size}.png"))

        # End of window size loop
    return results

def evaluate_tvfc_metrics(
    input_filenames: str,
    results_dirname: str,
    metrics: dict = None,
    random: np.random.Generator = None):
    """_summary_

    Args:
        input_filenames (str): _description_
        results_dirname (str): _description_
        metrics (list, optional): _description_. Defaults to ["mtd", "swc", "swd"].
    """
    if random is None:
        random = np.random
    if metrics is None:
        metrics = {
            "mtd": tools.mtd,
            "swc": tools.swc,
            "swd": tools.swd
        }

    results = {
        'filename': [],
        'metric': [],
        'window_size': [],
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

    evaluation_dir = os.path.join(results_dirname, "metrics-evaluation-analysis")
    os.mkdir(evaluation_dir)

    print_info(
        "##########################################################################", results_dirname)
    print_info("INFO: Caryying out tvFC metrics evaluation analysis", results_dirname)

    number_of_subjects = 30
    random_file_indices = random.choice(len(input_filenames), number_of_subjects, replace=False)
    selected_subject_nums = [
        os.path.basename(input_filenames[subject_idx]) for subject_idx in random_file_indices]
    print_info(f"INFO: Selected files {', '.join(selected_subject_nums)}", results_dirname)

    window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    for fileidx in random_file_indices:
        filename = input_filenames[fileidx]
        emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
        pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

        # generate Surrogate data with the same frequency spectrum,
        # autocorrelation, cross-frequency as empirical
        sc_data = tools.sc(emp_data, random=random)
        scc_data = tools.pr(emp_data, random=random)
        for metric_name, metric in metrics.items():
            analyze_surrogate_statistics(
                emp_data,
                results_dirname,
                metric_name=metric_name,
                metric=metric,
                window_sizes=window_sizes,
                sc_data=sc_data,
                scc_data=scc_data,
                pairs=pairs,
                plot=False,
                info=False,
                filename=os.path.basename(filename),
                results=results
            )

    pd.DataFrame(results).to_csv(
        os.path.join(evaluation_dir, "baysian-stats.csv"),
        index=False
    )
