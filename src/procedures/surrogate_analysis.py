""" Procedures of surrogate data analysis
"""
import os
from itertools import combinations
import numpy as np
import tools
from tools import print_info


def analyze_surrogate_statistics(
        filename: str,
        results_dirname: str,
        metric: str = "swd"):
    """ run procedures for analyzing empirical against surrogate statistics of given metric

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    if metric == "swc":
        tvfc = tools.swc
    else:
        tvfc = tools.swd

    print_info(
        "##########################################################################", results_dirname)
    print_info(f"INFO: analyzing {metric.upper()} surrogate statistics", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

    # generate Surrogate data with the same frequency spectrum,
    # and autocorrelation as empirical
    
    sc_data = tools.sc(emp_data)
    scc_data = tools.laumann(emp_data)

    surrogate_dir = os.path.join(results_dirname, f"{metric}-surrogate-analysis")
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
        title="SC Surrogate",
        out=os.path.join(surrogate_dir, "autcorrelation-SC-surrogate.png"))
    tools.plot_autocorrelation(
        scc_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="SCC Surrogate",
        out=os.path.join(surrogate_dir, "autcorrelation-SCC-surrogate.png"))


    window_sizes = [9, 19, 29, 39, 49, 59, 69,
                    79, 89, 99, emp_data.shape[-1]]
    for window_size in window_sizes:
        print_info(
            f"# window size = {window_size} ####################################################", results_dirname)
        # Compute and plot tvFC estimates
        estimates_empirical = tvfc(emp_data, window_size=window_size)
        estimates_sc = tvfc(sc_data, window_size=window_size)
        estimates_scc = tvfc(scc_data, window_size=window_size)

        tools.plot_overlapping_distributions(
            [estimates_empirical, estimates_sc],
            ["Empirical", "SC Surrogate"],
            xlabel="Estimate",
            ylabel="Density",
            title=f"w = {window_size}",
            out=os.path.join(
                surrogate_dir,
                f"tvFC-estimates-SC-overlapping-distribution-{window_size}.png"
            ))
        tools.plot_overlapping_distributions(
            [estimates_empirical, estimates_scc],
            ["Empirical", "SCC Surrogate"],
            xlabel="Estimate",
            ylabel="Density",
            title=f"w = {window_size}",
            out=os.path.join(
                surrogate_dir,
                f"tvFC-estimates-SCC-overlapping-distribution-{window_size}.png"
            ))

        if window_size > 1 and window_size < 100:
            edge_variance_empirical = np.var(estimates_empirical, axis=-1)
            edge_variance_scc = np.var(estimates_scc, axis=-1)
            tools.plot_overlapping_distributions(
                [
                    edge_variance_empirical,
                    edge_variance_scc,
                ],
                ["Empirical", "SCC Surrogate"],
                xlabel="Edge Variance",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"edge-variance-distribution-{window_size}.png"
                ))

            estimates_significance = np.abs(
                tools.significant_estimates(estimates_empirical))
            total_significance_count = np.sum(estimates_significance)
            significance_rate = total_significance_count / \
                np.size(estimates_significance)
            print_info("INFO: total number of significant tvFC estimates (before filtering): " +
                       f"{total_significance_count}, {significance_rate}", results_dirname)

            # Test time-averaged estimates null hypothesis (H1)
            interest_edges_h1 = tools.get_edges_of_interest(
                emp_data,
                sc_data,
                pairs,
                window_size=window_size,
                h1=True,
                metric=tvfc
            )
            estimates_significance_h1 = (
                estimates_significance.T * interest_edges_h1).T
            significance_count_h1 = np.sum(estimates_significance_h1)
            significance_rate_h1 = significance_count_h1 / \
                np.size(estimates_significance)
            h1_type1_error_rate = \
                (total_significance_count - significance_count_h1) / \
                total_significance_count
            print_info(
                f"INFO: significant edge count of H1: {np.sum(interest_edges_h1)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H1 " +
                       "(time-averaged estimates' null): " +
                       f"{significance_count_h1}, {significance_rate_h1}", results_dirname)
            print_info(
                f"INFO: H1 type 1 error rate: {h1_type1_error_rate}", results_dirname)

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
                    f"h1-filtered-overlapping-distributions-{window_size}.png"
                ))

            # Test edge variance null hypothesis (H2)
            interest_edges_h2 = tools.get_edges_of_interest(
                emp_data,
                scc_data,
                pairs,
                window_size=window_size,
                h2=True,
                metric=tvfc
            )
            estimates_significance_h2 = (
                estimates_significance.T * interest_edges_h2).T
            significance_count_h2 = np.sum(estimates_significance_h2)
            significance_rate_h2 = significance_count_h2 / \
                np.size(estimates_significance)
            h2_type1_error_rate = \
                (total_significance_count - significance_count_h2) / \
                total_significance_count
            print_info(f"INFO: significant edge count of H2 (w={window_size}): " +
                       f"{np.sum(interest_edges_h2)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H2 (edge variance null): " +
                       f"{significance_count_h2}, {significance_rate_h2}", results_dirname)
            print_info(
                f"INFO: H2 type 1 error rate (w={window_size}): {h2_type1_error_rate}", results_dirname)

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
                    f"h2-filtered-overlapping-distributions-{window_size}.png"
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
            all_type1_error_rate = \
                (total_significance_count - significance_count_h1h2) / \
                total_significance_count
            print_info(f"INFO: significant edge count of H1 & H2 (w={window_size}):" +
                       f" {np.sum(interest_edges_h1h2)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H1 & H2: " +
                       f"{significance_count_h1h2}, {significance_rate_h1h2}", results_dirname)
            print_info(
                f"INFO: H1 & H2 type 1 error rate (w={window_size}): {all_type1_error_rate}", results_dirname)

            insig_edge_indices = [
                i for i, is_edge_significant in enumerate(interest_edges_h1h2)
                if not is_edge_significant]
            sig_edge_indices_h1 = [
                i for i, is_edge_significant in enumerate(interest_edges_h1)
                if is_edge_significant]
            sig_edge_indices_h2 = [
                i for i, is_edge_significant in enumerate(interest_edges_h2)
                if is_edge_significant]
            sig_edge_indices_h1h2 = [
                i for i, is_edge_significant in enumerate(interest_edges_h1h2)
                if is_edge_significant]
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
                    f"h1h2-filtered-overlapping-distributions-{window_size}.png"
                ))

            estimates_significance = np.abs(tools.significant_estimates(
                estimates_empirical,
                mean=np.mean(estimates_empirical[insig_edge_indices]),
                std=np.std(estimates_empirical[insig_edge_indices])))
            total_significance_count = np.sum(estimates_significance)
            significance_rate = total_significance_count / \
                np.size(estimates_significance)
            print_info("INFO: total number of significant tvFC estimates (after filtering): " +
                       f"{total_significance_count}, {significance_rate}", results_dirname)
            false_significance = np.abs(tools.significant_estimates(
                estimates_empirical[insig_edge_indices]))
            false_positive_count = np.sum(false_significance)
            type_1_error_rate = false_positive_count / total_significance_count
            print_info(
                f"INFO: Filtered type 1 error rate (w={window_size}): {type_1_error_rate}", results_dirname)

            chance_significance_count_per_edge = total_significance_count / estimates_significance.shape[0]
            false_significance_rate = np.sum(
                estimates_significance[insig_edge_indices], axis=-1
            ) / (
                chance_significance_count_per_edge
            )
            edge_h1_significance_rate = np.sum(
                estimates_significance[sig_edge_indices_h1], axis=-1
            ) / (
                chance_significance_count_per_edge
            )
            edge_h2_significance_rate = np.sum(
                estimates_significance[sig_edge_indices_h2], axis=-1
            ) / (
                chance_significance_count_per_edge
            )
            edge_h1h2_significance_rate = np.sum(
                estimates_significance[sig_edge_indices_h1h2], axis=-1
            ) / (
                chance_significance_count_per_edge
            )
            discriminability_index_h1 = (
                np.median(edge_h1_significance_rate) - np.median(false_significance_rate)
            ) / (
                np.percentile(false_significance_rate, 75) - np.median(false_significance_rate)
            )
            discriminability_index_h2 = (
                np.median(edge_h2_significance_rate) - np.median(false_significance_rate)
            ) / (
                np.percentile(false_significance_rate, 75) - np.median(false_significance_rate)
            )
            discriminability_index_h1h2 = (
                np.median(edge_h1h2_significance_rate) - np.median(false_significance_rate)
            ) / (
                np.percentile(false_significance_rate, 75) - np.median(false_significance_rate)
            )
            print_info(f"INFO: Disriminability index of H1 (w={window_size}): " + \
                f"{discriminability_index_h1}", results_dirname)
            print_info(f"INFO: Disriminability index of H2 (w={window_size}): " + \
                f"{discriminability_index_h2}", results_dirname)
            print_info(f"INFO: Disriminability index of H1 & H2 (w={window_size}): " + \
                f"{discriminability_index_h1h2}", results_dirname)
            edge_discriminability_h1 = (np.sum(
                edge_h1_significance_rate[edge_h1_significance_rate > 1]
            ) + np.sum(
                false_significance_rate[false_significance_rate < 1]
            )) / (
                np.sum(edge_h1_significance_rate) + np.sum(false_significance_rate)
            )
            edge_discriminability_h2 = (np.sum(
                edge_h2_significance_rate[edge_h2_significance_rate > 1]
            ) + np.sum(
                false_significance_rate[false_significance_rate < 1]
            )) / (
                np.sum(edge_h2_significance_rate) + np.sum(false_significance_rate)
            )
            edge_discriminability_h1h2 = (np.sum(
                edge_h1h2_significance_rate[edge_h1h2_significance_rate > 1]
            ) + np.sum(
                false_significance_rate[false_significance_rate < 1]
            )) / (
                np.sum(edge_h1h2_significance_rate) + np.sum(false_significance_rate)
            )
            print_info(f"INFO: Edge disriminability index of H1 (w={window_size}): " + \
                f"{edge_discriminability_h1}", results_dirname)
            print_info(f"INFO: Edge disriminability index of H2 (w={window_size}): " + \
                f"{edge_discriminability_h2}", results_dirname)
            print_info(f"INFO: Edge disriminability index of H1 & H2 (w={window_size}): " + \
                f"{edge_discriminability_h1h2}", results_dirname)

            tools.plot_overlapping_distributions(
                [
                    edge_h1h2_significance_rate,
                    false_significance_rate
                ],
                [

                    "H1 & H2 Significant",
                    "H1 & H2 Insignificant",
                ],
                xlabel="Significance Rate per Edge",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"h1h2-discriminability-distributions-{window_size}.png"))
            tools.plot_overlapping_distributions(
                [
                    edge_h1_significance_rate,
                    false_significance_rate
                ],
                [
                    "H1 Significant",
                    "H1 & H2 Insignificant",
                ],
                xlabel="Significance Rate per Edge",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"h1-discriminability-distributions-{window_size}.png"))
            tools.plot_overlapping_distributions(
                [
                    edge_h2_significance_rate,
                    false_significance_rate
                ],
                [
                    "H2 Significant",
                    "H1 & H2 Insignificant",
                ],
                xlabel="Significance Rate per Edge",
                ylabel="Density",
                title=f"w = {window_size}",
                out=os.path.join(
                    surrogate_dir,
                    f"h2-discriminability-distributions-{window_size}.png"))

            # false_interest_edges_h2 = tools.get_edges_of_interest(
            #     scc_data,
            #     scc_data,
            #     pairs,
            #     window_size=window_size,
            #     h2=True,
            #     metric=tvfc
            #     alpha=0.05
            # )
            # false_sig_edge_indices_h2 = [
            #     i for i, is_edge_significant in enumerate(false_interest_edges_h2)
            #     if is_edge_significant]

            # session_length = estimates_empirical.shape[-1] // 4
            # session_idx = np.random.choice(4)
            # start = session_idx * session_length
            # end = start + session_length
            # stationarity_rate = 1 - tools.test_stationary(
            #     estimates_empirical[sig_edge_indices_h2, start:end])
            # print_info(
            #     f"INFO: Nonstationarity rate (w={window_size}): {stationarity_rate}", results_dirname)
            # stationarity_rate = 1 - tools.test_stationary(
            #     estimates_empirical[sig_edge_indices_h2, start:end], regression="ct")
            # print_info(
            #     f"INFO: Trend-nonstationarity rate (w={window_size}): {stationarity_rate}", results_dirname)
            # false_stationarity_rate = 1 - tools.test_stationary(
            #     estimates_scc[false_sig_edge_indices_h2, start:end])
            # print_info(
            #     f"INFO: False nonstationarity rate (w={window_size}): {false_stationarity_rate}", results_dirname)
            # false_stationarity_rate = 1 - tools.test_stationary(
            #     estimates_scc[false_sig_edge_indices_h2, start:end], regression="ct")
            # print_info(
            #     f"INFO: False trend-nonstationarity rate (w={window_size}): {false_stationarity_rate}", results_dirname)

        # End of window size loop
