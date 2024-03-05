""" Procedures of surrogate data analysis
"""
import os
from itertools import combinations
import numpy as np
import pandas as pd
import tools
from tools import print_info


def analyze_surrogate_statistics(
        data: str | np.ndarray,
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
            'sdv': [],
            'sdr_h1': [],
            'sdr_h2': [],
            'sdr_h1h2': [],
            'edr_h1': [],
            'edr_h2': [],
            'edr_h1h2': [],
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
        title="SC Surrogate",
        out=os.path.join(surrogate_dir, "autcorrelation-SC-surrogate.png"))
    tools.plot_autocorrelation(
        scc_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="SCC Surrogate",
        out=os.path.join(surrogate_dir, "autcorrelation-SCC-surrogate.png"))

    timeavg_estimates_empirical = metric(emp_data, emp_data.shape[-1], pairs=pairs)
    timeavg_estimates_sc = metric(sc_data, sc_data.shape[-1], pairs=pairs)
    
    if metric_name in ["mtd", "swd"]:
        # prepare pointwise estimates to reduce amount of computations
        pw_estimates_empirical = metric(emp_data, 1, pairs=pairs)
        pw_estimates_sc = metric(sc_data, 1, pairs=pairs)
        pw_estimates_scc = metric(scc_data, 1, pairs=pairs)

    if window_sizes is None:
        if metric_name == "swd":
            window_sizes = [1, 5, 9, 19, 29, 39, 49, 59, 69,
                            79, 89, 99, emp_data.shape[-1]]
        else:
            window_sizes = [5, 9, 19, 29, 39, 49, 59, 69,
                            79, 89, 99, emp_data.shape[-1]]
    for window_size in window_sizes:
        print_info(
            f"# window size = {window_size} ####################################################", results_dirname)
        # Compute and plot tvFC estimates
        if metric_name in ["mtd", "swd"]:
            estimates_empirical = tools.sliding_average(pw_estimates_empirical, window_size)
            estimates_sc = tools.sliding_average(pw_estimates_sc, window_size)
            estimates_scc = tools.sliding_average(pw_estimates_scc, window_size)
        else:
            estimates_empirical = metric(emp_data, window_size=window_size, pairs=pairs)
            estimates_sc = metric(sc_data, window_size=window_size, pairs=pairs)
            estimates_scc = metric(scc_data, window_size=window_size, pairs=pairs)

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
            total_significance_count_nofilter = np.sum(estimates_significance)
            significance_rate_nofilter = total_significance_count_nofilter / \
                np.size(estimates_significance)
            print_info("INFO: total number of significant tvFC estimates (before filtering): " +
                       f"{total_significance_count_nofilter}, {significance_rate_nofilter}", results_dirname)

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
            h1_type1_error_rate = \
                (total_significance_count_nofilter - significance_count_h1) / \
                total_significance_count_nofilter
            q = 1 - h1_type1_error_rate
            p = np.sum(interest_edges_h1) / estimates_empirical.shape[0]
            h1_null_deviation = p * np.log(p / q)
            print_info(
                f"INFO: significant edge count of H1: {np.sum(interest_edges_h1)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H1 " +
                       "(time-averaged estimates' null): " +
                       f"{significance_count_h1}, {significance_rate_h1}", results_dirname)
            print_info(
                f"INFO: H1 type 1 error rate: {h1_type1_error_rate}", results_dirname)
            print_info(
                f"INFO: H1 Deviation from null (chance): {h1_null_deviation}", results_dirname)

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
            h2_type1_error_rate = \
                (total_significance_count_nofilter - significance_count_h2) / \
                total_significance_count_nofilter
            q = 1 - h2_type1_error_rate
            p = np.sum(interest_edges_h2) / estimates_empirical.shape[0]
            h2_null_deviation = p * np.log(p / q)
            print_info(f"INFO: significant edge count of H2 (w={window_size}): " +
                       f"{np.sum(interest_edges_h2)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H2 (edge variance null): " +
                       f"{significance_count_h2}, {significance_rate_h2}", results_dirname)
            print_info(
                f"INFO: H2 type 1 error rate (w={window_size}): {h2_type1_error_rate}", results_dirname)
            print_info(
                f"INFO: H2 Deviation from null (chance): {h2_null_deviation}", results_dirname)

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
            all_type1_error_rate = \
                (total_significance_count_nofilter - significance_count_h1h2) / \
                total_significance_count_nofilter
            q = 1 - all_type1_error_rate
            p = np.sum(interest_edges_h1h2) / estimates_empirical.shape[0]
            h1h2_null_deviation = p * np.log(p / q)
            print_info(f"INFO: significant edge count of H1 & H2 (w={window_size}):" +
                       f" {np.sum(interest_edges_h1h2)}", results_dirname)
            print_info("INFO: significant tvFC estimates count of H1 & H2: " +
                       f"{significance_count_h1h2}, {significance_rate_h1h2}", results_dirname)
            print_info(
                f"INFO: H1 & H2 type 1 error rate (w={window_size}): {all_type1_error_rate}", results_dirname)
            print_info(
                f"INFO: H1 & H2 Deviation from null (chance): {h1h2_null_deviation}", results_dirname)

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
                    f"filtered-h1h2-overlapping-distributions-{window_size}.png"
                ))

            estimates_significance = np.abs(tools.significant_estimates(
                estimates_empirical,
                null=estimates_sc))
            total_significance_count_filter = np.sum(estimates_significance)
            significance_rate_filter = total_significance_count_filter / \
                np.size(estimates_significance)
            print_info("INFO: total number of significant tvFC estimates (after filtering): " +
                       f"{total_significance_count_filter}, {significance_rate_filter}", results_dirname)
            false_significance = np.abs(tools.significant_estimates(
                estimates_empirical[insig_edge_indices]))
            false_positive_count = np.sum(false_significance)
            type_1_error_rate = false_positive_count / total_significance_count_filter
            change_in_significance_rate = (
                significance_rate_filter - (false_positive_count / np.size(false_significance))
            ) / (false_positive_count / np.size(false_significance))
            q = 1 - type_1_error_rate
            p = np.sum(interest_edges_h1h2) / estimates_empirical.shape[0]
            null_deviation = p * np.log(p / q)
            print_info(
                f"INFO: Filtered type 1 error rate (w={window_size}): {type_1_error_rate}", results_dirname)
            print_info(
                f"INFO: H1 & H2 Deviation from null with new confidence levels: {null_deviation}", results_dirname)
            print_info(
                f"INFO: Change in significance rate: {change_in_significance_rate}", results_dirname)

            empirical_significance_rate = tools.scaled_significance_rate(
                estimates_significance
            )
            false_significance_rate = empirical_significance_rate[insig_edge_indices]
            edge_h1_significance_rate = empirical_significance_rate[sig_edge_indices_h1]
            edge_h2_significance_rate = empirical_significance_rate[sig_edge_indices_h2]
            edge_h1h2_significance_rate = empirical_significance_rate[sig_edge_indices_h1h2]

            null_significance = np.abs(tools.significant_estimates(
                estimates_sc,
                null=estimates_sc))
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

            sdv = tools.sdv(
                empirical_significance_rate,
                null_significance_rate
            )
            print_info(f"INFO: SDV of empirical data (w={window_size}): " + \
                f"{sdv}", results_dirname)
            results["sdv"].append(sdv)
            sdr_h1 = tools.sdr(
                edge_h1_significance_rate,
                false_significance_rate
            )
            sdr_h2 = tools.sdr(
                edge_h2_significance_rate,
                false_significance_rate
            )
            sdr_h1h2 =  tools.sdr(
                edge_h1h2_significance_rate,
                false_significance_rate
            )
            print_info(f"INFO: SDR of H1 (w={window_size}): " + \
                f"{sdr_h1}", results_dirname)
            print_info(f"INFO: SDR of H2 (w={window_size}): " + \
                f"{sdr_h2}", results_dirname)
            print_info(f"INFO: SDR of H1 & H2 (w={window_size}): " + \
                f"{sdr_h1h2}", results_dirname)
            results["sdr_h1"].append(sdr_h1)
            results["sdr_h2"].append(sdr_h2)
            results["sdr_h1h2"].append(sdr_h1h2)
            edr_h1 = tools.edr(
                edge_h1_significance_rate,
                false_significance_rate
            )
            edr_h2 = tools.edr(
                edge_h2_significance_rate,
                false_significance_rate
            )
            edr_h1h2 =  tools.edr(
                edge_h1h2_significance_rate,
                false_significance_rate
            )
            print_info(f"INFO: EDR of H1 (w={window_size}): " + \
                f"{edr_h1}", results_dirname)
            print_info(f"INFO: EDR of H2 (w={window_size}): " + \
                f"{edr_h2}", results_dirname)
            print_info(f"INFO: EDR of H1 & H2 (w={window_size}): " + \
                f"{edr_h1h2}", results_dirname)
            results["edr_h1"].append(edr_h1)
            results["edr_h2"].append(edr_h2)
            results["edr_h1h2"].append(edr_h1h2)

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
                    f"discriminability-h1h2-distributions-{window_size}.png"))
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
                    f"discriminability-h1-distributions-{window_size}.png"))
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
                    f"discriminability-h2-distributions-{window_size}.png"))

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
            # session_idx = random.choice(4)
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
        'sdv': [],
        'sdr_h1': [],
        'sdr_h2': [],
        'sdr_h1h2': [],
        'edr_h1': [],
        'edr_h2': [],
        'edr_h1h2': [],
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
        os.path.join(evaluation_dir, "discriminability.csv"),
        index=False
    )
