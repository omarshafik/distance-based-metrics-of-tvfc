""" Procedures of surrogate data analysis
"""
import os
from itertools import combinations
import numpy as np
import tools

def analyze_surrogate_statistics(
    filename: str,
    results_dirname: str):
    """ run procedures for analyzing empirical against surrogate statistics of SWD

    Args:
        filename (str): filename to use
        results_dirname (str): parent directory name of the results
            (results will stored in a new subdirectory)
    """
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

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

    surrogate_dir = os.path.join(results_dirname, "surrogate-analysis")
    os.mkdir(surrogate_dir)

    # plot static Correlation matrices
    tools.plot_correlation_matrices(
        [emp_data, sc_data],
        ["Empirical Correlation", "SC Surrogate Correlation"],
        out=os.path.join(surrogate_dir, "SC-correlation-plots.png"))
    tools.plot_correlation_matrices(
        [emp_data, scc_data],
        ["Empirical Correlation", "SCC Surrogate Correlation"],
        out=os.path.join(surrogate_dir, "SCC-correlation-plots.png"))

    tools.plot_autocorrelation(
        emp_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="Autocorrelation of Empirical Data",
        out=os.path.join(surrogate_dir, "autcorrelation-empirical.png"))
    tools.plot_autocorrelation(
        sc_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="Autocorrelation of SC Surrogate Data",
        out=os.path.join(surrogate_dir, "autcorrelation-SC-surrogate.png"))
    tools.plot_autocorrelation(
        scc_data,
        xlabel="Lag",
        ylabel="Correlation",
        title="Autocorrelation of SCC Surrogate Data",
        out=os.path.join(surrogate_dir, "autcorrelation-SCC-surrogate.png"))

    window_sizes = [9, 19, 29, 39, 49, 59, 69, 99, 299, 499, emp_data.shape[-1]]
    for window_size in window_sizes:
        print(f"# window size = {window_size} ####################################################")
        # Compute and plot SWD
        estimates_empirical = tools.swd(emp_data, window_size=window_size)
        estimates_sc = tools.swd(sc_data, window_size=window_size)
        estimates_scc = tools.swd(scc_data, window_size=window_size)

        tools.plot_overlapping_distributions(
            [estimates_empirical, estimates_sc],
            ["Empirical", "SC Surrogate"],
            xlabel="Estimate",
            ylabel="Density",
            title="Distributions of tvFC Estimates for Empirical and SC Surrogate Data",
            out=os.path.join(
                surrogate_dir,
                f"tvFC-estimates-SC-overlapping-distribution-{window_size}.png"
            ))
        tools.plot_overlapping_distributions(
            [estimates_empirical, estimates_scc],
            ["Empirical", "SCC Surrogate"],
            xlabel="Estimate",
            ylabel="Density",
            title="Distributions of tvFC Estimates for Empirical and SCC Surrogate Data",
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
                ["Empirical", "Surrogate"],
                xlabel="Edge Variance",
                ylabel="Density",
                title="Distributions of Edge Variances for Empirical and SCC Surrogate Data",
                out=os.path.join(
                    surrogate_dir,
                    f"edge-variance-distribution-{window_size}.png"
                ))

            estimates_significance = np.abs(tools.significant_estimates(estimates_empirical))
            total_significance_count = np.sum(estimates_significance)
            significance_rate = total_significance_count / np.size(estimates_significance)
            print("INFO: total number of significant tvFC estimates (before filtering): " + \
                f"{total_significance_count}, {significance_rate}")

            # Test time-averaged estimates null hypothesis (H1)
            interest_edges_h1 = tools.get_edges_of_interest(
                emp_data,
                sc_data,
                pairs,
                window_size=window_size,
                alpha=0.05,
                h1=True
            )
            estimates_significance_h1 = (estimates_significance.T * interest_edges_h1).T
            significance_count_h1 = np.sum(estimates_significance_h1)
            significance_rate_h1 = significance_count_h1 / np.size(estimates_significance)
            h1_type1_error_rate = \
                (total_significance_count - significance_count_h1) / total_significance_count
            print(f"INFO: significant edge count of H1 (w={window_size}): " + \
                  f"{significance_count_h1}")
            print("INFO: significant tvFC estimates count of H1 " + \
                "(time-averaged estimates' null): " + \
                f"{significance_count_h1}, {significance_rate_h1}")
            print(f"INFO: H1 type 1 error rate (w={window_size}): {h1_type1_error_rate}")

            # Test edge variance null hypothesis (H2)
            interest_edges_h2 = tools.get_edges_of_interest(
                emp_data,
                scc_data,
                pairs,
                window_size=window_size,
                alpha=0.05,
                h2=True
            )
            estimates_significance_h2 = (estimates_significance.T * interest_edges_h2).T
            significance_count_h2 = np.sum(estimates_significance_h2)
            significance_rate_h2 = significance_count_h2 / np.size(estimates_significance)
            h2_type1_error_rate = \
                (total_significance_count - significance_count_h2) / total_significance_count
            print(f"INFO: significant edge count of H2 (w={window_size}): " + \
                  f"{np.sum(interest_edges_h2)}")
            print("INFO: significant tvFC estimates count of H2 (edge variance null): " + \
                f"{significance_count_h2}, {significance_rate_h2}")
            print(f"INFO: H2 type 1 error rate (w={window_size}): {h2_type1_error_rate}")

            # find significantly variant/different tvFC estimates that belong to
            #   edges with significant statistics (time-averaged estimate or edge variance)
            interest_edges_h1h2 = interest_edges_h1 + interest_edges_h2
            interest_edges_h1h2[interest_edges_h1h2 != 0] = 1
            estimates_significance_h1h2 = (estimates_significance.T * interest_edges_h1h2).T
            significance_count_h1h2 = np.sum(estimates_significance_h1h2)
            significance_rate_h1h2 = significance_count_h1h2 / np.size(estimates_significance)
            all_type1_error_rate = \
                (total_significance_count - significance_count_h1h2) / total_significance_count
            print(f"INFO: significant edge count of H1 & H2 (w={window_size}):" + \
                f" {np.sum(interest_edges_h1h2)}")
            print("INFO: significant tvFC estimates count of H1 & H2: " + \
                f"{significance_count_h1h2}, {significance_rate_h1h2}")
            print(f"INFO: H1 & H2 type 1 error rate (w={window_size}): {all_type1_error_rate}")
        # End of window size loop
