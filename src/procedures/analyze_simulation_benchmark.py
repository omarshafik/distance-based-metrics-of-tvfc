import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tools
from tools import print_info, bioplausible


def simulatiom_benchmark(
    filename: str,
    results_dirname: str,
    metrics: dict = None,
    random: np.random.Generator = None
):
    if random is None:
        random = np.random
    if metrics is None:
        metrics = {
            "mtd": tools.mtd,
            "swc": tools.swc,
            "swd": tools.swd
        }

    print_info("##########################################################################", results_dirname)
    print_info("INFO: Carrying out simulation benchmarks", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    sc_data = tools.sc(emp_data, random=random)

    benchmark_dir = os.path.join(results_dirname, "simulation-benchmark")
    if not os.path.exists(benchmark_dir):
        os.mkdir(benchmark_dir)

    pi = np.pi
    phases = np.arange(pi / 16, pi, pi / 16)
    frequencies = np.arange(0.01, 0.11, 0.01)

    window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109]
    # for window_size in window_sizes:
    #     for metric_name, metric in metrics.items():
    #         sc_estimates = metric(sc_data, window_size)
    #         sinusoid_data = None
    #         sinusoid_pairs = None
    #         sinusoid_idx = 0
    #         for phase in phases:
    #             for frequency in frequencies:
    #                 sinusoid1 = tools.sinusoid(300, frequency, 0, 0.72)
    #                 sinusoid2 = tools.sinusoid(300, frequency, phase, 0.72)
    #                 if sinusoid_data is None:
    #                     sinusoid_data = np.array([sinusoid1, sinusoid2])
    #                     sinusoid_pairs = np.array([[sinusoid_idx, sinusoid_idx + 1]])
    #                 else:
    #                     sinusoid_data = np.append(sinusoid_data, [sinusoid1, sinusoid2], axis=0)
    #                     sinusoid_pairs = np.append(sinusoid_pairs, [[sinusoid_idx, sinusoid_idx + 1]], axis=0)
    #                 sinusoid_idx = sinusoid_idx + 2
    #         sinusoid_estimates = metric(sinusoid_data, window_size, pairs=sinusoid_pairs)
    #         sinusoid_significance = tools.significant_estimates(
    #             sinusoid_estimates,
    #             null=sc_estimates,
    #             alpha=0.05
    #         )
    #         phase_frequency_average_probability = np.reshape(
    #             np.mean(sinusoid_significance, axis=-1),
    #             (len(phases), len(frequencies)))
    #         sns.heatmap(
    #             phase_frequency_average_probability,
    #             cmap='seismic',
    #             vmin=-1,
    #             vmax=1,
    #             xticklabels = [str(round(frequency, 2)) for frequency in frequencies],
    #             yticklabels = ["pi / " + str(round(pi/phase, 2)) for phase in phases])
    #         if results_dirname is None:
    #             plt.show()
    #         else:
    #             figpath = os.path.join(benchmark_dir, f"phase-frequency-{metric_name}-window-{window_size}-heatmap.png")
    #             plt.savefig(figpath, dpi=1200)
    #             plt.close()
            
    #         # calculate uncertainty = -1 * sum(plog(p))
    #         non_zero_probabilities = np.abs(phase_frequency_average_probability[phase_frequency_average_probability != 0])
    #         uncertainty = np.abs(np.sum(non_zero_probabilities * np.log(non_zero_probabilities)))
    #         print_info(f"INFO: {metric_name.upper()} uncertainty w={window_size}: {uncertainty}", benchmark_dir)
    for metric_name, metric in metrics.items():
        bioplausible_significance_rate = None
        for window_size in window_sizes:
            sc_estimates = metric(sc_data, window_size)
            bioplausible_data = None
            bioplausible_pairs = None
            bioplausible_idx = 0
            for phase in phases:
                bioplausible_signal1 = bioplausible(emp_data , 0 , 0 , 300 )
                bioplausible_signal2 = bioplausible(emp_data, phase , 0 , 300)
                if bioplausible_data is None:
                    bioplausible_data = np.array([bioplausible_signal1, bioplausible_signal2])
                    bioplausible_pairs = np.array([[bioplausible_idx, bioplausible_idx + 1]])
                else:
                    bioplausible_data = np.append(bioplausible_data, [bioplausible_signal1, bioplausible_signal2], axis=0)
                    bioplausible_pairs = np.append(bioplausible_pairs, [[bioplausible_idx, bioplausible_idx + 1]], axis=0)
                bioplausible_idx = bioplausible_idx + 2
            bioplausible_estimates = metric(bioplausible_data , window_size , pairs = bioplausible_pairs)
            bioplausible_significance = tools.significant_estimates(
                bioplausible_estimates ,
                null=sc_estimates,
                alpha = 0.05
                )
            bioplausible_phase_significance_rate = np.mean(bioplausible_significance, axis=-1)
            bioplausible_significance_rate = np.array([bioplausible_phase_significance_rate]) if bioplausible_significance_rate is None else np.append(bioplausible_significance_rate , [bioplausible_phase_significance_rate], axis=0)
        
        sns.heatmap(
            bioplausible_significance_rate.T,
            cmap='seismic',
            vmin=-1,
            vmax=1,
            xticklabels = [str(window_size) for window_size in window_sizes],
            yticklabels = ["pi / " + str(round(pi/phase, 2)) for phase in phases]
        )
        if results_dirname is None:
            plt.show()
        else:
            figpath = os.path.join(benchmark_dir, f"bioplausible-{metric_name.upper()}-window-{window_size}-heatmap.png")
            plt.savefig(figpath, dpi=1200)
            plt.close()