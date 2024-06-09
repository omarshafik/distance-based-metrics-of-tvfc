import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import tools
from tools import print_info, spectrally_constrained_pair


def sinusoid_simulatiom_benchmark(
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
    print_info("Carrying out sinusoid simulation benchmarks", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    sc_data = tools.sc(emp_data, random=random)

    benchmark_dir = os.path.join(results_dirname, "simulation-benchmark")
    if not os.path.exists(benchmark_dir):
        os.mkdir(benchmark_dir)

    pi = np.pi
    phases = np.arange(pi / 16, pi, pi / 16)
    frequencies = np.arange(0.01, 0.11, 0.01)
    window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]

    sinusoid_results = {
        'metric': [],
        'window': [],
        'uncertainty': [],
        'sensitivity': []
    }

    for window_size in window_sizes:
        for metric_name, metric in metrics.items():
            sc_estimates = metric(sc_data, window_size)
            sinusoid_data = None
            sinusoid_pairs = None
            sinusoid_idx = 0
            for phase in phases:
                for frequency in frequencies:
                    sinusoid1 = tools.sinusoid(300, frequency, 0, 0.72)
                    sinusoid2 = tools.sinusoid(300, frequency, phase, 0.72)
                    if sinusoid_data is None:
                        sinusoid_data = np.array([sinusoid1, sinusoid2])
                        sinusoid_pairs = np.array([[sinusoid_idx, sinusoid_idx + 1]])
                    else:
                        sinusoid_data = np.append(sinusoid_data, [sinusoid1, sinusoid2], axis=0)
                        sinusoid_pairs = np.append(sinusoid_pairs, [[sinusoid_idx, sinusoid_idx + 1]], axis=0)
                    sinusoid_idx = sinusoid_idx + 2
            sinusoid_estimates = metric(sinusoid_data, window_size, pairs=sinusoid_pairs)
            sinusoid_significance = tools.significant_estimates(
                sinusoid_estimates,
                null=sc_estimates,
                alpha=0.05
            )
            phase_frequency_average_probability = np.reshape(
                np.mean(sinusoid_significance, axis=-1),
                (len(phases), len(frequencies)))
            sns.heatmap(
                phase_frequency_average_probability,
                cmap='seismic',
                vmin=-1,
                vmax=1,
                xticklabels = [str(round(frequency, 2)) for frequency in frequencies],
                yticklabels = ["pi / " + str(round(pi/phase, 2)) for phase in phases])
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(benchmark_dir, f"phase-frequency-{metric_name}-window-{window_size}-heatmap.png")
                plt.savefig(figpath, dpi=1200)
                plt.close()

            # calculate uncertainty = -1 * sum(plog(p))
            non_zero_probabilities = np.abs(phase_frequency_average_probability[phase_frequency_average_probability != 0])
            uncertainty = np.abs(np.sum(non_zero_probabilities * np.log(non_zero_probabilities)))
            # calculate sensitivity
            sensitivity = np.mean(np.abs(phase_frequency_average_probability))
            print_info(f"{metric_name.upper()}, w={window_size} uncertainty={uncertainty} sensitivity={sensitivity}", benchmark_dir)
            sinusoid_results["metric"].append(metric_name)
            sinusoid_results["window"].append(window_size)
            sinusoid_results["uncertainty"].append(uncertainty)
            sinusoid_results["sensitivity"].append(sensitivity)

    sinusoid_results_df = pd.DataFrame(sinusoid_results)
    sinusoid_results_filepath = os.path.join(benchmark_dir, "sinusoid.csv")
    sinusoid_results_df.to_csv(sinusoid_results_filepath)

    return sinusoid_results_filepath

def sc_simulatiom_benchmark(
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
    print_info("Carrying out spectrally-constrained simulation benchmarks", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    sc_data = tools.sc(emp_data, random=random)

    benchmark_dir = os.path.join(results_dirname, "simulation-benchmark")
    if not os.path.exists(benchmark_dir):
        os.mkdir(benchmark_dir)

    pi = np.pi
    phases = np.arange(pi / 16, pi, pi / 16)
    window_sizes = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]

    spectrally_const_results = {
        'metric': [],
        'window_size': [],
        'uncertainty': [],
        'sensitivity': []
    }
    
    # get node/parcel pairs
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

        # get all frequency spectra from empirical data
    freq_spectra = np.abs(np.fft.rfft(emp_data, axis=-1))

    # compute the euclidean distance betweem all pairs of spectra
    freq_spectra_distances = tools.swd(
        freq_spectra,
        window_size=freq_spectra.shape[-1],
        use_derivative=False,
        transform=None,
        kaiser_beta=0,
        pairs=pairs)

    freq_spectra_dtype = [('idx', int), ('value', float)]
    freq_spectra_distances = np.array(list(enumerate(freq_spectra_distances)), dtype=freq_spectra_dtype)
    # sort the spectra pairs by distance
    freq_spectra_distances.sort(order='value')
    sample_size = 10
    step_size = freq_spectra_distances.shape[0] // sample_size
    selected_freq_spectra_distances = freq_spectra_distances[::step_size]

    for metric_name, metric in metrics.items():
        for window_size in window_sizes:
            sc_estimates = metric(sc_data, window_size)
            spectra_distances = []
            spectrally_const_data = None
            spectrally_const_pairs = None
            spectrally_const_idx = 0
            for pair_idx, spectra_distance in selected_freq_spectra_distances:
                pair = pairs[pair_idx]
                emp_signal1 = emp_data[pair[0]]
                emp_signal2 = emp_data[pair[1]]
                spectra_distances.append(spectra_distance)
                for phase in phases:
                    spectrally_const_signal1, spectrally_const_signal2 = spectrally_constrained_pair([emp_signal1, emp_signal2], phase, 0, 300)
                    if spectrally_const_data is None:
                        spectrally_const_data = np.array([spectrally_const_signal1, spectrally_const_signal2])
                        spectrally_const_pairs = np.array([[spectrally_const_idx, spectrally_const_idx + 1]])
                    else:
                        spectrally_const_data = np.append(spectrally_const_data, [spectrally_const_signal1, spectrally_const_signal2], axis=0)
                        spectrally_const_pairs = np.append(spectrally_const_pairs, [[spectrally_const_idx, spectrally_const_idx + 1]], axis=0)
                    spectrally_const_idx = spectrally_const_idx + 2
            spectrally_const_estimates = metric(spectrally_const_data , window_size , pairs = spectrally_const_pairs)
            spectrally_const_significance = tools.significant_estimates(
                spectrally_const_estimates ,
                null=sc_estimates,
                alpha = 0.05
                )
            average_spectrally_const_significance_per_winsize = np.reshape(
                np.mean(spectrally_const_significance, axis=-1),
                (sample_size, len(phases)))

            sns.heatmap(
                average_spectrally_const_significance_per_winsize.T,
                cmap='seismic',
                vmin=-1,
                vmax=1,
                xticklabels = [str(round(spectra_distance, 3)) for spectra_distance in spectra_distances],
                yticklabels = [f"pi / {round(pi/phase, 2)}" for phase in phases]
            )
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(benchmark_dir, f"spectrally-const-{metric_name.upper()}-wsize-{window_size}-heatmap.png")
                plt.savefig(figpath, dpi=1200)
                plt.close()

            # calculate uncertainty = -1 * sum(plog(p))
            non_zero_probabilities = np.abs(average_spectrally_const_significance_per_winsize[average_spectrally_const_significance_per_winsize != 0])
            uncertainty = np.abs(np.sum(non_zero_probabilities * np.log(non_zero_probabilities)))

            # calculate sensitivity, i.e., the mean probability of detecting significant estimates
            sensitivity = np.mean(np.abs(average_spectrally_const_significance_per_winsize))
            print_info(f"{metric_name.upper()} w={window_size} spectrally const uncertainty={uncertainty} sensitivity={sensitivity}", benchmark_dir)
            spectrally_const_results["metric"].append(metric_name)
            spectrally_const_results["window_size"].append(window_size)
            spectrally_const_results["uncertainty"].append(uncertainty)
            spectrally_const_results["sensitivity"].append(sensitivity)

    spectrally_const_results_df = pd.DataFrame(spectrally_const_results)
    sc_results_filepath = os.path.join(benchmark_dir, "spectrally-constrained.csv")
    spectrally_const_results_df.to_csv(sc_results_filepath)

    return sc_results_filepath
