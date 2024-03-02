import os
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import tools
from tools import print_info


def get_sinusoid_relationship_significance(frequency, phase, null, window_size, metric):
    sinusoid1 = tools.sinusoid(300, frequency, 0, 0.72)
    sinusoid2 = tools.sinusoid(300, frequency, phase, 0.72)
    sinusoid = np.array([sinusoid1, sinusoid2])
    sinusoid_estimates = metric(sinusoid, window_size)
    sinusoid_significance = tools.significant_estimates(
        sinusoid_estimates,
        null=null,
        alpha=0.05
    )
    return np.mean(sinusoid_significance)


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

    window_sizes = [9, 29, 49, 69, 89, 109]
  
    pi = np.pi
    phases = np.arange(pi / 16, pi, pi / 16)
    frequencies = np.arange(0.01, 0.11, 0.01)
    
    for window_size in window_sizes:
        for metric_name, metric in metrics.items():
            sc_estimates = metric(sc_data, window_size)
            phase_frequency_significance_rate = np.reshape(Parallel(n_jobs=2)(
                delayed(get_sinusoid_relationship_significance)(
                    frequency, phase, sc_estimates, window_size, metric
                ) for phase in phases for frequency in frequencies
            ), (len(phases), len(frequencies)))
            print(f"INFO: plotting {metric_name.upper()} w={window_size} heatmap")
            sns.heatmap(
                phase_frequency_significance_rate,
                vmin=-1,
                vmax=1,
                xticklabels = [str(round(frequency, 2)) for frequency in frequencies],
                yticklabels = ["pi / " + str(round(pi/phase, 2)) for phase in phases])
            sns.color_palette("YlOrBr", as_cmap=True)
            if results_dirname is None:
                plt.show()
            else:
                figpath = os.path.join(benchmark_dir, f"phase-frequency-{metric_name}-window-{window_size}-heatmap.png")
                plt.savefig(figpath, dpi=1200)
                plt.close()
