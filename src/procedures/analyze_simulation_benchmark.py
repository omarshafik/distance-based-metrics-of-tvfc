import os
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import tools
from tools import print_info

filepath = "C:\\Users\\hussi\\Func-Sim-Research\\HCP\\3T_HCP1200_MSMAll_d100_ts2\\129634.txt"
results_dirpath = "C:\\Users\\hussi\\distance-based-metrics-of-tvfc-main\\new\\distance-based-metrics-of-tvfc\\results"

#Constructing parameter space --> parameter objects

def simulatiom_benchmark(
    filename: str,
    results_dirname: str):

    print_info("##########################################################################", results_dirname)
    print_info("INFO: Carrying out simulation benchmarks", results_dirname)
    emp_data = tools.prep_emp_data(np.loadtxt(filename).T)
    pairs = np.array(list(combinations(range(emp_data.shape[0]), 2)))

    benchmark_dir = os.path.join(results_dirname, "simulation-benchmark")
    if not os.path.exists(benchmark_dir):
        os.mkdir(benchmark_dir)

    #{plot windows 9 to 99 :10}
    window_sizes = [9, 29, 49, 69]
  
    #{Phase shifts pi/16 to pi : pi/32}
    pi = np.pi
    phases = np.linspace(pi / 16 , pi , num = 32)
    frequencies = np.linspace(0.01, 0.1, num = 10)

    #Loop over noise level
    
    for window_size in window_sizes:
        phase_frequency_significance_rate = []
        empirical_estimates = tools.swd(emp_data, window_size)
        for phase in phases:
            frequency_significance_rate = []
            #Loop over window size
            for frequency in frequencies:
                sinusoid1 = tools.sinusoid(300, frequency, 0, 0.72)
                sinusoid2 = tools.sinusoid(300, frequency, phase, 0.72)
                sinusoid = np.array([sinusoid1, sinusoid2])
            
                #Compute signal estimates
                sinusoid_estimates = tools.swd(sinusoid, window_size)
                #Compute significance of signal estimates(from empirical parameters)
                sinusoid_significance = tools.significant_estimates(
                    sinusoid_estimates,
                    null=empirical_estimates,
                    alpha=0.1
                )
                sinusoid_significance_rate = np.mean(
                    sinusoid_significance
                )
                frequency_significance_rate.append(sinusoid_significance_rate)
            phase_frequency_significance_rate.append(frequency_significance_rate)

        sns.heatmap(phase_frequency_significance_rate , xticklabels = frequencies , yticklabels = phases / pi)
        sns.color_palette("YlOrBr", as_cmap=True)
        plt.show()


simulatiom_benchmark(filepath, results_dirpath)