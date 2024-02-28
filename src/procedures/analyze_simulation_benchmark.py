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

    #{window sizes 1 to 200 with :2}
    window_sizes = list(range(1 , 50 , 2))

  
    #{Phase shifts pi/16 to pi : pi/32}
    pi = np.pi
    phase_shift = pi / 32
    phases = np.linspace(pi / 16 , pi , num = 32)
    #{plot windows 9 to 99 :10}

    #Loop over noise level
    empirical_pw_estimates = tools.swd(emp_data, 1)
    phase_window_significance_rate = np.array([])
    for phase in phases:
        window_significance_rate = np.array([])
        #Loop over window size
        for window_size in window_sizes:
            empirical_estimates = tools.sliding_average(empirical_pw_estimates, window_size)
            bioplausible_signal = tools.bioplausible(emp_data, phase, 0 , 500)
            #Compute signal estimates
            bioplausible_estimates = tools.swd(bioplausible_signal, window_size)
            #Compute significance of signal estimates(from empirical parameters)
            bioplausible_significance = np.abs(tools.significant_estimates(
                bioplausible_estimates,
                mean=np.mean(empirical_estimates),
                std=np.std(empirical_estimates)
            ))
            bioplausible_significance_rate = np.sum(
                bioplausible_significance
            ) / np.size(bioplausible_significance)
            window_significance_rate = np.append(window_significance_rate , bioplausible_significance_rate)
        phase_window_significance_rate = np.append( phase_window_significance_rate , window_significance_rate)
    sns.heatmap(phase_window_significance_rate , xticklabels = "Window Size" , yticklabels = "Phase  Shift")
    sns.color_palette("YlOrBr", as_cmap=True)
    plt.show()


simulatiom_benchmark(filepath, results_dirpath)