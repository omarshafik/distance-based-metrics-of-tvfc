"""
imports module
"""
from .analyze import *
from .common import (
    normalized,
    derivative,
    sliding_average,
    pad_timeseries,
    prep_emp_data,
    print_info,
    PRINT)
from .plot import (
    plot_distribution,
    plot_overlapping_distributions,
    plot_timeseries_and_estimates,
    plot_global_timeseries,
    plot_autocorrelation,
    plot_qq,
    plot_grid,
    scatter_fc_edges,
    plot_timeseries,
    plot_correlation_matrices)
from .test import (
    test_stationary,
    test_distribution,
    significance,
    get_edges_of_interest,
    significant_estimates,
    significant_time_points,
    likelihood,
    posterior,
    kl_divergence,
    scaled_significance_rate,
    sdv,
    sdr,
    edr)
from .simulate import (
    sc,
    sc_resample,
    pr,
    laumann,
    spectrally_constrained_pair,
    sinusoid
)
