"""
imports module
"""
from .within_subject_ensemble_analysis import (
    analyze_within_subject_ensemble_statistics,
    analyze_within_subject_swd_swc_correlation
)
from .surrogate_analysis import (
    metrics_surrogates_evaluation
)
from .within_subject_sample_analysis import analyze_sample_statistics
from .between_subjects_ensemble_analysis import analyze_between_subjects_ensemble_statistics
from .correlation_analysis import analyze_metrics_correlation
from .illustrations import generate_illustrations
from .analyze_simulation_benchmark import sinusoid_simulatiom_benchmark, sc_simulatiom_benchmark
