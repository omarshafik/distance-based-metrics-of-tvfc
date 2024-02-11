""" run all procedures
"""
import argparse
import os
from datetime import datetime
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import procedures
from tools import print_info, prep_emp_data, sc, pr, laumann

TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
RANDOM_SEED = int(datetime.now().timestamp())

parser = argparse.ArgumentParser(
    description='Python project of the study procedures in ' + \
        '\'Distance-Based Metrics of Time-Varying Functional Connectivity\'')
parser.add_argument('-i', '--indir', required=True, type=str,
    help='input fmri timeseries directory of files to process')
parser.add_argument('-o', '--outdir', type=str,
    help='output directory location for saving analysis results. \
        The default output directory for runnning the study is the same directory of the input file. \
        The default output directory for processing files is the $(CWD)/swd-results')
parser.add_argument('-s', '--random-seed', type=int, default=RANDOM_SEED,
    help='random seed to use')
args = parser.parse_args()

outdir = args.outdir
if not os.path.isdir(args.indir):
    raise ValueError(f"{args.indir} is not a directory")
if args.outdir is not None and not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)


# set the randomization seed; this will make all steps of the study replicable in any run
random = RandomState(MT19937(SeedSequence(args.random_seed)))
# Find all files in the directory (we assume all files are txt of nodes timeseries)
input_files = []
input_files.extend([
    os.path.join(args.indir, filename)
        for filename in os.listdir(args.indir)
        if os.path.isfile(os.path.join(args.indir, filename))])

file_to_process = input_files[random.choice(len(input_files), size=1)[0]]
data = prep_emp_data(np.loadtxt(file_to_process).T, smooth=0)
data_smoothed = prep_emp_data(np.loadtxt(file_to_process).T, smooth=10)
sc_data = sc(data, random=random)
pr_data = pr(data, random=random)
laumann_data = laumann(data, random=random)
sc_data_smoothed = sc(data_smoothed, random=random)
pr_data_smoothed = pr(data_smoothed, random=random)
laumann_data_smoothed = laumann(data_smoothed, random=random)

# create results (output) directory
if outdir is None:
    outdir = os.path.join(args.indir, "swd-results")
if not os.path.isdir(outdir):
    os.mkdir(outdir)
results_dir = os.path.join(outdir, TIMESTAMP)
pr_surrogates_results_dir = os.path.join(results_dir, "pr")
pr_smooth_surrogates_results_dir = os.path.join(results_dir, "pr-smoothed")
laumann_surrogates_results_dir = os.path.join(results_dir, "laumann")
laumann_smooth_surrogates_results_dir = os.path.join(results_dir, "laumann-smoothed")
os.mkdir(results_dir)
os.mkdir(pr_surrogates_results_dir)
os.mkdir(pr_smooth_surrogates_results_dir)
os.mkdir(laumann_surrogates_results_dir)
os.mkdir(laumann_smooth_surrogates_results_dir)

print_info(f"INFO: randomization seed: {args.random_seed}", results_dir)
print_info(f"INFO: Selected file {os.path.basename(file_to_process)}", results_dir)
procedures.analyze_within_subject_ensemble_statistics(file_to_process, results_dir, random=random)
procedures.analyze_within_subject_ensemble_statistics(file_to_process, results_dir, metric_name="swc", random=random)
# surrogate analysis using PR and unsmoothed empirical data
procedures.analyze_surrogate_statistics(
    data,
    pr_surrogates_results_dir,
    sc_data=sc_data,
    scc_data=pr_data,
    metric_name="mtd",
    random=random)
procedures.analyze_surrogate_statistics(
    data,
    pr_surrogates_results_dir,
    sc_data=sc_data,
    scc_data=pr_data,
    metric_name="swc",
    random=random)
procedures.analyze_surrogate_statistics(
    data,
    pr_surrogates_results_dir,
    sc_data=sc_data,
    scc_data=pr_data,
    random=random)
# surrogate analysis using PR and smoothed empirical data
procedures.analyze_surrogate_statistics(
    data_smoothed,
    pr_smooth_surrogates_results_dir,
    sc_data=sc_data_smoothed,
    scc_data=pr_data_smoothed,
    metric_name="mtd",
    random=random)
procedures.analyze_surrogate_statistics(
    data_smoothed,
    pr_smooth_surrogates_results_dir,
    sc_data=sc_data_smoothed,
    scc_data=pr_data_smoothed,
    metric_name="swc",
    random=random)
procedures.analyze_surrogate_statistics(
    data_smoothed,
    pr_smooth_surrogates_results_dir,
    sc_data=sc_data_smoothed,
    scc_data=pr_data_smoothed,
    random=random)
# surrogate analysis using Laumann and unsmoothed empirical data
procedures.analyze_surrogate_statistics(
    data,
    laumann_surrogates_results_dir,
    sc_data=sc_data,
    scc_data=laumann_data,
    metric_name="mtd",
    random=random)
procedures.analyze_surrogate_statistics(
    data,
    laumann_surrogates_results_dir,
    sc_data=sc_data,
    scc_data=laumann_data,
    metric_name="swc",
    random=random)
procedures.analyze_surrogate_statistics(
    data,
    laumann_surrogates_results_dir,
    sc_data=sc_data,
    scc_data=laumann_data,
    random=random)
# surrogate analysis using Laumann and smoothed empirical data
procedures.analyze_surrogate_statistics(
    data_smoothed,
    laumann_smooth_surrogates_results_dir,
    sc_data=sc_data_smoothed,
    scc_data=laumann_data_smoothed,
    metric_name="mtd",
    random=random)
procedures.analyze_surrogate_statistics(
    data_smoothed,
    laumann_smooth_surrogates_results_dir,
    sc_data=sc_data_smoothed,
    scc_data=laumann_data_smoothed,
    metric_name="swc",
    random=random)
procedures.analyze_surrogate_statistics(
    data_smoothed,
    laumann_smooth_surrogates_results_dir,
    scc_data=laumann_data_smoothed,
    sc_data=sc_data,
    random=random)
procedures.analyze_sample_statistics(file_to_process, results_dir, random=random)
procedures.analyze_between_subjects_ensemble_statistics(input_files, results_dir, random=random)
procedures.analyze_metrics_correlation(input_files, results_dir, random=random)
# procedures.evaluate_tvfc_metrics(input_files, results_dir, random=random)
