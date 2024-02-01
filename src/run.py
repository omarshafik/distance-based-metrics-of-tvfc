""" run all procedures
"""
import argparse
import os
from datetime import datetime
import numpy as np
import procedures
from tools import print_info

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
np.random.seed(args.random_seed)

# Find all files in the directory (we assume all files are txt of nodes timeseries)
input_files = []
input_files.extend([
    os.path.join(args.indir, filename)
        for filename in os.listdir(args.indir)
        if os.path.isfile(os.path.join(args.indir, filename))])

# create results (output) directory
if outdir is None:
    outdir = os.path.join(args.indir, "swd-results")
if not os.path.isdir(outdir):
    os.mkdir(outdir)
results_dir = os.path.join(outdir, TIMESTAMP)
os.mkdir(results_dir)

file_to_process = input_files[np.random.choice(len(input_files), size=1)[0]]
print_info(f"INFO: randomization seed: {args.random_seed}", results_dir)
print_info(f"INFO: Selected file {os.path.basename(file_to_process)}", results_dir)
procedures.analyze_within_subject_ensemble_statistics(file_to_process, results_dir)
procedures.analyze_surrogate_statistics(file_to_process, results_dir, metric="mtd")
procedures.analyze_surrogate_statistics(file_to_process, results_dir, metric="swc")
procedures.analyze_surrogate_statistics(file_to_process, results_dir)
procedures.analyze_sample_statistics(file_to_process, results_dir)
procedures.analyze_between_subjects_ensemble_statistics(input_files, results_dir)
procedures.analyze_within_subject_swd_swc_correlation(file_to_process, results_dir)
procedures.evaluate_tvfc_metrics(input_files, results_dir)
