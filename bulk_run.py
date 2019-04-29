"""
Run a list of bulk experiments.
"""

import argparse
import subprocess
import os

def get_job_key(args):
    return(str(args))

def count_jobs(args):

    # get the job key
    params = sorted([(k, v) for k, v in args.items() if k not in ["mode"] and v is not None])
    key = " ".join("{}={}".format(k,v) for k, v in params)

    # count the number folders that have our experiment name and arguments.
    counter = 0
    for r, d, f in os.walk("runs"):
        if key in os.path.basename(r):
            counter += 1

    return counter

def run_job(args):
    """ Runs job with given arguments. """
    subprocess.call(["python","train.py","train"] + ["--{}={}".format(k,v) for k,v in args.items()])

def process_job(repeats, **args):
    """ Process a job. """
    for _ in range(repeats):
        if count_jobs(args) < repeats:
            run_job(args)

parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
parser.add_argument('--num_repeats', type=int, default=3, help='Number of times to repeat each trial.')
args = parser.parse_args()

# trial 4.
for target_update in [100, 500, 1000, 2000]:
    for learning_rate in [0.000003, 0.00001, 0.00003]:
        process_job(
            repeats=args.num_repeats,
            experiment="trial_5",
            target_update=target_update,
            learning_rate=learning_rate
        )
