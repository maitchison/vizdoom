"""
Run a list of bulk experiments.
"""

import argparse
import subprocess
import os
import sys

def get_job_key(args):
    params = sorted([(k, v) for k, v in args.items() if k not in ["mode"] and v is not None])
    return " ".join("{}={}".format(k, v) for k, v in params)+" "


def count_jobs(args):

    # get the job key
    key = get_job_key(args)

    # count the number folders that have our experiment name and arguments.
    counter = 0
    for r, d, f in os.walk("runs"):
        if key in os.path.basename(r):
            counter += 1

    return counter

def run_job(args):
    """ Runs job with given arguments. """
    python = "python3" if sys.platform in ["linux", "linux2"] else "python"
    subprocess.call([python,"train.py","train"] + ["--{}={}".format(k,v) for k,v in args.items()])

def process_job(repeats, **args):
    """ Process a job. """
    for _ in range(repeats):
        if count_jobs(args) < repeats:
            run_job(args)

def count_job(repeats, **args):
    """ Process a job. """
    print("{:<40} {}/{}".format(get_job_key(args), count_jobs(args), repeats))

parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
parser.add_argument('mode', type=str, help='count | run')
parser.add_argument('trial', type=str, help='Trial to run')
parser.add_argument('--num_repeats', type=int, default=3, help='Number of times to repeat each trial.')

args = parser.parse_args()

# trial 5.

if args.trial == "trial_5":

    for target_update in [100, 500, 1000, 2000]:
        for learning_rate in [0.00003, 0.0001]:
            if args.mode == "run":
                process_job(
                    repeats=args.num_repeats,
                    experiment="trial_5",
                    target_update=target_update,
                    learning_rate=learning_rate
                )
            else:
                count_job(
                    repeats=args.num_repeats,
                    experiment="trial_5",
                    target_update=target_update,
                    learning_rate=learning_rate
                )

elif args.trial == "trial_6":

    for update_every in [8,4,2,1,1/2,1/4,1/8,1/16,1/32,1/64]:
        if args.mode == "run":
            process_job(
                repeats=args.num_repeats,
                experiment="trial_6",
                update_every=update_every,
                target_update=100,
                learning_rate=3e-5
            )
        else:
            count_job(
                repeats=args.num_repeats,
                experiment="trial_6",
                update_every=update_every,
                target_update=100,
                learning_rate=3e-5
            )

else:
    print("Invalid trial name {}".format(args.trial))

