"""
Run a list of bulk experiments.
"""

import argparse
import subprocess
import os
import sys

def clean(s):
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return "".join([x if x in valid_chars else "_" for x in s])

def get_job_key(args):
    params = sorted([(k, v) for k, v in args.items() if k not in ["mode"] and v is not None])
    return " ".join("{}={}".format(clean(k), clean(str(v))) for k, v in params)+" "


def count_jobs(experiment, job_name):

    # count the number folders that have our experiment name and arguments.
    counter = 0
    for r, d, f in os.walk(os.path.join("runs", experiment)):
        if job_name+" " in os.path.basename(r):
            counter += 1

    return counter

def run_job(experiment, job_name, args):
    """ Runs job with given arguments. """
    python = "python3" if sys.platform in ["linux", "linux2"] else "python"
    subprocess.call([python,"train.py","train"] +
                    ["--experiment={}".format(experiment)]+
                    ["--job_name={}".format(job_name)]+
                    ["--{}={}".format(k,v) for k,v in args.items()])

def process_job(experiment, job_name, repeats, **args):
    """ Process a job. """
    for _ in range(repeats):
        if count_jobs(experiment, job_name) < repeats:
            run_job(experiment, job_name, args)

def show_job_count(experiment, job_name, repeats):
    """ Count job. """
    print("{:<40} {}/{}".format(experiment+' '+job_name, count_jobs(experiment, job_name), repeats))

parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
parser.add_argument('mode', type=str, help='count | run')
parser.add_argument('trial', type=str, help='Trial to run')
parser.add_argument('--repeats', type=int, default=3, help='Number of times to repeat each trial.')

args = parser.parse_args()

# trial 5.
jobs = []
if args.trial == "trial_5":
    for target_update in [100, 500, 1000, 2000]:
        for learning_rate in [0.00003, 0.0001]:
            jobs.append(
                ("learning_rate={} target_update={}".format(learning_rate, target_update), {
                    'target_update':target_update,
                    'learning_rate':learning_rate
                })
            )
elif args.trial == "trial_6":
    for update_every in [8.0, 4.0, 2.0, 1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]:
        jobs.append(
            ("update_every={}".format(update_every), {
                'update_every': update_every,
                'target_update':100,
                'learning_rate':3e-5
            })
        )
elif args.trial == "quick":
    for target_update in [10,100,1000,10000]:
        jobs.append(
            ("target_update={}".format(target_update), {
                'target_update': target_update,
                'config_file_path':'scenarios/basic.cfg',
                'epochs':2,
                'learning_rate':1e-4
            })
        )
elif args.trial == "trial_7":
    for num_stacks in [1,2,4]:
        jobs.append(
            ("num_stacks={}".format(num_stacks), {
                'target_update': 100,
                'num_stacks': num_stacks,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering.cfg",
                'test_episodes_per_epoch':100,
            })
        )
elif args.trial == "trial_9":
    for num_stacks in [1, 2, 4]:
        jobs.append(
            ("num_stacks={}".format(num_stacks), {
                'target_update': 100,
                'num_stacks': num_stacks,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':200,
                'test_episodes_per_epoch':25,   #faster to train, can always run more later...
            })
    )
elif args.trial == "test_envs":
    for env in [
        "scenarios/basic.cfg",
        "scenarios/simpler_basic.cfg",
        "scenarios/deadly_corridor.cfg",
        "scenarios/deathmatch.cfg",
        "scenarios/defend_the_center.cfg",
        "scenarios/defend_the_line.cfg",
        "scenarios/health_gathering.cfg",
        "scenarios/health_gathering_supreme.cfg",
        "scenarios/predict_position.cfg",
        "scenarios/rocket_basic.cfg",
        "scenarios/take_cover.cfg",
    ]:
        jobs.append(
            ("env={}".format(env), {
                'config_file_path': env,
                'target_update': 100,               # maybe this has to be tuned?
                'learning_rate': 3e-5,
                'epochs': 100,                      # some of these may require more than 100k training steps.
                'test_episodes_per_epoch': 10,      # faster for training, and can re-evaluate later.
             })
        )

else:
    print("Invalid trial name '{}'".format(args.trial))
    exit(-1)

for job_name, job_args in jobs:
    if args.mode == "count":
        show_job_count(
            experiment=args.trial,
            job_name=job_name,
            repeats=args.repeats
        )
    elif args.mode == "run":
        process_job(
            experiment=args.trial,
            job_name=job_name,
            repeats=args.repeats,
            **job_args
        )
    else:
        print("Invalid mode {}".format(args.mode))