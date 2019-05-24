"""
Run a list of bulk experiments.
"""


"""
Recover system

1/ identify jobs that have not completed (i.e. no complete results)  
2/ check file's last update to see if it's being worked on
3/ if it hasn't been touched in 60 minutes try to resume it using train.py resume

Extension system

run train.py resume, but with --epochs=xxx, this will delete the completed files and modify the config to the new epochs

Problem:
Resume jobs should really use the train.py file from the folder, but this will not have the resume option built in?
In this case I'll need to copy by hand the new python file in, that's ok for these results as they won't be published.

Maybe build an actual job system? With text files being piked up? So I can have computers on watch mode?


Models system
Create different models and test them

something simple and low res
something high res
maybe more filters?

Maybe some system to check the dims of the activations, i.e. to see how many filters are required at each layer?
This could be done with PCA and looking at how many dims required for 95th percentile?

This would mean doing PCA on a batch... which may be enough? I mean it can only have 32 dims right? hmm? maybe
put the largest batch I can through, say 512 and see what happens. Or run many small batches and check dims.

Delete orphined jobs:
bulk_run purge exp_3 --timeout=60 

"""

import argparse
import subprocess
import os
import sys
import numpy as np


def clean(s):
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return "".join([x if x in valid_chars else "_" for x in s])


def get_job_key(args):
    params = sorted([(k, v) for k, v in args.items() if k not in ["mode"] and v is not None])
    return " ".join("{}={}".format(clean(k), clean(str(v))) for k, v in params)


def count_jobs(experiment, job_name):

    # count the number folders that have our experiment name and arguments.
    counter = 0
    for r, d, f in os.walk(os.path.join("runs", experiment)):
        if job_name+" [" in os.path.basename(r) and "gcs" not in r:
            counter += 1

    return counter


def get_python():
    return "python3" if sys.platform in ["linux", "linux2"] else "python"


def get_train_script():
    if args.train_script is not None:
        return args.train_script
    else:
        # use the trial train.py if it exists, otherwise use default
        trial_train = os.path.join("runs",args.trial,"train.py")
        if os.path.exists(trial_train):
            return trial_train
        else:
            return "train.py"


def run_job(experiment, job_name, kwargs):
    """ Runs job with given arguments. """
    print("Using training script {}".format(get_train_script()))
    subprocess.call([get_python(),get_train_script(),"train"] +
                    ["--experiment={}".format(experiment)]+
                    ["--job_name={}".format(job_name)]+
                    ["--{}={}".format(k,v) for k,v in kwargs.items()])


def process_job(experiment, job_name, repeats, **kwargs):
    """ Process a job. """
    for _ in range(repeats):
        if count_jobs(experiment, job_name) < repeats:
            run_job(experiment, job_name, kwargs)


def process_eval(experiment, eval_results_suffix, **kwargs):
    """ Process a job. """

    jobs_to_evaluate = []

    for r, d, f in os.walk(os.path.join("runs", experiment)):
        folder = os.path.basename(r)
        if len(folder.split()) < 2:
            continue

        job_name = folder.split()[0]
        job_id = folder.split()[1][1:-1]
        for file_name in f:
            if file_name == "results_complete.dat":

                # check if the results file exists
                results_file_path = os.path.join(r, "results{}.dat".format(eval_results_suffix))
                if os.path.exists(results_file_path):
                    continue

                jobs_to_evaluate.append((
                    job_name, job_id
                ))

    for job_name, job_id in jobs_to_evaluate:
        subprocess.call([get_python(), get_train_script(), "eval"] +
                        ["--experiment={}".format(experiment)] +
                        ["--job_name={}".format(job_name)] +
                        ["--job_id={}".format(job_id)] +
                        ["--eval_results_suffix={}".format(eval_results_suffix)] +
                        ["--{}={}".format(k, v) for k, v in kwargs.items()])



def show_job_count(experiment, job_name, repeats):
    """ Count job. """
    print("{:<40} {}/{}".format(experiment+' '+job_name, count_jobs(experiment, job_name), repeats))


parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
parser.add_argument('mode', type=str, help='count | run | search | eval')
parser.add_argument('trial', type=str, help='Trial to run')
parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each trial.')
parser.add_argument('--train_script', type=str, default=None, help='Script to use to train.')

args = parser.parse_args()

jobs = []

if args.trial == "frame_repeat":
    for frame_repeat in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 35*2, 35*3, 35*4, "g_10_0.5", "g_10_1", "g_10_2", "g_10_4 "]:
        jobs.append(
            ("frame_repeat={}".format(frame_repeat), {
            'frame_repeat':             frame_repeat,
            'target_update':            100,
            'learning_steps_per_epoch': 5000,
            'update_every':             4,
            'replay_memory_size':       10000,
            'batch_size':               32,
            'num_stacks':               4,
            'learning_rate':            4e-4,
            'health_as_reward':         True,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'epochs':                   200,
            'test_episodes_per_epoch':  25,
        }))
elif args.trial == "frame_repeat_eval":
    for frame_repeat in reversed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 35*2, 35*3, 35*4, "g_10_0.5", "g_10_1", "g_10_2", "g_10_4"]):
        jobs.append(
            ("_frame_repeat={}".format(frame_repeat), {
            'test_frame_repeat':        frame_repeat,
            'test_episodes_per_epoch':  100,
        }))
    args.trial = "frame_repeat"
elif args.trial == "end_epsilon":
    # look into epsilon decay
    for end_eps in [0.2,0.1,0.05,0.025,0]:
        jobs.append(
            ("end_eps={}".format(end_eps), {
            'end_eps': end_eps,
            'target_update':            100,
            'learning_steps_per_epoch': 5000,
            'update_every':             4,
            'replay_memory_size':       30000,
            'batch_size':               32,
            'num_stacks':               2,
            'learning_rate':            3e-4,
            'health_as_reward':         True,
            'frame_repeat':             10,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'epochs':                   200,
            'max_pool':                 False,
            'test_episodes_per_epoch':  25,
        }))
elif args.trial == "end_epsilon_step":
    # look into epsilon stage
    # todo: use best final eps? or search a little around it...
    for end_eps_step in [x*1000 for x in [12.5, 25, 50, 100, 200, 400, 600, 800, 1000]]:
        jobs.append(
            ("end_eps_step={}".format(end_eps_step), {
            'end_eps':                  0.05,
            'end_eps_step':             end_eps_step,
            'target_update':            100,
            'learning_steps_per_epoch': 5000,
            'update_every':             4,
            'replay_memory_size':       30000,
            'batch_size':               32,
            'num_stacks':               2,
            'learning_rate':            3e-4,
            'health_as_reward':         True,
            'frame_repeat':             10,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'epochs':                   200,
            'max_pool':                 False,
            'test_episodes_per_epoch':  25,
        }))
elif args.trial == "use_color":
    for use_color in [True, False]:
        jobs.append(
            ("use_color={}".format(use_color), {
            'use_color':                use_color,
            'end_eps':                  0.10,
            'target_update':            100,
            'learning_steps_per_epoch': 5000,
            'update_every':             4,
            'replay_memory_size':       10000,
            'batch_size':               32,
            'num_stacks':               4,
            'learning_rate':            3e-4,
            'health_as_reward':         True,
            'frame_repeat':             10,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'epochs':                   200,
            'max_pool':                 True,
            'test_episodes_per_epoch':  25,
        }))
elif args.trial == "search_1":
    for i in range(args.repeats):
        # pick random parameters
        jobs.append(
            ("sample", {
            'num_stacks':               np.random.choice([1, 2, 4, 8]),
            'discount_factor':          np.random.choice([1, 0.99, 0.98]),
            'replay_memory_size':       np.random.choice([3000,10000,30000]),
            'target_update':            np.random.choice([-1, 50, 100, 200, 400, 800]),
            'hidden_units':             np.random.choice([32, 64, 128, 256, 512, 1024, 2048]),
            'learning_rate':            np.random.choice([0.1e-4, 0.3e-4, 1e-4, 3e-4, 10e-4, 30e-4]),
            'max_pool':                 np.random.choice([True, False]),
            'use_color':                np.random.choice([True, False]),
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'frame_repeat':             10,
            'learning_steps_per_epoch': 5000,
            'test_episodes_per_epoch':  25,
            'update_every':             4,
            'epochs':                   200,
            'batch_size':               32,
            'health_as_reward':         True,
            'terminate_early':          True
            }))
    if args.mode == "run":
        args.mode = "search"
else:
    print("Invalid trial name '{}'".format(args.trial))
    exit(-1)


if args.mode == "count":
    for job_name, job_args in jobs:
        show_job_count(
            experiment=args.trial,
            job_name=job_name,
            repeats=args.repeats
        )
elif args.mode == "run":
    for runs in range(1,args.repeats+1):
        for job_name, job_args in jobs:
            process_job(
                experiment=args.trial,
                job_name=job_name,
                repeats=runs,
                **job_args
            )
elif args.mode == "eval":
    for results_suffix, job_args in jobs:
        process_eval(
            experiment=args.trial,
            eval_results_suffix=results_suffix,
            **job_args
        )
elif args.mode == "search":
    for job_name, job_args in jobs:
        run_job(
            experiment=args.trial,
            job_name=job_name,
            kwargs=job_args
        )

else:
    print("Invalid mode {}".format(args.mode))
