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
import random
import configparser
import platform
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
    for r, d, f in os.walk(os.path.join(args.output_path, experiment)):
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
        trial_train = os.path.join(args.output_path, args.trial,"train.py")
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
                    (['--output_path={}'.format(args.output_path)] if args.output_path is not None else []) +
                    ["--{}={}".format(k,v) for k,v in kwargs.items()])


def process_job(experiment, job_name, repeats, **kwargs):
    """ Process a job. """
    for _ in range(repeats):
        if count_jobs(experiment, job_name) < repeats:
            run_job(experiment, job_name, kwargs)


def process_eval(experiment, eval_results_suffix, **kwargs):
    """ Process a job. """

    jobs_to_evaluate = []

    for r, d, f in os.walk(os.path.join(args.output_path, experiment)):

        folder = os.path.basename(r)
        if len(folder.split()) < 2:
            continue

        job_name = folder.split()[0].strip()
        job_id = folder.split()[1][1:-1]
        for file_name in f:
            if file_name == "results_complete.dat":

                # check if the results file exists
                results_file_path = os.path.join(r, "results{}.dat".format(eval_results_suffix))
                if os.path.exists(results_file_path):
                    continue

                jobs_to_evaluate.append((job_id, job_name, results_file_path ))

    random.shuffle(jobs_to_evaluate)

    for (job_id, job_name, results_file_path ) in jobs_to_evaluate:
        # check (again) if the results file exists... just in case someone else if working on this...
        if os.path.exists(results_file_path):
            continue

        subprocess.call([get_python(), get_train_script(), "eval"] +
                        ["--experiment={}".format(experiment)] +
                        ["--job_name={}".format(job_name)] +
                        ["--job_id={}".format(job_id)] +
                        ["--eval_results_suffix={}".format(eval_results_suffix)] +
                        (['--output_path={}'.format(args.output_path)] if args.output_path is not None else []) +
                        ["--{}={}".format(k, v) for k, v in kwargs.items()])


def safe_cast(x):
    try:
        return int(str(x))
    except:
        try:
            return float(str(x))
        except:
            return x


def get_default_argument(argument):

    default = None

    # first some default values
    if argument == "optimizer":
        default = "rmsprop"
    elif argument == "output_path":
        default = "runs"
    elif argument == "include_xy":
        default = False

    hostname = platform.node()

    # check ini file override
    if hostname in ini_file:
        if argument in ini_file[hostname]:
            default = safe_cast(ini_file[hostname][argument])

    return default


def show_job_count(experiment, job_name, repeats):
    """ Count job. """
    print("{:<40} {}/{}".format(experiment+' '+job_name, count_jobs(experiment, job_name), repeats))


ini_file = configparser.ConfigParser()
ini_file.read("config.ini")

parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
parser.add_argument('mode', type=str, help='count | run | search | eval')
parser.add_argument('trial', type=str, help='Trial to run')
parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each trial.')
parser.add_argument('--train_script', type=str, default=None, help='Script to use to train.')
parser.add_argument('--threads', type=int, help='CPU threads for workers.')
parser.add_argument('--output_path', type=str, default=get_default_argument("output_path"), help='Path to output experiment results to.')

args = parser.parse_args()

jobs = []

frame_repeat_list = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 35*2, 35*3, 35*4,
    "g_7_0", "g_7_0.5", "g_7_1", "g_7_1.5", "g_7_2", "g_7_3", "g_7_4", "g_7_8",
    "g_10_0", "g_10_0.25", "g_10_0.5", "g_10_0.75", "g_10_1", "g_10_2", "g_10_4", "g_10_8",
    "p_5", "p_7", "p_10", "p_20", "p_40"
]

frame_repeat_list_2 = [7, "f_7_0", "f_7_1", "f_7_2", "f_7_4"]

if args.trial == "frame_delay":
    for frame_repeat in frame_repeat_list_2:
        jobs.append(
            ("frame_repeat={}".format(frame_repeat), {
            'frame_repeat':             frame_repeat,
            'target_update':            100,
            'learning_steps_per_epoch': 5000,
            'update_every':             4,
            'replay_memory_size':       10000,
            'batch_size':               32,
            'num_stacks':               4,
            'learning_rate':            1e-4,
            'health_as_reward':         True,
            'include_xy':               True,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'epochs':                   200,
            'test_episodes_per_epoch':  25,
        }))
elif args.trial == "frame_delay_eval":
    for frame_repeat in reversed(frame_repeat_list_2):
        jobs.append(
            ("_frame_repeat={}".format(frame_repeat), {
            'test_frame_repeat':        frame_repeat,
            'test_episodes_per_epoch':  1000,
        }))
    args.trial = "frame_repeat_2"
elif args.trial == "frame_repeat":
    for frame_repeat in frame_repeat_list:
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
    for frame_repeat in reversed(frame_repeat_list):
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
elif args.trial == "include_xy":
    for learning_rate in [1e-4, 3e-4]:
        for include_xy in [True, False]:
            jobs.append(
                ("include_xy={} learning_rate={}".format(include_xy, learning_rate), {
                'include_xy':               include_xy,
                'end_eps':                  0.10,
                'target_update':            100,
                'learning_steps_per_epoch': 5000,
                'update_every':             4,
                'replay_memory_size':       10000,
                'batch_size':               32,
                'num_stacks':               4,
                'learning_rate':            learning_rate,
                'health_as_reward':         True,
                'frame_repeat':             10,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':                   200,
                'max_pool':                 True,
                'test_episodes_per_epoch':  25,
            }))
elif args.trial == "weight_decay_2":
    for weight_decay in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0]:
        jobs.append(
            ("weight_decay={}".format(weight_decay), {
            'weight_decay':             weight_decay,
            'end_eps':                  0.10,
            'target_update':            100,
            'learning_steps_per_epoch': 5000,
            'update_every':             4,
            'replay_memory_size':       10000,
            'batch_size':               32,
            'num_stacks':               4,
            'learning_rate':            1e-4,
            'health_as_reward':         True,
            'frame_repeat':             10,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'epochs':                   200,
            'max_pool':                 True,
            'test_episodes_per_epoch':  25,
        }))
elif args.trial == "optimizer":
    for optimizer in ["adam", "rmsprop", "rmsprop_centered"]:
        for learning_rate in [3e-5, 1e-4, 3e-4, 1e-3]:
            jobs.append(
                ("optimizer={} lr={}".format(optimizer, learning_rate), {
                'optimizer':                optimizer,
                'learning_rate':            learning_rate,
                'end_eps':                  0.10,
                'target_update':            100,
                'learning_steps_per_epoch': 5000,
                'update_every':             4,
                'replay_memory_size':       10000,
                'batch_size':               32,
                'num_stacks':               4,
                'health_as_reward':         True,
                'frame_repeat':             10,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':                   200,
                'max_pool':                 True,
                'test_episodes_per_epoch':  25,
            }))
elif args.trial == "model":
    for learning_rate in [1e-4, 3e-5, 3e-4]:
        for model in ["basic", "tall", "fat", "deep"]:
            jobs.append(
                ("model={} lr={}".format(model, learning_rate), {
                'model':                    model,
                'learning_rate':            learning_rate,
                'end_eps':                  0.10,
                'target_update':            100,
                'learning_steps_per_epoch': 5000,
                'update_every':             4,
                'replay_memory_size':       10000,
                'batch_size':               32,
                'num_stacks':               4,
                'health_as_reward':         True,
                'frame_repeat':             10,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':                   200,
                'max_pool':                 True,
                'test_episodes_per_epoch':  25,
            }))
elif args.trial == "health_gathering_supreme":
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
            'include_xy':               np.random.choice([True, False]),
            'end_eps':              np.random.choice([0, 0.1, 0.03, 0.01]),
            'weight_decay':             np.random.choice([0, 1e-5, 1e-4, 1e-3]),
            'optimizer':                np.random.choice(["adam", "rmsprop", "rmsprop_centered"]),
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'frame_repeat':             np.random.choice([7, 10, 14]),
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
elif args.trial == "take_cover":
    for i in range(args.repeats):
        # pick random parameters
        jobs.append(
            ("sample", {
            'num_stacks':               np.random.choice([1, 2, 4, 8]),
            'discount_factor':          np.random.choice([1, 0.99, 0.98]),
            'replay_memory_size':       np.random.choice([3000,10000,30000]),
            'target_update':            np.random.choice([-1, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]),
            'hidden_units':             np.random.choice([32, 64, 128, 256, 512, 1024, 2048]),
            'learning_rate':            np.random.choice([0.1e-4, 0.3e-4, 1e-4, 3e-4, 10e-4, 30e-4]),
            'max_pool':                 np.random.choice([True, False]),
            'use_color':                np.random.choice([True, False]),
            'include_xy':               np.random.choice([True, False]),
            'end_eps':              np.random.choice([0, 0.1, 0.03, 0.01]),
            'weight_decay':             np.random.choice([0, 1e-5, 1e-4, 1e-3]),
            'optimizer':                np.random.choice(["adam", "rmsprop", "rmsprop_centered"]),
            'config_file_path': "scenarios/take_cover.cfg",
            'frame_repeat':             np.random.choice([7, 10, 14]),
            'learning_steps_per_epoch': 5000,
            'test_episodes_per_epoch':  25,
            'update_every':             4,
            'epochs':                   200,
            'batch_size':               32,
            'health_as_reward':         np.random.choice([True, False]),
            'terminate_early':          True
            }))
    if args.mode == "run":
        args.mode = "search"


# --------------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------------

elif args.trial == "health_gathering_supreme_2":
    for i in range(args.repeats):
        # pick random parameters
        jobs.append(
            ("sample", {
            'num_stacks':               np.random.choice([1, 2, 4]),
            'discount_factor':          np.random.choice([1, 0.98, 0.95]),
            'replay_memory_size':       10000,
            'target_update':            np.random.choice([100, 200]),
            'hidden_units':             np.random.choice([256, 512, 1024]),
            'learning_rate':            np.random.choice([1e-4, 3e-4, 1e-5]),
            'max_pool':                 True,
            'use_color':                True,
            'include_xy':               False,
            'end_eps':                  0,
            'weight_decay':             np.random.choice([0, 1e-6, 1e-5]),
            'optimizer':                "rmsprop",
            'config_file_path':         "scenarios/health_gathering_supreme.cfg",
            'frame_repeat':             np.random.choice([7, 10, 14]),
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
elif args.trial == "take_cover_2":
    for i in range(args.repeats):
        # pick random parameters
        jobs.append(
            ("sample", {
            'learning_rate':            np.random.choice([1e-4, 3e-4, 1e-3]),
            'num_stacks':               np.random.choice([1, 2, 4]),
            'hidden_units':             np.random.choice([64, 128, 256]),
            'target_update':            np.random.choice([200, 400, 800]),
            'end_eps':                  np.random.choice([0, 0.01]),
            'frame_repeat':             10,

            'optimizer':                "rmsprop_centered",     # centered is better for this task
            'max_pool':                 False,                  # max pool has little effect, and it's faster with this off.
            'use_color':                True,
            'include_xy':               False,
            'weight_decay':             0,
            'discount_factor':          1,
            'replay_memory_size':       10000,
            'learning_steps_per_epoch': 5000,
            'test_episodes_per_epoch':  25,
            'update_every':             4,
            'epochs':                   200,
            'batch_size':               32,
            'health_as_reward':         False,
            'terminate_early':          True,
            'config_file_path': "scenarios/take_cover.cfg"
            }))
    if args.mode == "run":
        args.mode = "search"

elif args.trial == "dynamic_tc":
    for dynamic_frame_repeat in [True]:
        # pick random parameters
        jobs.append(
            ("TC {} [10]".format(dynamic_frame_repeat), {
            'dynamic_frame_repeat':     dynamic_frame_repeat,

            'learning_rate':            3e-4,
            'num_stacks':               2,
            'hidden_units':             128,
            'target_update':            800,
            'end_eps':                  0.1,
            'frame_repeat':             10,
            'optimizer':                "rmsprop_centered",     # centered is better for this task
            'max_pool':                 False,                  # max pool has little effect, and it's faster with this off.
            'use_color':                True,
            'include_xy':               False,
            'weight_decay':             0,
            'discount_factor':          1,
            'replay_memory_size':       10000,
            'learning_steps_per_epoch': 25000,
            'test_episodes_per_epoch':  100,
            'update_every':             4,
            'epochs':                   100,
            'batch_size':               32,
            'max_simultaneous_actions': 2,
            'health_as_reward':         False,
            'config_file_path': "scenarios/take_cover.cfg"
            }))
elif args.trial == "dynamic_hgs":
    for dynamic_frame_repeat in [True]:
        # pick random parameters
        jobs.append(
            ("HGS {} [10]".format(dynamic_frame_repeat), {
            'dynamic_frame_repeat':     dynamic_frame_repeat,

            'num_stacks':               4,
            'discount_factor':          1,
            'replay_memory_size':       10000,
            'target_update':            200,
            'hidden_units':             512,
            'learning_rate':            1e-4,
            'max_pool':                 True,
            'use_color':                True,
            'include_xy':               False,
            'end_eps':                  0.1,
            'weight_decay':             0,
            'optimizer':                "rmsprop",
            'config_file_path':         "scenarios/health_gathering_supreme.cfg",
            'frame_repeat':             10,
            'learning_steps_per_epoch': 25000,
            'test_episodes_per_epoch':  100,
            'update_every':             4,
            'epochs':                   100,
            'batch_size':               32,
            'max_simultaneous_actions': 2,
            'health_as_reward':         True,
            }))



else:
    print("Invalid trial name {}".format(args.trial))
    exit(-1)

# set thread limit for worker
if args.threads is not None:
    for job in jobs:
        job[1]["threads"] = args.threads


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
    random.shuffle(jobs) # process jobs in random order.
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
