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
    return " ".join("{}={}".format(clean(k), clean(str(v))) for k, v in params)


def count_jobs(experiment, job_name):

    # count the number folders that have our experiment name and arguments.
    counter = 0
    for r, d, f in os.walk(os.path.join("runs", experiment)):
        if job_name+" [" in os.path.basename(r) and "gcs" not in r:
            counter += 1

    return counter

def run_job(experiment, job_name, kwargs):
    """ Runs job with given arguments. """
    python = "python3" if sys.platform in ["linux", "linux2"] else "python"
    subprocess.call([python,args.train_script,"train"] +
                    ["--experiment={}".format(experiment)]+
                    ["--job_name={}".format(job_name)]+
                    ["--{}={}".format(k,v) for k,v in kwargs.items()])

def process_job(experiment, job_name, repeats, **kwargs):
    """ Process a job. """
    for _ in range(repeats):
        if count_jobs(experiment, job_name) < repeats:
            run_job(experiment, job_name, kwargs)

def show_job_count(experiment, job_name, repeats):
    """ Count job. """
    print("{:<40} {}/{}".format(experiment+' '+job_name, count_jobs(experiment, job_name), repeats))

parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
parser.add_argument('mode', type=str, help='count | run')
parser.add_argument('trial', type=str, help='Trial to run')
parser.add_argument('--repeats', type=int, default=2, help='Number of times to repeat each trial.')
parser.add_argument('--train_script', type=str, default="train.py", help='Script to use to train.')

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
                'test_episodes_per_epoch':20,   #faster to train, can always run more later...
            })
    )
elif args.trial == "trial_11":
    # very close to original paper...
    jobs.append(
        ("original (no ta)", {
            'target_update': -1,
            'num_stacks': 4,
            'learning_rate': 0.00001,
            'health_as_reward': False,
            'config_file_path': "scenarios/health_gathering_supreme.cfg",
            'learning_steps_per_epoch': 5000,
            'epochs':200,
            'test_episodes_per_epoch':20,
        })
    )
elif args.trial == "trial_12":
    # see how target update effects things
    for target_update in [-1, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        jobs.append(
            ("target_update={}".format(target_update), {
                'target_update': target_update,
                'num_stacks': 4,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':200,
                'test_episodes_per_epoch':20,
            })
        )
elif args.trial == "trial_13":
    # see how exp buffer effects things
    for replay_memory_size in [100, 500, 1000, 2000, 5000, 10000, 20000]:
        jobs.append(
            ("replay_memory_size={}".format(replay_memory_size), {
                'replay_memory_size': replay_memory_size,
                'target_update': 100,
                'num_stacks': 4,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':200,
                'test_episodes_per_epoch':20,
            })
        )
elif args.trial == "trial_14":
    for update_every in [2, 1, 1 / 2]:
        jobs.append(
            ("[default] update_every={}".format(update_every), {
                'target_update': 100,
                'num_stacks': 4,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'update_every': update_every,
                'epochs': 200,
                'test_episodes_per_epoch': 20,  # faster to train, can always run more later...
            }))
        jobs.append(
            ("[per_env] update_every={}".format(update_every), {
                'target_update': int(100 / update_every),
                'update_every': update_every,
                'num_stacks': 4,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs': 200,
                'test_episodes_per_epoch': 20,  # faster to train, can always run more later...
            }))
        jobs.append(
            ("[exp_replay] update_every={}".format(update_every), {
                'target_update': int(100 / update_every),
                'replay_memory_size': int(10000 / update_every),
                'update_every': update_every,
                'num_stacks': 4,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs': 200,
                'test_episodes_per_epoch': 20,  # faster to train, can always run more later...
            }))
elif args.trial == "trial_15":
    for update_every in [2,1,0.5]:
        for target_update in [1000, 2000, 5000, 10000]:
            jobs.append(
            ("update_every={} target_update={}".format(update_every, target_update), {
                'target_update': target_update,
                'update_every': update_every,
                'num_stacks': 4,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs': 1000,
                'test_episodes_per_epoch': 10,
            }))

elif args.trial == "exp_1":
    for update_every in [8, 4, 2, 1, 1/2, 1/4]:                           #[4, 2, 1, 1/2, 1/4]:
        for target_update in [25, 50, 100, 200, 500, 1000, 2000, 5000]:   #[25, 50, 100, 200]:
            for replay_memory_size in [10000]:                            #[2500, 5000, 10000, 20000, 40000]:
                jobs.append(
                    ("ue={} rms={} ta={}".format(update_every, replay_memory_size, target_update), {
                    'target_update': target_update,
                    'update_every': update_every,
                    'replay_memory_size': replay_memory_size,
                    'batch_size': 32,
                    'num_stacks': 4,
                    'learning_rate': 0.0001,
                    'health_as_reward': True,
                    'config_file_path': "scenarios/health_gathering_supreme.cfg",
                    'epochs':200,
                    'test_episodes_per_epoch':20,   #faster to train, can always run more later...
                })
    )
elif args.trial == "trial_16":
    for update_every in [4, 2, 1, 1/2, 1/4]:
        for target_update in [1000]:
            for learning_rate in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
                jobs.append(
                    ("ue={} ta={} lr={}".format(update_every, target_update, learning_rate), {
                    'target_update': target_update,
                    'update_every': update_every,
                    'replay_memory_size': 10000,
                    'batch_size': 32,
                    'num_stacks': 4,
                    'learning_rate': learning_rate,
                    'health_as_reward': True,
                    'config_file_path': "scenarios/health_gathering_supreme.cfg",
                    'epochs':200,
                    'test_episodes_per_epoch':10,   #faster to train, can always run more later...
                })
    )

elif args.trial == "trial_17":
    # batch vs update every
    for update_every in [ 2, 1, 1/2]:
            for batch_size in [16, 32, 64]:
                jobs.append(
                    ("ue={} bs={}".format(update_every, batch_size), {
                    'target_update': 1000,
                    'update_every': update_every,
                    'replay_memory_size': 10000,
                    'batch_size': batch_size,
                    'num_stacks': 4,
                    'learning_rate': 0.0001,
                    'health_as_reward': True,
                    'config_file_path': "scenarios/health_gathering_supreme.cfg",
                    'epochs':200,
                    'test_episodes_per_epoch':10,   #faster to train, can always run more later...
                })
    )

elif args.trial == "trial_18":
    # stride vs max_pool
    for max_pool in [True, False]:
        jobs.append(
            ("max_pool={}".format(max_pool), {
                'target_update': 100,
                'update_every': 2,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 4,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs': 200,
                'max_pool': max_pool,
                'test_episodes_per_epoch': 10,  # faster to train, can always run more later...
            }))

elif args.trial == "exp_2":
    for update_every in [4, 2, 1, 1/2]:                                             # todo: include (1/4)
        for learning_rate in [1e-4 * 2 ** x for x in [-3, -2, -1, 0, 1, 2, 3]]:
            jobs.append(
                ("update_every={} learning_rate={}".format(update_every, learning_rate), {
                'target_update': 1000,
                'learning_steps_per_epoch': 5000,
                'update_every': update_every,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 4,
                'learning_rate': learning_rate,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':50,
                'max_pool': True,
                'test_episodes_per_epoch':100,   #faster to train, can always run more later...
            }))

elif args.trial == "exp_2b":
    args.trial = "exp_2"
    for update_every in [1/4]:
        for learning_rate in [1e-4 * 2 ** x for x in [-4, -3]]:
            jobs.append(
                ("update_every={} learning_rate={}".format(update_every, learning_rate), {
                'target_update': 1000,
                'learning_steps_per_epoch': 5000,
                'update_every': update_every,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 4,
                'learning_rate': learning_rate,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':50,
                'max_pool': True,
                'test_episodes_per_epoch':100,   #faster to train, can always run more later...
            }))

elif args.trial == "exp_2c":
    args.trial = "exp_2"
    for update_every in [2]:
        for learning_rate in [1e-4 * 2 ** x for x in [-3, -2, -1, 0, 1, 2, 3]]:
            jobs.append(
                ("update_every={} learning_rate={}".format(update_every, learning_rate), {
                'target_update': 1000,
                'learning_steps_per_epoch': 5000,
                'update_every': update_every,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 4,
                'learning_rate': learning_rate,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':50,
                'max_pool': True,
                'test_episodes_per_epoch':100,   #faster to train, can always run more later...
            }))

elif args.trial == "exp_2e":
    # repeats around best results
    args.trial = "exp_2"
    for update_every in [32, 16, 8, 4, 2, 1, 1/2, 1/4]:
        for learning_rate in [5e-5 * update_every * 2 ** x for x in [-2, -1, 0, 1, 2]]:
            jobs.append(
                ("update_every={} learning_rate={}".format(update_every, learning_rate), {
                'target_update': 1000,
                'learning_steps_per_epoch': 5000,
                'update_every': update_every,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 4,
                'learning_rate': learning_rate,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':50,
                'max_pool': True,
                'test_episodes_per_epoch':25,   #faster to train, can always run more later...
            }))

elif args.trial == "exp_2f":
    # extended search
    args.trial = "exp_2"
    for update_every in [32, 16, 8, 4, 2, 1, 1/2, 1/4]:
        for learning_rate in [5e-5 * update_every * 2 ** x for x in [-3, -2, -1, 0, 1, 2, 3]]:
            jobs.append(
                ("update_every={} learning_rate={}".format(update_every, learning_rate), {
                'target_update': 1000,
                'learning_steps_per_epoch': 5000,
                'update_every': update_every,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 4,
                'learning_rate': learning_rate,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':50,
                'max_pool': True,
                'test_episodes_per_epoch':25,   #faster to train, can always run more later...
            }))


elif args.trial == "agent_mode":
    for agent_mode in ["default","random","stationary"]:                                             # todo: include (1/4)
            jobs.append(
                ("agent_mode={}".format(agent_mode), {
                'target_update': 1000,
                'learning_steps_per_epoch': 1000,
                'update_every': 4,
                'replay_memory_size': 10000,
                'batch_size': 32,
                'num_stacks': 1,
                'learning_rate': 0.0001,
                'health_as_reward': True,
                'config_file_path': "scenarios/health_gathering_supreme.cfg",
                'epochs':10,
                'max_pool': False,
                'test_episodes_per_epoch':100,
            }))

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


if args.mode == "count":
    for job_name, job_args in jobs:
        show_job_count(
            experiment=args.trial,
            job_name=job_name,
            repeats=args.repeats
        )
elif args.mode == "run":
    # get 1 pass on each job first, then head back and do the repeat runs...
    for job_name, job_args in jobs:
        process_job(
            experiment=args.trial,
            job_name=job_name,
            repeats=1,
            **job_args
        )
    for job_name, job_args in jobs:
        process_job(
            experiment=args.trial,
            job_name=job_name,
            repeats=args.repeats,
            **job_args
        )
else:
    print("Invalid mode {}".format(args.mode))
