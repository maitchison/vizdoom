"""
Author

Test the effect of update frequency in DQN.

Based on...

this paper has some hyper paramaters too http://cs229.stanford.edu/proj2017/final-reports/5238810.pdf
keras implementation of some maps. https://github.com/flyyufelix/VizDoom-Keras-RL
"""

# force single threads, makes program more CPU efficient but doesn't really hurt performance.
# this does seem to affect pytorch cpu speed a bit though.
import os
import sys

if __name__ == '__main__':

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    for arg in sys.argv:
        if "--threads" in arg.lower():
            thread_count = arg.split("=")[-1]
            print("Setting thread count to {}".format(thread_count))
            os.environ["OMP_NUM_THREADS"] = thread_count
            os.environ["OPENBLAS_NUM_THREADS"] = thread_count
            os.environ["MKL_NUM_THREADS"] = thread_count
            os.environ["OPENBLAS_NUM_THREADS"] = thread_count

import ast
import argparse
import vizdoom
import vizdoom.vizdoom as vzd
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import pickle
import matplotlib.pyplot as plt
import os.path
import cv2
import uuid

import configparser
import subprocess

import shutil
import logging
import platform
import keyboard

from operator import itemgetter
from pympler import tracker

# --------------------------------------------------------
# Globals
# --------------------------------------------------------

MAX_GRAD = 1000     # this should be 1, the fact we have super high gradients is concerning, maybe
                    # I should make sure input is normalised to [0,1] and change the reward shaping structure
                    # to not be so high.  Also log any extreme rewards...

AUX_INPUTS = 64     # number of auxilary inputs to model.
MAX_BUTTONS = 32    # maximum number of button inputs

preview_screen_resolution = vzd.ScreenResolution.RES_640X480
train_screen_resolution = vzd.ScreenResolution.RES_160X120

time_stats = {}
prev_loss = 0
max_q = 0
max_grad = 0
last_total_shaping_reward = 0
learning_steps = 0
previous_health = 0
kb = keyboard.KBHit()
console_logger = None
health_history = []
observation_history = []    # observational history, plus health and time tick.
data_history = []    # observational history, plus health and time tick.
game = None
game_hq = None      # for video previews

starting_locations = set()
criterion = nn.MSELoss()

# --------------------------------------------------------

def safe_cast(x):
    try:
        return int(str(x))
    except:
        try:
            return float(str(x))
        except:
            return x


def clean(s):
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return "".join([x if x in valid_chars else "_" for x in s])


def get_job_key(args):
    params = sorted([(k, v) for k, v in args.items() if k not in ["mode"] and v is not None])
    return " ".join("{}={}".format(clean(k), clean(str(v))) for k, v in params)+" "


class Config:
    """
    Contains configuration information

    Attributes:
        mode: The mode we are current in
        ...
    """

    def __init__(self):

        self.mode = "train"
        self.num_stacks = 1
        self.use_color = True
        self.epochs = 40
        self.learning_rate = 0.00001
        self.discount_factor = 1
        self.learning_steps_per_epoch = 1000
        self.test_episodes_per_epoch = 20
        self.replay_memory_size = 10000
        self.end_eps = 0.1
        self.start_eps = 1.0
        self.hidden_units =  1024
        self.update_every = 1
        self.batch_size = 64
        self.target_update = 1000
        self.first_update_step = 1000
        self.frame_repeat = 10
        self.test_frame_repeat = None           # if not none overrides frame_repeat for testing.
        self.resolution = (84, 84)
        self.verbose = False
        self.max_pool = False
        self.dynamic_frame_repeat = False
        self.dfr_decision_cost = 0.0
        self.end_eps_step = None
        self.config_file_path = "scenarios/health_gathering.cfg"
        self.job_id = uuid.uuid4().hex[:16]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.experiment = "experiments"
        self.computer_specs = platform.uname()
        self.hostname = platform.node()
        self.include_aux_rewards = True
        self.health_as_reward = False
        self.export_video = True
        self.job_name = "job"
        self.agent_mode = "default"
        self.pytorch_version = torch.__version__
        self.python_vesion = sys.version
        self.rand_seed = None
        self.eval_results_suffix = None
        self.model = None
        self.optimizer = "rmsprop"
        self.weight_decay = 0.0
        self.include_xy = False
        self.output_path = "runs"
        self.max_simultaneous_actions = None
        self.weighted_random_actions = False
        self.gate_epoch = None
        self.gate_score = None
        self.gradient_clip = 0
        self.novelty = 0.0
        self.id_factor = 0.0
        # this makes config file loading require vizdoom... which I don't want.
        # self.screen_resolution = vzd.ScreenResolution.RES_160X120

    def apply_args(self, args):
        """
        apply args to config
        :param args: argparse arguments
        :return: none
        """

        for k,v in args.__dict__.items():
            if v is not None:
                setattr(self, k, v)

        self.mode = self.mode.lower()
        self.args = args

    @property
    def num_channels(self):
        color_channels = 3 if self.use_color else 1
        xy_channels = 2 if self.include_xy else 0
        return color_channels + xy_channels

    @property
    def job_folder(self):
        """ path to job folder. """

        if self.mode == "benchmark":
            experiment = "benchmarks"
        elif self.mode == "test":
            experiment = "test"
        else:
            experiment = self.experiment

        prefix = "_" if config.mode == "train" else ""

        return os.path.join(self.output_path, experiment, prefix + self.job_subfolder)

    @property
    def final_job_folder(self):

        if self.mode == "benchmark":
            experiment = "benchmarks"
        elif self.mode == "test":
            experiment = "test"
        else:
            experiment = self.experiment

        return os.path.join(self.output_path, experiment, self.job_subfolder)

    @property
    def job_subfolder(self):
        """ job folder without runs folder and experiment folder. """
        return "{} [{}]".format(self.job_name, self.job_id)

    @property
    def scenario(self):
        return os.path.splitext(os.path.basename(self.config_file_path))[0]

    @property
    def start_eps_decay(self):
        return min(self.total_steps * 0.1, self.learning_steps_per_epoch * 10, 10*1000)

    @property
    def end_eps_decay(self):
        if self.end_eps_step is None:
            # find a good default
            return min(self.total_steps * 0.6, self.learning_steps_per_epoch * 60, 100*1000)
        else:
            return self.end_eps_step

    @property
    def total_steps(self):
        return self.epochs * self.learning_steps_per_epoch

    def make_job_folder(self):
        """create the job folder"""

        # on a networked folder this sometimes errors so we try a couple of times.
        error = None
        for _ in range(3):
            try:
                os.makedirs(self.job_folder, exist_ok=True)
                os.makedirs(os.path.join(self.job_folder, "models"), exist_ok=True)
                os.makedirs(os.path.join(self.job_folder, "videos"), exist_ok=True)
                return
            except e:
                error = e

        raise e

    def rename_job_folder(self):
        """ moves job to completed folder. """

        # first make sure that we have completed writing the final files

        completed_file = os.path.join(self.job_folder, "results_complete.dat")

        # wait 5 minutes for final file to copy across before renaming folder.
        for _ in range(5):
            if os.path.exists(completed_file) and os.stat(completed_file).st_size > 0:
                break
            sleep(60)
        else:
            logging.critical("Warning: timeout while waiting for file {} to finish.".format(completed_file))

        for _ in range(3):
            try:
                os.rename(self.job_folder, self.final_job_folder)
                break
            except Exception as e:
                print("\nFailed to rename job folder: {}\n".format(e))
                sleep(60)  # give Dropbox a chance to sync up.
        else:
            print("Error moving completed job to {}.".format(config.final_job_folder))


def track_time_taken(method):

    def timed(*args, **kwargs):
        name = kwargs.get('log_name', method.__name__.upper())
        start_time = time()
        result = method(*args, **kwargs)
        time_taken = time() - start_time
        if name not in time_stats:
            time_stats[name] = (0,0.0)
        stat = time_stats[name]
        time_stats[name] = (stat[0] + 1, stat[1] + time_taken)

        return result

    return timed


def show_time_stats(log_level = logging.DEBUG):
    logging.log(log_level, "Timing stats:")
    logging.log(log_level, "\t{:<25} {:<8} {:<8}".format("Function","Calls","Time (ms) "))
    for k,v in time_stats.items():
        count, total_time = v
        per_time = total_time / count
        logging.log(log_level, "\t{:<25} {:<8} {:<8.1f}".format(k.lower(), count, per_time * 1000))

xy_cache = {}

def get_xy_matrix(h, w):
    """ Returns 2xhxw matrix with xy locations. """

    key = (h,w)

    if key in xy_cache:
        return xy_cache[key]

    xy = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            xy[:, x, y] = (x / w, y / h)

    xy_cache[key] = xy
    return xy

# Converts and down-samples the input image
@track_time_taken
def preprocess(img):

    img = np.swapaxes(img, 0, 2)
    img = cv2.resize(np.float32(img)/255, dsize=config.resolution[::-1], interpolation=cv2.INTER_AREA)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    # convert to grayscale if needed
    if not config.use_color:
        img = np.mean(img, axis=0, keepdims=True)

    # add xy if needed
    if config.include_xy:
        c, h, w = img.shape
        img = np.concatenate((img, get_xy_matrix(h, w)), axis=0)

    # convert from float32 to uint8
    img = np.uint8(img * 255)

    return img

class NoveltyMemory:

    def __init__(self, capacity):
        self.s = np.zeros((capacity, config.hidden_units))
        self.pos = 0
        self.size = 0
        self.mu = 0
        self.sigma = 1
        self.counter = 0

    @track_time_taken
    def check_normalization_constants(self):
        # calculate distance for many samples.
        print(": novelty constants (current) mean:{:.3f} std:{:.3f}".format(self.mu, self.sigma))
        sample_count = 100
        sample = [self.get_distance(self.s[x], ignore_closest=True) for x in np.random.choice(range(self.size), size=sample_count)]
        print(": novelty constants (sample) mean:{:.3f} std:{:.3f}".format(np.mean(sample), np.std(sample)))

    @track_time_taken
    def add_novelty_sample(self, sample):

        self.s[self.pos] = sample
        self.pos = (self.pos + 1) % len(self.s)
        self.size = min(self.size + 1, len(self.s))
        self.counter += 1

        # update our normalization constants a bit...
        if self.counter % 4 == 0:
            sample_count = 4
            sample = [self.get_distance(self.s[x], ignore_closest=True) for x in np.random.choice(range(self.size), size=sample_count)]

            alpha = 0.99
            self.mu = alpha * self.mu + (1-alpha) * np.mean(sample)
            self.sigma = alpha * self.sigma + (1 - alpha) * np.std(sample, ddof=1) # this is a sample not the population...

        # every now and them make constants more accurate, probably not needed though?
        if self.counter % 5000 == 0:
            self.check_normalization_constants()

    def get_distance(self, sample, ignore_closest=False):
        """ returns distance between this example and all examples in cache.
            ignore_closest: closets match will be disgarded which is helpful for testing distance of a cached sample
                without including itself.
        """

        sample = np.asarray(sample)

        # these slots are not yet filled... so filter them out
        others = self.s if self.size == len(self.s) else self.s[:self.size]

        # L2 seems to work the best in my other experiments on density estimation
        distances = np.linalg.norm(sample - others, ord=2, axis=1)

        distances = sorted(distances)
        if ignore_closest:
            top_5 = distances[1:6]
        else:
            top_5 = distances[:5]

        if len(top_5) == 0:
            return 0
        else:
            return np.mean(top_5)


    def get_novelty(self, sample):
        """ returns novelty [-1..1] of given sample where sample is the current hidden state.
            a score of 1.0 indicates highly novel, where as a score of 0 or less indicates typical.
        """
        distance = self.get_distance(sample)
        return np.tanh((distance - self.mu) / self.sigma)

class ReplayMemory:

    def __init__(self, capacity):
        state_shape = (capacity, config.num_channels, config.resolution[0], config.resolution[1])
        novel_shape = (capacity,)
        data_shape = (capacity, AUX_INPUTS)
        self.s1 = np.zeros(state_shape, dtype=np.uint8)
        self.s2 = np.zeros(state_shape, dtype=np.uint8)

        self.nov = np.zeros(novel_shape, dtype=np.float32)
        self.d1 = np.zeros(data_shape, dtype=np.float32)
        self.d2 = np.zeros(data_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        logging.info("State buffer size: {} = {:.1f}m".format(state_shape, np.prod(state_shape)*1/1024/1024))
        logging.info("Data buffer size: {} = {:.1f}m".format(data_shape, np.prod(data_shape)*4/1024/1024))

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, d1, action, s2, d2, isterminal, reward):

        assert s1.dtype == np.uint8, "Please convert frames to uint8 before adding to experience replay."
        assert s2 is None or s2.dtype == np.uint8, "Please convert frames to uint8 before adding to experience replay."

        self.s1[self.pos, :, :, :] = s1[0]
        self.d1[self.pos] = d1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2[0]
            self.d2[self.pos] = d2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack(self, data, samples):
        result = []
        for sample in samples:
            # get the [F,C,H,W] sample and convert to [F*C,H,W]
            frames = data[sample-config.num_stacks+1:sample+1]
            if len(frames.shape) != 1:
                frames = np.concatenate(frames, axis=0)
            result.append(frames)
        # convert to [B,F*C,H,W]
        result = np.asarray(result)
        return result

    @track_time_taken
    def get_sample(self, sample_size):
        """
        Fetches a sample from the experience replay
        :param sample_size:
        :return: numpy array of shape ...
        """
        samples = np.random.choice(self.size - config.num_stacks, size=sample_size, replace=True) + config.num_stacks
        return (
            self._stack(self.s1, samples),
            self._stack(self.d1, samples),
            self.a[samples],
            self._stack(self.s2, samples),
            self._stack(self.d2, samples),
            self.isterminal[samples],
            self.r[samples]
        )


def prod(X):
    y = 1
    for x in X:
        y *= x
    return y


def save_frames(filename, x):
    print("Saving {}".format(filename))
    print("Shape:",x.shape)
    for i in range(len(x)):
        plt.imsave("{}-{:03d}.png".format(filename,i+1), x[i])


def get_net(available_actions_count):
    """ Construct network according to config setting. """
    if config.model == "basic":
        return Net(available_actions_count)
    elif config.model == "dual":
        return DualNet(available_actions_count)
    else:
        raise Exception("Invalid model name {}.".format(config.model))

class Net(nn.Module):
    """
    This is a basic 4 layer conv network used in many tests.
    """
    def __init__(self, available_actions_count):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(
            config.num_channels * config.num_stacks, 32,    #3 color channels, 4 previous frames.
            kernel_size=5,
            stride=1 if config.max_pool else 2
        )
        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3,
            stride=1 if config.max_pool else 2
        )
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3,
            stride=1
        )
        self.conv4 = nn.Conv2d(
            64, 64, kernel_size=3,
            stride=1
        )
        # maxpool and stride have slightly different final shapes.
        final_shape = [64, 7, 1]
        self.fc1 = nn.Linear(prod(final_shape) + AUX_INPUTS * config.num_stacks, config.hidden_units)
        self.fc2 = nn.Linear(config.hidden_units, available_actions_count)

        self.pa = nn.Linear(config.hidden_units*2, available_actions_count)

        self.num_actions = available_actions_count

    def hidden(self, x, d):
        x = x.to(config.device)  # BxCx120x45
        d = d.to(config.device)  # Bx4x3

        x = x.float() / 255  # convert from unit8 to float32 format

        x = F.relu(self.conv1(x))
        if config.max_pool:
            x = F.max_pool2d(x, 2, padding=0)
        x = F.relu(self.conv2(x))
        if config.max_pool:
            x = F.max_pool2d(x, 2, padding=0)
        x = F.max_pool2d(x, kernel_size=[1, 3], padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, padding=1)
        x = F.relu(self.conv4(x))
        x = x.view(-1, prod(x.shape[1:]))  # Bx352
        x = torch.cat((x, d), 1)  # Bx355
        x = F.leaky_relu(self.fc1(x))
        return x

    def predict_action(self, s1, d1, s2, d2):
        h1 = self.hidden(s1, d1)
        h2 = self.hidden(s2, d2)
        pred = torch.sigmoid(self.pa(torch.cat((h1,h2), 1)))
        return pred.cpu()


    @track_time_taken
    def forward(self, x, d):

        x = self.hidden(x, d)
        x = self.fc2(x)

        return x.cpu()


class DualNet(nn.Module):
    """
    Dualing DQN Net.
    See https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/
    """
    def __init__(self, available_actions_count):
        super(DualNet, self).__init__()

        self.conv1 = nn.Conv2d(
            config.num_channels * config.num_stacks, 32,    #3 color channels, 4 previous frames.
            kernel_size=5,
            stride=1 if config.max_pool else 2
        )
        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3,
            stride=1 if config.max_pool else 2
        )
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3,
            stride=1
        )
        self.conv4 = nn.Conv2d(
            64, 64, kernel_size=3,
            stride=1
        )
        # maxpool and stride have slightly different final shapes.
        final_shape = [64, 7, 1]
        self.fc1 = nn.Linear(prod(final_shape) + AUX_INPUTS * config.num_stacks, config.hidden_units)


        self.fc_v = nn.Linear(config.hidden_units, 1)
        self.fc_a = nn.Linear(config.hidden_units, available_actions_count)

        self.num_actions = available_actions_count

    @track_time_taken
    def forward(self, x, d):

        x = x.to(config.device)   # BxCx120x45
        d = d.to(config.device)   # Bx4x3

        x = x.float()/255         # convert from unit8 to float32 format

        x = F.relu(self.conv1(x))
        if config.max_pool:
            x = F.max_pool2d(x, 2, padding=0)
        x = F.relu(self.conv2(x))
        if config.max_pool:
            x = F.max_pool2d(x, 2, padding=0)
        x = F.max_pool2d(x, kernel_size=[1,3], padding=0)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, padding=1)
        x = F.relu(self.conv4(x))
        x = x.view(-1, prod(x.shape[1:])) # Bx352
        x = torch.cat((x, d), 1)  # Bx355
        x = F.leaky_relu(self.fc1(x))

        a = self.fc_a(x)
        v = self.fc_v(x).expand(x.size(0), self.num_actions)

        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)


        return x.cpu()


def learn(s1, d1, s2, d2, a, target_q):
    s1 = torch.from_numpy(s1)
    d1 = torch.from_numpy(d1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1), Variable(target_q)

    optimizer.zero_grad()
    output = policy_model(s1, d1)
    loss = criterion(output, target_q)

    if config.id_factor != 0 and s2 is not None:
        s2 = Variable(torch.from_numpy(s2))
        d2 = torch.from_numpy(d2)
        one_hot_actions = torch.from_numpy(np.float32([action_id_to_list(x) for x in a]))
        pred = policy_model.predict_action(s1, d1*0, s2, d2*0) # data has action information in it...

        id_loss = config.id_factor * criterion(pred, one_hot_actions)

        #print("loss was:  {} added is: {}".format(loss, id_loss))

        loss += id_loss

    loss.backward()

    if config.gradient_clip != 0:
        for param in policy_model.parameters():
            param.grad.data.clamp_(-config.gradient_clip, config.gradient_clip)

    optimizer.step()
    return loss


def get_q_values(s,d):
    global max_q
    s = Variable(torch.from_numpy(s))
    d = Variable(torch.from_numpy(d))
    q = policy_model(s,d)
    max_q = max(max_q, float(torch.max(q)))
    return q


def get_target_q_values(s,d):
    global max_q
    s = Variable(torch.from_numpy(s))
    d = Variable(torch.from_numpy(d))
    q = target_model(s,d)
    max_q = max(max_q, float(torch.max(q)))
    return q


def get_hidden(s, d):
    s = Variable(torch.from_numpy(s))
    d = Variable(torch.from_numpy(d))
    h = target_model.hidden(s, d).cpu()
    return h

def get_best_action(s,d):
    if config.agent_mode == "random":
        return sample_random_action()
    elif config.agent_mode == "stationary":
        return 0
    elif config.agent_mode == "default":
        s = Variable(torch.from_numpy(s))
        d = Variable(torch.from_numpy(d))
        q = policy_model(s,d).detach()
        m, index = torch.max(q, 1)
        action = index.data.numpy()[0]
        return action
    else:
        raise Exception("Invalid agent_mode {}".format(config.agent_mode))


@track_time_taken
def perform_learning_step():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > config.batch_size:

        global learning_steps
        learning_steps += 1

        s1, d1, a, s2, d2, isterminal, r = memory.get_sample(config.batch_size)

        q = get_q_values(s2, d2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_target_q_values(s1, d1).data.numpy() if config.target_update > 0 else get_q_values(s1, d1).data.numpy()

        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r

        target_q[np.arange(target_q.shape[0]), a] = r + config.discount_factor * (1 - isterminal) * q2

        this_loss = learn(s1, d1, s2, d2, a, target_q)
        global prev_loss
        prev_loss = 0.95 * prev_loss + 0.05 * float(this_loss)

        # update target net every so often.
        if config.target_update > 0 and learning_steps % config.target_update == 0:
            update_target()


def exploration_rate(step):
    """# Define exploration rate change over time"""

    if step < config.start_eps_decay:
        return config.start_eps
    elif step > config.end_eps_decay:
        return config.end_eps
    else:
        # linear decay
        return config.start_eps + (step - config.start_eps_decay) / (config.end_eps_decay - config.start_eps_decay) *\
               (config.end_eps - config.start_eps)


@track_time_taken
def env_step(action, frame_repeat_code):
    """
    Take environmental step
    :param action: action, list of 0,1 for each button.
    :param frame_repeat_code: code, either integer frame skip, or code for random skips.
    :return:
    """
    frame_repeat = convert_frame_repeat(frame_repeat_code)
    return game.make_action(action, frame_repeat)


def push_state(s,d):
    observation_history.append(s)
    data_history.append(d)
    if len(observation_history) > config.num_stacks:
        del observation_history[:-config.num_stacks]
        del data_history[:-config.num_stacks]


def action_list_to_id(action_list):
    mul = 1
    result = 0
    for x in action_list:
        result += mul * x
        mul *= 2
    return result

def action_id_to_list(action_id):
    return [int(x) for x in bin(action_id)[2:].zfill(len(actions))]

def get_observation():
    return preprocess(game.get_state().screen_buffer)


def get_data():

    number_of_actions = len(game.get_last_action())
    max_game_varaibles = AUX_INPUTS - number_of_actions

    game_variables = [game.get_episode_time()] + [game.get_game_variable(x) for x in game.get_available_game_variables()]

    if len(game_variables) > max_game_varaibles:
        raise Exception("Too many game variables for model, found {} but only room for {}".format(len(game_variables), max_game_varaibles))

    # pad game variables with 0s.
    while len(game_variables) < max_game_varaibles:
        game_variables += [0.0]

    # add the last action as a bit vector
    game_variables += game.get_last_action()

    return np.float32(game_variables)

def sample_random_action():
    """ Returns a random action sample from action space. """

    if config.weighted_random_actions:
        probs = [p for _, p in actions]
        probs = 1 / np.float64(probs)   # sample by inverse length (ie. an action 3x longer is sampled at 1/3 rate.
        probs /= np.sum(probs)          # normalize,
        return np.random.choice(len(actions), p=probs)
    else:
        return randint(0, len(actions) - 1)


@track_time_taken
def perform_environment_step(step):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    s1, d1 = get_obs()

    # With probability eps make a random action.
    eps = exploration_rate(step)

    if random() <= eps:
        a = sample_random_action()
    else:
        a = get_best_action(*get_stack())

    global last_total_shaping_reward

    # give some time for this to catch up...
    reward = env_step(*actions[a])

    if config.health_as_reward:
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        delta_health = current_health - health_history[-1]
        reward = delta_health
        health_history.append(current_health)

    # Retrieve the shaping reward
    fixed_shaping_reward = game.get_game_variable(vzd.GameVariable.USER1)  # Get value of scripted variable
    shaping_reward = vzd.doom_fixed_to_double(fixed_shaping_reward)  # If value is in DoomFixed format project it to double
    shaping_reward = shaping_reward - last_total_shaping_reward
    last_total_shaping_reward += shaping_reward

    if abs(shaping_reward) > 1000:
        logging.critical("Unusually large shaping reward found {}.".format(shaping_reward))

    reward += shaping_reward

    if config.dynamic_frame_repeat:
        reward -= config.dfr_decision_cost

    isterminal = game.is_episode_finished()

    if isterminal:
        s2, d2 = None, None
    else:
        push_state(get_observation(), get_data())
        s2, d2 = get_obs()

    # calculate the novelty - note we do this here, not once in the memory.
    if config.novelty != 0:
        h = get_hidden(*get_stack()).detach()[0]
        nov = feature_cache.get_novelty(h)

        # there are a few ideas here,
        # we can't store 10k samples so just take 10% or them
        # maybe novel states should be weighted more highly? not sure?

        if randint(0, 9) == 0:
            feature_cache.add_novelty_sample(h)

        novelty_reward = config.novelty * nov
        reward += novelty_reward

    memory.add_transition(s1, d1, a, s2, d2, isterminal, reward)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom():
    """
    Initialize game (global varaible).
    :return:
    """

    global game
    global game_hq

    if game is not None:
        game.close()

    if game_hq is not None:
        game_hq.close()

    logging.info("Initializing Doom.")
    game = vzd.DoomGame()
    game.load_config(config.config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_screen_resolution(train_screen_resolution)
    game.set_seed(123)
    game.init()

    if config.export_video:
        game_hq = vzd.DoomGame()
        game_hq.load_config(config.config_file_path)
        game_hq.set_window_visible(False)
        game_hq.set_mode(vzd.Mode.PLAYER)
        game_hq.set_screen_format(vzd.ScreenFormat.CRCGCB)
        game_hq.set_screen_resolution(preview_screen_resolution)
        game_hq.init()

    initialize_actions(config.frame_repeat, verbose=True)

    logging.info("Doom initialized.")


def initialize_actions(base_skip, verbose=False):

    global actions

    base_skip = safe_cast(base_skip)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()

    if config.dynamic_frame_repeat:
        # extend actions with repeat counts
        actions = []

        low_skip = max(1, base_skip // 3)
        med_skip = base_skip
        high_skip = base_skip*3

        for skip in [low_skip, med_skip, high_skip]:
            actions += [(list(a), skip) for a in it.product([0, 1], repeat=n)]
    else:
        actions = [(list(a), base_skip) for a in it.product([0, 1], repeat=n)]

    if config.max_simultaneous_actions is not None:
        original_action_count = len(actions)
        actions = [(a, skip) for a, skip in actions if sum(a) <= config.max_simultaneous_actions]
        if verbose:
            logging.critical("Using {}/{} action combinations (with max actions: {} from {} buttons)".format(
                len(actions), original_action_count, config.max_simultaneous_actions, n
            ))

@track_time_taken
def update_target():
    target_model.load_state_dict(policy_model.state_dict())


def tidy_args(args):
    result = {}
    for k,v in args.__dict__.items():
        if v is not None:
            result[k] = v
    return result


def handle_keypress():
    if kb.kbhit():
        c = kb.getch().lower()
        if c == "p":
            print()
            logging.critical("***** Pausing. Press 'R' to resume. ****")
            while kb.getch().lower() != 'r':
                sleep(0.1)
        elif c == "t":
            print()
            show_time_stats(logging.CRITICAL)
        elif c == "i":
            print()
            logging.critical("***** ID {} - {} [{}]".format(config.experiment, config.job_name, config.job_id))
            logging.critical("Outputting results to {}".format(config.job_folder))
        elif c == "a":
            print("Actions stats not implemented yet.")
            #print_action_stats(actions_taken)
        elif c == "h":
            print()
            logging.critical("**** H for help")
            logging.critical("**** I to show jobname")
            logging.critical("**** T for timing stats")
            logging.critical("**** P to pause")
            logging.critical("**** Q to quit")
        elif c == "q":
            exit(-1)
        elif c == "m":
            mem = tracker.SummaryTracker()
            results = sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10]
            for obj, count, size in results:
                print("{:<50} {:>20} {:>20.1f}m".format(obj, count, size/1024/1024))
        elif c == "c":
            print("="*60)
            print("Config")
            print("=" * 60)
            for k, v in config.__dict__.items():
                if k == "args":
                    print("{}:{}".format(k, tidy_args(v)))
                else:
                    print("{}:{}".format(k, v))
            print("-" * 60)
        elif c == 'z':
            logging.critical("**** Open subprocesses: {}".format([current_processes[p.pid][0] for p in get_open_processes()]))
        elif c == 'j':
            logging.critical("Joining subprocesses: {}".format([current_processes[p.pid][0] for p in get_open_processes()]))
            join_open_processes()
            logging.critical("Done.")
        else:
            print()
            logging.critical("\nInvalid input {}.\n".format(c))


def get_final_score(health_as_reward=None):
    if health_as_reward is None:
        health_as_reward = config.health_as_reward
    if health_as_reward:
        # use integral of health over time assuming agent would have lasted the full steps.

        current_tick = data_history[-1][0]

        game_duration = game.get_episode_timeout()

        if game_duration == 0:
            # default to 2100 (60 seconds) if game has no max duration.
            game_duration = 2100

        final_score = np.mean(health_history) * (current_tick / game_duration)

    else:
        final_score = game.get_total_reward()

    return final_score


def get_player_location():
    return (
        round(game.get_game_variable(vzd.GameVariable.POSITION_X),2),
        round(game.get_game_variable(vzd.GameVariable.POSITION_Y),2),
        round(game.get_game_variable(vzd.GameVariable.ANGLE),2)
    )


def reset_agent(episode=0):
    """ Resets agent and stats to start of episode. """

    global previous_health
    global last_total_shaping_reward
    global health_history

    previous_health = 100
    last_total_shaping_reward = 0
    health_history = [100]

    if config.rand_seed is not None:
        game.set_seed(config.rand_seed+episode)  # make sure we get a different start each time.
        game.new_episode()
    else:
        game.set_seed(randint(1, 99999999))
        game.new_episode()

        # there is a bug? with vizdoom 1.1.7 where it will give the same starting location 50% of the time on
        # windows only, which makes the results / training much higher on that platform.  This forces a unique starting
        # location each time. Some maps, however, always start in the same place.

        requires_duplicate_detection = "health_gathering" in config.config_file_path

        if requires_duplicate_detection:

            counter = 0
            while get_player_location() in starting_locations and counter < 1000:
                game.set_seed(randint(1, 99999999))
                game.new_episode()
                counter += 1
            if get_player_location() in starting_locations:
                logging.critical("Warning! Location duplicate found: {}".format(get_player_location()))

        starting_locations.add(get_player_location())

    # full observation buffer with first observation.
    for k in range(config.num_stacks):
        push_state(get_observation(), get_data())


def save_video(path, frames, frame_rate=60):
    """ Saves given frames (list of np arrays) to video file. """

    folder, filename = os.path.split(path)

    logging.info("Exporting video example {}.".format(filename))

    os.makedirs(folder, exist_ok=True)

    channels, height, width = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(folder, filename), fourcc, frame_rate, (width, height))

    for frame in frames:
        # we are in CHW but want to be in HWC
        frame = np.swapaxes(frame, 0, 2) #WHC
        frame = np.swapaxes(frame, 0, 1) #HWC
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


current_processes = {}

def get_open_processes():
    """ Returns list of currently open processes. """
    open_processes = []
    for temp_file, p in current_processes.values():
        if p.poll() is None:
            open_processes.append(p)
    return open_processes

def join_open_processes():
    """ Waits for all currently open processes to finish. """
    for temp_file, p in current_processes.values():
        if p.poll() is None:
            p.wait()



def _cached_write(temp_file, destination_file, f):
    """ Executes function f, creating temp_filename then moves results to destination as a background process
        This helps performance when destination path is a networked drive.
    """

    # not supported on windows yet...
    if os.name == "nt":
        f(destination_file)
    else:
        f(temp_file)
        p = subprocess.Popen(["mv", temp_file, destination_file])
        current_processes[p.pid] = (temp_file, p)


def get_stack():
    """ Returns [1xCxHxW], [1xCxD] """
    return (
        np.concatenate(observation_history)[np.newaxis, :, :, :],
        np.concatenate(data_history)[np.newaxis, :]
    )


def get_obs():
    """ Returns a tuple containing last observation and data"""
    return (
        np.concatenate(observation_history[-1:])[np.newaxis, :, :, :],
        np.concatenate(data_history[-1:])[np.newaxis, :]
    )


def convert_frame_repeat(code):

    # we convert code to an integer if possible
    try:
        code = int(code)
    except:
        pass

    repeat = 0

    if type(code) == int:
        repeat = code
    elif code[0] == 'g':
        # gaussian
        _, mu, sigma = code.split("_")
        repeat = np.random.normal(float(mu), float(sigma))
    elif code[0] == 'f':
        # gaussian on delay instead of frame_rate
        _, mu, sigma = code.split("_")
        delay = np.random.normal(float(mu) / 35, float(sigma) / 35)  # doom runs at 35 fps
        if delay < 1e-3:  # cap to skips of 1,000
            delay = 1e-3
        repeat = round(1 / delay)
    elif code[0] == 'u':
        # uniform
        _, b, a = code.split("_")
        repeat = np.random.uniform(float(a), float(b))
    elif code[0] == 'p':
        # poisson
        _, lamb = code.split("_")
        repeat = np.random.poisson(float(lamb))
    else:
        logging.critical("Invalid format string for frame_repeat {}".format(code))
        exit()

    # make sure we take get at least 1 step forward, otherwise progress won't happen.
    repeat = round(repeat)
    if repeat < 1:
        repeat = 1

    return repeat


def eval_model(generate_video=False):

    policy_model.eval()
    test_scores = []
    actions_taken = []
    test_scores_health = []
    test_scores_reward = []
    test_scores_exploration = []

    if config.test_frame_repeat is not None:
        initialize_actions(config.test_frame_repeat)
    else:
        initialize_actions(config.frame_repeat)

    for test_episode in trange(config.test_episodes_per_epoch, leave=False):

        reset_agent(test_episode)
        step = 0

        cells_explored = set()

        frames = []
        actions_this_episode = []

        while not game.is_episode_finished():
            handle_keypress()
            step += 1

            push_state(get_observation(), get_data())

            s1, d1 = get_stack()

            best_action_index = get_best_action(s1, d1)

            actions_this_episode.append(actions[best_action_index])

            reward = env_step(*actions[best_action_index])

            x,y,z = \
                game.get_game_variable(vzd.GameVariable.POSITION_X), \
                game.get_game_variable(vzd.GameVariable.POSITION_Y), \
                game.get_game_variable(vzd.GameVariable.POSITION_Z)  \

            # break map into 4x4 cells
            CELL_SIZE = 4
            cells_explored.add((int(x/CELL_SIZE), int(y/CELL_SIZE), int(z/CELL_SIZE)))

            health_history.append(game.get_game_variable(vzd.GameVariable.HEALTH))

            if not game.is_episode_finished():
                img = game.get_state().screen_buffer
                img = np.swapaxes(img, 0, 2)
                img = cv2.resize(np.float32(img) / 255, dsize=(480,640), interpolation=cv2.INTER_NEAREST)
                img = np.swapaxes(img, 0, 2)
                img = np.uint8(img * 255)
                if generate_video:
                    frames.append(img)

        # make sure to record both the reward score, and the health as reward score.
        test_scores.append(get_final_score())
        actions_taken.append(actions_this_episode[:])
        test_scores_health.append(get_final_score(health_as_reward=True))
        test_scores_reward.append(get_final_score(health_as_reward=False))
        test_scores_exploration.append(len(cells_explored))

        if generate_video:
            save_video("./example-{}-{}-{}.mp4".format(config.job_id, test_episode, platform.node()), frames, frame_rate=6)

    return np.array(test_scores), np.array(test_scores_health), np.array(test_scores_reward), np.array(test_scores_exploration), actions_taken


def export_video(epoch):

    global game
    global game_hq

    # activate game_hq
    game, game_hq = game_hq, game

    policy_model.eval()
    frames = []

    reset_agent(0)
    step = 0
    best_action_index = 0
    best_action, best_skip = actions[best_action_index]

    frame_repeat_cooldown = 0

    while not game.is_episode_finished():

        if frame_repeat_cooldown <= 0:
            # only make decisions at the correct frame rate
            push_state(get_observation(), get_data())
            s, d = get_stack()
            best_action_index = get_best_action(s, d)
            best_action, best_skip = actions[best_action_index]
            frame_repeat_cooldown = convert_frame_repeat(best_skip)

        # we generate all frames for smooth video, even though
        # actions may stick for multiple frames.
        frames.append(game.get_state().screen_buffer)

        # show how long before next decision
        frames[-1][:, 9:20, 10:10 + frame_repeat_cooldown] = 255
        frames[-1][:, 10:19, 11:9 + frame_repeat_cooldown] = 0

        _ = game.make_action(best_action, 1)
        step += 1
        frame_repeat_cooldown -= 1

    # activate game again...
    game, game_hq = game_hq, game

    try:
        destination_file = os.path.join(config.job_folder, "videos", "epoch-{0:03d}.mp4".format(epoch))
        temp_file = os.path.join("temp", "{}-video-{:03d}.mp4".format(config.job_id,epoch if epoch is not None else ""))
        _cached_write(temp_file, destination_file, lambda x: save_video(x, frames, frame_rate=35))
    except Exception as e:
        logging.critical("Error saving video: {}".format(e))


def get_optimizer():
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop_centered":
        return torch.optim.RMSprop(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                            centered=True)
    elif config.optimizer == "adam":
        return torch.optim.Adam(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise Exception("Invalid optimizer name {}".format(config.optimizer))


def print_action_stats(actions_taken):
    """ Prints stats on actions taken by agent.
    :param actions_taken: List of lists of tuple (button presses, frame_skip)
    """

    button_history = []
    frame_skip_history = []

    for episode_actions in actions_taken:
        for buttons, frame_skip in episode_actions:
            button_history.append(tuple(buttons))
            frame_skip_history.append(frame_skip)

    # print button combinations
    button_options = set(tuple(buttons) for buttons, frame_skip in actions)
    frame_skip_options = set(frame_skip for buttons, frame_skip in actions)

    for button in button_options:
        percent = button_history.count(button) / len(button_history) * 100
        logging.critical("{:<20} {:.1f}%".format(str(button), percent))
    if len(frame_skip_options) > 1:
        for frame_skip in frame_skip_options:
            percent = frame_skip_history.count(frame_skip) / len(frame_skip_history) * 100
            logging.critical("{:<20} {:.1f}%".format(str(frame_skip), percent))




def train_agent(continue_from_save=False):
    """ Run a test with given parameters, returns stats in dictionary. """


    logging.critical("=" * 60)
    logging.critical("Running Experiment {} {} [{}]".format(config.experiment, config.job_name, config.job_id))
    logging.critical("=" * 60)

    global actions
    global memory
    global feature_cache

    # setup doom
    initialize_vizdoom()

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=config.replay_memory_size)
    feature_cache = NoveltyMemory(capacity=config.replay_memory_size//10)

    # create a file showing the run parameters
    with open(os.path.join(config.job_folder, "config.txt"), "w") as f:
        f.write("{\n")
        for k, v in config.__dict__.items():
            if k == "args":
                f.write("\t{}:{},\n".format(k, tidy_args(v)))
            else:
                f.write("\t{}:{},\n".format(k, v))
        f.write("}")

    # make a copy of the script that generated the results
    this_script = sys.argv[0]
    shutil.copy(this_script, config.job_folder)

    results = {}
    results["test_scores"] = []
    results["test_scores_health"] = []
    results["test_scores_reward"] = []
    results["test_scores_mean"] = []
    results["test_scores_exploration"] = []
    results["test_actions"] = []
    results["config"] = config
    results["args"] = config.args

    global target_model
    global policy_model

    start_epoch = 0

    if continue_from_save:
        # load config
        pass
        # load model
        pass
        # load results
        # set globals (learning_step)
        # also set start_epoch
        pass
    else:
        target_model = get_net(len(actions))
        policy_model = get_net(len(actions))

    if config.device.lower()[:4] == "cuda":
        for model in [target_model, policy_model]:
            device_str = config.device.split(":")
            if len(device_str) == 2:
                device_id = int(device_str[1])
            else:
                device_id = 0

            model.cuda(device_id)

    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    logging.debug("Actions: {}".format(actions))

    global optimizer

    optimizer = get_optimizer()

    time_start = time()

    passed_gates = set()
    failed_gate = False

    for epoch in range(start_epoch, config.epochs):

        # clear timing stats
        global time_stats
        time_stats = {}

        global max_q
        global max_grad

        max_q = 0
        max_grad = 0

        logging.critical("------------- Epoch {}/{} (eps={:.3f}) -------------".format(
            epoch + 1, config.epochs,
            exploration_rate(epoch*config.learning_steps_per_epoch))
        )
        train_episodes_finished = 0
        total_train_episodes_finished = 0
        train_scores = []

        logging.info("Training...")
        policy_model.train()

        reset_agent(epoch)

        for learning_step in trange(config.learning_steps_per_epoch, leave=False):

            handle_keypress()

            step = learning_step + epoch*config.learning_steps_per_epoch

            perform_environment_step(step)

            if step >= int(config.first_update_step * config.update_every):
                if config.update_every < 1:
                    for i in range(int(1 / config.update_every)):
                        perform_learning_step()
                else:
                    if step % int(config.update_every) == 0:
                        perform_learning_step()

            if game.is_episode_finished():
                score = get_final_score()
                train_scores.append(score)
                reset_agent(total_train_episodes_finished)
                train_episodes_finished += 1
                total_train_episodes_finished += 1

        logging.info("\tTraining episodes played: {}".format(train_episodes_finished))

        train_scores = np.array(train_scores)

        logging.info("\tResults: mean: {:.1f} +/- {:.1f} [min:{:.1f} max:{:.1f}]".format(
            train_scores.mean(),
            train_scores.std(),
            train_scores.min(),
            train_scores.max()
        ))

        show_time_stats()

        if config.export_video and (((epoch+1) % 25 == 0 or epoch == config.epochs-1 or epoch == 0)):
            logging.info("Exporting video...")
            export_video(epoch+1)

        logging.info("Testing...")
        test_scores, test_scores_health, test_scores_reward, test_scores_exploration, actions_taken = eval_model()

        # -----------------------------------------------------------------------------
        # logging...
        # -----------------------------------------------------------------------------

        if max_q > 10000:
            logging.warning("")
            logging.warning("******* Warning MaxQ was too high ***************")
            logging.warning("MaxQ: {:.2f}".format(max_q))
            logging.warning("*************************************************")
            logging.warning("")
        else:
            logging.info("\tMaxQ: {:.2f}".format(max_q))

        if max_grad > 10000:
            logging.warning("")
            logging.warning("******* Warning MaxGrad was too high ***************")
            logging.warning("MaxGrad: {:.2f}".format(max_grad))
            logging.warning("*************************************************")
            logging.warning("")
        else:
            logging.info("\tMaxGrad: {:.2f}".format(max_grad))

        test_scores = np.array(test_scores)

        logging.info("\tResults: mean: {:.1f} +/- {:.1f} [min:{:.1f} max:{:.1f}]".format(
            test_scores.mean(),
            test_scores.std(),
            test_scores.min(),
            test_scores.max()
        ))

        results["test_scores_mean"].append(test_scores.mean())
        results["test_scores"].append(test_scores)
        results["test_scores_health"].append(test_scores_health)
        results["test_scores_reward"].append(test_scores_reward)
        results["test_scores_exploration"].append(test_scores_exploration)
        results["test_actions"].append(actions_taken)

        elapsed_time = (time() - time_start)

        logging.info("\tTotal elapsed time: {:.2f} min".format(elapsed_time / 60.0))

        progress = ((epoch+1) / config.epochs)

        est_total_time = elapsed_time / progress
        est_remaining_time = est_total_time - elapsed_time

        logging.critical("Estimated remaining time: {:.0f} min ({:.2f}h total)".format(est_remaining_time/60, est_total_time/60/60))

        logging.critical("Scores: {}".format([round(x,2) for x in results["test_scores_mean"]]))
        logging.critical("Exploration: {}".format([round(np.mean(x), 2) for x in results["test_scores_exploration"]]))

        results["elapsed_time"] = ((time() - time_start) / 60.0)

        save_results(results,"_partial")
        save_model(epoch+1)

        if args.gate_epoch is not None and args.gate_score is not None:
            for req_gate, req_score in zip(config.gate_epoch, config.gate_score):
                if req_gate not in passed_gates and epoch >= req_gate:
                    avg_score = np.mean(np.mean(results["test_scores_reward"], axis=1)[-3:])
                    if avg_score < req_score:
                        logging.critical(
                            "Agent has not performed well enough to continue.  Reward at epoch {} is {:.1f} but needed to be {:.0f}".format(
                                epoch , avg_score, req_score))
                        failed_gate = True
                        break
                    else:
                        logging.critical(
                            "Agent passed gate at epoch {} with {:.1f} / {:.0f}".format(
                                epoch, avg_score, req_score))
                        passed_gates.add(req_gate)

        if failed_gate:
            break


    save_results(results, "_complete")
    save_model()

    game.close()

    # this stops all logging, and released the log.txt file allowing the
    # folder to be renamed.
    for log in logging.getLogger().handlers:
        log.close()
        logging.getLogger().removeHandler(log)

    # wait for processes to finish
    if len(get_open_processes()) > 0:
        print("Waiting for processes:",end='')
    while len(get_open_processes()) > 0:
        print(format([current_processes[p.pid][0] for p in get_open_processes()]))
        sleep(30)

    if config.mode == "train":
        sleep(10)  # give Dropbox a chance to sync up, and logs etc to finish up.
        config.rename_job_folder()

    return results


def save_results(results, suffix=""):
    # save raw results to a pickle file for processing

    # make sure we are not currently saving an old file
    while True:
        processes = [current_processes[p.pid][0] for p in get_open_processes()]
        waiting_on_results = False
        for process in processes:
            if "results" in process:
                waiting_on_results = True
        if not waiting_on_results:
            break
        logging.critical("Waiting on results file to write out.")
        sleep(30)


    temp_file = os.path.join("temp","results-{}.tmp".format(config.job_id))
    destination_file = os.path.join(config.job_folder, "results"+suffix+".dat")

    _cached_write(temp_file, destination_file, lambda x: pickle.dump(results, open(x, "wb")))

    with open(os.path.join(config.job_folder, "results"+suffix+".txt"), "w") as f:
        f.write(str(results["test_scores_mean"]))

    generate_graphs(results)


@track_time_taken
def save_model(epoch=None):

    os.makedirs(os.path.join(config.job_folder, "models"), exist_ok=True)

    if epoch is None:
        destination_file = os.path.join(config.job_folder, "model_complete.dat")
    else:
        filename = "model_{0:03d}.dat".format(epoch)
        destination_file = os.path.join(config.job_folder, "models", filename)

    temp_file = os.path.join("temp","{}-{}.tmp".format(config.job_id, epoch if epoch is not None else ""))

    os.makedirs("temp", exist_ok=True)

    # copy the file across
    _cached_write(temp_file, destination_file, lambda x: torch.save(policy_model, x))

def restore_model(epoch=None):
    """ restores model from checkpoint. """
    if epoch is None:
        model_path = os.path.join(config.job_folder, "model_complete.dat")
    else:
        filename = "model_{0:03d}.dat".format(epoch)
        model_path = os.path.join(config.job_folder, "models", filename)

    global policy_model
    global target_model

    target_model = get_net(len(actions))
    policy_model = get_net(len(actions))

    policy_model = torch.load(model_path, map_location=config.device)
    target_model = torch.load(model_path, map_location=config.device)

    # upgrade older save files
    for model in [policy_model, target_model]:
        # if the model was saved in an earlyer version of pytorch it won't have these set
        # so set them here
        if "padding_mode" not in model.conv1.__dict__.keys():
            model.conv1.padding_mode = 'constant'
            model.conv2.padding_mode = 'constant'
            model.conv3.padding_mode = 'constant'
            model.conv4.padding_mode = 'constant'

    if config.device.lower() == "cuda":
        for model in [target_model, policy_model]:
            model.cuda()

    target_model.eval()


def smooth(X, epsilon=0.9):
    if X == []:
        return X
    y = X[0]
    result = []
    for x in X:
        y = y * epsilon + x * (1-epsilon)
        result.append(y)
    return result


def get_best_epoch(results):
    """ Load the model with the best performance, returns epoch loaded. """
    smooth_scores = smooth(results["test_scores_mean"], 0.8)
    best_epoch = np.argmax(smooth_scores)
    return best_epoch


def run_eval():
    """ Evaluate the model the best model with given settings. """

    global config

    override_test_frame_repeat = config.test_frame_repeat
    override_device = config.device
    override_test_episodes_per_epoch = config.test_episodes_per_epoch
    override_rand_seed = config.rand_seed
    override_eval_results_suffix = config.eval_results_suffix
    override_output_path = config.output_path

    config_filename = os.path.join(config.job_folder, "results_partial.dat")
    results = pickle.load(open(config_filename, "rb"))
    config = results["config"]

    # copy across the defined test_frame_repeat (if defined)
    if override_test_frame_repeat is not None:
        config.test_frame_repeat = override_test_frame_repeat

    if config.test_episodes_per_epoch is not None:
        config.test_episodes_per_epoch = override_test_episodes_per_epoch

    config.rand_seed = override_rand_seed

    # older files will not have these config variables so put defaults in here.
    for param_name, param_default in [
        ("dynamic_frame_repeat", False),
        ("dfr_decision_cost", 0.0),
        ("test_frame_repeat", None),
        ("use_color", True),
        ("max_simultaneous_actions", None),
        ("model", "basic"),
        ("include_xy", False)]:
        if param_name not in config.__dict__.keys():
            config.__dict__[param_name] = param_default



    config.job_name = config.job_name.strip()

    # make sure we use the correct device.
    config.device = override_device

    config.output_path = override_output_path

    # put results suffix on.
    config.eval_results_suffix = override_eval_results_suffix

    config.mode = "eval"

    # initialize vizdoom
    initialize_vizdoom()

    # load the best model
    best_epoch = get_best_epoch(results)

    logging.critical("=" * 100)
    logging.critical("Evaluating Experiment {} {} [{}] - epoch {}".format(config.experiment, config.job_name, config.job_id, best_epoch))
    logging.critical("=" * 100)

    print("Using testing frame skip: {}".format(config.test_frame_repeat))

    restore_model(best_epoch)

    results = {}
    results["test_scores"] = []
    results["test_scores_health"] = []
    results["test_scores_reward"] = []
    results["test_scores_mean"] = []
    results["test_scores_exploration"] = []
    results["test_actions"] = []
    results["config"] = config
    results["args"] = config.args

    test_scores, test_scores_health, test_scores_reward, test_actions, test_scores_exploration = eval_model()

    results["test_scores_mean"].append(test_scores.mean())
    results["test_scores"].append(test_scores)
    results["test_scores_health"].append(test_scores_health)
    results["test_scores_reward"].append(test_scores_reward)
    results["test_actions"].append(test_actions)
    results["test_scores_exploration"].append(test_scores_exploration)
    results["best_epoch"] = best_epoch

    logging.critical("Scores: {}".format([round(x, 2) for x in results["test_scores_mean"]]))

    save_results(results, config.eval_results_suffix)


def run_full_eval():
    """ (re)runs model evaluations. """

    # get config settings from previous folder
    global config
    config_filename = os.path.join(config.job_folder, "results_partial.dat")
    config = pickle.load(open(config_filename,"rb"))["config"]

    # old config files didn't have job_id as an attribute...
    if 'job_id' not in config.__dict__:
        config.job_id = config.uid[:16]

    config.mode = "eval"

    initialize_vizdoom()

    results = {}
    results["test_scores"] = []
    results["test_scores_health"] = []
    results["test_scores_reward"] = []
    results["test_scores_mean"] = []
    results["test_actions"] = []
    results["config"] = config
    results["args"] = config.args

    time_start = time()

    for epoch in range(config.epochs):
        restore_model(epoch+1)
        test_scores, test_scores_health, test_scores_reward, test_actions = eval_model()

        results["test_scores_mean"].append(test_scores.mean())
        results["test_scores"].append(test_scores)
        results["test_scores_health"].append(test_scores_health)
        results["test_scores_reward"].append(test_scores_reward)
        results["test_actions"].append(test_actions)

        elapsed_time = (time() - time_start)

        logging.info("\tTotal elapsed time: {:.2f} min".format(elapsed_time / 60.0))

        progress = ((epoch+1) / config.epochs)

        est_total_time = elapsed_time / progress
        est_remaining_time = est_total_time - elapsed_time

        logging.critical("Estimated remaining time: {:.0f} min ({:.2f}h total)".format(est_remaining_time/60, est_total_time/60/60))

        logging.critical("Scores: {}".format([round(x,2) for x in results["test_scores_mean"]]))

        results["elapsed_time"] = ((time() - time_start) / 60.0)

        save_results(results,"_partial")

    save_results(results, "_complete")


def run_benchmark():
    """ Runs a standard benchmark to see how fast learning happens. """

    # set test settings
    config.num_stacks = 1
    config.epochs = 1
    config.learning_rate = 0.001 # basic needs a much faster learning rate because of small rewards.
    config.discount_factor = 1
    config.learning_steps_per_epoch = 1000
    config.test_episodes_per_epoch = 1
    config.replay_memory_size = 10000
    config.end_eps = 0.0
    config.start_eps = 0.0
    config.hidden_units = 1024
    config.update_every = 1
    config.batch_size = 64
    config.target_update = 1000
    config.first_update_step = 0
    config.frame_repeat = 10
    config.verbose=False
    config.config_file_path = "scenarios/basic.cfg"
    config.export_video = False

    # apply any custom arguments
    config.apply_args(args)

    start_time = time()
    train_agent()
    total_time_taken = time() - start_time

    step_time = total_time_taken / config.total_steps

    show_time_stats(logging.CRITICAL)

    logging.log(logging.CRITICAL, "Took {:.1f} seconds at rate of {:.1f} steps / second.".format(total_time_taken, 1 / step_time))


def run_simple_test(args):
    """ Runs a simple test to make sure model is able to learn the basic environment. """

    # set test settings
    config.num_stacks = 1
    config.epochs = 10
    config.learning_rate = 0.001 # basic needs a much faster learning rate because of small rewards.
    config.discount_factor = 1
    config.learning_steps_per_epoch = 500
    config.test_episodes_per_epoch = 20
    config.replay_memory_size = 10000
    config.end_eps = 0.1
    config.start_eps = 1.0
    config.hidden_units = 128
    config.update_every = 1
    config.batch_size = 64
    config.target_update = 100
    config.first_update_step = 100
    config.frame_repeat = 10
    config.verbose = False
    config.config_file_path = "scenarios/basic.cfg"
    config.health_as_reward = False

    # apply any custom arguments
    config.apply_args(args)

    result = train_agent()

    avg_score = np.mean(result["test_scores_mean"][5:])

    print("Average score for last half of training was {:.2f}".format(avg_score))

    # 80-90 is reasonable, but we pass 70 or above.
    if avg_score >= 70:
        print("Test passed.")
        exit(0)
    else:
        print("Test failed!")
        exit(-1)


def generate_graphs(data):
    """ Generates graph from results file. """

    ys = np.mean(data["test_scores"], axis=1)
    xs = range(len(ys))
    plt.plot(xs, ys, label="mean")

    ys = np.max(data["test_scores"], axis=1)
    xs = range(len(ys))
    plt.plot(xs, ys, label="max")

    ys = np.min(data["test_scores"], axis=1)
    xs = range(len(ys))
    plt.plot(xs, ys, label="min")

    plt.savefig(os.path.join(config.job_folder, "training.png"))


def enable_logging():
    global console_logger
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO if config.verbose else logging.ERROR)
    logging.getLogger().addHandler(console_logger)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_to_int_array(v):
    try:
        return [int(x) for x in ast.literal_eval(v)]
    except:
        raise argparse.ArgumentTypeError('Array of integers expected.')

def str_to_float_array(v):
    try:
        return [float(x) for x in ast.literal_eval(v)]
    except:
        raise argparse.ArgumentTypeError('Array of floats expected.')


def get_default_argument(argument):

    default = None

    # first some default values
    if argument == "optimizer":
        default = "rmsprop"
    elif argument == "output_path":
        default = "runs"
    elif argument == "include_xy":
        default = False

    # check ini file override
    if config.hostname in ini_file:
        if argument in ini_file[config.hostname]:
            default = safe_cast(ini_file[config.hostname][argument])

    return default


if __name__ == '__main__':

    config = Config()

    ini_file = configparser.ConfigParser()
    ini_file.read("config.ini")


    # handle parameters
    parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
    parser.add_argument('mode', type=str, help='train | test | benchmark | info | eval | full_eval')
    parser.add_argument('--num_stacks', type=int, help='Agent is shown this number of frames of history.')
    parser.add_argument('--use_color', type=str2bool, help='Enables color model.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    parser.add_argument('--discount_factor', type=float, help='Discount factor.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for.')
    parser.add_argument('--learning_steps_per_epoch', type=int, help='Number of environment steps per epoch.')
    parser.add_argument('--end_eps', type=float, help='Final epsilon rate.')
    parser.add_argument('--end_eps_step', type=float, help='Step after which epsilon will decay to end epsilon.')
    parser.add_argument('--replay_memory_size', type=int, help='Number of environment steps per epoch.')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in model.')
    parser.add_argument('--batch_size', type=int, help='Samples per minibatch.')
    parser.add_argument('--device', type=str, help='CPU | CUDA')
    parser.add_argument('--verbose', type=str2bool, help='enable verbose output')
    parser.add_argument('--update_every', type=float, help='apply update every x learning steps')
    parser.add_argument('--experiment', type=str, help='name of subfolder to put experiment in.')
    parser.add_argument('--job_name', type=str, help='name of job.')
    parser.add_argument('--target_update', type=int, help='how often to update target network')
    parser.add_argument('--test_episodes_per_epoch', type=int, help='how many tests epsodes to run each epoch')
    parser.add_argument('--config_file_path', type=str, help="config file to use for VizDoom.")
    parser.add_argument('--health_as_reward', type=str2bool, help="use change in health as reward instead of default reward.")
    parser.add_argument('--frame_repeat', type=str, help="number of frames to skip.")
    parser.add_argument('--test_frame_repeat', type=str, help="number of frames to skip during testing.")
    parser.add_argument('--include_aux_rewards', type=str2bool, help="use auxualry reward during training (these are not counted during evaluation).")
    parser.add_argument('--export_video', type=str2bool, help="exports one video per epoch showing agents performance.")
    parser.add_argument('--job_id', type=str, help="unique id for job.")
    parser.add_argument('--max_pool', type=str2bool, help="enable maxpooling.")
    parser.add_argument('--terminate_early', type=str2bool, help="agent stops training if progress has not been made.")
    parser.add_argument('--agent_mode', type=str, help="default | random | stationary")
    parser.add_argument('--rand_seed', type=int, help="random seed for environment initialization")
    parser.add_argument('--threads', type=int, help="Number of threads to use during training")
    parser.add_argument('--eval_results_suffix', type=str, default="", help="Filename suffix for evaluation results.")
    parser.add_argument('--model', type=str, default="basic", help="Name of model to use basic | dual")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay for optimizer")
    parser.add_argument('--optimizer', type=str, default=get_default_argument("optimizer"), help="adam | rmsprop | rmsprop_centered")
    parser.add_argument('--include_xy', type=str2bool, default=get_default_argument("include_xy"), help="if true includes xy location as a channel.")
    parser.add_argument('--output_path', type=str, default=get_default_argument("output_path"), help="path to store experiment results.")
    parser.add_argument('--dynamic_frame_repeat', type=str2bool, help="Enables dynamic frame repeating. ")
    parser.add_argument('--dfr_decision_cost', type=float, default=0.0, help="Cost per decision for dynamic frame repeat. Encourages taking larger frame skips.")
    parser.add_argument('--max_simultaneous_actions', type=int, default=4, help="Maximum number of buttons agent can push at a time.")
    parser.add_argument('--cuda_device', type=int, help="id of CUDA device to use.")
    parser.add_argument('--weighted_random_actions', type=str2bool, help="Weights random action sampling by 1/frame_repeat for each action.")
    parser.add_argument('--gate_epoch', type=str_to_int_array, help="list of epochs to test score at.")
    parser.add_argument('--gate_score', type=str_to_float_array, help="list of minimum score at corresponding epoch to pass gate.")
    parser.add_argument('--gradient_clip', type=float, default=0, help="enable gradient clipping.")
    parser.add_argument('--novelty', type=float, default=0.0, help="novelty reward, 0 to disable.")
    parser.add_argument('--id_factor', type=float, default=0.0, help="inverse dynamics factor.")

    args = parser.parse_args()

    config.mode = config.mode.lower()

    config.apply_args(args)

    if not os.path.exists(config.output_path):
        raise Exception("Output folder {} not found, please create it.".format(config.output_path))

    config.make_job_folder()

    log_name = "log.txt" if args.mode in ["training", "benchmark", "test"] else args.mode+".txt"

    # setup logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(config.job_folder,log_name),
                        filemode='w')

    enable_logging()


    logging.info("Using device: {}".format(config.device))

    if args.cuda_device is not None:
        logging.critical(" - using cuda device {}".format(args.cuda_device))
        torch.cuda.device(args.cuda_device)


    # apply mode
    if config.mode == "full_eval":
        run_full_eval()
    if config.mode == "eval":
        run_eval()
    elif config.mode == "train":
        train_agent()
    elif config.mode == "continue":
        train_agent(continue_from_save=True)
    elif config.mode == "info":
        print("Hostname:", config.hostname)
        print("Python:", sys.version)
        print("PyTorch:", torch.__version__)
        print("ViZDoom:", vzd.__version__)
        print("Device:", config.device)
        np.__config__.show()
    elif config.mode == "benchmark":
        run_benchmark()
    elif config.mode == "test":
        run_simple_test(args)
    else:
        print("Invalid mode {}.".format(config.mode))
