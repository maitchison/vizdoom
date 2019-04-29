#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# E. Culurciello
# August 2017

# todo:
# add momentium
# add adam
# different models...
# more resolution..
# check what game actually looks like
# better network
# test effect of screen resolution
# why is testing slow sometimes... oh.. it's frame repeat...


# a multi dimensional optimizer graph. bin the hyperparemters into 5 and graph the average of the best 5?

from vizdoom import *
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
import os.path
import matplotlib.pyplot as plt

epochs = 20

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
learning_steps_per_epoch = 2000
replay_memory_size = 10000
end_epsilon = 0.1

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 10
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = False
load_model = False
skip_learning = False

#screen_resolution = ScreenResolution.RES_640X480
#screen_resolution = ScreenResolution.RES_320X240
screen_resolution = ScreenResolution.RES_160X120 # this is the lowest resolution we can go

#screen_format = ScreenFormat.GRAY8
screen_format = ScreenFormat.CRCGCB
#screen_format = ScreenFormat.RGB24

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

# Configuration file path
#config_file_path = "scenarios/simpler_basic.cfg"
config_file_path = "scenarios/health_gathering.cfg"
#config_file_path = "scenarios/health_gathering_supreme.cfg"


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution, anti_aliasing=False)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x).cpu()


class __Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64*2*4, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*2*4)
        x = F.relu(self.fc1(x))
        return self.fc2(x).cpu()


criterion = nn.MSELoss()

def get_q_values(state):
    state = torch.from_numpy(state)
    state = Variable(state)
    return model(state)


def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action

def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = end_epsilon
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(screen_format)
    game.set_screen_resolution(screen_resolution)
    game.init()
    print("Doom initialized.")
    return game


def run_benchmark():
    """ Run a test with given parameters, returns stats in dictionary. """

    print("Get VizDoom step rate (single CPU)")

    time_start = time()
    test_iterations = 5000

    times = []

    game.new_episode()
    for i in range(test_iterations):
        x = game.get_state().screen_buffer

        a = randint(0, len(actions) - 1)

        st = time()

        _ = game.make_action(actions[a], 100)

        times.append(time()-st)

        if game.is_episode_finished():
            game.new_episode()

    avg_time = (time()-time_start) / test_iterations

    times = [x * 1000 for x in times]

    print("Environment runs at {:.1f} FPS".format(1/avg_time))
    print("Step time is min: {:.2f} max: {:.2f} mean: {:.2f} median: {:.2f} ms".format(min(times), max(times), np.mean(times), np.median(times)))

    plt.hist(times, bins=50)
    plt.show()
    plt.plot(range(len(times)), times)
    plt.show()

    pickle.dump(times, open("times.dat","wb"))


    return

    print("Running learning Benchmark")
    train_episodes_finished = 0
    train_scores = []

    time_start = time()

    print("Training...")
    game.new_episode()
    for learning_step in trange(learning_steps_per_epoch, leave=False):
        perform_learning_step(0)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1

    print("\n\t%d training episodes played." % train_episodes_finished)

    train_scores = np.array(train_scores)

    print("\tResults: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    print("\tTotal elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))




if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    # random sample from reasonable values.
    tests = [{'epochs':1}]

    for test_params in tests:

        try:
            result = run_benchmark()

        except Exception as e:
            print("error:", e)

    game.close()