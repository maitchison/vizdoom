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
frame_repeat = 5
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = False
load_model = False
skip_learning = False

#screen_resolution = ScreenResolution.RES_640X480
#screen_resolution = ScreenResolution.RES_320X240
screen_resolution = ScreenResolution.RES_160X120 # this is the lowest resolution we can go

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

# Configuration file path
config_file_path = "scenarios/simpler_basic.cfg"


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
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


def learn(s1, target_q):
    s1 = torch.from_numpy(s1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1), Variable(target_q)
    output = model(s1)
    loss = criterion(output, target_q)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def get_q_values(state):
    state = torch.from_numpy(state)
    state = Variable(state)
    return model(state)


def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q = get_q_values(s2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


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

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(screen_resolution)
    game.init()
    print("Doom initialized.")
    return game


def run_test(**kwargs):
    """ Run a test with given parameters, returns stats in dictionary. """

    results = {}
    results["test_scores"] = []
    results["test_scores_mean"] = []
    results["params"] = kwargs

    for k,v in kwargs.items():
        globals()[k] = v

    global model
    model = Net(len(actions))
    if device.lower() == "cuda":
        model.cuda()

    global optimizer
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    print("-"*60)
    print("Parameters ",kwargs)
    print("-" * 60)

    time_start = time()

    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        game.new_episode()
        for learning_step in trange(learning_steps_per_epoch, leave=False):
            perform_learning_step(epoch)
            if game.is_episode_finished():
                score = game.get_total_reward()
                train_scores.append(score)
                game.new_episode()
                train_episodes_finished += 1

        print("\n\t%d training episodes played." % train_episodes_finished)

        train_scores = np.array(train_scores)

        print("\tResults: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        print("\nTesting...")
        test_episode = []
        test_scores = []
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            while not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                state = state.reshape([1, 1, resolution[0], resolution[1]])
                best_action_index = get_best_action(state)

                game.make_action(actions[best_action_index], frame_repeat)
            r = game.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
        print("\n\tResults: mean: %.1f +/- %.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
              "max: %.1f" % test_scores.max())

        results["test_scores_mean"].append(test_scores.mean())
        results["test_scores"].append(test_scores)

        if save_model:
            print("Saving the network weigths to:", model_savefile)
            torch.save(model, model_savefile)

        print("\tTotal elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        print("\nScores:", results["test_scores_mean"],"\n")

        results["elapsed_time"] = ((time() - time_start) / 60.0)

    return results

def save_results(results):

    # create an easy to read CVS file

    if not os.path.isfile("results.csv"):
        with open("results.csv", "w") as f:
            f.write("learning_rate, discount_factor, replay_memory_size, end_epsilon, batch_size, frame_repeat, time (min), epochs\n")

    with open("results.csv", "a") as f:
        f.write(
            str(results["params"].get("learning_rate", learning_rate))+"," +
            str(results["params"].get("discount_factor", discount_factor)) + "," +
            str(results["params"].get("replay_memory_size", replay_memory_size)) + "," +
            str(results["params"].get("end_epsilon", end_epsilon)) + "," +
            str(results["params"].get("batch_size", batch_size)) + "," +
            str(results["params"].get("frame_repeat", frame_repeat)) + "," +
            str(results["elapsed_time"])+","+
            ",".join(str(x) for x in results["test_scores_mean"]) + "\n"
        )

    # save raw results to a pickle file for processing
    try:
        db = pickle.load(open("results.dat","rb"))
    except FileNotFoundError:
        db = []
    db.append(results)
    pickle.dump(db, open("results.dat","wb"))

if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    # random sample from reasonable values.
    tests = []
    for _ in range(1000):
        test = {}
        test["learning_rate"] = np.random.choice(10**np.linspace(-2,-6,100))
        test["discount_factor"] = np.random.choice(1 - 10 ** np.linspace(-1, -3, 100))
        test["replay_memory_size"] = np.random.choice(10 ** np.linspace(2, 5, 100))
        test["end_epsilon"] = np.random.choice(10 ** np.linspace(0, -3, 100))
        test["batch_size"] = np.random.choice([16,32,64])
        test["frame_repeat"] = np.random.choice(list(range(1,40)))

        tests.append(test)

    for test_params in tests:

        try:
            result = run_test(**test_params)
            save_results(result)
        except Exception as e:
            print("********** Test failed....")
            print("Params:",test_params)
            print("error:", e)

    game.close()
