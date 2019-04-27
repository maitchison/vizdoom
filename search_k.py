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


"""
Search for best update / experience ratio

On Basic: (maybe this is too easy, but will give me an idea of learning speed...)
"""

"""

Try to learn navigation:
[*] Start with very simple agent and get a baseline (i.e. the default model)
[*] Make basic speed optimizations
[ ] Create an an agent with 3 layers and larger retina
[ ] Add LSTM?
[ ] Look for grid cells
[ ] Try periodic LSTM
[ ] Copy model from vector-based paper, see if grid-cells develop.

Also
[ ] create video replays to see what's happening
[ ] look for other benchmarks on this map (looks like A2C gets ~100 after 10 million itterations.


Performance:

Speed numbers:
Initial: 30-40 it/s
No AA: 60 it/s

This is still too slow... profile. (might be skip size?)
Maybe run forward models on CPU? as they will be batch of 1? Ah... probably not, still to slow right? as model gets more
complicated.  Alternatively run multiple threads collecting env steps

try: multi core env?

Wasn't there a paper about predicting the future but only the parts of the future that our actions influence

The way home is interesting as rewards a very rare, so learning the dynamics of the system could help a lot!

Turns out adam was unstable... maybe learning rate was just too high...

Test:
[*] switch to color
[ ] make sure color works with basic
[ ] switch on batch norm
[ ] see if color helps way home (but train for 50 epochs...
[ ]

"""

from vizdoom import *
import vizdoom as vzd
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

SHOW_REWARDS = False
SHOW_MAXQ = False

epochs = 10

# Q-learning settings
learning_rate = 0.0003
discount_factor = 0.99
learning_steps_per_epoch = 200 # we probably want 10 million steps, let's make an epoch a 100,000 steps, so 100 epochs
replay_memory_size = 10000
end_epsilon = 0.02
update_every = 1 # perform gradient descent every k steps.  The number of times each env sample is used is
                 # batch_size / update_every.

# NN learning settings
batch_size = 32

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (84, 84)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = False
load_model = False
skip_learning = False

prev_loss = 0
prev_max_q = 0
bad_q = False

# this is the lowest resolution we can go, and should be fine.
screen_resolution = ScreenResolution.RES_160X120

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

# Configuration file path
config_file_path = "scenarios/basic.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = np.swapaxes(img, 0, 2)
    img = skimage.transform.resize(img, resolution, anti_aliasing=False)
    img = np.swapaxes(img, 0, 2)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 3
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
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def prod(X):
    y = 1
    for x in X:
        y *= x
    return y

class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(prod([64,7,7]), 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = x.to(device)
        #x = F.relu(self.conv1_bn(self.conv1(x)))
        #x = F.relu(self.conv2_bn(self.conv2(x)))
        #x = F.relu(self.conv3_bn(self.conv3(x))) # we are now Bx64x7x7

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # we are now Bx64x7x7
        x = x.view(-1, prod(x.shape[1:]))
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
    global bad_q
    state = torch.from_numpy(state)
    state = Variable(state)
    q = model(state)
    max_q = float(torch.max(q))
    if max_q > 1000 and not bad_q:
        print("Error MaxQ = {}".format(max_q))
        bad_q = True
    return q


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
        this_loss = learn(s1, target_q)
        global prev_loss
        prev_loss = 0.95 * prev_loss + 0.05 * float(this_loss)

def exploration_rate(epoch):
    """# Define exploration rate change over time"""
    start_eps = 0.5
    end_eps = end_epsilon
    const_eps_epochs = 0
    eps_decay_epochs = min(0.5 * epochs, 10)

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps


def perform_learning_step(epoch, step):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""


    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 3, resolution[0], resolution[1]])
        a = get_best_action(s1)
        global prev_max_q
        prev_max_q = float(torch.max(get_q_values(s1)))

    reward = game.make_action(actions[a], frame_repeat)
    if SHOW_REWARDS and reward > 0:
        print("Reward {} at step {}".format(reward, step))

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    if update_every < 1:
        for i in range(int(1/update_every)):
            learn_from_memory()
    else:
        if step % update_every == 0:
            learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
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

    print("Actions:",actions)

    global optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("-"*60)
    print("Parameters ",kwargs)
    print("-" * 60)

    time_start = time()

    global bad_q
    bad_q = False

    for epoch in range(epochs):
        print("\nEpoch {} (eps={:.3f})\n-------------".format(epoch + 1, exploration_rate(epoch)))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        model.train()
        game.new_episode()
        for learning_step in trange(learning_steps_per_epoch, leave=False):
            if SHOW_MAXQ and learning_step % 1000 == 0:
                print("maxq: {:.2f} loss: {:.5f}".format(prev_max_q, float(prev_loss)))
            perform_learning_step(epoch, learning_step)
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
        model.eval()
        test_scores = []
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            step = 0
            while not game.is_episode_finished():
                step += 1
                state = preprocess(game.get_state().screen_buffer)
                state = state.reshape([1, 3, resolution[0], resolution[1]])
                if random() < 0:
                    best_action_index = randint(0, len(actions) - 1)
                else:
                    best_action_index = get_best_action(state)
                #print("step", step, "action", best_action_index, "state",state.shape,"max q", torch.max(get_q_values(state)))
                reward = game.make_action(actions[best_action_index], frame_repeat)
                if SHOW_REWARDS and reward > 0:
                    print("Reward! {} at step {}".format(reward, step))

            r = game.get_total_reward()
            test_scores.append(r)

        if bad_q:
            print("******* Warning MaxQ was too high ***************")

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

    if not os.path.isfile("results_k.csv"):
        with open("results_k.csv", "w") as f:
            f.write("learning_rate, update_every, discount_factor, replay_memory_size, end_epsilon, batch_size, frame_repeat, time (min), epochs\n")

    with open("results_k.csv", "a") as f:
        f.write(
            str(results["params"].get("learning_rate", learning_rate))+"," +
            str(results["params"].get("update_every", learning_rate)) + "," +
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
        db = pickle.load(open("results_k.dat","rb"))
    except FileNotFoundError:
        db = []
    db.append(results)
    pickle.dump(db, open("results_k.dat","wb"))

if __name__ == '__main__':

    # this doesn't hurt performance much and leaves other CPUs avalaible for more workers.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    # random sample from reasonable values.
    #tests = [{}] # start with default settings test.
    tests = []
    for lr in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        for ue in [1/4, 1/2, 1, 2, 4]:
            tests.append({"update_every": ue, "learning_rate":lr})

    for test_params in tests:
        try:
            result = run_test(**test_params)
            save_results(result)
        except Exception as e:
            print("********** Test failed....")
            print("Params:",test_params)
            print("error:", e)
    game.close()
