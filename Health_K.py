"""
Author

Test the effect of update frequency in DQN.

Based on...

Model based on VizDoom Paper

Still not done:
[ ] check frame is orientated correctly, we want this to be wide
[ ] stack 4 states
[ ] add game features to network
[ ] reward shaping (+100 medkit, -100 vile, but don't include in final score) <in my paper maybe remove this...>

Also
[ ] per epoch saving (model weights etc, just in case of crash...)
[ ] run a short test with 1 10th the steps to make sure I have all the data I need.
[ ] implement fixed targets

Divergence
RMSProp (and adam) seem to diverge
What I've tried
[*] moving zero_grad to before the model forward (thought it helped, but it didn't)
[*] waiting 4k steps before first update (didn't help)
[*] add gradient clamping... (nope... doesn't help)
[-] ignore it and see if it fixes itself? (might just run it through once and see what happens, maybe I get same reuslts as them, and ignore it?)
    (my guess is that this wouldn't work?)
[*] use the fixed targets things, why are we not doing this??
    (yes this worked)
[ ] add batch norm? hmm... changes model a lot

Things to get stuff working
[ ] Try 1k model updates

... owch... maxQ again...

Things to make it faster
[...] batch norm (converge faster)


# this paper has some hyper paramaters too http://cs229.stanford.edu/proj2017/final-reports/5238810.pdf
# keras implementation of some maps. https://github.com/flyyufelix/VizDoom-Keras-RL
"""

# force single threads, makes progream more CPU efficent but doesn't really hurt performance.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
import matplotlib.pyplot as plt
import os.path
import cv2
import uuid
import sys
import argparse
import shutil

# --------------------------------------------------------
# Debug settings
# --------------------------------------------------------

run_name = "results_health_k"

SHOW_REWARDS = False
SHOW_MAXQ = False
show_model_updates = False
SHOW_FIRST_FRAME = False


# --------------------------------------------------------
# Main one
# --------------------------------------------------------

learn_factor = 1

num_stacks = 1
num_channels = 3 # for color

epochs = 100//learn_factor

# Q-learning settings
learning_rate = 0.00001         # maybe this is too slow! ? oh right... rewards very high i.e. 100 in this one.
                                # (default 00001)
discount_factor = 1
learning_steps_per_epoch = 5000//learn_factor  # we probably want 10 million steps, let's make an epoch a 100,000 steps, so 100 epochs
replay_memory_size = 10000

end_eps = 0.1
start_eps = 1.0
start_eps_decay = 4000
end_eps_decay = 104000

hidden_units = 1024

update_every = 1
batch_size = 64

# Training regime
test_episodes_per_epoch = 200//learn_factor

target_update = 1000//learn_factor           # 10k was DQN paper
first_update_step = 1000//learn_factor        # make sure we have some experience before we start making updates.

# Other parameters
frame_repeat = 10
resolution = (120, 45)

time_stats = {}

# Configuration file path
config_file_path = "scenarios/health_gathering_supreme.cfg"

# --------------------------------------------------------

"""


# --------------------------------------------------------
# Basic Test
# --------------------------------------------------------

epochs = 10

# Q-learning settings
learning_rate = 0.0003         # maybe this is too slow! ? oh right... rewards very high i.e. 100 in this one.
discount_factor = 1
learning_steps_per_epoch = 1000  # we probably want 10 million steps, let's make an epoch a 100,000 steps, so 100 epochs
replay_memory_size = 10000

end_eps = 0.1
start_eps = 1.0
start_eps_decay = 0
end_eps_decay = 4000

hidden_units = 128

update_every = 1
batch_size = 64

# Training regime
test_episodes_per_epoch = 50

target_update = 500             # 10k was DQN paper
first_update_step = 500        # make sure we have some experience before we start making updates.

# Other parameters
frame_repeat = 10
resolution = (120, 45)

# Configuration file path
config_file_path = "scenarios/basic.cfg"

# --------------------------------------------------------
"""

prev_loss = 0
prev_max_q = 0
max_q = 0
last_total_shaping_reward = 0

# this is the lowest resolution we can go, and should be fine.
screen_resolution = vzd.ScreenResolution.RES_160X120

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

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

def print_time_stats():
    for k,v in time_stats.items():
        count, total_time = v
        per_time = total_time / count
        print("{:<25}: called {:<8} times with cost of {:<8.3f}ms".format(k, count, per_time * 1000))


# Converts and down-samples the input image
@track_time_taken
def preprocess(img):
    # note: this is quite slow, might switch to another method of resizing?

    original_img = img

    img = np.swapaxes(img, 0, 2)
    img = cv2.resize(np.float32(img)/255, dsize=resolution[::-1], interpolation=cv2.INTER_LINEAR)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    global SHOW_FIRST_FRAME
    if show_first_frame:
        print("original input shape:", original_img.shape)
        original_img = np.swapaxes(original_img, 0, 2)
        original_img = np.swapaxes(original_img, 0, 1)
        print("preprocessed shape:",img.shape)
        plt.imshow(original_img)
        plt.show()
        plt.imshow(np.swapaxes(img,0,2))
        plt.show()
        show_first_frame = False

    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = num_channels * num_stacks
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
        self.conv1 = nn.Conv2d(num_channels * num_stacks, 32, kernel_size=7, stride=4) #3 color channels, 4 previous frames.
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(prod([32,11,1]), hidden_units)
        self.fc2 = nn.Linear(hidden_units, available_actions_count)

    @track_time_taken
    def forward(self, x):
        x = x.to(device)          # Bx3x120x45
        x = F.relu(self.conv1(x)) # Bx32x29x10
        x = F.relu(self.conv2(x)) # Bx32x13x3
        x = F.relu(self.conv3(x)) # Bx32x11x1
        x = x.view(-1, prod(x.shape[1:]))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.cpu()

criterion = nn.MSELoss()

def learn(s1, target_q):
    s1 = torch.from_numpy(s1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1), Variable(target_q)

    optimizer.zero_grad()
    output = policy_model(s1)
    loss = criterion(output, target_q)
    loss.backward()
    for param in policy_model.parameters(): #clamp gradients...
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

def get_q_values(state):
    global max_q
    state = torch.from_numpy(state)
    state = Variable(state)
    q = policy_model(state)
    max_q = max(max_q, float(torch.max(q)))
    return q

def get_target_q_values(state):
    global max_q
    state = torch.from_numpy(state)
    state = Variable(state)
    q = target_model(state).detach()
    max_q = max(max_q, float(torch.max(q)))
    return q

def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action


@track_time_taken
def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q = get_q_values(s2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_target_q_values(s1).data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r

        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2

        this_loss = learn(s1, target_q)
        global prev_loss
        prev_loss = 0.95 * prev_loss + 0.05 * float(this_loss)

def exploration_rate(step):
    """# Define exploration rate change over time"""

    if step < start_eps_decay:
        return start_eps
    elif step > end_eps_decay:
        return end_eps
    else:
        # linear decay
        return start_eps + (step - start_eps_decay) / (end_eps_decay - start_eps_decay) * (end_eps - start_eps)


@track_time_taken
def env_step(action, frame_repeat):
    return game.make_action(action, frame_repeat)

def push_state_history(s):
    pass

@track_time_taken
def perform_learning_step(step):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(step)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, num_channels * num_stacks, resolution[0], resolution[1]])
        a = get_best_action(s1)
        global prev_max_q
        prev_max_q = float(torch.max(get_q_values(s1)))

    global last_total_shaping_reward

    # give some time for this to catch up...
    reward = env_step(actions[a], frame_repeat)

    # Retrieve the shaping reward
    fixed_shaping_reward = game.get_game_variable(vzd.GameVariable.USER1)  # Get value of scripted variable
    shaping_reward = vzd.doom_fixed_to_double(fixed_shaping_reward)  # If value is in DoomFixed format project it to double
    shaping_reward = shaping_reward - last_total_shaping_reward
    last_total_shaping_reward += shaping_reward

    reward += shaping_reward

    #if SHOW_REWARDS and reward > 0:
    #   print("Reward {} at step {}".format(reward, step))
    if SHOW_REWARDS and shaping_reward != 0:
        print("Shaping reward of {}".format(shaping_reward))


    isterminal = game.is_episode_finished()

    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    memory.add_transition(s1, a, s2, isterminal, reward)



# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_screen_resolution(screen_resolution)
    game.init()
    print("Doom initialized.")
    return game


@track_time_taken
def update_target():
    target_model.load_state_dict(policy_model.state_dict())

def run_test(**kwargs):
    """ Run a test with given parameters, returns stats in dictionary. """

    results = {}
    results["test_scores"] = []
    results["test_scores_mean"] = []
    results["params"] = kwargs

    for k,v in kwargs.items():
        globals()[k] = v

    global target_model
    global policy_model

    target_model = Net(len(actions))
    policy_model = Net(len(actions))
    if device.lower() == "cuda":
        for model in [target_model, policy_model]:
            model.cuda()

    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    print("Actions:",actions)

    global optimizer
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=learning_rate)

    print("-"*60)
    print("Parameters ",kwargs)
    print("-" * 60)

    time_start = time()

    global last_total_shaping_reward

    for epoch in range(epochs):

        # clear timing stats
        global time_stats
        time_stats = {}

        global max_q
        max_q = 0

        print("\nEpoch {} (eps={:.3f})\n-------------".format(epoch + 1, exploration_rate(epoch*learning_steps_per_epoch)))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        policy_model.train()
        target_model.train()
        game.new_episode()
        last_total_shaping_reward = 0
        for learning_step in trange(learning_steps_per_epoch, leave=False):

            step = learning_step  + epoch*learning_steps_per_epoch

            perform_learning_step(step)

            # update target net every so often.
            if step % target_update == 0:
                if show_model_updates:
                    print("Model update.")
                update_target()


            if step >= first_update_step:
                if update_every < 1:
                    for i in range(int(1 / update_every)):
                        learn_from_memory()
                else:
                    if step % update_every == 0:
                        learn_from_memory()

            if game.is_episode_finished():
                last_total_shaping_reward = 0
                score = game.get_total_reward()
                train_scores.append(score)
                game.new_episode()
                train_episodes_finished += 1

        print("\n\t%d training episodes played." % train_episodes_finished)

        train_scores = np.array(train_scores)

        print("\tResults: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        print("Timing stats:")
        print_time_stats()

        print("\nTesting...")
        policy_model.eval()
        target_model.eval()
        test_scores = []
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            game.new_episode()
            step = 0
            while not game.is_episode_finished():
                step += 1
                state = preprocess(game.get_state().screen_buffer)
                state = state.reshape([1, num_channels * num_stacks, resolution[0], resolution[1]])
                if random() < 0:
                    best_action_index = randint(0, len(actions) - 1)
                else:
                    best_action_index = get_best_action(state)
                #print("step", step, "action", best_action_index, "state",state.shape,"max q", torch.max(get_q_values(state)))
                reward = env_step(actions[best_action_index], frame_repeat)
                if SHOW_REWARDS and reward > 0:
                    print("Reward! {} at step {}".format(reward, step))

            r = game.get_total_reward()
            test_scores.append(r)

        if max_q > 10000:
            print()
            print("******* Warning MaxQ was too high ***************")
            print("MaxQ:",max_q)
            print("*************************************************")
            print()
        else:
            print("\n\tMaxQ:", max_q)

        test_scores = np.array(test_scores)
        print("\n\tResults: mean: %.1f +/- %.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
              "max: %.1f" % test_scores.max())

        results["test_scores_mean"].append(test_scores.mean())
        results["test_scores"].append(test_scores)

        """
        if save_model:
            print("Saving the network weigths to:", model_savefile)
            torch.save(model, model_savefile)
        """

        elapsed_time = (time() - time_start)

        print("\tTotal elapsed time: %.2f minutes" % (elapsed_time / 60.0))

        progress = ((epoch+1) / epochs)

        est_total_time = elapsed_time / progress
        est_remaining_time = est_total_time - elapsed_time

        print("\tEstimated remaining time: {:.2f}h ({:.2f}h total)".format(est_remaining_time/60/60, est_total_time/60/60))

        print("\nScores:", results["test_scores_mean"],"\n")

        results["elapsed_time"] = ((time() - time_start) / 60.0)

        save_partial_results(results)

    return results

def save_partial_results(results):

    # save raw results to a pickle file for processing
    try:
        db = pickle.load(open(run_name + "_partial.dat", "rb"))
    except FileNotFoundError:
        db = []
    db.append(results)
    pickle.dump(db, open(run_name + "_partial.dat", "wb"))

    # save the model
    print("Saving the network weigths.")
    torch.save(target_model, run_name + "_model.dat")

def save_results(results):

    # create an easy to read CVS file
    if not os.path.isfile(run_name+".csv"):
        with open(run_name+".csv", "w") as f:
            f.write("learning_rate, update_every, discount_factor, replay_memory_size, end_epsilon, batch_size, frame_repeat, time (min), epochs\n")

    with open(run_name+".csv", "a") as f:
        f.write(
            str(results["params"].get("learning_rate", learning_rate))+"," +
            str(results["params"].get("update_every", learning_rate)) + "," +
            str(results["params"].get("discount_factor", discount_factor)) + "," +
            str(results["params"].get("replay_memory_size", replay_memory_size)) + "," +
            str(results["params"].get("end_epsilon", end_eps)) + "," +
            str(results["params"].get("batch_size", batch_size)) + "," +
            str(results["params"].get("frame_repeat", frame_repeat)) + "," +
            str(results["elapsed_time"])+","+
            ",".join(str(x) for x in results["test_scores_mean"]) + "\n"
        )

    # save raw results to a pickle file for processing
    try:
        db = pickle.load(open(run_name+".dat","rb"))
    except FileNotFoundError:
        db = []
    db.append(results)
    pickle.dump(db, open(run_name+".dat","wb"))

def train_agent():

    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    result = run_test(**kwargs)
    save_results(result)

    game.close()



def run_experiment(**params):
    """ Runs an experiment with given parameters. """

    uid = uuid.uuid4()
    scenario_name = "blah"

    experiment_name = "{} {} ".format(scenario_name, uid.hex[:16])

    print("=" * 60)
    print("Running Experiment {}".format(experiment_name))
    print("=" * 60)

    run_folder = "experiments"
    job_folder = os.path.join(run_folder, experiment_name)

    # create the job folder
    os.makedirs(job_folder, exist_ok=True)

    # create a file showing the run parameters
    with open(os.path.join(job_folder, "params.txt"),"w") as f:
        f.write(params)

    # make a copy of the script that generated the results
    this_script = sys.argv[0]
    shutil.copy(this_script,os.path.join(job_folder, this_script))

    # run the job
    train_agent(**params)



def run_benchmark():
    # pass

def run_simple_test(**params):
    """ Runs a simple test to make sure model is able to learn the basic environment. """
    # pass

if __name__ == '__main__':

    # handle parameters
    parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
    parser.add_argument('mode', type=str, nargs='+', help='train | test | benchmark')
    parser.add_argument('--num_stack', default=4, type=int, help='Agent is shown this number of frames of history.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate.')
    parser.add_argument('--discount_factor', default=1, type=float, help='Discount factor.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train for.')
    parser.add_argument('--learning_steps_per_epoch', default=5000, type=int, help='Number of environment steps per epoch.')
    parser.add_argument('--replay_memory_size', default=10000, type=int, help='Number of environment steps per epoch.')
    parser.add_argument('--hidden_units', default=1024, type=int, help='Number of hidden units in model.')
    parser.add_argument('--batch_size', default=64, type=int, help='Samples per minibatch.')




    end_eps = 0.1
    start_eps = 1.0
    start_eps_decay = 4000
    end_eps_decay = 104000


    update_every = 1

    # Training regime
    test_episodes_per_epoch = 200 // learn_factor

    target_update = 1000 // learn_factor  # 10k was DQN paper
    first_update_step = 1000 // learn_factor  # make sure we have some experience before we start making updates.

    # Other parameters
    frame_repeat = 10
    resolution = (120, 45)

    # figure out mode
    if mode == "train":
        pass
    elif mode == "benchmark":
        pass
    elif mode == "test":
            pass



