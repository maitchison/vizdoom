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


Things to make it faster
[ ] batch norm (converge faster)
[ ] faster resize
[ ] multi thread preprocessing or something? (better to just run agents in parallel

"""

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

# --------------------------------------------------------
# Debug settings
# --------------------------------------------------------

run_name = "results_health_k"

SHOW_REWARDS = False
SHOW_MAXQ = False

epochs = 200

# --------------------------------------------------------
# Hyper-parameters
# --------------------------------------------------------

# Q-learning settings
learning_rate = 0.00001         # maybe this is too slow! ? oh right... rewards very high i.e. 100 in this one.
discount_factor = 1
learning_steps_per_epoch = 5000  # we probably want 10 million steps, let's make an epoch a 100,000 steps, so 100 epochs
replay_memory_size = 10000

end_eps = 0.1
start_eps = 1.0
start_eps_decay = 4000
end_eps_decay = 104000

hidden_units = 1024

update_every = 1
batch_size = 64

# Training regime
test_episodes_per_epoch = 200

target_update = 10000           # 10k was DQN paper
first_update_step = 1000        # make sure we have some experience before we start making updates.

# Other parameters
frame_repeat = 10
resolution = (120, 45)
episodes_to_watch = 10

prev_loss = 0
prev_max_q = 0
max_q = 0

# this is the lowest resolution we can go, and should be fine.
screen_resolution = vzd.ScreenResolution.RES_160X120

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

# Configuration file path
config_file_path = "scenarios/health_gathering.cfg"

# Converts and down-samples the input image
def preprocess(img):
    # note: this is quite slow, might switch to another method of resizing?
    img = np.swapaxes(img, 0, 2)
    img = skimage.transform.resize(img, resolution, anti_aliasing=False)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4) #3 color channels, 4 previous frames.
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(prod([32,11,1]), hidden_units)
        self.fc2 = nn.Linear(hidden_units, available_actions_count)

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

    # updaste target net every so often.
    if step % target_update == 0:
        target_model.load_state_dict(policy_model.state_dict())

    if step >= first_update_step:
        if update_every < 1:
            for i in range(int(1/update_every)):
                learn_from_memory()
        else:
            if step % update_every == 0:
                learn_from_memory()


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

    for epoch in range(epochs):

        global max_q
        max_q = 0

        print("\nEpoch {} (eps={:.3f})\n-------------".format(epoch + 1, exploration_rate(epoch*learning_steps_per_epoch)))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        policy_model.train()
        game.new_episode()
        for learning_step in trange(learning_steps_per_epoch, leave=False):

            step = learning_step  + epoch*learning_steps_per_epoch

            if SHOW_MAXQ and learning_step % 1000 == 0:
                print("maxq: {:.2f} loss: {:.5f}".format(prev_max_q, float(prev_loss)))
            perform_learning_step(step)
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
        policy_model.eval()
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

        if max_q > 1000:
            print()
            print("******* Warning MaxQ was too high ***************")
            print("MaxQ:",max_q)
            print("*************************************************")
            print()

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

    return results

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

def test_exploration_rate():
    """
    Draw a graph showing exploration rate over time.
    :return:
    """
    xs = list(range(200000))
    ys = [exploration_rate(x) for x in xs]
    plt.plot(xs, ys)
    plt.show()

def run_original():
    """ Run the agent from the original paper."""
    result = run_test()
    save_results(result)

def run_test_suite():
    """ Runs tests at various levels update frequency. """

    # run tests...
    # tests = [{}] # start with default settings test.
    tests = []
    for lr in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        for ue in [1 / 4, 1 / 2, 1, 2, 4]:
            tests.append({"update_every": ue, "learning_rate": lr})

    for test_params in tests:
        try:
            result = run_test(**test_params)
            save_results(result)
        except Exception as e:
            print("********** Test failed....")
            print("Params:", test_params)
            print("error:", e)


if __name__ == '__main__':

    # this doesn't hurt performance much and leaves other CPUs available for more workers.
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    run_original()

    game.close()
