"""
Author

Test the effect of update frequency in DQN.

Based on...

this paper has some hyper paramaters too http://cs229.stanford.edu/proj2017/final-reports/5238810.pdf
keras implementation of some maps. https://github.com/flyyufelix/VizDoom-Keras-RL
"""

# force single threads, makes progream more CPU efficent but doesn't really hurt performance.
# this does seem to affect pytorch cpu speed a bit though.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
import logging
import platform
import keyboard


# --------------------------------------------------------
# Debug settings
# --------------------------------------------------------

SHOW_REWARDS = False
SHOW_MAXQ = False
SHOW_FIRST_FRAME = False

# --------------------------------------------------------
# Globals
# --------------------------------------------------------

MAX_GRAD = 1000     # this should be 1, the fact we have super high gradients is concerning, maybe
                    # I should make sure input is normalised to [0,1] and change the reward shaping structure
                    # to not be so high.  Also log any extreme rewards...

aux_inputs = 3      # number of auxilary inputs to model.

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


# --------------------------------------------------------

def clean(s):
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return "".join([x if x in valid_chars else "_" for x in s])

def get_job_key(args):
    params = sorted([(k, v) for k, v in args.items() if k not in ["mode"] and v is not None])
    return " ".join("{}={}".format(clean(k), clean(str(v))) for k, v in params)+" "

class Config:

    def __init__(self):
        self.mode = "train"
        self.num_stacks = 1
        self.num_channels = 3   # 3 channels for color
        self.epochs = 40
        self.learning_rate = 0.00001
        self.discount_factor = 1
        self.learning_steps_per_epoch = 1000
        self.test_episodes_per_epoch = 20
        self.replay_memory_size = 10000
        self.end_eps = 0.1
        self.start_eps = 1.0
        self.hidden_units = 1024
        self.update_every = 1
        self.batch_size = 64
        self.target_update = 1000
        self.first_update_step = 1000
        self.frame_repeat = 10
        self.resolution = (120, 45)
        self.verbose = False
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
    def job_folder(self):
        """ path to job folder. """

        if self.mode == "benchmark":
            experiment = "benchmarks"
        elif self.mode == "test":
            experiment = "test"
        else:
            experiment = self.experiment

        prefix = "_" if config.mode == "train" else ""

        return os.path.join("runs", experiment, prefix + self.job_subfolder)

    @property
    def final_job_folder(self):

        if self.mode == "benchmark":
            experiment = "benchmarks"
        elif self.mode == "test":
            experiment = "test"
        else:
            experiment = self.experiment

        return os.path.join("runs", experiment, self.job_subfolder)

    @property
    def job_subfolder(self):
        """ job folder without runs folder and experiment folder. """
        return "{} [{}]".format(self.job_name, self.job_id)

    @property
    def scenario(self):
        return os.path.splitext(os.path.basename(self.config_file_path))[0]

    @property
    def start_eps_decay(self):
        return min(self.total_steps * 0.1, self.learning_steps_per_epoch * 10, 10000)

    @property
    def end_eps_decay(self):
        return min(self.total_steps * 0.6, self.learning_steps_per_epoch * 60, 100000)

    @property
    def total_steps(self):
        return self.epochs * self.learning_steps_per_epoch

    def make_job_folder(self):
        """create the job folder"""
        os.makedirs(self.job_folder, exist_ok=True)

    def rename_job_folder(self):
        """ moves job to completed folder. """
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
        logging.log(log_level, "\t{:<25} {:<8} {:<8.3f}".format(k.lower(), count, per_time * 1000))



# Converts and down-samples the input image
@track_time_taken
def preprocess(img):
    # note: this is quite slow, might switch to another method of resizing?

    original_img = img

    img = np.swapaxes(img, 0, 2)
    img = cv2.resize(np.float32(img)/255, dsize=config.resolution[::-1], interpolation=cv2.INTER_AREA)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    # convert from float32 to uint8
    img = np.uint8(img * 255)

    global SHOW_FIRST_FRAME
    if SHOW_FIRST_FRAME:
        original_img = np.swapaxes(original_img, 0, 2)
        original_img = np.swapaxes(original_img, 0, 1)
        plt.imshow(original_img)
        plt.show()
        plt.imshow(np.swapaxes(img,0,2))
        plt.show()
        SHOW_FIRST_FRAME = False
    return img


class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, config.num_channels * config.num_stacks, config.resolution[0], config.resolution[1])
        data_shape = (capacity, aux_inputs * config.num_stacks)
        self.s1 = np.zeros(state_shape, dtype=np.uint8)
        self.s2 = np.zeros(state_shape, dtype=np.uint8) # save memory...
        self.d1 = np.zeros(data_shape, dtype=np.float32)
        self.d2 = np.zeros(data_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

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

    @track_time_taken
    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return (
            self.s1[i],
            self.d1[i],
            self.a[i],
            self.s2[i],
            self.d2[i],
            self.isterminal[i],
            self.r[i]
        )


def prod(X):
    y = 1
    for x in X:
        y *= x
    return y

def save_frames(filename, x):
    print("Saving {}".format(filename))
    print("Shape:",x.shape)
    print("min/max/mean/median",print(np.min(x), np.max(x), np.mean(x), np.median(x)))
    for i in range(len(x)):
        plt.imsave("{}-{:03d}.png".format(filename,i+1), x[i])


class Net(nn.Module):

    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(config.num_channels * config.num_stacks, 32, kernel_size=7, stride=4) #3 color channels, 4 previous frames.
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(prod([32,11,1]) + aux_inputs * config.num_stacks, config.hidden_units)
        self.fc2 = nn.Linear(config.hidden_units, available_actions_count)

    @track_time_taken
    def forward(self, x, d):

        x = x.to(config.device)   # BxCx120x45
        d = d.to(config.device)   # Bx4x3

        x = x.float()/255         # convert from unit8 to float32 format

        # for debuging, save frames
        #frame_index = int(d.cpu().data.numpy()[0][1])
        #frames = x.cpu().data.numpy().reshape(-1,config.resolution[0], config.resolution[1])
        #save_frames("frame-{0:03d}".format(frame_index), frames)

        x = F.relu(self.conv1(x)) # Bx32x29x10
        x = F.relu(self.conv2(x)) # Bx32x13x3
        x = F.relu(self.conv3(x)) # Bx32x11x1
        x = x.view(-1, prod(x.shape[1:])) # Bx352
        x = torch.cat((x, d), 1)  # Bx355
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x.cpu()

criterion = nn.MSELoss()

def learn(s1, d1, target_q):
    s1 = torch.from_numpy(s1)
    d1 = torch.from_numpy(d1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1), Variable(target_q)

    optimizer.zero_grad()
    output = policy_model(s1, d1)
    loss = criterion(output, target_q)
    loss.backward()
    for param in policy_model.parameters(): #clamp gradients...
        if param.grad is not None:
            grads = param.grad.data.cpu().numpy() # note: could keep this all on GPU if I wanted until we need to clamp..
            global max_grad
            max_grad = max(max_grad, np.max(np.abs(grads)))
            """
            if max_grad > MAX_GRAD:
                logging.debug("Gradients on tensor with dims {} are very large (min:{:.1f} max:{:.1f} mean:{:.1f} std:{:.1f})]".format(
                    param.shape,
                    np.min(grads), np.max(grads), np.mean(grads), np.std(grads),
                    -MAX_GRAD, MAX_GRAD
                ))
                # don't actually clamp, this might be causing problems...
                # param.grad.data.clamp_(-MAX_GRAD, MAX_GRAD)
            """
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

def get_best_action(s,d):
    s = Variable(torch.from_numpy(s))
    d = Variable(torch.from_numpy(d))
    q = policy_model(s,d).detach()
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action


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

        this_loss = learn(s1, d1, target_q)
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
def env_step(action, frame_repeat):
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


def get_observation():
    return preprocess(game.get_state().screen_buffer)


def get_data():
    return np.float32([
        game.get_game_variable(vzd.GameVariable.HEALTH),
        game.get_episode_time(),
        action_list_to_id(game.get_last_action())
    ])


@track_time_taken
def perform_environment_step(step):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    s1, d1 = get_stack()

    # With probability eps make a random action.
    eps = exploration_rate(step)

    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        a = get_best_action(s1, d1)

    global last_total_shaping_reward

    # give some time for this to catch up...
    reward = env_step(actions[a], config.frame_repeat)

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

    if SHOW_REWARDS and shaping_reward != 0:
        logging.debug("Shaping reward of {}".format(shaping_reward))

    isterminal = game.is_episode_finished()

    if isterminal:
        s2, d2 = None, None
    else:
        push_state(get_observation(), get_data())
        s2, d2 = get_stack()

    memory.add_transition(s1, d1, a, s2, d2, isterminal, reward)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom():
    """
    Initialize game (global varaible).
    :return:
    """

    global game
    global game_hq
    global actions

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
    game.init()

    if config.export_video:
        game_hq = vzd.DoomGame()
        game_hq.load_config(config.config_file_path)
        game_hq.set_window_visible(False)
        game_hq.set_mode(vzd.Mode.PLAYER)
        game_hq.set_screen_format(vzd.ScreenFormat.CRCGCB)
        game_hq.set_screen_resolution(preview_screen_resolution)
        game_hq.init()

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    logging.info("Doom initialized.")


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
        elif c == "h":
            print()
            logging.critical("**** H for help")
            logging.critical("**** I to show jobname")
            logging.critical("**** T for timing stats")
            logging.critical("**** P to pause")
            logging.critical("**** Q to quit")
        elif c == "q":
            exit(-1)
        else:
            print()
            logging.critical("\nInvalid input {}.\n".format(c))


def get_final_score(health_as_reward=None):
    if health_as_reward is None:
        health_as_reward = config.health_as_reward
    if health_as_reward:
        # use integral of health over time assuming agent would have lasted 2100 steps.
        final_score = sum(health_history) / (2100 / config.frame_repeat)
    else:
        final_score = game.get_total_reward()

    return final_score


def reset_agent():
    """ Resets agent and stats to start of episode. """

    global previous_health
    global last_total_shaping_reward
    global health_history

    previous_health = 100
    last_total_shaping_reward = 0
    health_history = [100]

    game.new_episode()

    # full observation buffer with first observation.
    for k in range(config.num_stacks):
        push_state(get_observation(), get_data())

def save_video(folder, filename, frames):
    """ Saves given frames (list of np arrays) to video file. """

    logging.info("Exporting video example {}.".format(filename))

    os.makedirs(folder, exist_ok=True)

    channels, height, width = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(os.path.join(folder, filename), fourcc, 60, (width, height))

    for frame in frames:
        # we are in CHW but want to be in HWC
        frame = np.swapaxes(frame, 0, 2) #WHC
        frame = np.swapaxes(frame, 0, 1) #HWC
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def get_stack():
    """ Returns [1xCxHxW], [1xCxD] """
    return (
        np.concatenate(observation_history)[np.newaxis, :, :, :],
        np.concatenate(data_history)[np.newaxis, :]
    )

def eval_model():

    policy_model.eval()
    test_scores = []
    test_scores_health = []
    test_scores_reward = []

    for test_episode in trange(config.test_episodes_per_epoch, leave=False):
        reset_agent()
        step = 0
        while not game.is_episode_finished():
            handle_keypress()
            step += 1

            push_state(get_observation(), get_data())

            s1, d1 = get_stack()

            if random() < 0:
                best_action_index = randint(0, len(actions) - 1)
            else:
                best_action_index = get_best_action(s1, d1)
            reward = env_step(actions[best_action_index], config.frame_repeat)
            health_history.append(game.get_game_variable(vzd.GameVariable.HEALTH))
            if SHOW_REWARDS and reward > 0:
                logging.info("Reward! {} at step {}".format(reward, step))

        # make sure to record both the reward score, and the health as reward score.
        test_scores.append(get_final_score())
        test_scores_health.append(get_final_score(health_as_reward=True))
        test_scores_reward.append(get_final_score(health_as_reward=False))

    return np.array(test_scores), np.array(test_scores_health), np.array(test_scores_reward)


def export_video(epoch):

    global game
    global game_hq

    # activate game_hq
    game, game_hq = game_hq, game

    policy_model.eval()
    frames = []

    reset_agent()
    step = 0
    while not game.is_episode_finished():
        if step % config.frame_repeat == 0:
            # only make decisions at the correct frame rate
            push_state(get_observation(), get_data())

            s, d = get_stack()

            best_action_index = get_best_action(s, d)
        # we generate all frames for smooth video, even though
        # actions may stick for multiple frames.
        frames.append(game.get_state().screen_buffer)
        _ = game.make_action(actions[best_action_index], 1)
        step += 1

    # activate game again...
    game, game_hq = game_hq, game

    try:
        save_video(os.path.join(config.job_folder, "videos"), "epoch-{0:03d}.mp4".format(epoch), frames)
    except Exception as e:
        logging.critical("Error saving video: {}".format(e))


def train_agent(continue_from_save = False):
    """ Run a test with given parameters, returns stats in dictionary. """


    logging.critical("=" * 60)
    logging.critical("Running Experiment {} {} [{}]".format(config.experiment, config.job_name, config.job_id))
    logging.critical("=" * 60)

    global actions
    global memory

    # setup doom
    initialize_vizdoom()

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=config.replay_memory_size)

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
        target_model = Net(len(actions))
        policy_model = Net(len(actions))

    if config.device.lower() == "cuda":
        for model in [target_model, policy_model]:
            model.cuda()

    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    logging.debug("Actions: {}".format(actions))

    global optimizer
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=config.learning_rate)

    time_start = time()

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
        train_scores = []

        logging.info("Training...")
        policy_model.train()

        reset_agent()

        for learning_step in trange(config.learning_steps_per_epoch, leave=False):

            handle_keypress()

            step = learning_step + epoch*config.learning_steps_per_epoch

            perform_environment_step(step)

            if step >= config.first_update_step:
                if config.update_every < 1:
                    for i in range(int(1 / config.update_every)):
                        perform_learning_step()
                else:
                    if step % int(config.update_every) == 0:
                        perform_learning_step()

            if game.is_episode_finished():
                score = get_final_score()
                train_scores.append(score)
                reset_agent()
                train_episodes_finished += 1

        logging.info("\tTraining episodes played: {}".format(train_episodes_finished))

        train_scores = np.array(train_scores)

        logging.info("\tResults: mean: {:.1f} +/- {:.1f} [min:{:.1f} max:{:.1f}]".format(
            train_scores.mean(),
            train_scores.std(),
            train_scores.min(),
            train_scores.max()
        ))

        show_time_stats()

        if config.export_video and ((epoch+1) % 10 == 0 or epoch == config.epochs-1) or epoch == 0:
            logging.info("Exporting video...")
            export_video(epoch+1)

        logging.info("Testing...")
        test_scores, test_scores_health, test_scores_reward = eval_model()

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

        elapsed_time = (time() - time_start)

        logging.info("\tTotal elapsed time: {:.2f} min".format(elapsed_time / 60.0))

        progress = ((epoch+1) / config.epochs)

        est_total_time = elapsed_time / progress
        est_remaining_time = est_total_time - elapsed_time

        logging.critical("Estimated remaining time: {:.0f} min ({:.2f}h total)".format(est_remaining_time/60, est_total_time/60/60))

        logging.critical("Scores: {}".format([round(x,2) for x in results["test_scores_mean"]]))

        results["elapsed_time"] = ((time() - time_start) / 60.0)

        save_results(results,"_partial")
        save_model(epoch+1)

    save_results(results, "_complete")
    save_model()

    game.close()

    # this stops all logging, and released the log.txt file allowing the
    # folder to be renamed.
    for log in logging.getLogger().handlers:
        log.close()
        logging.getLogger().removeHandler(log)

    if config.mode == "train":
        sleep(10)  # give Dropbox a chance to sync up, and logs etc to finish up.
        config.rename_job_folder()

    return results

def save_results(results, suffix=""):
    # save raw results to a pickle file for processing
    pickle.dump(results, open(os.path.join(config.job_folder, "results"+suffix+".dat"), "wb"))
    with open(os.path.join(config.job_folder, "results"+suffix+".txt"), "w") as f:
        f.write(str(results["test_scores_mean"]))
    generate_graphs(results)


def save_model(epoch=None):
    if epoch is None:
        torch.save(policy_model, os.path.join(config.job_folder, "model_complete.dat"))
    else:
        filename = "model_{0:03d}.dat".format(epoch)
        os.makedirs(os.path.join(config.job_folder, "models"), exist_ok=True)
        torch.save(policy_model, os.path.join(config.job_folder, "models", filename))

def restore_model(epoch=None):
    """ restores model from checkpoint. """
    if epoch is None:
        model_path = os.path.join(config.job_folder, "model_complete.dat")
    else:
        filename = "model_{0:03d}.dat".format(epoch)
        model_path = os.path.join(config.job_folder, "models", filename)

    global policy_model
    global target_model

    target_model = Net(len(actions))
    policy_model = Net(len(actions))

    policy_model = torch.load(model_path)
    target_model = torch.load(model_path)

    if config.device.lower() == "cuda":
        for model in [target_model, policy_model]:
            model.cuda()

    target_model.eval()

def run_evaluation():
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
    results["config"] = config
    results["args"] = config.args

    time_start = time()

    for epoch in range(config.epochs):
        restore_model(epoch+1)
        test_scores, test_scores_health, test_scores_reward = eval_model()

        results["test_scores_mean"].append(test_scores.mean())
        results["test_scores"].append(test_scores)
        results["test_scores_health"].append(test_scores_health)
        results["test_scores_reward"].append(test_scores_reward)

        elapsed_time = (time() - time_start)

        logging.info("\tTotal elapsed time: {:.2f} min".format(elapsed_time / 60.0))

        progress = ((epoch+1) / config.epochs)

        est_total_time = elapsed_time / progress
        est_remaining_time = est_total_time - elapsed_time

        logging.critical("Estimated remaining time: {:.0f} min ({:.2f}h total)".format(est_remaining_time/60, est_total_time/60/60))

        logging.critical("Scores: {}".format([round(x,2) for x in results["test_scores_mean"]]))

        results["elapsed_time"] = ((time() - time_start) / 60.0)

        save_results(results,"_partial")
        save_model(epoch+1)

    save_results(results, "_complete")



def run_benchmark():
    """ Runs a standard benchmark to see how fast learning happens. """

    # set test settings
    config.num_stacks = 1
    config.num_channels = 3  # 3 channels for color
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
    config.resolution = (120, 45)
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
    config.num_channels = 3  # 3 channels for color
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
    config.resolution = (120, 45)
    config.verbose=False
    config.config_file_path = "scenarios/basic.cfg"

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

if __name__ == '__main__':

    config = Config()

    # handle parameters
    parser = argparse.ArgumentParser(description='Run VizDoom Tests.')
    parser.add_argument('mode', type=str, help='train | test | benchmark | info | eval')
    parser.add_argument('--num_stacks', type=int, help='Agent is shown this number of frames of history.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    parser.add_argument('--discount_factor', type=float, help='Discount factor.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for.')
    parser.add_argument('--learning_steps_per_epoch', type=int, help='Number of environment steps per epoch.')
    parser.add_argument('--replay_memory_size', type=int, help='Number of environment steps per epoch.')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in model.')
    parser.add_argument('--batch_size', type=int, help='Samples per minibatch.')
    parser.add_argument('--device', type=str, help='CPU | CUDA')
    parser.add_argument('--verbose', type=bool, help='enable verbose output')
    parser.add_argument('--update_every', type=float, help='apply update every x learning steps')
    parser.add_argument('--experiment', type=str, help='name of subfolder to put experiment in.')
    parser.add_argument('--job_name', type=str, help='name of job.')
    parser.add_argument('--target_update', type=int, help='how often to update target network')
    parser.add_argument('--test_episodes_per_epoch', type=int, help='how many tests epsodes to run each epoch')
    parser.add_argument('--config_file_path', type=str, help="config file to use for VizDoom.")
    parser.add_argument('--health_as_reward', type=bool, help="use change in health as reward instead of default reward.")
    parser.add_argument('--frame_repeat', type=int, help="number of frames to skip.")
    parser.add_argument('--include_aux_rewards', type=bool, help="use auxualry reward during training (these are not counted during evaluation).")
    parser.add_argument('--export_video', type=bool, help="exports one video per epoch showing agents performance.")
    parser.add_argument('--job_id', type=str, help="unique id for job.")

    args = parser.parse_args()

    config.mode = config.mode.lower()

    config.apply_args(args)
    config.make_job_folder()

    # setup logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(config.job_folder,'log.txt'),
                        filemode='w')

    enable_logging()

    logging.info("Using device: {}".format(config.device))

    # apply mode
    if config.mode == "eval":
        run_evaluation()
    elif config.mode == "train":
        train_agent()
    elif config.mode == "continue":
        train_agent(continue_from_save=True)
    elif config.mode == "info":
        print("PyVorch:", torch.__version__)
        print("Device:", config.device)
        np.__config__.show()
    elif config.mode == "benchmark":
        run_benchmark()
    elif config.mode == "test":
        run_simple_test(args)
    else:
        print("Invalid mode {}.".format(config.mode))