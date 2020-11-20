import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

try: # server-compatible import statement
    from models import *
    from env import *
except:
    from .models import *
    from .env import *

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 15000
batch_size    = 32
max_episodes  = 10000
t_max         = 600
min_buffer    = 5000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 1000
print_interval= 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model_DDQN.pt')
#model_path = "./model_DDQN.pt"
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer = []    # stores (s,a,s') transitions
        self.MAX_SIZE = buffer_limit
        self.ptr = 0    # ptr that points to the oldest transition

    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        if len(self.buffer) < self.MAX_SIZE:
            # store transition tuple normally
            self.buffer.append(transition)
        else:
            # replace old transitions
            self.buffer[self.ptr] = transition
            self.ptr += 1
            self.ptr = self.ptr % self.MAX_SIZE


    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        indices = np.random.choice(len(self.buffer), batch_size)    # indices to sample from buffer
        transitions = [self.buffer[i] for i in indices]
        states = list(map(lambda x: x[0], transitions))
        actions = list(map(lambda x: x[1], transitions))
        rewards = list(map(lambda x: x[2], transitions))
        next_states = list(map(lambda x: x[3], transitions))
        dones = list(map(lambda x: x[4], transitions))

        states = torch.tensor(states, dtype = torch.float32)

        next_states = torch.tensor(next_states, dtype=torch.float32)

        actions = torch.tensor(actions)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        dones = torch.tensor(dones, dtype=torch.float32)

        return (states, actions, rewards, next_states, dones)




    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)

# Double DQN - select action with online weights. Evaluate Q function with target
def compute_loss(model, target, states, actions, rewards, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''

    # compute TD target using target network. (batch_size, num_actions)
    #print(rewards.shape)

    V_s, A_s = model.forward(states)
    V_s_, A_s_ = model.forward(next_states)

    #
    V_s_eval, A_s_eval = target.forward(next_states)

    Q_vals_next_state = torch.add(V_s_eval,
        (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

    Q_vals_online = torch.add(V_s_,
        (A_s_ - A_s_.mean(dim=1,keepdim=True)))

    Q_vals_state_action = torch.add(V_s, (A_s - A_s.mean(dim=1,keepdim=True)))


    # action selection
    #Q_vals_online = model.forward(next_states)

    # argmax actions using online model
    next_actions = torch.argmax(Q_vals_online, dim=1, keepdim=True)

    # next states
    Q_vals_next_state = torch.gather(Q_vals_next_state, 1, next_actions)
    #print(max_Q_vals_next_state.shape)
    # (batch_size, 1)
    TD_target = rewards + (1 - dones) * gamma * Q_vals_next_state
    #print(TD_target.shape)
    # Q vals for the current state (batch_size, num_actions)
    #Q_vals_state_action = model.forward(states)

    # return (batch_size, 1 ) vector
    Q_vals_curr_state_action = torch.gather(Q_vals_state_action, 1, actions)
    #print(Q_vals_curr_state_action.shape)
    # (batch_size, 1) vector
    loss = nn.MSELoss()

    mse = loss(Q_vals_curr_state_action, TD_target)

    return mse



def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon


def car_speeds(prev_state, cur_state):
    # append previous speeds of cars
    # initial state cars don't have intial speed, so it is 0.
    old_car_pos = prev_state[0]
    new_car_pos = cur_state[0]

    old_ptr = 0
    new_ptr = 0

    # loop through each lane
    no_lanes = prev_state[0].shape[0]
    no_cols = prev_state[0].shape[1]

    speed_mat = np.zeros((1, no_lanes, no_cols))
    for lane in range(no_lanes):
        old_car_lane_pos = old_car_pos[lane]
        new_car_lane_pos = new_car_pos[lane]

        old_car_idx = np.where(old_car_lane_pos == 1)[0]
        new_car_idx = np.where(new_car_lane_pos == 1)[0]

        #keep rolling while cars out of position
        aligned = False
        roll = 0
        while (not aligned):
            aligned = True
            for i in range(roll, len(old_car_idx)):
                if (new_car_idx[i] >= old_car_idx[i]):
                    aligned = False
                    new_car_idx = np.roll(new_car_idx, 1)
                    roll += 1
                    break

        #new_car_idx = np.roll(new_car_idx, roll)

        for idx in range(len(old_car_idx)):
            speed = (old_car_idx[idx] - new_car_idx[idx]) % no_cols
            speed_mat[0][lane][new_car_idx[idx]] = speed

    return speed_mat







def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).

    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    input_shape = (5, env.observation_space.shape[1], env.observation_space.shape[2])

    model = model_class(input_shape, env.action_space).to(device)
    target = model_class(input_shape, env.action_space).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):

        epsilon = compute_epsilon(episode)
        state = env.reset()
        speed = np.zeros((1, env.observation_space.shape[1], env.observation_space.shape[2]))
        state = np.concatenate((state, speed), axis = 0)
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)
            # Apply the action to the environment
            next_state, reward, done, info = env.step(action, state)

            car_speed = car_speeds(state, next_state)

            next_state = np.concatenate((next_state, car_speed), axis=0)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))
            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.1:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model

def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = model.act(state)
            state, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
    avg_rewards = np.mean(rewards)
    print("{} episodes avg rewards : {:.1f}".format(max_episodes, avg_rewards))
    return avg_rewards

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`.
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)

def get_env():
    '''
    Get the sample test cases for training and testing.
    '''
    return construct_random_lane_env()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    env = get_env()

    env = EnvWrapper(env)
    if args.train:
        model = train(DuellingDQN, env)
        save_model(model)
    else:
        model = get_model()
    test(model, env, max_episodes=600)