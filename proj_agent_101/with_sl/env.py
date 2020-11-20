import gym
from gym.utils import seeding
from gym_grid_driving.envs.grid_driving import LaneSpec
import numpy as np
from enum import Enum

class EnvWrapper:
    REWARD_NEARER = 0.1
    REWARD_FURTHER = 0.05
    def __init__(self, env):
        self.env = env
        self.action_space = int(env.action_space.n)
        self.observation_space = env.observation_space
        self.REWARD_NEARER = 0.1
        self.REWARD_FURTHER = -0.1
        #self.REWARD_CRASH = -10
        self.real_reward = 0

    def reset(self):
        return self.env.reset()


    def agent_nearer(self, state, next_state):
        agent_pos_cur = state[1]
        agent_pos_next = next_state[1]

        reward = 0
        agent_pos_2d_cur = list(zip(*np.where(agent_pos_cur == 1)))[0]
        agent_pos_2d_next = list(zip(*np.where(agent_pos_next == 1)))[0]


        agent_pos_cur_x, agent_pos_cur_y  = agent_pos_2d_cur

        agent_pos_next_x, agent_pos_next_y = agent_pos_2d_next

        if agent_pos_next_x < agent_pos_cur_x:
            reward += self.REWARD_NEARER

        if agent_pos_next_y < agent_pos_cur_y:
            dist = abs(agent_pos_next_y - agent_pos_cur_y)
            reward += dist * self.REWARD_NEARER

        else:
            reward = self.REWARD_FURTHER

        return reward


    def step(self, action, state):
        next_state, reward, done, info = self.env.step(action)
        self.real_reward = reward
        # set custom reward
        # find agent position
        if (not done):
            reward = self.agent_nearer(state, next_state)

        # it seems that negative reward for agent crash is not good for training
        # if (self.env.world.agent_state.name == "crashed"):
        #     reward = self.REWARD_CRASH

        return next_state, reward, done, info




def construct_random_lane_env():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-3, -1]),
                        LaneSpec(cars=8, speed_range=[-3, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-3, -1]),
                        LaneSpec(cars=8, speed_range=[-3, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-3, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=8, speed_range=[-3, -1])],
               'random_lane_speed': True
            }
    return gym.make('GridDriving-v0', **config)
