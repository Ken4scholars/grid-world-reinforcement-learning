import random
from functools import partial

import numpy as np
import os
import torch


script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.load = partial(torch.load, map_location=device)

'''
An example to import a Python file.

Uncomment the following lines (both try-except statements) to import everything inside models.py
'''
try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

try: # server-compatible import statement
    from models import *
    from rlnet import car_speeds
    from agent import RLAgent
except:
    from .models import *
    from .rlnet import car_speeds
    from .agent import RLAgent


class ExampleAgent(Agent):
    '''
    An example agent that just output a random action.
    '''
    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.

        For example, you might want to load the appropriate neural networks weight
        in this method.
        '''
        #test_case_id = kwargs.get('test_case_id')
        self.model = kwargs.get('model')

        '''
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def reset(self, state, *args, **kwargs):
        '''
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        '''
        '''
        # Uncomment to help debugging
        print('>>> RESET >>>')
        print('state:', state)
        '''
        # for speed of cars
        self.internal_state = state
        self.reset_state = True
        self.agent_state = None

    def step(self, state, *args, **kwargs):
        '''
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''
        # ((1, 10, 50))


        if (self.reset_state):
            car_speed = np.zeros((1, 10, 50))
            self.reset_state = False
        else:
            car_speed = car_speeds(self.internal_state, state)

        agent_state = np.concatenate((state, car_speed), axis = 0)
        self.agent_state = agent_state

        action = self.model.act(agent_state)

        self.internal_state = state

        return action

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''



def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    # model_class, model_state_dict, input_shape, num_actions = 'SLNetwork', torch.load(model_path), (5, 10, 50), 5
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)

    return RLAgent(model = model)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_random_lane_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
        agent.initialize(**agent_init)
        for run in range(runs):
            state = env.reset()
            agent.reset(state)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
            if episode_rewards < 10:
                print('Got reward {} after {} / {} steps with state {}'.format(episode_rewards,
                                                                               t + 1, t_max, env.world.agent_state))
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_random_lane_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    timed_test(task)
