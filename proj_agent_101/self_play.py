from functools import partial

import numpy as np
import torch
from gym_grid_driving.envs.grid_driving import AgentState

try:
    from .env import construct_random_lane_env, EnvWrapper
    from .agent import RLAgent
    from .models import DuellingDQN
except:
    from env import construct_random_lane_env, EnvWrapper
    from agent import RLAgent
    from models import DuellingDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.load = partial(torch.load, map_location=device)
model_path = 'model.pt'


def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`.
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)


def create_agent():
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return RLAgent(model=model)


def get_readable_action(a):
    action_space = {0: 'up', 1: 'down', 2: 'forward[-3]', 3: 'forward[-2]', 4: 'forward[-1]'}
    if isinstance(a, int):
        return action_space[a]
    return action_space[a.name]


def play(agent, env, runs=1000, t_max=40, savepoints=50):
    rewards = []
    good_states = []
    good_actions = []
    states_file = 'states.npy'
    actions_file = 'actions.npy'
    for run in range(runs):
        state = env.reset()
        agent.reset(state)
        episode_rewards = 0.0
        states = []
        actions = []
        for t in range(t_max):
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            full_state = {
                'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                'done': done, 'info': info
            }
            agent.update(**full_state)
            states.append(agent.agent_state.tolist())
            actions.append(action)
            state = next_state
            episode_rewards += reward
            if done:
                break
        if t > 3:
            if env.world.agent_state != AgentState.alive:
                actions = actions[:-1]
                states = states[:-1]
            good_actions += actions
            good_states += states
        if (run + 1) % savepoints == 0:
            print('[Episode {}]: reward: {}'.format(run + 1, sum(rewards) / len(rewards)))
            if len(good_states) == 0:
                continue
            try:
                with open(states_file, 'rb') as f:
                    prev_states = np.load(f)
                with open(actions_file, 'rb') as f:
                    prev_actions = np.load(f)
            except Exception as e:
                print(e)
                prev_states = None
                prev_actions = None
            good_states = np.array(good_states)
            good_actions = np.array(good_actions)
            with open(states_file, 'wb') as f:
                f.truncate(0)
                if prev_states is not None and prev_actions is not None:
                    good_states = np.vstack((prev_states, good_states))
                np.save(f, good_states)

            with open(actions_file, 'wb') as f:
                f.truncate(0)
                if prev_states is not None and prev_actions is not None:
                    good_actions = np.concatenate((prev_actions, good_actions))
                np.save(f, good_actions)
            good_states = []
            good_actions = []
        rewards.append(episode_rewards)
    avg_rewards = sum(rewards ) / len(rewards)
    print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
    return avg_rewards

def run():
    env = construct_random_lane_env()
    agent = create_agent()
    print(play(agent, env))