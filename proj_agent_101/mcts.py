from copy import deepcopy
from functools import partial

import torch
import gym
from gym.utils import seeding
import gym_grid_driving
import math

from gym_grid_driving.envs.grid_driving import DenseReward

random = None
SUBMISSION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.load = partial(torch.load, map_location=device)
model_path = 'model.pt'


def randomPolicy(state, model):
    '''
    Policy followed in MCTS simulation for playout
    '''
    reward = model.value(state.state)
    return reward


class GridWorldState:

    def __init__(self, state, reward=0, is_done=False):
        '''
        Data structure to represent state of the environment
        self.env : Environment of gym_grid_environment simulator
        self.state : State of the gym_grid_environment
        self.is_done : Denotes whether the GridWorldState is terminal
        self.num_lanes : Number of lanes in gym_grid_environment
        self.width : Width of lanes in gym_grid_environment
        self.reward : Reward of the state
        '''
        self.state = deepcopy(state)
        self.is_done = is_done  # if is_done else False
        # if self.state.agent.position.x < 0:
        #     self.is_done = True
        #     self.state.agent.position.x = 0
        # self.reward = reward

    def simulateStep(self, env, action):
        '''
        Simulates action at self.state and returns the next state
        '''
        env = deepcopy(env)
        state_desc = env.step(action=action)
        newState = GridWorldState(state=state_desc[0], reward=state_desc[1], is_done=state_desc[2])
        return newState
    #
    # def isDone(self):
    #     '''
    #     Returns whether the state is terminal
    #     '''
    #     return self.is_done
    #
    # def getReward(self):
    #     '''
    #     Returns reward of the state
    #     '''
    #     return self.reward


class Node:
    def __init__(self, state, parent=None):
        '''
        Data structure for a node of the MCTS tree
        self.state : GridWorld state represented by the node
        self.parent : Parent of the node in the MCTS tree
        self.numVisits : Number of times the node has been visited
        self.totalReward : Sum of all rewards backpropagated to the node
        self.isDone : Denotes whether the node represents a terminal state
        self.allChildrenAdded : Denotes whether all actions from the node have been explored
        self.children : Set of children of the node in the MCTS tree
        '''
        self.state = state
        self.parent = parent
        self.numVisits = 0
        self.totalReward = state.reward #0
        self.isDone = state.isDone()
        self.children = {}

    @property
    def allChildrenAdded(self):
        return len(self.children)


class MonteCarloTreeSearch:
    def __init__(self, env, numiters, model, explorationParam, playoutPolicy=randomPolicy, random_seed=None):
        '''
        self.numiters : Number of MCTS iterations
        self.explorationParam : exploration constant used in computing value of node
        self.playoutPolicy : Policy followed by agent to simulate rollout from leaf node
        self.root : root node of MCTS tree
        '''
        self.env = env
        self.numiters = numiters
        self.explorationParam = explorationParam
        self.playoutPolicy = playoutPolicy
        self.root = None
        self.model = model
        global random
        random, seed = seeding.np_random(random_seed)

    def buildTreeAndReturnBestAction(self, initialState):
        '''
        Function to build MCTS tree and return best action at initialState
        '''
        self.root = Node(state=initialState, parent=None)
        for i in range(self.numiters):
            self.addNodeAndBackpropagate(i)
        bestChild = self.chooseBestActionNode(self.root, 0)
        for action, cur_node in self.root.children.items():
            if cur_node is bestChild:
                return action

    def addNodeAndBackpropagate(self, iteration):
        '''
        Function to run a single MCTS iteration
        '''
        node = self.addNode(iteration)
        reward = self.playoutPolicy(node.state, self.env)
        self.backpropagate(node, reward)

    def addNode(self, iteration):
        import random
        '''
        Function to add a node to the MCTS tree
        '''
        cur_node = self.root
        while not cur_node.isDone:
            if cur_node.allChildrenAdded:
                cur_node = self.chooseBestActionNode(cur_node, self.explorationParam)
            else:
                actions = self.env.actions
                # if iteration < self.numiters / 2:
                #     sample_k = int(np.log10(iteration + 1) * 2) + 1
                #     # print(sample_k)
                #     actions = random.sample(actions, sample_k)
                for action in actions:
                    if action not in cur_node.children:
                        childnode = cur_node.state.simulateStep(env=self.env, action=action)
                        newNode = Node(state=childnode, parent=cur_node)
                        cur_node.children[action] = newNode
                        if len(actions) == len(cur_node.children):
                            cur_node.allChildrenAdded = True
                        return newNode
        return cur_node

    def backpropagate(self, node, reward):
        '''
        FILL ME : This function should implement the backpropation step of MCTS.
                  Update the values of relevant variables in Node Class to complete this function
        '''
        while node:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def chooseBestActionNode(self, node, explorationValue):
        global random
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            '''
            FILL ME : Populate the list bestNodes with all children having maximum value

                       Value of all nodes should be computed as mentioned in question 3(b).
                       All the nodes that have the largest value should be included in the list bestNodes.
                       We will then choose one of the nodes in this list at random as the best action node. 
            '''
            value = child.totalReward / child.numVisits + \
                    explorationValue * (math.sqrt(math.log(node.numVisits, math.e) / child.numVisits))
            if value > bestValue:
                bestValue = value
                bestNodes = [child]
            elif value == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)


if not SUBMISSION:
    from env import construct_random_lane_env
    import torch
    import numpy as np


    def get_model():
        '''
        Load `model` from disk. Location is specified in `model_path`.
        '''
        model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
        model = eval(model_class)(input_shape, num_actions).to(device)
        model.load_state_dict(model_state_dict)


    def get_action(a):
        if isinstance(a, int):
            return a
        action_space = {'up': 0, 'down': 1, 'forward[-3]': 2, 'forward[-2]': 3, 'forward[-1]': 4}
        return action_space[action.name]

    simulations = 3
    simulations_file = 'simulations.npy'
    actions_file = 'actions.npy'

    model = get_model()

    for i in range(simulations):
        env = construct_random_lane_env()
        RANDOM_SEED = np.random.randint(0, 100)
        numiters = 1
        env.render()
        done = False
        data = []
        actions = []
        mcts = MonteCarloTreeSearch(env=env, numiters=numiters, model=model, explorationParam=1, random_seed=RANDOM_SEED)
        while not env.done:
            state = GridWorldState(env.state, is_done=done)
            action = mcts.buildTreeAndReturnBestAction(initialState=state)
            print(action)
            done = env.step(state=deepcopy(state.state), action=action)[2]
            env.render()
            data.append(env.world.tensor_state)
            actions.append(get_action(action))
            if done == True:
                break
        data = np.array(data)
        actions = np.array(actions)
        print(f"simulation {i} done with gathered data of shape {data.shape}")
        try:
            with open(simulations_file, 'rb') as f:
                prev_data = np.load(f)
            with open(actions_file, 'rb') as f:
                prev_actions = np.load(f)
        except:
            prev_data = None
            prev_actions = None
        with open(simulations_file, 'wb') as f:
            f.truncate(0)
            if prev_data is not None and prev_actions is not None:
                data = np.vstack((prev_data, data))
            np.save(f, data)

        with open(actions_file, 'wb') as f:
            f.truncate(0)
            if prev_data is not None and prev_actions is not None:
                actions = np.concatenate((prev_actions, actions))
            np.save(f, actions)


