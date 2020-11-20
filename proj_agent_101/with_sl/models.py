import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)

class DQN(Base):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

class AtariDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        r = random.uniform(0,1)
        if r < epsilon:
        # return random action
            return int(np.random.choice(self.num_actions, 1)[0])
        else:
        # return greedy action
            Q_vals = self.forward(state)
            return int(torch.argmax(Q_vals, 1)[0])

class DuellingDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.A = nn.Sequential(
            nn.Linear(self.feature_size(), 400),
            nn.ReLU(),
            nn.Linear(400, self.num_actions)
        )
        self.V = nn.Sequential(
            nn.Linear(self.feature_size(), 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        V = self.V(x)
        A = self.A(x)
        return V,A


    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''

        r = random.uniform(0,1)

        if r < epsilon:
            # return random action
            return int(np.random.choice(self.num_actions, 1)[0])
        else:
            # return greedy action
            _, advantage = self.forward(state)
            return int(torch.argmax(advantage, 1)[0])


class SLNetwork(DuellingDQN):
    def forward(self, x):
        return F.softmax(super().forward(x)[1], dim=1)

    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''

        r = random.uniform(0, 1)

        if r < epsilon:
            # return random action
            return int(np.random.choice(self.num_actions, 1)[0])
        else:
            # return greedy action
            advantage = self.forward(state)
            return int(torch.argmax(advantage, 1)[0])
