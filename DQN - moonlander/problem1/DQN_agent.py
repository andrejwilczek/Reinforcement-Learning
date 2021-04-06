# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import numpy as np
import gym
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pickle


class DQN(object):

    def __init__(self, n_actions, dim_states, buffer_size, discount_factor):
        self.Agent = Agent(n_actions)
        self.Network = MyNetwork(dim_states, n_actions, discount_factor)
        self.TargetNetwork = pickle.loads(pickle.dumps(self.Network))
        self.ReplayBuffer = ExperienceReplayBuffer(buffer_size)


class Agent(object):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def epsilonGreedy(self, Q_tensor, epsilon):
        p = np.random.uniform(0, 1, 1)
        if p < epsilon:
            self.last_action = np.random.randint(0, self.n_actions)
            return self.last_action
        else:
            self.last_action = Q_tensor.max(1)[1].item()
            return self.last_action


Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError(
                'Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)

### Neural Network ###


class MyNetwork(nn.Module):
    """ Create a feedforward neural network """

    def __init__(self, input_size, output_size, discount_factor):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, 128)
        self.input_layer_activation = nn.ReLU()
        self.discount_factor = discount_factor

        # Create hidden layer
        self.hidden_layer_1 = nn.Linear(128, 128)
        # self.hidden_layer_2 = nn.Linear(128, 64)
        self.hidden_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, state_tensor, grad=True):
        # Function used to compute the forward pass
        # state_tensor = torch.tensor([state],
        #                             requires_grad=grad,
        #                             dtype=torch.float32)

        # Compute first layer
        l1 = self.input_layer(state_tensor)
        l1 = self.input_layer_activation(l1)

        # Compute hidden layer
        h1 = self.hidden_layer_1(l1)
        h1 = self.hidden_layer_activation(h1)

        # h2 = self.hidden_layer_2(h1)
        # h2 = self.hidden_layer_activation(h2)

        # Compute output layer
        out = self.output_layer(h1)
        return out

    def target_values(self, next_states, rewards, dones, Q_target):
        targets = []
        Q_np = Q_target.detach().numpy()[0]

        for k, done in enumerate(dones):
            if not done:
                y = rewards[k] + self.discount_factor * np.max(Q_np[k, :])
            else:
                y = rewards[k]
            targets.append(y)
        target_tensor = torch.tensor([targets],
                                     requires_grad=True,
                                     dtype=torch.float32)
        return target_tensor

    def backward(self, target_values, states, actions):
        states_tensor = torch.tensor([states],
                                     requires_grad=True,
                                     dtype=torch.float32)
        Q = self.forward(states_tensor)[0]
        Q_loss = torch.zeros(1, len(actions),
                             requires_grad=False,
                             dtype=torch.float32)
        for i, action in enumerate(actions):
            Q_loss[0, i] = Q[i, action]

        loss = nn.functional.mse_loss(
            Q_loss, target_values)

        return loss


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
