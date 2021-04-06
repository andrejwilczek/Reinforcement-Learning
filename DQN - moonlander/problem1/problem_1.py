# Andrej Wilczek 880707-7477
# Kildo Alias 971106-7430

# Load packages
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQN, Experience
import copy
import math
import pickle
import time


##########################################################################

from collections import namedtuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
mpl.rc('animation', html='jshtml')


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def epsilon_decay_linear(emax, emin, Z, i):
    efunc = emax - ((emax-emin)*i)/(Z-1)
    epsilon = max(emin, efunc)
    return epsilon


def epsilon_decay_exp(emax, emin, Z, i):
    efunc = emax*(emin/emax)**(i/(Z-1))
    epsilon = max(emin, efunc)
    return epsilon


if __name__ == "__main__":

    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Parameters
    N_episodes = 500                             # Number of episodes
    discount_factor = 1                       # Value of the discount factor
    n_ep_running_average = 50                    # Running average of 50 episodes
    n_actions = env.action_space.n               # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality
    N_batchsize = 32                             # Traing batchsize
    buffersize = 15000
    C = int(buffersize/N_batchsize)

    # Epsilon decay
    emax = 0.99
    emin = 0.05
    Z = 0.95*N_episodes

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # Init DQN class
    DQN = DQN(n_actions, dim_state, buffersize, discount_factor)

    ### Create Experience replay buffer ###
    buffer = DQN.ReplayBuffer

    ### Create network ###
    network = DQN.Network

    ### Create optimizer ###
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    ### Create target network ###
    target_network = DQN.TargetNetwork

    # Agent initialization
    agent = DQN.Agent

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        # frames = []
        state = env.reset()
        total_episode_reward = 0.
        epsilon = epsilon_decay_exp(emax, emin, Z, i)
        t = 0
        while not done:
            if t % C == 0:
                # target_network = copy.deepcopy(network)
                target_network = pickle.loads(pickle.dumps(network))
                # print('network updated')

            #! Animation:
            if i % 20 == 0:
                env.render()
            # frames.append(img)

            # Calculate Q-tensor
            state_tensor = torch.tensor([state],
                                        requires_grad=True,
                                        dtype=torch.float32)

            Q = network.forward(state_tensor, grad=True)

            # Get epsilon greedy actrion
            action = agent.epsilonGreedy(Q, epsilon)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            # Append experience to the buffer
            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            # Update episode reward
            total_episode_reward += reward

            ### TRAINING ###
            # Perform training only if we have more than N_batchsize elements in the buffer
            if len(buffer) >= N_batchsize:
                # Sample a batch of 3 elements
                states, actions, rewards, next_states, dones = buffer.sample_batch(
                    n=N_batchsize)
                # Training process, set gradients to 0
                optimizer.zero_grad()

                next_states_tensor = torch.tensor([next_states],
                                                  requires_grad=True,
                                                  dtype=torch.float32)
                # Compute output of the network given the states batch
                target_Q = target_network.forward(next_states_tensor)
                target_values = target_network.target_values(
                    next_states, rewards, dones, target_Q)

                loss = network.backward(target_values, states, actions)
                # Compute gradient
                loss.backward()
                # Clip gradient norm to 1
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.)
                # Perform backward pass (backpropagation)
                optimizer.step()
            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    torch.save(network, 'LunarLanderNetwork.pth')

    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes+1)],
               episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes+1)],
               episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()
