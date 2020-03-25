#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nick Tacca

Policy Gradient: REINFORCE with Baseline
"""

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from helpers import simulator, visualization

class Policy(nn.Module):
    
    """Defining neural network architecture for policy and value estimate"""
    
    def __init__(self, n_states, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, n_output)

        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        output = F.softmax(self.linear4(x), dim=-1)
        return output

actions = np.linspace(-10, 10, 5)
#state = np.array([np.pi, 0, 0, 0])
state = np.array([0, 0, 0, 0])

n_states = state.shape[0]
n_actions = len(actions)

s = torch.FloatTensor(state)

policy = Policy(n_states, 512, 512, 256, n_actions)
#policy.load_state_dict(torch.load("policy.pt"))
#policy.eval()
s_value_func = Policy(n_states, 512, 512, 256, 1)
#s_value_func.load_state_dict(torch.load("value.pt"))
#s_value_func.eval()

optimizer_theta = optim.Adam(policy.parameters(), lr=1e-4)
gamma = 0.999

seed = 1
torch.manual_seed(seed)

def reinforce_agent():

#    state = np.array([np.pi, 0, 0, 0])
    state = np.array([0, 0, 0, 0])

    state_sequence = []
    log_act_prob = []
    action_list = []
    total_rewards = []
    
    # One episode of training --> 50 actions
    while len(action_list) < 50:
        s = torch.from_numpy(state).unsqueeze(0).float()
        state_sequence.append(deepcopy(s))
        action_probs = policy(s)
        m = Categorical(action_probs)
        action = m.sample()
        m_log_prob = m.log_prob(action)
        log_act_prob.append(m_log_prob)
        a = actions[action]
        action_list.append(a)
        snext, rnext, done = simulator(s.detach().numpy().squeeze(0), a, 0.1, 0.01)
        policy.reward.append(rnext)
        total_rewards.append(rnext)
        state = snext

    R = 0
    Gt = []

    # Discount rewards
    for reward in policy.reward[::-1]:
        R = reward + gamma * R
        Gt.insert(0, R)
    
#    # Standardize rewards
#    Gt = (Gt - np.mean(Gt)) / np.std(Gt)
    
    # Divide by std dev
#    Gt /= np.std(Gt)

    total_loss_v = 0
    total_loss_theta = 0
    # Update networks
    for i in range(len(Gt)):
        G = Gt[i]
        V = s_value_func(state_sequence[i])
        delta = G - V
        
        # Update value network
        optimizer_v = optim.Adam(policy.parameters(), lr=1e-3)
        optimizer_v.zero_grad()
        loss_v = -delta
        total_loss_v += loss_v
        loss_v.backward(retain_graph = True)
        clip_grad_norm_(loss_v, 0.9)
        optimizer_v.step()

        # Update policy network
        optimizer_theta.zero_grad()
        loss_theta = -log_act_prob[i] * delta
        total_loss_theta += loss_theta
        loss_theta.backward(retain_graph = True)
        clip_grad_norm_(loss_theta, 0.9)
        optimizer_theta.step()
    
    # Start from zero for new episode
    del policy.log_act_probs[:]
    del policy.reward[:]
    
    return total_rewards, total_loss_theta, total_loss_v

def rolling_mean(x, N):
    """ Determines rolling mean of x over a given N window"""
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return np.array((cumsum[N:] - cumsum[:-N]) / float(N))

def main():

    """ Main function for training agent and updating networks. To balance
    the pole starting in the "up" position, 500 episodes is sufficient.
    To swing the pole up from the "down" position, 100000 episodes
    is required"""
    
    total_rewards = []
    total_loss_theta = []
    total_loss_v = []

    for episode in range(500):
        rewards, loss_theta, loss_v = reinforce_agent()
        total_rewards.append(np.sum(rewards))
        total_loss_theta.append(loss_theta)
        total_loss_v.append(loss_v)
        
        # Print total rewards
        print("\rEp: {} Total Rewards: {:.2f}".format(
            episode + 1, total_rewards[episode], end=""))
        
        if np.mean(total_rewards[-10:]) == 0:
            break

    torch.save(policy.state_dict(), "policy.pt")
    torch.save(s_value_func.state_dict(), "value.pt")
    
    # Rolling means
    window = 100
    mean_total_rewards = rolling_mean(total_rewards, window)
    mean_total_loss_theta = rolling_mean(total_loss_theta, window)
    mean_total_loss_v = rolling_mean(total_loss_v, window)
    
    # Plots
    line1, = plt.plot(total_rewards, label = "Total Rewards")
    line2, = plt.plot(mean_total_rewards, label = "Rolling Mean")
    legend = plt.legend(handles = [line1, line2], loc=1)
    plt.gca().add_artist(legend)
    plt.ylabel('Cummulative Rewards')
    plt.xlabel('Episodes')
    plt.grid()
    plt.show()
    
    line1, = plt.plot(total_loss_theta, label = "Total Policy Loss")
    line2, = plt.plot(mean_total_loss_theta, label = "Rolling Mean")
    legend = plt.legend(handles = [line1, line2], loc=1)
    plt.gca().add_artist(legend)
    plt.ylabel('Cummulative Policy Loss')
    plt.xlabel('Episodes')
    plt.grid()
    plt.show()
    
    line1, = plt.plot(total_loss_v, label = "Total Value Loss")
    line2, = plt.plot(mean_total_loss_v, label = "Rolling Mean")
    legend = plt.legend(handles = [line1, line2], loc=1)
    plt.gca().add_artist(legend)
    plt.ylabel('Cummulative Value Loss')
    plt.xlabel('Episodes')
    plt.grid()
    plt.show()
    
    # Visualize after training
#    state = np.array([np.pi, 0, 0, 0])
    state = np.array([0, 0, 0, 0])
    
    for k in range(600):
        s = torch.from_numpy(state).unsqueeze(0).float()
        action_probs = policy(s)
        m = Categorical(action_probs)
        action = m.sample()
        a = actions[action]
        snext, rnext, done = simulator(s.detach().numpy().squeeze(0), a, 0.1, 0.01)
        visualization(s.detach().numpy().squeeze(0))
        state = snext
    
    plt.show()
    
if __name__ == '__main__':
    main()
