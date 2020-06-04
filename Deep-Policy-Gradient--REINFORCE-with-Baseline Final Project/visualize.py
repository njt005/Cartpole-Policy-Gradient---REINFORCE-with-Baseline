#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:48:37 2020

@author: Nick Tacca
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.distributions import Categorical
from helpers import simulator, visualization
from cartpole2 import Policy
 
def main():   

    # Loading saved model
    # state = np.array([np.pi, 0, 0, 0])
    state = np.array([0, 0, 0, 0])
    actions = np.linspace(-10, 10, 9)
    n_states = state.shape[0]
    n_actions = len(actions)
    policy = Policy(n_states, 500, 500, n_actions)
    policy.load_state_dict(torch.load("policy.pt"))
    policy.eval()
    
    # Visualize after training
    
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
