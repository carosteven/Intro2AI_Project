import matplotlib
import math
import random
import numpy as np
from  collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
from nav_obstacle_env import Nav_Obstacle_Env
import models

# env = Wheeled_Robot_Sim(state_type='')
env = Nav_Obstacle_Env()
env.state_type = 'sensor'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Get number of actions from env
n_actions = len(env.available_actions)
# Get number of state observations
state = env.reset()
n_observations = len(state)
# print(state)
# checkpoint_path = 'model - no pushing.pt'
checkpoint_path = 'checkpoint_DDQN_sensor.pt'


policy_net = models.SensorDQN(n_observations, n_actions).to(device)
print(n_observations, n_actions)
policy_net.eval()

checkpoint = torch.load(checkpoint_path, map_location=device)
policy_net.load_state_dict(checkpoint['policy_state_dict'])

def select_action(state):
    return policy_net(state).max(1).indices.view(1,1)

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False
print(checkpoint['epoch'])
frame = 0
action_freq = 25
while not done:
    if frame % action_freq == 0:
        action = select_action(state)
        env.step(env.available_actions[action])
        
    elif frame % action_freq == action_freq-1:
        state, reward, done, info = env.step(None)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print(reward, end='\r')

    else:
        env.step(None)
    if done:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
    frame += 1
print(reward)