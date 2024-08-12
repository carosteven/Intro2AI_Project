import matplotlib.pyplot as plt
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
checkpoint_path = 'checkpoint_DDQN_Multi_novel.pt'


policy_net_lo = models.SensorDQN(n_observations, n_actions).to(device)
policy_net_hi = models.SensorDQN(n_observations, n_actions).to(device)
policy_net_lo.eval()
policy_net_hi.eval()

checkpoint = torch.load(checkpoint_path, map_location=device)
policy_net_lo.load_state_dict(checkpoint['policy_lo_state_dict'])
policy_net_hi.load_state_dict(checkpoint['policy_hi_state_dict'])

print(len(checkpoint['time_stats']))
plt.plot(checkpoint['time_stats'])

plt.show()

def select_action(state, policy):
    with torch.no_grad():
        qvalues = policy(state)
    action = torch.argmax(qvalues).item()
    return action

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False
print(checkpoint['epoch'])
frame = 0
total_actions = 0
action_freq = 25
low_to_high_freq = 3
action_tracker = []
while not done:
    if frame % action_freq == 0:
        if total_actions % low_to_high_freq == 0:
            action = select_action(state, policy_net_hi)
            env.step(env.available_actions[action])
            action_tracker.append(action)
        else:
            action = select_action(state, policy_net_lo)
            env.step(env.available_actions[action])
            action_tracker.append(action)
    
    elif frame % action_freq == action_freq-1:
        observation, reward, done, _ = env.step(None)
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        if total_actions % low_to_high_freq != 0:
            # print(reward, end='\r')
            pass
        if total_actions % low_to_high_freq == low_to_high_freq-1:
            print(action_tracker, end='\r')
            action_tracker = []
        total_actions += 1

    else:
        env.step(None)

    if done:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
    frame += 1
print(reward)
