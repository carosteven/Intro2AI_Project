import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from itertools import count
from time import time
from nav_obstacle_env import Nav_Obstacle_Env

import logging
logging.getLogger('pymunk').propagate = False
logging.basicConfig(filename='ql.log',level=logging.DEBUG)

env = Nav_Obstacle_Env()
epsilon = 0.1
gamma = 0.8
action_freq = 25
DEFAULT_ACT = env.available_actions[0]
train = False
continue_training = False

def init_Q_n(Q, n, s):
    for a in env.available_actions:
        Q[(s, a)] = 0
        n[(s, a)] = 0
    return Q, n

def policy(Q, s):
    # Find argmax_a from Q
    rewards = [Q[(s,i)] for i in env.available_actions]
    best_action = np.argmax(rewards)
    return best_action

def sample_next_action(Q, s):
    method = np.random.choice(['exploration', 'exploitation'], p=[epsilon, 1-epsilon])

    if method == 'exploration':
        next_action = np.random.choice(len(env.available_actions))

    elif method == 'exploitation':
        next_action = policy(Q, s)

    return next_action

def get_state():
    sensor = (round(env.left_sensor_data[2]/3)*3 if env.left_sensor_data is not None else 100, round(env.right_sensor_data[2]/3)*3 if env.right_sensor_data is not None else 100)

    velocity = round(env._agent['robot'].velocity.length/5.0)*5
    direction = (round(env._agent['robot'].rotation_vector.perpendicular().x), round(env._agent['robot'].rotation_vector.perpendicular().y))

    state = []
    # state.append(velocity)
    for stat in [sensor, direction]:
        for coord in stat:
            state.append(coord)
    return tuple(state)

def train_model(epochs=1, Q={}, n={}, reward_stats=[], time_stats=[]):
    # reward_stats = []
    # time_stats = []
    logging.info("Training Started")
    for epoch in tqdm(range(epochs)):
        reward_stats.append(0)
        time_stats.append(time())
        env.reset()

        state = get_state()
         # Initialize Q and n (if necessary)
        if Q.get((state, DEFAULT_ACT)) == None:
            Q, n = init_Q_n(Q, n, state)

        epi = 0
        done = False
        # Loop unitl reaches goal
        # for frame in tqdm(range(100000)):
        for frame in count():
            if frame % action_freq == 0:
                action = env.available_actions[sample_next_action(Q, state)]
                env.step(action)

                epi += 1

            elif frame % action_freq == action_freq - 1:
                # Observe s' and r
                _, reward, done, _ = env.step(None)
                reward_stats[-1] += reward
                print(reward_stats[-1], end='\r')
                next_state = get_state()

                # Initialize Q and n (if necessary)
                if Q.get((next_state, DEFAULT_ACT)) == None:
                    Q, n = init_Q_n(Q, n, next_state)

                # Update counts
                n[(state, action)] += 1

                # Learning rate
                a = 1 / (n[(state, action)])
                
                # Update Q-value
                next_rewards = [Q[(next_state, i)] for i in env.available_actions]
                Q[(state, action)] = Q[(state, action)] + a*(reward + gamma*max(next_rewards) - Q[(state, action)])
                
                state = next_state
                
            else:
                env.step(None)
            
            if done:
                break

        time_stats[-1] = time() - time_stats[-1]
        logging.info(f"Epoch {epoch} complete. Time: {time_stats[-1]} Reward: {reward_stats[-1]}")
        f = open('Q.pckl', 'wb')
        pickle.dump([Q, n, reward_stats, time_stats], f)
        f.close()

    return [Q, n]

def test_model(Q_actual, n_actual):
    env.__init__()
    Q = Q_actual
    n = n_actual
    frame = 0
    while not env._done:
        # env.step()
        print(f"Score: {round(env.reward, 2)}", end='\r')

        current_state = get_state()
        if Q.get((current_state, DEFAULT_ACT)) == None:
                Q, n = init_Q_n(Q, n, current_state)
        if frame % action_freq == 0:
            action = env.available_actions[policy(Q, current_state)]
            env.step(action)
        else:
            env.step(None)
        frame += 1


train = False

if train == True:
    if continue_training == True:
        f = open('Q.pckl', 'rb')
        Q, n, r, t = pickle.load(f)
        f.close()
        logging.info("Resuming training...")

    else:
        Q = {}
        n = {}
        r = []
        t = []

    Q, n = train_model(100, Q, n, r, t)
    # f = open('Q.pckl', 'wb')
    # pickle.dump([Q, n], f)
    # f.close()

else:
    f = open('Q_old.pckl', 'rb')
    Q, n,r,t = pickle.load(f)
    f.close()
    print(r)
    plt.plot(r)
    plt.show()
    test_model(Q, n)

