# Intro2AI Project: Navigating a Two-Wheeled Robot with Reinforcement Learning

This project explores the application of reinforcement learning algorithms to train a two-wheeled robot to navigate an unknown environment with obstacles. The algorithms used in this project include Tabular Q-Learning, Deep Q-Learning (DQN), and Multi-Frequency Deep Q-Learning (MFDQN). The project aims to compare the effectiveness of these algorithms in terms of the time taken to reach the goal and cumulative reward achieved.

## Project Overview

The objective of this project is to evaluate three different reinforcement learning algorithms in learning a policy to control a two-wheeled robot. The robot starts at a random position and must navigate to a goal while avoiding obstacles. The performance of each algorithm is assessed based on the time to reach the goal and the cumulative reward.

### Algorithms Evaluated

- **Tabular Q-Learning**: A model-free reinforcement learning algorithm that uses a table to store q-values for state-action pairs. It employs an ϵ-greedy policy for exploration and updates q-values using the Bellman equation.

- **Deep Q-Learning (DQN)**: An extension of Q-Learning that uses a neural network to approximate the q-function. This implementation uses the double deep Q-learning algorithm for stability and experience replay for efficient learning.

- **Multi-Frequency Deep Q-Learning (MFDQN)**: Enhances DQN by introducing multiple policy networks operating at different frequencies. This approach aims to improve efficiency by balancing high-level and low-level actions.

## Project Structure

- **Environment Setup**: The simulated environment uses PyMunk, a 2D physics engine, with a 600x600 unit square area. The robot, goal, and obstacles are initialized with random positions and sizes at the start of each episode.

- **State Space**: The robot uses sensors to detect the environment, with both discrete and continuous representations of state space utilized in different algorithms.

- **Action Space**: The robot can perform four discrete actions: moving both wheels forward, moving both wheels backward, and turning clockwise or counterclockwise.

- **Rewards**: The agent receives positive rewards for reaching the goal, penalties for collisions, and incremental rewards based on movement toward the goal.

## Experimental Setup

The project consists of training each algorithm over 100 epochs, with the environment reset at the start of each epoch. The algorithms are evaluated based on the average time taken to reach the goal and the cumulative reward per epoch.

## Results

- **Tabular Q-Learning** showed consistent cumulative reward performance but was slower compared to deep learning-based approaches.
  
- **Deep Q-Learning (DQN)** achieved the fastest time to the goal and maintained relatively high cumulative rewards.

- **Multi-Frequency Deep Q-Learning (MFDQN)** demonstrated interesting behavior, with the high-level policy focusing on movement and the low-level policy on obstacle avoidance, though it had lower cumulative rewards.

## Conclusion

The project concluded that Deep Q-Learning is the most effective algorithm for training the robot to navigate the environment quickly. However, Tabular Q-Learning excelled in maximizing cumulative rewards. The MFDQN approach offered insights into multi-frequency action strategies but requires further tuning for optimal performance.
