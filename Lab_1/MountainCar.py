import gym
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits import mplot3d
import numpy as np

from Lab_2 import plotting

n_states = 40
iter_max = 10000

initial_lr = 1.0  # Learning rate
min_lr = 0.3
gamma = 1.0
t_max = 10000
epsilon = 0.2

env_name = "MountainCar-v0"
env = gym.make(env_name)
obs = env.reset()
env.render()
# n_states = 40
episodes = 10
# initial_lr = 1.0
# min_lr = 0.05
# gamma = 0.99
max_stps = 10000
# epsilon = 0.5
env = env.unwrapped
env.seed()
np.random.seed(0)

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    # ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()

def discretization(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_den = (env_high - env_low) / n_states
    pos_den = env_den[0]
    vel_den = env_den[1]
    pos_high = env_high[0]
    pos_low = env_low[0]
    vel_high = env_high[1]
    vel_low = env_low[1]
    pos_scaled = int((obs[0] - pos_low) / pos_den)
    vel_scaled = int((obs[1] - vel_low) / vel_den)

    return pos_scaled, vel_scaled


q_table = np.zeros((n_states, n_states, env.action_space.n))
total_steps = 0
q_table_arr = []
actions_arr = []
velocity_arr = []
positions_arr = []
for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    total_reward = 0
    alpha = max(min_lr, initial_lr * (gamma ** (episode // 100)))
    steps = 0
    counter = 0
    while True:
        env.render()
        pos, vel = discretization(env, obs)
        if np.random.uniform(low=0, high=1) < epsilon:
            a = np.random.choice(env.action_space.n)
        else:
            a = np.argmax(q_table[pos][vel])
        obs, reward, terminate, _ = env.step(a)
        total_reward += abs(obs[0] + 0.5)
        pos_, vel_ = discretization(env, obs)
        q_table[pos][vel][a] = (1 - alpha) * q_table[pos][vel][a] + alpha * (
                reward + gamma * np.max(q_table[pos_][vel_]))

        # print(q_table[pos][vel][a])
        q_table_arr.append(q_table[pos][vel][a])
        actions_arr.append(a)
        velocity_arr.append(vel)
        positions_arr.append(pos)
        # print(pos, ' ', vel, ' ', a)
        counter += 1
        steps += 1

        if terminate:
            break
    print('Episode: ', episode)
    print('Q-table: ', q_table_arr)
    print('Actions: ', actions_arr)
    print('Velocity: ', velocity_arr)
    print('Positions: ', positions_arr)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions_arr, velocity_arr, actions_arr, c='r', marker='o')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Action')
    plt.show()
    q_table_arr = []
    actions_arr = []
    velocity_arr = []
    positions_arr = []

while True:
    env.render()