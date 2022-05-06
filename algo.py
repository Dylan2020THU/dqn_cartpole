# Born time: 2022-5-6
# Latest update: 2022-5-6
# RL brain
# Hengxi


import torch.nn as nn
import torch

import gym
from collections import deque
import itertools
import numpy as np
import random

from policy import Network


GAMMA = 0.99
MINI_BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQUENCY = 1000

env = gym.make("CartPole-v1")

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())  # ?

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
obs = env.reset()
for i in range(MIN_REPLAY_SIZE):
    a = env.action_space.sample()
    obs_, r, done, info = env.step(a)  # info will not be used
    memory = (obs, a, r, done, obs_)
    replay_buffer.append(memory)
    obs = obs_

    if done:
        obs = env.reset()

# Main Training Loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()

    if random_sample <= epsilon:
        a = env.action_space.sample()
    else:
        a = online_net.act(obs)

    obs_, r, done, info = env.step(a)  # info will not be used
    memory = (obs, a, r, done, obs_)  # memories: tuple
    replay_buffer.append(memory)
    obs = obs_

    episode_reward += r

    if done:
        obs = env.reset()

        reward_buffer.append(episode_reward)

    # After solved, watch it play
    if len(reward_buffer) >= 100:
        if np.mean(reward_buffer) >= 30000:
            while True:
                a = online_net.act(obs)
                obs, r, done, info = env.step(a)
                env.render()

                if done:
                    env.reset()

    # Start Gradient Step
    memories = random.sample(replay_buffer, MINI_BATCH_SIZE)

    all_obs = np.asarray([memo_i[0] for memo_i in memories])  # np.asarray
    all_a = np.asarray([memo_i[1] for memo_i in memories])
    all_r = np.asarray([memo_i[2] for memo_i in memories])
    all_done = np.asarray([memo_i[3] for memo_i in memories])
    all_obs_ = np.asarray([memo_i[4] for memo_i in memories])

    all_obs_tensor = torch.as_tensor(all_obs, dtype=torch.float32)  # torch.float32 ?
    all_a_tensor = torch.as_tensor(all_a, dtype=torch.int64).unsqueeze(-1)
    all_r_tensor = torch.as_tensor(all_r, dtype=torch.float32).unsqueeze(-1)
    all_done_tensor = torch.as_tensor(all_done, dtype=torch.float32).unsqueeze(-1)
    all_obs__tensor = torch.as_tensor(all_obs_, dtype=torch.float32)

    # Compute Targets
    target_q_values = target_net(all_obs__tensor)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # ?

    targets = all_r_tensor + GAMMA * (1 - all_done_tensor) * max_target_q_values

    # Compute Loss
    q_values = online_net(all_obs_tensor)

    a_q_values = torch.gather(input=q_values, dim=1, index=all_a_tensor)

    loss = nn.functional.smooth_l1_loss(a_q_values, targets)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(online_net.state_dict())  # ?

    # Print the training progress
    if step % 1000 == 0:
        print()
        print("Step: {}".format(step))
        print("Avg reward: {}".format(np.mean(reward_buffer)))
